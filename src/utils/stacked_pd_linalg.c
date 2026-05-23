/*
 * Copyright 2026 Daniel Cederberg and William Zhang
 *
 * This file is part of the SparseDiffEngine project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "utils/stacked_pd_linalg.h"

#include "utils/cblas_wrapper.h"
#include "utils/iVec.h"
#include "utils/permuted_dense.h"
#include "utils/permuted_dense_linalg.h"
#include "utils/stacked_pd.h"
#include "utils/tracked_alloc.h"
#include "utils/utils.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

// ----------------------------------------------------------------------------------
// copy_sparsity and DA: C = D @ A where A is stacked_pd. The logic is blockwise.
// ----------------------------------------------------------------------------------
matrix *copy_sparsity_spd_alloc(const stacked_pd *A)
{
    int n_blocks = A->n_blocks;
    spd_per_block_shape *shapes = (spd_per_block_shape *) SP_CALLOC(
        n_blocks > 0 ? n_blocks : 1, sizeof(*shapes));
    for (int k = 0; k < n_blocks; k++)
    {
        shapes[k].m0 = A->blocks[k]->m0;
        shapes[k].n0 = A->blocks[k]->n0;
        shapes[k].row_perm = A->blocks[k]->row_perm;
        shapes[k].col_perm = A->blocks[k]->col_perm;
    }
    matrix *C =
        new_stacked_pd_from_shapes_unchecked(A->base.m, A->base.n, n_blocks, shapes);
    free(shapes);
    return C;
}

void DA_spd_fill_values(const double *d, const stacked_pd *A, stacked_pd *C)
{
    for (int k = 0; k < A->n_blocks; k++)
    {
        DA_pd_fill_values(d, A->blocks[k], C->blocks[k]);
    }
}

// ----------------------------------------------------------------------------------
// Transpose C = A^T where A is stacked_pd. Blockwise logic followed by coalescing:
// we first transpose each block independently, yielding a raw spd with the same
// block count and swapped row/col perms as the final output but potentially
// overlapping row perms. We then coalesce that raw spd to get the final output with
// disjoint row perms. We keep the raw spd around on `out->pre_coalesce` for reuse
// by `transpose_spd_fill_values` since its per-block transposes are also needed
// there.
// ----------------------------------------------------------------------------------
matrix *transpose_spd_alloc(const stacked_pd *A)
{
    int n_blocks = A->n_blocks;
    permuted_dense **AT_blocks =
        (permuted_dense **) SP_MALLOC(n_blocks * sizeof(permuted_dense *));
    for (int k = 0; k < n_blocks; k++)
    {
        AT_blocks[k] = (permuted_dense *) transpose_pd_alloc(A->blocks[k]);
    }

    /* Raw spd: dimensions swapped, identity src_block_idx_*; rows may
       overlap so use the unchecked constructor. */
    matrix *raw = new_stacked_pd_unchecked(A->base.n, A->base.m, n_blocks, AT_blocks,
                                           NULL, NULL);
    free(AT_blocks);

    matrix *C = coalesce_spd_alloc((stacked_pd *) raw);
    ((stacked_pd *) C)->pre_coalesce = (stacked_pd *) raw;
    return C;
}

void transpose_spd_fill_values(const stacked_pd *A, stacked_pd *C)
{
    stacked_pd *raw = C->pre_coalesce;
    for (int k = 0; k < A->n_blocks; k++)
    {
        transpose_pd_fill_values(A->blocks[k], raw->blocks[k]);
    }
    coalesce_spd_fill_values(raw, C);
}

// ====================================================================================
// Shared "spd blockwise + coalesce-accumulate" skeleton.
//
// Used by kernels of the shape C = f(spd_iter, X) where spd_iter is a
// stacked_pd we iterate, and each per-block partial f(spd_iter->blocks[k], X)
// is a PD. The partials' row_perms (and cells) may overlap across k, so we
// collect them via new_stacked_pd_unchecked, coalesce with the _unchecked
// variant, and use the accumulating scatter at fill time. The raw spd is
// stashed on out->pre_coalesce so fill can refresh per-block values without
// reallocating structure.
//
// (Cm, Cn) is the output global shape. For "spd is the BTA's left operand"
// callers (BTA_spd_pd / BTA_spd_csc / BTA_spd_spd) and the ATA self-case,
// Cm = spd_iter->base.n. For BTA_csc_spd we iterate A but output rows come
// from B_csc, so the caller passes Cm = B_csc->n explicitly.
//
// Kernel pairs build on this skeleton: BTA_spd_pd / BTDA_spd_pd,
// BTA_spd_csc / BTDA_spd_csc, BTA_spd_spd / BTDA_spd_spd,
// BTA_csc_spd / BTDA_csc_spd, and ATA_spd / ATDA_spd.
// ====================================================================================
/* Streaming alloc: each call walks the input blocks twice — once to gather
   shapes and max scratch sizes, once (inside coalesce_spd_alloc_from_shapes)
   to materialize the coalesced output. No per-block PD allocations. The
   matching fill op is the existing _fill_values, which writes into the
   shared scratch PD; the scatter loop here moves those values into the
   final coalesced output blocks.
 */
/* SP_MALLOC + memcpy an int array. Used by shape callbacks to give the
   per_block_shapes struct its own row_perm / col_perm copies (uniform
   ownership: stacked_pd_free frees them unconditionally). */
static int *clone_int_array(const int *src, int n)
{
    int *dst = (int *) SP_MALLOC((n > 0 ? n : 1) * sizeof(int));
    if (n > 0) memcpy(dst, src, n * sizeof(int));
    return dst;
}

typedef spd_per_block_shape (*spd_per_block_shape_fn)(const permuted_dense *blk,
                                                      const void *ctx);

typedef void (*spd_per_block_fill_fn)(const permuted_dense *blk, const double *d,
                                      const void *ctx, permuted_dense *Ck);

/* Build the inverse of src_block_idx_p / src_block_idx (CSR map from
   output blocks to source blocks): a CSR map from each source block to
   the output blocks it contributes to. */
static void build_src_to_outs_inverse(int n_input_blocks, int n_out_blocks,
                                      const int *src_block_idx_p,
                                      const int *src_block_idx_data, int **p_out,
                                      int **data_out)
{
    int total = src_block_idx_p[n_out_blocks];

    int *p = (int *) SP_CALLOC(n_input_blocks + 1, sizeof(int));
    for (int t = 0; t < total; t++)
    {
        p[src_block_idx_data[t] + 1]++;
    }
    for (int k = 0; k < n_input_blocks; k++)
    {
        p[k + 1] += p[k];
    }

    int *data = (int *) SP_MALLOC((total > 0 ? total : 1) * sizeof(int));
    int *cursor = (int *) SP_CALLOC(n_input_blocks, sizeof(int));
    for (int out_k = 0; out_k < n_out_blocks; out_k++)
    {
        int lo = src_block_idx_p[out_k];
        int hi = src_block_idx_p[out_k + 1];
        for (int t = lo; t < hi; t++)
        {
            int src_k = src_block_idx_data[t];
            data[p[src_k] + cursor[src_k]++] = out_k;
        }
    }
    free(cursor);

    *p_out = p;
    *data_out = data;
}

static matrix *spd_blockwise_alloc_coalesce(const stacked_pd *spd_iter, int Cm,
                                            int Cn, spd_per_block_shape_fn shape_fn,
                                            const void *ctx)
{
    int n_blocks = spd_iter->n_blocks;

    /* Pass 1: shape pass — calls into the per-wrapper shape callback for
       each input block. Each callback (a) computes the would-be output
       shape, (b) pre-sizes any input-side kernel_dwork the matching fill
       op will read at fill time, and (c) reports the dest-side scratch
       sizes the matching fill op needs from its destination PD. */
    spd_per_block_shape *shapes = (spd_per_block_shape *) SP_MALLOC(
        (n_blocks > 0 ? n_blocks : 1) * sizeof(spd_per_block_shape));
    size_t max_scratch_X = 0;
    size_t max_scratch_dwork = 0;
    size_t max_scratch_iwork = 0;
    for (int k = 0; k < n_blocks; k++)
    {
        shapes[k] = shape_fn(spd_iter->blocks[k], ctx);
        size_t sz = (size_t) shapes[k].m0 * (size_t) shapes[k].n0;
        if (sz > max_scratch_X) max_scratch_X = sz;
        if (shapes[k].scratch_dwork_needed > max_scratch_dwork)
            max_scratch_dwork = shapes[k].scratch_dwork_needed;
        if (shapes[k].scratch_iwork_needed > max_scratch_iwork)
            max_scratch_iwork = shapes[k].scratch_iwork_needed;
    }

    /* Pass 2: symbolic coalesce + output materialization. Output blocks
       are views into one shared X buffer that the output stacked_pd owns. */
    matrix *C_mat =
        coalesce_spd_alloc_from_shapes_unchecked(shapes, n_blocks, Cm, Cn);
    stacked_pd *C = (stacked_pd *) C_mat;

    /* Build the inverse src→outs map so fill can iterate "for each input
       block, scatter into the output blocks it contributes to". */
    build_src_to_outs_inverse(n_blocks, C->n_blocks, C->src_block_idx_p,
                              C->src_block_idx, &C->src_to_outs_p,
                              &C->src_to_outs_data);

    /* Scratch buffers for the streaming fill — sized once at max across
       blocks; fill never grows them. */
    C->scratch_X_capacity = max_scratch_X;
    C->scratch_X = (double *) SP_MALLOC((max_scratch_X > 0 ? max_scratch_X : 1) *
                                        sizeof(double));
    C->scratch_dwork_capacity = max_scratch_dwork;
    C->scratch_dwork = max_scratch_dwork > 0
                           ? (double *) SP_MALLOC(max_scratch_dwork * sizeof(double))
                           : NULL;
    C->scratch_iwork_capacity = max_scratch_iwork;
    C->scratch_iwork = max_scratch_iwork > 0
                           ? (int *) SP_MALLOC(max_scratch_iwork * sizeof(int))
                           : NULL;

    /* Inverse maps for the scratch PD. Sized at the global (m, n); reset
       and repopulated each iteration from per_block_shapes[k].row_perm /
       col_perm. Some fill ops (BTA_pd_spd) read scratch->col_inv to map
       global col indices to scratch-local positions. */
    C->scratch_row_inv = (int *) SP_MALLOC(Cm * sizeof(int));
    C->scratch_col_inv = (int *) SP_MALLOC(Cn * sizeof(int));

    /* Keep shapes on the output so fill can re-aim the scratch PD. The
       output stacked_pd takes ownership (and frees in its destructor). */
    C->per_block_shapes = shapes;
    C->n_input_blocks = n_blocks;

    return &C->base;
}

static void spd_blockwise_fill_coalesce_accumulate(const stacked_pd *spd_iter,
                                                   const double *d, const void *ctx,
                                                   stacked_pd *C,
                                                   spd_per_block_fill_fn op)
{
    if (C->base.nnz == 0) return;

    /* Stack-allocated scratch PD view, re-aimed at each input block's
       shape each iteration. No SP_MALLOC anywhere in this loop —
       per the project invariant on _fill_values. */
    permuted_dense scratch = {0};
    scratch.X = C->scratch_X;
    scratch.base.x = C->scratch_X;
    scratch.owns_X = false;
    scratch.kernel_dwork = C->scratch_dwork;
    scratch.kernel_dwork_size = C->scratch_dwork_capacity;
    scratch.kernel_iwork = C->scratch_iwork;
    scratch.kernel_iwork_size = C->scratch_iwork_capacity;
    scratch.row_inv = C->scratch_row_inv;
    scratch.col_inv = C->scratch_col_inv;
    scratch.base.m = C->base.m;
    scratch.base.n = C->base.n;

    memset(C->base.x, 0, C->base.nnz * sizeof(double));

    for (int k = 0; k < spd_iter->n_blocks; k++)
    {
        const spd_per_block_shape *sh = &C->per_block_shapes[k];
        scratch.m0 = sh->m0;
        scratch.n0 = sh->n0;
        scratch.row_perm = (int *) sh->row_perm;
        scratch.col_perm = (int *) sh->col_perm;
        scratch.base.nnz = sh->m0 * sh->n0;

        /* Repopulate row_inv / col_inv for this block. memset(-1) works on
           int arrays because two's-complement makes 0xFFFFFFFF == -1. */
        memset(scratch.row_inv, -1, (size_t) C->base.m * sizeof(int));
        memset(scratch.col_inv, -1, (size_t) C->base.n * sizeof(int));
        for (int ii = 0; ii < sh->m0; ii++)
        {
            scratch.row_inv[sh->row_perm[ii]] = ii;
        }
        for (int jj = 0; jj < sh->n0; jj++)
        {
            scratch.col_inv[sh->col_perm[jj]] = jj;
        }

        /* Some fill ops (BTA_pd_spd) read leading metadata from the
           destination PD's kernel_iwork[0..init_len) at fill time. The
           shape callback computed those values at alloc time; copy them
           into the shared scratch_iwork before invoking the op. */
        for (int j = 0; j < sh->iwork_init_len; j++)
        {
            scratch.kernel_iwork[j] = sh->iwork_init[j];
        }

        op(spd_iter->blocks[k], d, ctx, &scratch);

        /* Scatter scratch into each output block this source contributes to. */
        int lo = C->src_to_outs_p[k];
        int hi = C->src_to_outs_p[k + 1];
        for (int t = lo; t < hi; t++)
        {
            int out_idx = C->src_to_outs_data[t];
            scatter_one_source_into_one_output_accumulate(&scratch,
                                                          C->blocks[out_idx]);
        }
    }
}

// ------------------------------------------------------------------------------------
// C = ATDA for stacked_pd A. Let A = [A1; A2; A3] where Ai has n columns (the same
// number as A). Then ATDA = A1^T D1 A1 + A2^T D2 A2 + A3^T D3 A3. Term i and j
// overlap if Ai and Aj have overlapping column entries. We therefore first compute
// Ai^T Di Ai and then take care of the coalescing.
// ------------------------------------------------------------------------------------

/*
A ^ T A decomposes as Σ_k B_k ^ T B_k, where summands with overlapping col_perms(C_k)
share cells.The output groups cols of A by signature sig_C(c) = {k: c ∈ C_k}; each
   unique signature becomes one output PD with row_perm = group cols
   and col_perm = ⋃ C_k for k in the signature. No structural zeros.

   The shape callback returns the (n0, n0) per-block Gram shape (col_perm =
   Ak->col_perm) and pre-sizes Ak's kernel_dwork — both the same setup
   ATA_pd_alloc would do, minus the per-block PD allocation. The matching
   fill (ATDA_pd_fill_values) reads Ak->kernel_dwork and writes into a
   caller-supplied destination's X; no destination kernel state is read. */
static spd_per_block_shape wrapper_ATA_pd_shape(const permuted_dense *Ak,
                                                const void *ctx)
{
    (void) ctx;
    /* Same pre-sizing ATA_pd_alloc does at permuted_dense_linalg.c:83. */
    permuted_dense_ensure_kernel_dwork(Ak, (size_t) Ak->m0 * Ak->n0);
    spd_per_block_shape shape = {0};
    shape.m0 = Ak->n0;
    shape.n0 = Ak->n0;
    shape.row_perm = clone_int_array(Ak->col_perm, Ak->n0);
    shape.col_perm = clone_int_array(Ak->col_perm, Ak->n0);
    return shape;
}

static void wrapper_ATDA_pd(const permuted_dense *Ak, const double *d,
                            const void *ctx, permuted_dense *Ck)
{
    (void) ctx;
    ATDA_pd_fill_values(Ak, d, Ck);
}

matrix *ATA_spd_alloc(const stacked_pd *A)
{
    return spd_blockwise_alloc_coalesce(A, A->base.n, A->base.n,
                                        wrapper_ATA_pd_shape, NULL);
}

void ATDA_spd_fill_values(const stacked_pd *A, const double *d, stacked_pd *C)
{
    spd_blockwise_fill_coalesce_accumulate(A, d, NULL, C, wrapper_ATDA_pd);
}

// ---------------------------------------------------------------------------------
// BTA_pd_spd: C = B^T @ A where B is a permuted_dense and A is a stacked_pd. C is
// also permuted_dense. C has row_perm = B->col_perm, and its col_perm is the sorted
// union of the col_perms of the contributing A-blocks. An A-block contributes if its
// row_perm overlaps B's row_perm. We compute this as
// C = \sum_k (B[r_k, :])^T @ A_k, where r_k = A_k->row_perm.
//
// This is the canonical (cache-friendly) kernel: the gather of B is a single
// memcpy per gathered row, since B is row-major. BA_pd_spd_* is a thin wrapper
// that transposes its B argument and calls into this kernel.
// ----------------------------------------------------------------------------------
matrix *BTA_pd_spd_alloc(const permuted_dense *B, const stacked_pd *A)
{
    int n_blocks = A->n_blocks;

    // ------------------------------------------------------------------------------
    // Find column permutations of C. An A-block contributes if its row_perm overlaps
    // B's row_perm.
    // ------------------------------------------------------------------------------
    const int **col_perms = (const int **) SP_MALLOC(n_blocks * sizeof(int *));
    int *lens = (int *) SP_MALLOC(n_blocks * sizeof(int));

    int n_contributing_A_blocks = 0;
    for (int k = 0; k < n_blocks; k++)
    {
        permuted_dense *Ak = A->blocks[k];
        if (has_overlap(B->row_perm, B->m0, Ak->row_perm, Ak->m0, 0))
        {
            col_perms[n_contributing_A_blocks] = Ak->col_perm;
            lens[n_contributing_A_blocks] = Ak->n0;
            n_contributing_A_blocks++;
        }
    }

    iVec *col_union = iVec_new(8);
    sorted_union_int_arrays(col_perms, lens, n_contributing_A_blocks, col_union);

    // -------------------------------------------------------------------------------
    // Allocate C with row_perm = B->col_perm and col_perm = col_union.
    // -------------------------------------------------------------------------------
    matrix *C = new_permuted_dense(B->base.n, A->base.n, B->n0, col_union->len,
                                   B->col_perm, col_union->data, NULL);

    iVec_free(col_union);
    free(col_perms);
    free(lens);

    // -------------------------------------------------------------------------------
    // Set up workspace. We need:
    // 1. (s_max, max_n0_A) - worst-case intersection size and worst-case A-block
    //    column count across all k. Used to size all other scratch buffers.
    // 2. (idx_B, idx_A) - arrays to find rows of B that overlap with rows of A_k.
    // 3. Bg and Ag to hold the gathered rows of B and gathered rows of Ak.
    // 4. Cg = Bg^T @ Ag which is then scattered into C.
    // 5. out_cols - work array for cache-friendly scattering
    // -------------------------------------------------------------------------------
    permuted_dense *C_pd = (permuted_dense *) C;
    int s_max = 0;
    int max_n0_A = 0;
    for (int k = 0; k < n_blocks; k++)
    {
        permuted_dense *Ak = A->blocks[k];
        int s = B->m0 < Ak->m0 ? B->m0 : Ak->m0;
        if (s > s_max) s_max = s;
        if (Ak->n0 > max_n0_A) max_n0_A = Ak->n0;
    }

    C_pd->kernel_iwork_size = (size_t) 2 + (size_t) 2 * s_max + (size_t) max_n0_A;
    C_pd->kernel_iwork = (int *) SP_MALLOC(C_pd->kernel_iwork_size * sizeof(int));
    C_pd->kernel_iwork[0] = s_max;
    C_pd->kernel_iwork[1] = max_n0_A;

    size_t dwork_total = (size_t) B->n0 * s_max + (size_t) s_max * max_n0_A +
                         (size_t) B->n0 * max_n0_A;
    permuted_dense_ensure_kernel_dwork(C_pd, dwork_total);

    return C;
}

void BTA_pd_spd_fill_values(const permuted_dense *B, const stacked_pd *A,
                            permuted_dense *C)
{
    /* return if C is empty */
    if (C->base.nnz == 0)
    {
        return;
    }

    /* reset values of C */
    memset(C->X, 0, (size_t) C->m0 * C->n0 * sizeof(double));

    /* partition workspace according to how it was set up */
    int s_max = C->kernel_iwork[0];
    int max_n0_A = C->kernel_iwork[1];
    int *idx_B = C->kernel_iwork + 2;
    int *idx_A = idx_B + s_max;
    int *out_cols = idx_A + s_max;
    double *Bg = C->kernel_dwork;
    double *Ag = Bg + (size_t) s_max * B->n0;
    double *Cg = Ag + (size_t) s_max * max_n0_A;

    /* for each block Ak of A */
    for (int k = 0; k < A->n_blocks; k++)
    {
        permuted_dense *Ak = A->blocks[k];

        /* find rows of B that overlap with rows of A_k */
        int s = sorted_intersect_indices(B->row_perm, B->m0, Ak->row_perm, Ak->m0,
                                         idx_B, idx_A);
        if (s == 0)
        {
            continue;
        }

        /* Bg = B[idx_B, :] where idx_B contains the overlapping row indices. One
           contiguous memcpy per gathered row (row-major-friendly). */
        for (int p = 0; p < s; p++)
        {
            memcpy(Bg + p * B->n0, B->X + idx_B[p] * B->n0, B->n0 * sizeof(double));
        }

        /* Ag = A[idx_A, :] where idx_A contains the overlapping row indices */
        for (int p = 0; p < s; p++)
        {
            memcpy(Ag + p * Ak->n0, Ak->X + idx_A[p] * Ak->n0,
                   Ak->n0 * sizeof(double));
        }

        /* Cg = Bg^T @ Ag. Bg is (s, B->n0) row-major (lda = B->n0); we want
           output Cg = (B->n0, Ak->n0). */
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, B->n0, Ak->n0, s, 1.0,
                    Bg, B->n0, Ag, Ak->n0, 0.0, Cg, Ak->n0);

        /* precompute scatter positions once per block */
        for (int j = 0; j < Ak->n0; j++)
        {
            out_cols[j] = C->col_inv[Ak->col_perm[j]];
        }

        /* C += Cg  (scatter + add into C) */
        for (int i = 0; i < B->n0; i++)
        {
            double *C_row = C->X + i * C->n0;
            const double *Cg_row = Cg + i * Ak->n0;
            for (int j = 0; j < Ak->n0; j++)
            {
                C_row[out_cols[j]] += Cg_row[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// BTDA_pd_spd: C = B^T @ diag(d) @ A. No separate alloc — output sparsity
// is identical to BTA_pd_spd (D doesn't add/remove nonzeros), so callers
// pre-allocate C via BTA_pd_spd_alloc, same convention as BTA_pd_pd /
// BTDA_pd_pd.
// ---------------------------------------------------------------------------------
void BTDA_pd_spd_fill_values(const permuted_dense *B, const double *d,
                             const stacked_pd *A, permuted_dense *C)
{
    /* skip if C is empty (no contributing A-blocks) */
    if (C->base.nnz == 0)
    {
        return;
    }

    /* TODO: must remove this allocation. Very important. The DA
       intermediate spd is allocated and freed on every Hessian
       iteration — violates the no-alloc-in-fill policy. Fix is to
       fold diag(d) directly into BTA_pd_spd_fill_values (either via a
       shared internal helper that takes an optional d, or by stashing
       a persistent DA scratch on C via a new aux slot — mirror of the
       transpose_cache pattern). */
    /* C = BT @ (DA) */
    stacked_pd *DA = (stacked_pd *) copy_sparsity_spd_alloc(A);
    DA_spd_fill_values(d, A, DA);
    BTA_pd_spd_fill_values(B, DA, C);
    free_matrix(&DA->base);
}

// ---------------------------------------------------------------------------------
// BTA_spd_pd / BTDA_spd_pd: C = B^T @ (diag(d) @) A where B is stacked_pd, A is
// permuted_dense. Output C is stacked_pd. We implement this using blockwise
// logic followed by coalescing. The blockwise logic is not obvious
// mathematically, but it is correct when we think of each of B's blocks as a
// global matrix and B as a sum of these blocks.
//
// B = B1 + B2 + B3 where each Bi is a global permuted dense. Then
// C = B^T D A = C1 + C2 + C3 where Ci = Bi^T D A.
//
// TODO: each BTDA_pd_pd_fill_values call internally allocates a DA intermediate
// (see permuted_dense_linalg.c BTDA_pd_pd_fill_values). That means the BTDA
// variant here allocates n_blocks DA temps per fill, all on the hot Hessian
// path. Must be fixed — same future remedy as the per-block BTDA: fold
// diag(d) directly into BTA_pd_pd's gather step.
// ---------------------------------------------------------------------------------
/* Same shape derivation as BTA_pd_pd_alloc (permuted_dense_linalg.c:105),
   minus the per-block PD allocation. Preserves the input-side dwork
   pre-sizing on both Bk and A, and reports the destination's iwork need
   (2 * s_max) so the streaming alloc sizes the shared scratch_iwork once
   at the max across blocks. */
static spd_per_block_shape wrapper_BTA_pd_pd_shape(const permuted_dense *Bk,
                                                   const void *ctx)
{
    const permuted_dense *A = (const permuted_dense *) ctx;
    spd_per_block_shape shape = {0};

    if (!has_overlap(A->row_perm, A->m0, Bk->row_perm, Bk->m0, 0))
    {
        /* empty output block */
        return shape;
    }

    int s_max = A->m0 < Bk->m0 ? A->m0 : Bk->m0;
    permuted_dense_ensure_kernel_dwork(A, (size_t) s_max * A->n0);
    permuted_dense_ensure_kernel_dwork(Bk, (size_t) s_max * Bk->n0);

    shape.m0 = Bk->n0;
    shape.n0 = A->n0;
    shape.row_perm = clone_int_array(Bk->col_perm, Bk->n0);
    shape.col_perm = clone_int_array(A->col_perm, A->n0);
    shape.scratch_iwork_needed = (size_t) 2 * s_max;
    return shape;
}

static void wrapper_BTDA_pd_pd(const permuted_dense *Bk, const double *d,
                               const void *ctx, permuted_dense *Ck)
{
    BTDA_pd_pd_fill_values(Bk, d, (const permuted_dense *) ctx, Ck);
}

matrix *BTA_spd_pd_alloc(const stacked_pd *B, const permuted_dense *A)
{
    return spd_blockwise_alloc_coalesce(B, B->base.n, A->base.n,
                                        wrapper_BTA_pd_pd_shape, A);
}

void BTDA_spd_pd_fill_values(const stacked_pd *B, const double *d,
                             const permuted_dense *A, stacked_pd *C)
{
    spd_blockwise_fill_coalesce_accumulate(B, d, A, C, wrapper_BTDA_pd_pd);
}

// ---------------------------------------------------------------------------------
// BTA_spd_csc / BTDA_spd_csc: C = B^T @ (diag(d) @) A where B is stacked_pd,
// A is CSC. Output C is stacked_pd. Same two-stage skeleton as BTA_spd_pd /
// BTDA_spd_pd, just with BTA_pd_csc / BTDA_pd_csc as the per-block kernel.
// Unlike the pd-on-the-right path, BTDA_pd_csc reuses the per-block
// B_k->kernel_dwork sized by BTA_pd_csc_alloc — no temp DA allocation.
// ---------------------------------------------------------------------------------
/* Same shape derivation as BTA_pd_csc_alloc (permuted_dense_linalg.c:391),
   minus the per-block PD allocation. col_perm is computed from A's nnz
   pattern intersected with Bk->row_inv — freshly SP_MALLOC'd and owned by
   the shape (freed when the per_block_shapes lifetime ends). */
static spd_per_block_shape wrapper_BTA_pd_csc_shape(const permuted_dense *Bk,
                                                    const void *ctx)
{
    const CSC_matrix *A = (const CSC_matrix *) ctx;
    spd_per_block_shape shape = {0};

    iVec *col_active = iVec_new(8);
    for (int j = 0; j < A->n; j++)
    {
        int start = A->p[j];
        int len = A->p[j + 1] - start;
        /* Inline of permuted_dense_linalg.c's idxs_hits_set: any row index
           of column j that lies in Bk's row_perm (Bk->row_inv != -1)
           contributes — column j of the output is active. */
        bool hits = false;
        for (int e = 0; e < len; e++)
        {
            if (Bk->row_inv[A->i[start + e]] != -1)
            {
                hits = true;
                break;
            }
        }
        if (hits)
        {
            iVec_append(col_active, j);
        }
    }

    int n0 = col_active->len;
    int *col_perm_owned = clone_int_array(col_active->data, n0);
    iVec_free(col_active);

    /* Pre-size Bk's dwork (same as BTA_pd_csc_alloc:413). */
    permuted_dense_ensure_kernel_dwork(Bk, (size_t) Bk->m0 * Bk->n0);

    shape.m0 = Bk->n0;
    shape.n0 = n0;
    shape.row_perm = clone_int_array(Bk->col_perm, Bk->n0);
    shape.col_perm = col_perm_owned;
    return shape;
}

static void wrapper_BTDA_pd_csc(const permuted_dense *Bk, const double *d,
                                const void *ctx, permuted_dense *Ck)
{
    BTDA_pd_csc_fill_values(Bk, d, (const CSC_matrix *) ctx, Ck);
}

matrix *BTA_spd_csc_alloc(const stacked_pd *B, const CSC_matrix *A)
{
    return spd_blockwise_alloc_coalesce(B, B->base.n, A->n, wrapper_BTA_pd_csc_shape,
                                        A);
}

void BTDA_spd_csc_fill_values(const stacked_pd *B, const double *d,
                              const CSC_matrix *A, stacked_pd *C)
{
    spd_blockwise_fill_coalesce_accumulate(B, d, A, C, wrapper_BTDA_pd_csc);
}

// ---------------------------------------------------------------------------------
// BTA_spd_spd / BTDA_spd_spd: C = B^T @ (diag(d) @) A where both B and A are
// stacked_pd. Output C is stacked_pd. Same skeleton as BTA_spd_pd/csc — per
// B-block partial is B_k^T @ A_spd via BTA_pd_spd / BTDA_pd_spd, which
// internally handles the iteration over A's blocks. The "double nesting"
// over (B_k, A_j) pairs is hidden inside BTA_pd_spd_alloc.
// ---------------------------------------------------------------------------------
/* Same shape derivation as BTA_pd_spd_alloc (this file, ~L344), minus
   the per-block PD allocation. col_perm is the sorted union of
   overlapping A-blocks' col_perms — freshly SP_MALLOC'd and owned by the
   shape. The fill (BTDA_pd_spd_fill_values) reads C->kernel_iwork[0..1]
   for s_max and max_n0_A; we store those in iwork_init so the streaming
   fill can seed the shared scratch_iwork before each per-block call. */
static spd_per_block_shape wrapper_BTA_pd_spd_shape(const permuted_dense *Bk,
                                                    const void *ctx)
{
    const stacked_pd *A = (const stacked_pd *) ctx;
    int n_blocks_A = A->n_blocks;
    spd_per_block_shape shape = {0};

    const int **col_perms = (const int **) SP_MALLOC(n_blocks_A * sizeof(int *));
    int *lens = (int *) SP_MALLOC(n_blocks_A * sizeof(int));

    int n_contributing_A_blocks = 0;
    int s_max = 0;
    int max_n0_A = 0;
    for (int k = 0; k < n_blocks_A; k++)
    {
        permuted_dense *Ak = A->blocks[k];
        int s = Bk->m0 < Ak->m0 ? Bk->m0 : Ak->m0;
        if (s > s_max) s_max = s;
        if (Ak->n0 > max_n0_A) max_n0_A = Ak->n0;
        if (has_overlap(Bk->row_perm, Bk->m0, Ak->row_perm, Ak->m0, 0))
        {
            col_perms[n_contributing_A_blocks] = Ak->col_perm;
            lens[n_contributing_A_blocks] = Ak->n0;
            n_contributing_A_blocks++;
        }
    }

    iVec *col_union = iVec_new(8);
    sorted_union_int_arrays(col_perms, lens, n_contributing_A_blocks, col_union);
    free(col_perms);
    free(lens);

    int n0 = col_union->len;
    int *col_perm_owned = clone_int_array(col_union->data, n0);
    iVec_free(col_union);

    shape.m0 = Bk->n0;
    shape.n0 = n0;
    shape.row_perm = clone_int_array(Bk->col_perm, Bk->n0);
    shape.col_perm = col_perm_owned;
    shape.scratch_iwork_needed = (size_t) 2 + (size_t) 2 * s_max + (size_t) max_n0_A;
    shape.scratch_dwork_needed = (size_t) Bk->n0 * (size_t) s_max +
                                 (size_t) s_max * (size_t) max_n0_A +
                                 (size_t) Bk->n0 * (size_t) max_n0_A;
    shape.iwork_init[0] = s_max;
    shape.iwork_init[1] = max_n0_A;
    shape.iwork_init_len = 2;
    return shape;
}

static void wrapper_BTDA_pd_spd(const permuted_dense *Bk, const double *d,
                                const void *ctx, permuted_dense *Ck)
{
    BTDA_pd_spd_fill_values(Bk, d, (const stacked_pd *) ctx, Ck);
}

matrix *BTA_spd_spd_alloc(const stacked_pd *B, const stacked_pd *A)
{
    return spd_blockwise_alloc_coalesce(B, B->base.n, A->base.n,
                                        wrapper_BTA_pd_spd_shape, A);
}

void BTDA_spd_spd_fill_values(const stacked_pd *B, const double *d,
                              const stacked_pd *A, stacked_pd *C)
{
    spd_blockwise_fill_coalesce_accumulate(B, d, A, C, wrapper_BTDA_pd_spd);
}

// ---------------------------------------------------------------------------------
// BTA_csc_spd / BTDA_csc_spd: C = B^T @ (diag(d) @) A where B is CSC and A is
// stacked_pd. Output C is stacked_pd.
//
// Math: A = Σ_k S_{r_k} A_k S_{c_k}^T, so
//   B^T A = Σ_k B^T A_k
// where each per-block partial B^T @ A_k is computed by BTA_csc_pd. Unlike
// the spd-on-the-left family, we iterate A's blocks (not B's), and output
// rows come from B's cols — hence the explicit Cm = B->n passed to the
// shared helper. Cells overlap across k (both row_perms and col_perms can
// collide), so we use the unchecked raw + accumulating coalesce.
// ---------------------------------------------------------------------------------
/* Same shape derivation as BTA_csc_pd_alloc (permuted_dense_linalg.c:444),
   minus the per-block PD allocation. row_perm is computed from B's nnz
   pattern intersected with Ak->row_inv — freshly SP_MALLOC'd and owned. */
static spd_per_block_shape wrapper_BTA_csc_pd_shape(const permuted_dense *Ak,
                                                    const void *ctx)
{
    const CSC_matrix *B = (const CSC_matrix *) ctx;
    spd_per_block_shape shape = {0};

    iVec *row_active = iVec_new(8);
    for (int i = 0; i < B->n; i++)
    {
        int start = B->p[i];
        int len = B->p[i + 1] - start;
        bool hits = false;
        for (int e = 0; e < len; e++)
        {
            if (Ak->row_inv[B->i[start + e]] != -1)
            {
                hits = true;
                break;
            }
        }
        if (hits)
        {
            iVec_append(row_active, i);
        }
    }

    int m0 = row_active->len;
    int *row_perm_owned = clone_int_array(row_active->data, m0);
    iVec_free(row_active);

    /* Pre-size Ak's dwork (same as BTA_csc_pd_alloc:467). */
    permuted_dense_ensure_kernel_dwork(Ak, (size_t) Ak->m0 * Ak->n0);

    shape.m0 = m0;
    shape.n0 = Ak->n0;
    shape.row_perm = row_perm_owned;
    shape.col_perm = clone_int_array(Ak->col_perm, Ak->n0);
    return shape;
}

static void wrapper_BTDA_csc_pd(const permuted_dense *Ak, const double *d,
                                const void *ctx, permuted_dense *Ck)
{
    BTDA_csc_pd_fill_values((const CSC_matrix *) ctx, d, Ak, Ck);
}

matrix *BTA_csc_spd_alloc(const CSC_matrix *B, const stacked_pd *A)
{
    return spd_blockwise_alloc_coalesce(A, B->n, A->base.n, wrapper_BTA_csc_pd_shape,
                                        B);
}

void BTDA_csc_spd_fill_values(const CSC_matrix *B, const double *d,
                              const stacked_pd *A, stacked_pd *C)
{
    spd_blockwise_fill_coalesce_accumulate(A, d, B, C, wrapper_BTDA_csc_pd);
}

// ---------------------------------------------------------------------------------
// BA_pd_spd: C = B @ A where B is permuted_dense and A is stacked_pd. Thin
// wrapper over the canonical BTA_pd_spd_* kernel: use B's lazily-cached
// transpose and call BTA. The cache is populated on first call (in alloc)
// and reused across subsequent fills.
//
// Contract: B's perms must be immutable between alloc and fill (the cache
// records B's perms at alloc time and is not re-validated at fill). For
// callers where B's perms change between calls — notably the kron-spd path
// that reuses a mutating scratch — bypass this wrapper and call
// BTA_pd_spd_* directly. BA_dense_kron_spd does exactly that
// (stacked_pd_kron_linalg.c) and is the only such caller today.
// ---------------------------------------------------------------------------------
matrix *BA_pd_spd_alloc(const permuted_dense *B, const stacked_pd *A)
{
    permuted_dense *BT = permuted_dense_ensure_transpose_cache(B);
    return BTA_pd_spd_alloc(BT, A);
}

void BA_pd_spd_fill_values(const permuted_dense *B, const stacked_pd *A,
                           permuted_dense *C)
{
    permuted_dense *BT = B->transpose_cache;
    transpose_pd_fill_values(B, BT);
    BTA_pd_spd_fill_values(BT, A, C);
}

// ====================================================================================
// "B is stacked_pd on the LEFT" BA kernels: C = B @ A for A in {CSC, PD, spd}.
// Output is always stacked_pd, with each output block = B_k @ A. Because B's
// blocks have disjoint row_perms (spd invariant) and the output rows come
// directly from B's rows, the output's blocks are also disjoint by construction
// — no coalesce needed. We use spd_map_filter_blocks (stacked_pd.h) which builds
// a valid stacked_pd directly, filtering out empty contributions.
//
// Not used in any production atom today (stacked_pds appear as derived Jacobians,
// never as constant left operands of B @ A). Kept as siblings of the BTA_spd_*
// family for API symmetry and because BA_spd_matrices is the natural counterpart
// to BTA_spd_matrices. Cost of carrying them is small.
// ====================================================================================
static matrix *wrapper_BA_pd_csc(permuted_dense *Bk, const void *ctx)
{
    return BA_pd_csc_alloc(Bk, (const CSC_matrix *) ctx);
}

matrix *BA_spd_csc_alloc(const stacked_pd *B, const CSC_matrix *A)
{
    return spd_map_filter_blocks(B, B->base.m, A->n, wrapper_BA_pd_csc, A);
}

void BA_spd_csc_fill_values(const stacked_pd *B, const CSC_matrix *A, stacked_pd *C)
{
    for (int k = 0; k < C->n_blocks; k++)
    {
        /* Bq is the block in B that contributes to Ck */
        int q = C->src_block_idx[C->src_block_idx_p[k]];
        const permuted_dense *Bq = B->blocks[q];
        BA_pd_csc_fill_values(Bq->X, Bq->n0, Bq->col_inv, A, C->blocks[k]);
    }
}

static matrix *wrapper_BA_pd_pd(permuted_dense *Bk, const void *ctx)
{
    return BA_pd_pd_alloc(Bk, (const permuted_dense *) ctx);
}

matrix *BA_spd_pd_alloc(const stacked_pd *B, const permuted_dense *A)
{
    return spd_map_filter_blocks(B, B->base.m, A->base.n, wrapper_BA_pd_pd, A);
}

void BA_spd_pd_fill_values(const stacked_pd *B, const permuted_dense *A,
                           stacked_pd *C)
{
    for (int k = 0; k < C->n_blocks; k++)
    {
        int q = C->src_block_idx[C->src_block_idx_p[k]];
        const permuted_dense *Bq = B->blocks[q];
        BA_pd_pd_fill_values(Bq, A, C->blocks[k]);
    }
}

static matrix *wrapper_BA_pd_spd(permuted_dense *Bk, const void *ctx)
{
    return BA_pd_spd_alloc(Bk, (const stacked_pd *) ctx);
}

matrix *BA_spd_spd_alloc(const stacked_pd *B, const stacked_pd *A)
{
    return spd_map_filter_blocks(B, B->base.m, A->base.n, wrapper_BA_pd_spd, A);
}

void BA_spd_spd_fill_values(const stacked_pd *B, const stacked_pd *A, stacked_pd *C)
{
    for (int k = 0; k < C->n_blocks; k++)
    {
        int q = C->src_block_idx[C->src_block_idx_p[k]];
        const permuted_dense *Bq = B->blocks[q];
        BA_pd_spd_fill_values(Bq, A, C->blocks[k]);
    }
}
