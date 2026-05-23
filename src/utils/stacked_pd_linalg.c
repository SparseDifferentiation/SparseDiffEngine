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
    permuted_dense **C_blocks =
        (permuted_dense **) sp_malloc(n_blocks * sizeof(permuted_dense *));
    for (int k = 0; k < n_blocks; k++)
    {
        C_blocks[k] = (permuted_dense *) copy_sparsity_pd_alloc(A->blocks[k]);
    }

    matrix *C = new_stacked_pd(A->base.m, A->base.n, n_blocks, C_blocks, NULL, NULL);
    free(C_blocks);
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
        (permuted_dense **) sp_malloc(n_blocks * sizeof(permuted_dense *));
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
typedef matrix *(*spd_per_block_alloc_fn)(const permuted_dense *blk,
                                          const void *ctx);

typedef void (*spd_per_block_fill_fn)(const permuted_dense *blk, const double *d,
                                      const void *ctx, permuted_dense *Ck);

static matrix *spd_blockwise_alloc_coalesce(const stacked_pd *spd_iter, int Cm,
                                            int Cn, spd_per_block_alloc_fn op,
                                            const void *ctx)
{
    int n_blocks = spd_iter->n_blocks;
    permuted_dense **partials =
        (permuted_dense **) sp_malloc(n_blocks * sizeof(permuted_dense *));
    for (int k = 0; k < n_blocks; k++)
    {
        partials[k] = (permuted_dense *) op(spd_iter->blocks[k], ctx);
    }
    matrix *raw = new_stacked_pd_unchecked(Cm, Cn, n_blocks, partials, NULL, NULL);
    free(partials);
    matrix *C = coalesce_spd_alloc_unchecked((stacked_pd *) raw);
    ((stacked_pd *) C)->pre_coalesce = (stacked_pd *) raw;
    return C;
}

static void spd_blockwise_fill_coalesce_accumulate(const stacked_pd *spd_iter,
                                                   const double *d, const void *ctx,
                                                   stacked_pd *C,
                                                   spd_per_block_fill_fn op)
{
    if (C->base.nnz == 0) return;

    stacked_pd *raw = C->pre_coalesce;
    for (int k = 0; k < spd_iter->n_blocks; k++)
    {
        op(spd_iter->blocks[k], d, ctx, raw->blocks[k]);
    }
    memset(C->base.x, 0, C->base.nnz * sizeof(double));
    coalesce_spd_fill_values_accumulate(raw, C);
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
   The output's `pre_coalesce` slot holds per-source scratch PDs (one symmetric
   PD per source block, row_perm = col_perm = C_k) that
   ATDA_spd_fill_values writes into.

   Note on per-block kernel choice: we use ATA_pd_alloc (rather than
   new_permuted_dense directly) because it also pre-sizes
   A->blocks[k]->kernel_dwork — that scratch buffer is what
   ATDA_pd_fill_values reads from during the fill phase. */
static matrix *wrapper_ATA_pd(const permuted_dense *Ak, const void *ctx)
{
    (void) ctx;
    return ATA_pd_alloc(Ak);
}

static void wrapper_ATDA_pd(const permuted_dense *Ak, const double *d,
                            const void *ctx, permuted_dense *Ck)
{
    (void) ctx;
    ATDA_pd_fill_values(Ak, d, Ck);
}

matrix *ATA_spd_alloc(const stacked_pd *A)
{
    return spd_blockwise_alloc_coalesce(A, A->base.n, A->base.n, wrapper_ATA_pd,
                                        NULL);
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
    const int **col_perms = (const int **) sp_malloc(n_blocks * sizeof(int *));
    int *lens = (int *) sp_malloc(n_blocks * sizeof(int));

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
    C_pd->kernel_iwork = (int *) sp_malloc(C_pd->kernel_iwork_size * sizeof(int));
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
static matrix *wrapper_BTA_pd_pd(const permuted_dense *Bk, const void *ctx)
{
    return BTA_pd_pd_alloc(Bk, (const permuted_dense *) ctx);
}

static void wrapper_BTDA_pd_pd(const permuted_dense *Bk, const double *d,
                               const void *ctx, permuted_dense *Ck)
{
    BTDA_pd_pd_fill_values(Bk, d, (const permuted_dense *) ctx, Ck);
}

matrix *BTA_spd_pd_alloc(const stacked_pd *B, const permuted_dense *A)
{
    return spd_blockwise_alloc_coalesce(B, B->base.n, A->base.n, wrapper_BTA_pd_pd,
                                        A);
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
static matrix *wrapper_BTA_pd_csc(const permuted_dense *Bk, const void *ctx)
{
    return BTA_pd_csc_alloc(Bk, (const CSC_matrix *) ctx);
}

static void wrapper_BTDA_pd_csc(const permuted_dense *Bk, const double *d,
                                const void *ctx, permuted_dense *Ck)
{
    BTDA_pd_csc_fill_values(Bk, d, (const CSC_matrix *) ctx, Ck);
}

matrix *BTA_spd_csc_alloc(const stacked_pd *B, const CSC_matrix *A)
{
    return spd_blockwise_alloc_coalesce(B, B->base.n, A->n, wrapper_BTA_pd_csc, A);
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
static matrix *wrapper_BTA_pd_spd(const permuted_dense *Bk, const void *ctx)
{
    return BTA_pd_spd_alloc(Bk, (const stacked_pd *) ctx);
}

static void wrapper_BTDA_pd_spd(const permuted_dense *Bk, const double *d,
                                const void *ctx, permuted_dense *Ck)
{
    BTDA_pd_spd_fill_values(Bk, d, (const stacked_pd *) ctx, Ck);
}

matrix *BTA_spd_spd_alloc(const stacked_pd *B, const stacked_pd *A)
{
    return spd_blockwise_alloc_coalesce(B, B->base.n, A->base.n, wrapper_BTA_pd_spd,
                                        A);
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
static matrix *wrapper_BTA_csc_pd(const permuted_dense *Ak, const void *ctx)
{
    return BTA_csc_pd_alloc((const CSC_matrix *) ctx, Ak);
}

static void wrapper_BTDA_csc_pd(const permuted_dense *Ak, const double *d,
                                const void *ctx, permuted_dense *Ck)
{
    BTDA_csc_pd_fill_values((const CSC_matrix *) ctx, d, Ak, Ck);
}

matrix *BTA_csc_spd_alloc(const CSC_matrix *B, const stacked_pd *A)
{
    return spd_blockwise_alloc_coalesce(A, B->n, A->base.n, wrapper_BTA_csc_pd, B);
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
