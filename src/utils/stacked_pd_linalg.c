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
        (permuted_dense **) SP_MALLOC(n_blocks * sizeof(permuted_dense *));
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
   ATDA_spd_fill_values writes into. */
matrix *ATA_spd_alloc(const stacked_pd *A)
{
    int n = A->base.n;
    int n_blocks = A->n_blocks;

    /* Pseudo-spd: per source block k, a square PD with
       row_perm = col_perm = C_k and uninitialized X. We use the
       PD-level ATA_pd_alloc here (rather than new_permuted_dense
       directly) because it also pre-sizes A->blocks[k]->kernel_dwork — that
       scratch buffer is what ATDA_pd_fill_values reads from during the
       fill phase. */
    permuted_dense **scratch_blocks =
        (permuted_dense **) SP_MALLOC(n_blocks * sizeof(permuted_dense *));
    for (int k = 0; k < n_blocks; k++)
    {
        scratch_blocks[k] = (permuted_dense *) ATA_pd_alloc(A->blocks[k]);
    }
    matrix *scratch =
        new_stacked_pd_unchecked(n, n, n_blocks, scratch_blocks, NULL, NULL);
    free(scratch_blocks);

    /* prepare coalescing of overlapping scratch blocks */
    matrix *C = coalesce_spd_alloc_unchecked((stacked_pd *) scratch);
    ((stacked_pd *) C)->pre_coalesce = (stacked_pd *) scratch;
    return C;
}

void ATDA_spd_fill_values(const stacked_pd *A, const double *d, stacked_pd *C)
{
    stacked_pd *scratch = C->pre_coalesce;

    /* compute Ai^T @ Di @ Ai into the scratch PDs. */
    for (int k = 0; k < A->n_blocks; k++)
    {
        ATDA_pd_fill_values(A->blocks[k], d, scratch->blocks[k]);
    }

    /* zero C (one memset is sufficient since, by design, the values of
       different blocks are stored consecutively in memory). */
    memset(C->base.x, 0, C->base.nnz * sizeof(double));

    /* scatter + add each Ai^T Di Ai into its destination C block. */
    coalesce_spd_fill_values_accumulate(scratch, C);
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
// BA_pd_spd: C = B @ A where B is permuted_dense and A is stacked_pd. Thin
// wrapper over the canonical BTA_pd_spd_* kernel: use B's lazily-cached
// transpose and call BTA. The cache is populated on first call (in alloc)
// and reused across subsequent fills, so this wrapper does not allocate
// a fresh transpose per call.
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
