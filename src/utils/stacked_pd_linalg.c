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

// --------------------------------------------------------------------------------
// BA_spd_csc: C = B @ A where B is spd, A is CSC. Suppose B = [B1; B2; B3]. Then
// C = [B1 @ A; B2 @ A; B3 @ A], so we can compute C by computing each Bi @ A.
// --------------------------------------------------------------------------------
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

        /* Ck = Bq @ A */
        BA_pd_csc_fill_values(Bq->X, Bq->n0, Bq->col_inv, A, C->blocks[k]);
    }
}

// ---------------------------------------------------------------------------------
// BA_spd_pd: C = B @ A where B is spd, A is PD. Same blockwise logic as BA_spd_csc.
// ---------------------------------------------------------------------------------
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
        /* Bq is the block in B that contributes to Ck */
        int q = C->src_block_idx[C->src_block_idx_p[k]];
        const permuted_dense *Bq = B->blocks[q];

        /* Ck = Bq @ A */
        BA_pd_pd_fill_values(Bq, A, C->blocks[k]);
    }
}

// -----------------------------------------------------------------------
// BA_spd_spd: C = B @ A where both are stacked_pd. Same blockwise logic.
// -----------------------------------------------------------------------
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
    /* Suppose B = [B1; B2; B3]. Then C = BA = [B1 @ A; B2 @ A; B3 @ A], so we can
       multiply each block of B with A. */
    for (int k = 0; k < C->n_blocks; k++)
    {
        /* Bq is the block in B that contributes to Ck */
        int q = C->src_block_idx[C->src_block_idx_p[k]];
        const permuted_dense *Bq = B->blocks[q];

        /* Ck = Bq @ A */
        BA_pd_spd_fill_values(Bq, A, C->blocks[k]);
    }
}

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
        const matrix *Ak = (matrix *) A->blocks[k];
        C_blocks[k] = (permuted_dense *) Ak->copy_sparsity(Ak);
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

/* ------------------------------------------------------------------ */
/* Transpose                                                           */
/* ------------------------------------------------------------------ */

matrix *transpose_spd_alloc(const stacked_pd *A)
{
    int n_blocks = A->n_blocks;
    permuted_dense **AT_blocks =
        (permuted_dense **) SP_MALLOC(n_blocks * sizeof(permuted_dense *));
    for (int k = 0; k < n_blocks; k++)
    {
        matrix *blk = (matrix *) A->blocks[k];
        AT_blocks[k] = (permuted_dense *) blk->transpose_alloc(blk);
    }

    /* Raw spd: dimensions swapped, identity src_block_idx_*; rows may
       overlap so use the unchecked constructor. */
    matrix *raw = new_stacked_pd_unchecked(A->base.n, A->base.m, n_blocks, AT_blocks,
                                           NULL, NULL);
    free(AT_blocks);

    matrix *C = coalesce_spd_alloc((stacked_pd *) raw);
    ((stacked_pd *) C)->work = (stacked_pd *) raw;
    return C;
}

void transpose_spd_fill_values(const stacked_pd *A, stacked_pd *C)
{
    stacked_pd *raw = C->work;
    for (int k = 0; k < A->n_blocks; k++)
    {
        matrix *A_blk = (matrix *) A->blocks[k];
        matrix *raw_blk = (matrix *) raw->blocks[k];
        A_blk->transpose_fill_values(A_blk, raw_blk);
    }
    coalesce_spd_fill_values(raw, C);
}

/* ------------------------------------------------------------------ */
/* ATA / ATDA                                                          */
/* ------------------------------------------------------------------ */

matrix *ATA_spd_alloc(const stacked_pd *A)
{
    int n = A->base.n;
    int n_blocks = A->n_blocks;

    /* Pseudo-spd: per source block k, a square PD with
       row_perm = col_perm = C_k and uninitialized X. We use the
       PD-level ATA_pd_alloc here (rather than new_permuted_dense
       directly) because it also pre-sizes A->blocks[k]->dwork — that
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

    /* Group cols by signature. Cells overlap by design (B_k^T B_k
       summands share cells), so use the unchecked alloc. */
    matrix *out = coalesce_spd_alloc_unchecked((stacked_pd *) scratch);
    ((stacked_pd *) out)->work = (stacked_pd *) scratch;
    return out;
}

/* Scatter + add each scratch PD into its destination block in C. Multiple scratch
   PDs may target the same Ck (their cells overlap by design — that's why we
   accumulate). Caller is responsible for zeroing C->base.x first. */
static inline void accumulate_scratch_blocks(const stacked_pd *scratch,
                                             stacked_pd *C)
{
    /* for each block Ck of C */
    for (int k = 0; k < C->n_blocks; k++)
    {
        permuted_dense *Ck = C->blocks[k];
        int s_lo = C->src_block_idx_p[k];
        int s_hi = C->src_block_idx_p[k + 1];

        /* for each block Eq = Ai^T D Ai in scratch contributing to block Ck in C */
        for (int qq = s_lo; qq < s_hi; qq++)
        {
            const permuted_dense *Eq = scratch->blocks[C->src_block_idx[qq]];

            /* for each row in Eq */
            for (int i = 0; i < Eq->m0; i++)
            {
                int Ck_row_idx = Ck->row_inv[Eq->row_perm[i]];
                if (Ck_row_idx < 0)
                {
                    /* this row in Eq does not contribute to Ck */
                    continue;
                }

                double *Ck_row = Ck->X + Ck_row_idx * Ck->n0;
                double *Eq_row = Eq->X + i * Eq->n0;

                /* place each col in Eq_row at its correct position in Ck_row */
                for (int j = 0; j < Eq->n0; j++)
                {
                    int out_j = Ck->col_inv[Eq->col_perm[j]];
                    assert(out_j >= 0);
                    Ck_row[out_j] += Eq_row[j];
                }
            }
        }
    }
}

void ATDA_spd_fill_values(const stacked_pd *A, const double *d, stacked_pd *C)
{
    /* Let A = [A1; A2; A3] where Ai has n columns (the same number as A). Then
       ATDA = A1^T D1 A1 + A2^T D2 A2 + A3^T D3 A3. Term i and j overlap if Ai
       and Aj have overlapping column entries. We therefore first compute
       Ai^T Di Ai and then take care of the accumulation. */

    stacked_pd *scratch = C->work;

    /* compute Ai^T @ Di @ Ai into the scratch PDs. */
    for (int k = 0; k < A->n_blocks; k++)
    {
        ATDA_pd_fill_values(A->blocks[k], d, scratch->blocks[k]);
    }

    /* zero C (one memset is sufficient since, by design, the values of
       different blocks are stored consecutively in memory). */
    memset(C->base.x, 0, C->base.nnz * sizeof(double));

    /* scatter + add each Ai^T Di Ai into its destination C block. */
    accumulate_scratch_blocks(scratch, C);
}

/* ------------------------------------------------------------------ */
/* BA_pd_spd: C = B @ A where B is a permuted_dense and A is a         */
/* stacked_pd. Output is a single permuted_dense.                      */
/* ------------------------------------------------------------------ */

matrix *BA_pd_spd_alloc(const permuted_dense *B, const stacked_pd *A)
{
    int n_blocks = A->n_blocks;

    /* Collect contributing A-blocks (those whose row_perm intersects
       B->col_perm). Two passes so we know the count for SP_MALLOC. */
    int n_contrib = 0;
    for (int k = 0; k < n_blocks; k++)
    {
        if (has_overlap(B->col_perm, B->n0, A->blocks[k]->row_perm, A->blocks[k]->m0,
                        0))
        {
            n_contrib++;
        }
    }

    iVec *col_union = iVec_new(8);
    if (n_contrib > 0)
    {
        const int **arrs =
            (const int **) SP_MALLOC((size_t) n_contrib * sizeof(int *));
        int *lens = (int *) SP_MALLOC((size_t) n_contrib * sizeof(int));
        int idx = 0;
        for (int k = 0; k < n_blocks; k++)
        {
            permuted_dense *Ak = A->blocks[k];
            if (has_overlap(B->col_perm, B->n0, Ak->row_perm, Ak->m0, 0))
            {
                arrs[idx] = Ak->col_perm;
                lens[idx] = Ak->n0;
                idx++;
            }
        }
        sorted_union_int_arrays(arrs, lens, n_contrib, col_union);
        free(arrs);
        free(lens);
    }

    matrix *C = new_permuted_dense(B->base.m, A->base.n, B->m0, col_union->len,
                                   B->row_perm, col_union->data, NULL);
    iVec_free(col_union);
    return C;
}

void BA_pd_spd_fill_values(const permuted_dense *B, const stacked_pd *A,
                           permuted_dense *C)
{
    if (C->m0 == 0 || C->n0 == 0)
    {
        return;
    }
    memset(C->X, 0, (size_t) C->m0 * C->n0 * sizeof(double));

    /* Compute scratch maxes in a single pass over A->blocks. */
    int max_inter = 0;
    int max_n0 = 0;
    for (int k = 0; k < A->n_blocks; k++)
    {
        int s = B->n0 < A->blocks[k]->m0 ? B->n0 : A->blocks[k]->m0;
        if (s > max_inter) max_inter = s;
        if (A->blocks[k]->n0 > max_n0) max_n0 = A->blocks[k]->n0;
    }
    if (max_inter == 0 || max_n0 == 0)
    {
        return;
    }

    int *idx_B = (int *) SP_MALLOC((size_t) max_inter * sizeof(int));
    int *idx_A = (int *) SP_MALLOC((size_t) max_inter * sizeof(int));
    double *B_gather =
        (double *) SP_MALLOC((size_t) B->m0 * (size_t) max_inter * sizeof(double));
    double *A_gather =
        (double *) SP_MALLOC((size_t) max_inter * (size_t) max_n0 * sizeof(double));
    double *temp =
        (double *) SP_MALLOC((size_t) B->m0 * (size_t) max_n0 * sizeof(double));

    for (int k = 0; k < A->n_blocks; k++)
    {
        permuted_dense *Ak = A->blocks[k];
        int s = sorted_intersect_indices(B->col_perm, B->n0, Ak->row_perm, Ak->m0,
                                         idx_B, idx_A);
        if (s == 0) continue;

        /* Gather B's cols: B_gather[i, p] = B->X[i, idx_B[p]]. */
        for (int i = 0; i < B->m0; i++)
        {
            for (int p = 0; p < s; p++)
            {
                B_gather[i * s + p] = B->X[i * B->n0 + idx_B[p]];
            }
        }

        /* Gather A_k's rows: A_gather[p, :] = A_k->X[idx_A[p], :]. */
        for (int p = 0; p < s; p++)
        {
            memcpy(A_gather + p * Ak->n0, Ak->X + idx_A[p] * Ak->n0,
                   (size_t) Ak->n0 * sizeof(double));
        }

        /* temp (m0_B × n0_k) = B_gather @ A_gather. */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B->m0, Ak->n0, s, 1.0,
                    B_gather, s, A_gather, Ak->n0, 0.0, temp, Ak->n0);

        /* Scatter+add into C via col_inv. */
        for (int j = 0; j < Ak->n0; j++)
        {
            int out_col = C->col_inv[Ak->col_perm[j]];
            for (int i = 0; i < B->m0; i++)
            {
                C->X[i * C->n0 + out_col] += temp[i * Ak->n0 + j];
            }
        }
    }

    free(idx_B);
    free(idx_A);
    free(B_gather);
    free(A_gather);
    free(temp);
}
