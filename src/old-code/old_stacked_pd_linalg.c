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
#include "old-code/old_stacked_pd_linalg.h"

#include "utils/permuted_dense.h"
#include "utils/permuted_dense_linalg.h"
#include "utils/stacked_pd.h"
#include "utils/stacked_pd_linalg.h"

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
