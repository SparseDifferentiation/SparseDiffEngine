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
#include "utils/stacked_pd.h"

#include "utils/matrix.h"
#include "utils/permuted_dense.h"
#include "utils/tracked_alloc.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void stacked_pd_free(matrix *self)
{
    stacked_pd *spd = (stacked_pd *) self;
    for (int k = 0; k < spd->n_blocks; k++)
    {
        free_matrix((matrix *) spd->blocks[k]);
    }
    free(spd->blocks);
    free(spd->src_block_idx);
    free(spd);
}

#ifndef NDEBUG
/* Pairwise verify that block row permutations are disjoint via a
   two-pointer merge over the sorted arrays. */
static void assert_disjoint_row_perms(int n_blocks, permuted_dense *const *blocks)
{
    for (int a = 0; a < n_blocks; a++)
    {
        for (int b = a + 1; b < n_blocks; b++)
        {
            const int *ra = blocks[a]->row_perm;
            int la = blocks[a]->m0;
            const int *rb = blocks[b]->row_perm;
            int lb = blocks[b]->m0;
            int i = 0;
            int j = 0;
            while (i < la && j < lb)
            {
                assert(ra[i] != rb[j] && "stacked_pd: block row_perms overlap");
                if (ra[i] < rb[j])
                {
                    i++;
                }
                else
                {
                    j++;
                }
            }
        }
    }
}
#endif

matrix *new_stacked_pd(int m, int n, int n_blocks, permuted_dense **blocks,
                       const int *src_block_idx)
{
    stacked_pd *spd = (stacked_pd *) SP_CALLOC(1, sizeof(stacked_pd));
    spd->base.m = m;
    spd->base.n = n;
    int nnz = 0;
    for (int k = 0; k < n_blocks; k++)
    {
        nnz += blocks[k]->base.nnz;
    }
    spd->base.nnz = nnz;
    spd->base.free_fn = stacked_pd_free;

    spd->n_blocks = n_blocks;
    spd->blocks = (permuted_dense **) SP_MALLOC(n_blocks * sizeof(permuted_dense *));
    spd->src_block_idx = (int *) SP_MALLOC(n_blocks * sizeof(int));
    if (n_blocks > 0)
    {
        memcpy(spd->blocks, blocks, n_blocks * sizeof(permuted_dense *));
        if (src_block_idx == NULL)
        {
            for (int k = 0; k < n_blocks; k++)
            {
                spd->src_block_idx[k] = k;
            }
        }
        else
        {
            memcpy(spd->src_block_idx, src_block_idx, n_blocks * sizeof(int));
        }
    }

#ifndef NDEBUG
    assert_disjoint_row_perms(n_blocks, blocks);
#endif

    return &spd->base;
}

matrix *BA_spd_csc_alloc(const stacked_pd *B, const CSC_matrix *A)
{
    permuted_dense **tmp_blocks = NULL;
    int *tmp_src = NULL;
    if (B->n_blocks > 0)
    {
        tmp_blocks =
            (permuted_dense **) SP_MALLOC(B->n_blocks * sizeof(permuted_dense *));
        tmp_src = (int *) SP_MALLOC(B->n_blocks * sizeof(int));
    }

    int out_n = 0;
    for (int k = 0; k < B->n_blocks; k++)
    {
        matrix *Ck = BA_pd_csc_alloc(B->blocks[k], A);
        permuted_dense *Ck_pd = (permuted_dense *) Ck;
        if (Ck_pd->n0 == 0)
        {
            free_matrix(Ck);
        }
        else
        {
            tmp_blocks[out_n] = Ck_pd;
            tmp_src[out_n] = k;
            out_n++;
        }
    }

    matrix *C = new_stacked_pd(B->base.m, A->n, out_n, tmp_blocks, tmp_src);
    free(tmp_blocks);
    free(tmp_src);
    return C;
}

void BA_spd_csc_fill_values(const stacked_pd *B, const CSC_matrix *A, stacked_pd *C)
{
    for (int k = 0; k < C->n_blocks; k++)
    {
        int src = C->src_block_idx[k];
        const permuted_dense *Bk = B->blocks[src];
        BA_pd_csc_fill_values(Bk->X, Bk->n0, Bk->col_inv, A, C->blocks[k]);
    }
}
