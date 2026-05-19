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
#include "utils/stacked_pd_linalg.h"
#include "utils/tracked_alloc.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void stacked_pd_free(matrix *self)
{
    stacked_pd *spd = (stacked_pd *) self;
    for (int k = 0; k < spd->n_blocks; k++)
    {
        /* blocks have owns_X = false; their free skips free(X) and the
           shared buffer is freed below. */
        free_matrix((matrix *) spd->blocks[k]);
    }
    free(spd->blocks);
    free(spd->src_block_idx_p);
    free(spd->src_block_idx);
    if (spd->work != NULL)
    {
        free_matrix((matrix *) spd->work);
    }
    free_CSR_matrix(spd->csr_cache); /* NULL-safe */
    free(spd->base.x);               /* shared values buffer, NULL-safe */
    free(spd);
}

/* Vtable adapters — each downcasts matrix * to stacked_pd * and
   delegates to the corresponding public spd kernel. */

static matrix *stacked_pd_vtable_copy_sparsity(const matrix *self)
{
    return copy_sparsity_spd_alloc((const stacked_pd *) self);
}

static void stacked_pd_vtable_DA_fill_values(const double *d, const matrix *self,
                                             matrix *out)
{
    DA_spd_fill_values(d, (const stacked_pd *) self, (stacked_pd *) out);
}

static matrix *stacked_pd_vtable_ATA_alloc(matrix *self)
{
    return ATA_spd_alloc((const stacked_pd *) self);
}

static void stacked_pd_vtable_ATDA_fill_values(const matrix *self, const double *d,
                                               matrix *out)
{
    ATDA_spd_fill_values((const stacked_pd *) self, d, (stacked_pd *) out);
}

static matrix *stacked_pd_vtable_transpose_alloc(const matrix *self)
{
    return transpose_spd_alloc((const stacked_pd *) self);
}

static void stacked_pd_vtable_transpose_fill_values(const matrix *self, matrix *out)
{
    transpose_spd_fill_values((const stacked_pd *) self, (stacked_pd *) out);
}

static void stacked_pd_vtable_refresh_csc_values(matrix *self)
{
    (void) self; /* spd has no CSC cache to refresh */
}

/* Lazy CSR view. First call builds the CSR structure (p, i) and fills
   values; subsequent calls only refresh values from the current block X
   buffers. Free is handled by stacked_pd_free. */
static CSR_matrix *stacked_pd_to_csr(matrix *self)
{
    stacked_pd *spd = (stacked_pd *) self;
    int m = spd->base.m;

    if (spd->csr_cache == NULL)
    {
        int total_nnz = 0;
        for (int k = 0; k < spd->n_blocks; k++)
        {
            total_nnz += spd->blocks[k]->base.nnz;
        }
        spd->csr_cache = new_CSR_matrix(m, spd->base.n, total_nnz);

        int *p = spd->csr_cache->p;
        for (int r = 0; r <= m; r++)
        {
            p[r] = 0;
        }
        for (int k = 0; k < spd->n_blocks; k++)
        {
            permuted_dense *blk = spd->blocks[k];
            for (int i = 0; i < blk->m0; i++)
            {
                p[blk->row_perm[i] + 1] = blk->n0;
            }
        }
        for (int r = 0; r < m; r++)
        {
            p[r + 1] += p[r];
        }

        /* col indices: sorted within each row because each block's col_perm
           is sorted by construction, and each row appears in at most one
           block (disjoint row_perms). */
        for (int k = 0; k < spd->n_blocks; k++)
        {
            permuted_dense *blk = spd->blocks[k];
            for (int i = 0; i < blk->m0; i++)
            {
                int row = blk->row_perm[i];
                memcpy(spd->csr_cache->i + p[row], blk->col_perm,
                       (size_t) blk->n0 * sizeof(int));
            }
        }
    }

    /* Refresh values every call (block X buffers may have changed). */
    int *p_arr = spd->csr_cache->p;
    for (int k = 0; k < spd->n_blocks; k++)
    {
        permuted_dense *blk = spd->blocks[k];
        for (int i = 0; i < blk->m0; i++)
        {
            int row = blk->row_perm[i];
            memcpy(spd->csr_cache->x + p_arr[row], blk->X + i * blk->n0,
                   (size_t) blk->n0 * sizeof(double));
        }
    }
    return spd->csr_cache;
}

matrix *spd_map_filter_blocks(const stacked_pd *B, int Cm, int Cn, spd_block_op op,
                              const void *ctx)
{
    if (B->n_blocks == 0)
    {
        return new_stacked_pd(Cm, Cn, 0, NULL, NULL, NULL);
    }

    permuted_dense **C_blocks =
        (permuted_dense **) SP_MALLOC(B->n_blocks * sizeof(permuted_dense *));
    int *C_src = (int *) SP_MALLOC(B->n_blocks * sizeof(int));

    int out_nb = 0;
    for (int k = 0; k < B->n_blocks; k++)
    {
        matrix *Ck = op(B->blocks[k], ctx);
        if (Ck->nnz == 0)
        {
            free_matrix(Ck);
        }
        else
        {
            C_blocks[out_nb] = (permuted_dense *) Ck;
            C_src[out_nb] = k;
            out_nb++;
        }
    }

    int *C_src_p = (int *) SP_MALLOC((out_nb + 1) * sizeof(int));
    for (int k = 0; k <= out_nb; k++)
    {
        C_src_p[k] = k;
    }

    matrix *C = new_stacked_pd(Cm, Cn, out_nb, C_blocks, C_src_p, C_src);
    free(C_blocks);
    free(C_src);
    free(C_src_p);
    return C;
}

/* Per-block delegation: apply PD's index_alloc to each source block. The
   per-block result is a PD whose row_perm holds output positions
   (0..n_idxs-1) — disjoint across source blocks since source row_perms are
   disjoint. Empty per-block results (no requested row hits the block) are
   dropped; src_block_idx_* records which source blocks survived. */
typedef struct
{
    const int *indices;
    int n_idxs;
} pd_index_ctx;

static matrix *wrapper_pd_index(permuted_dense *Bk, const void *ctx)
{
    const pd_index_ctx *c = (const pd_index_ctx *) ctx;
    matrix *blk = (matrix *) Bk;
    return blk->index_alloc(blk, c->indices, c->n_idxs);
}

static matrix *stacked_pd_vtable_index_alloc(matrix *self, const int *indices,
                                             int n_idxs)
{
    stacked_pd *src = (stacked_pd *) self;
    pd_index_ctx ctx = {indices, n_idxs};
    return spd_map_filter_blocks(src, n_idxs, src->base.n, wrapper_pd_index, &ctx);
}

static void stacked_pd_vtable_index_fill_values(matrix *self, const int *indices,
                                                int n_idxs, matrix *out)
{
    stacked_pd *src = (stacked_pd *) self;
    stacked_pd *out_spd = (stacked_pd *) out;
    for (int k = 0; k < out_spd->n_blocks; k++)
    {
        int sk = out_spd->src_block_idx[k];
        matrix *src_blk = (matrix *) src->blocks[sk];
        matrix *out_blk = (matrix *) out_spd->blocks[k];
        src_blk->index_fill_values(src_blk, indices, n_idxs, out_blk);
    }
}

/* Per-block delegation: apply PD's promote_alloc to each source block.
   Input spd represents a 1-row matrix, so at most one block has m0=1 and
   all others have m0=0. PD's promote yields m0==size on the populated
   block and m0==0 on empty blocks; we drop the empty ones, preserving
   spd's disjoint-row invariant on the output. */
static matrix *wrapper_pd_promote(permuted_dense *Bk, const void *ctx)
{
    matrix *blk = (matrix *) Bk;
    return blk->promote_alloc(blk, *(const int *) ctx);
}

static matrix *stacked_pd_vtable_promote_alloc(matrix *self, int size)
{
    stacked_pd *src = (stacked_pd *) self;
    return spd_map_filter_blocks(src, size, src->base.n, wrapper_pd_promote, &size);
}

static void stacked_pd_vtable_promote_fill_values(matrix *self, matrix *out)
{
    stacked_pd *src = (stacked_pd *) self;
    stacked_pd *out_spd = (stacked_pd *) out;
    for (int k = 0; k < out_spd->n_blocks; k++)
    {
        int sk = out_spd->src_block_idx[k];
        matrix *src_blk = (matrix *) src->blocks[sk];
        matrix *out_blk = (matrix *) out_spd->blocks[k];
        src_blk->promote_fill_values(src_blk, out_blk);
    }
}

/* Per-block delegation: each PD block's diag_vec_alloc rescales row_perm
   entries r -> r*(n+1) and preserves col_perm; fill_values memcpys the
   block X. The op is structure-preserving (no blocks become empty, none
   merge), so output spd has the same n_blocks and an identity
   src_block_idx_*. Disjoint-row invariant on output follows from input:
   source row_perms partition a subset of {0..n-1}; scaling by (n+1) keeps
   the sets pairwise disjoint inside {0..n*n-1}. */
static matrix *stacked_pd_vtable_diag_vec_alloc(matrix *self)
{
    stacked_pd *src = (stacked_pd *) self;
    int nb = src->n_blocks;
    int n = src->base.m;
    int out_m = n * n;

    permuted_dense **tmp_blocks = NULL;
    if (nb > 0)
    {
        tmp_blocks =
            (permuted_dense **) SP_MALLOC((size_t) nb * sizeof(permuted_dense *));
        for (int k = 0; k < nb; k++)
        {
            matrix *blk = (matrix *) src->blocks[k];
            matrix *out_blk = blk->diag_vec_alloc(blk);
            tmp_blocks[k] = (permuted_dense *) out_blk;
        }
    }

    matrix *out = new_stacked_pd(out_m, src->base.n, nb, tmp_blocks, NULL, NULL);
    free(tmp_blocks);
    return out;
}

static void stacked_pd_vtable_diag_vec_fill_values(matrix *self, matrix *out)
{
    stacked_pd *src = (stacked_pd *) self;
    stacked_pd *out_spd = (stacked_pd *) out;
    for (int k = 0; k < out_spd->n_blocks; k++)
    {
        matrix *src_blk = (matrix *) src->blocks[k];
        matrix *out_blk = (matrix *) out_spd->blocks[k];
        src_blk->diag_vec_fill_values(src_blk, out_blk);
    }
}

/* Per-block delegation: apply PD's broadcast_alloc to each source block.
   Output is structurally a valid spd because input row_perms are
   pairwise disjoint -> per-mode output ranges are also pairwise disjoint
   across input blocks. Empty per-block outputs (m0=0 in input) are
   dropped; src_block_idx_* records the survivors. */
typedef struct
{
    broadcast_type type;
    int d1;
    int d2;
} pd_broadcast_ctx;

static matrix *wrapper_pd_broadcast(permuted_dense *Bk, const void *ctx)
{
    const pd_broadcast_ctx *c = (const pd_broadcast_ctx *) ctx;
    matrix *blk = (matrix *) Bk;
    return blk->broadcast_alloc(blk, c->type, c->d1, c->d2);
}

static matrix *stacked_pd_vtable_broadcast_alloc(matrix *self, broadcast_type type,
                                                 int d1, int d2)
{
    stacked_pd *src = (stacked_pd *) self;
    pd_broadcast_ctx ctx = {type, d1, d2};
    return spd_map_filter_blocks(src, d1 * d2, src->base.n, wrapper_pd_broadcast,
                                 &ctx);
}

static void stacked_pd_vtable_broadcast_fill_values(matrix *self,
                                                    broadcast_type type, int d1,
                                                    int d2, matrix *out)
{
    stacked_pd *src = (stacked_pd *) self;
    stacked_pd *out_spd = (stacked_pd *) out;
    for (int k = 0; k < out_spd->n_blocks; k++)
    {
        int sk = out_spd->src_block_idx[k];
        matrix *src_blk = (matrix *) src->blocks[sk];
        matrix *out_blk = (matrix *) out_spd->blocks[k];
        src_blk->broadcast_fill_values(src_blk, type, d1, d2, out_blk);
    }
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

matrix *new_stacked_pd_unchecked(int m, int n, int n_blocks, permuted_dense **blocks,
                                 const int *src_block_idx_p,
                                 const int *src_block_idx)
{
    assert((src_block_idx_p == NULL) == (src_block_idx == NULL) &&
           "stacked_pd: src_block_idx_p and src_block_idx must both be NULL or "
           "both non-NULL");

    stacked_pd *spd = (stacked_pd *) SP_CALLOC(1, sizeof(stacked_pd));
    spd->base.m = m;
    spd->base.n = n;
    int nnz = 0;
    for (int k = 0; k < n_blocks; k++)
    {
        nnz += blocks[k]->base.nnz;
    }
    spd->base.nnz = nnz;
    spd->base.is_stacked_pd = true;
    spd->base.free_fn = stacked_pd_free;
    spd->base.copy_sparsity = stacked_pd_vtable_copy_sparsity;
    spd->base.DA_fill_values = stacked_pd_vtable_DA_fill_values;
    spd->base.ATA_alloc = stacked_pd_vtable_ATA_alloc;
    spd->base.ATDA_fill_values = stacked_pd_vtable_ATDA_fill_values;
    spd->base.transpose_alloc = stacked_pd_vtable_transpose_alloc;
    spd->base.transpose_fill_values = stacked_pd_vtable_transpose_fill_values;
    spd->base.refresh_csc_values = stacked_pd_vtable_refresh_csc_values;
    spd->base.to_csr = stacked_pd_to_csr;
    spd->base.index_alloc = stacked_pd_vtable_index_alloc;
    spd->base.index_fill_values = stacked_pd_vtable_index_fill_values;
    spd->base.promote_alloc = stacked_pd_vtable_promote_alloc;
    spd->base.promote_fill_values = stacked_pd_vtable_promote_fill_values;
    spd->base.diag_vec_alloc = stacked_pd_vtable_diag_vec_alloc;
    spd->base.diag_vec_fill_values = stacked_pd_vtable_diag_vec_fill_values;
    spd->base.broadcast_alloc = stacked_pd_vtable_broadcast_alloc;
    spd->base.broadcast_fill_values = stacked_pd_vtable_broadcast_fill_values;

    spd->n_blocks = n_blocks;
    spd->blocks = (permuted_dense **) SP_MALLOC(n_blocks * sizeof(permuted_dense *));
    spd->src_block_idx_p = (int *) SP_MALLOC((n_blocks + 1) * sizeof(int));
    spd->src_block_idx_p[0] = 0;
    if (n_blocks > 0)
    {
        memcpy(spd->blocks, blocks, n_blocks * sizeof(permuted_dense *));
    }

    if (src_block_idx_p == NULL)
    {
        /* identity: each output block has itself as its one source */
        spd->src_block_idx = (int *) SP_MALLOC(n_blocks * sizeof(int));
        for (int k = 0; k < n_blocks; k++)
        {
            spd->src_block_idx_p[k + 1] = k + 1;
            spd->src_block_idx[k] = k;
        }
    }
    else
    {
        int total = src_block_idx_p[n_blocks];
        memcpy(spd->src_block_idx_p, src_block_idx_p, (n_blocks + 1) * sizeof(int));
        spd->src_block_idx = (int *) SP_MALLOC(total * sizeof(int));
        if (total > 0)
        {
            memcpy(spd->src_block_idx, src_block_idx, total * sizeof(int));
        }
    }

    /* Absorb each block's X into a single shared values buffer owned by
       this spd. After this loop:
         spd->base.x          = the contiguous buffer (block-major layout)
         block[k]->X          = view into spd->base.x at the block's offset
         block[k]->owns_X     = false (block's free skips free(X))
       This honors the same flat-base.x invariant PD and sparse_matrix
       already honor, so atoms that loop over base.x work uniformly. */
    int total_nnz = 0;
    for (int k = 0; k < n_blocks; k++)
    {
        total_nnz += spd->blocks[k]->m0 * spd->blocks[k]->n0;
    }
    spd->base.x = (total_nnz > 0)
                      ? (double *) SP_MALLOC((size_t) total_nnz * sizeof(double))
                      : NULL;

    size_t offset = 0;
    for (int k = 0; k < n_blocks; k++)
    {
        size_t sz = (size_t) spd->blocks[k]->m0 * spd->blocks[k]->n0;
        if (sz == 0)
        {
            continue;
        }
        memcpy(spd->base.x + offset, spd->blocks[k]->X, sz * sizeof(double));
        free(spd->blocks[k]->X);
        spd->blocks[k]->X = spd->base.x + offset;
        spd->blocks[k]->base.x = spd->blocks[k]->X;
        spd->blocks[k]->owns_X = false;
        offset += sz;
    }

    return &spd->base;
}

matrix *new_stacked_pd(int m, int n, int n_blocks, permuted_dense **blocks,
                       const int *src_block_idx_p, const int *src_block_idx)
{
    matrix *out = new_stacked_pd_unchecked(m, n, n_blocks, blocks, src_block_idx_p,
                                           src_block_idx);
#ifndef NDEBUG
    assert_disjoint_row_perms(n_blocks, blocks);
#endif
    return out;
}
