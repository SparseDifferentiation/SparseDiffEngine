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

#include "utils/cblas_wrapper.h"
#include "utils/iVec.h"
#include "utils/matrix.h"
#include "utils/permuted_dense.h"
#include "utils/tracked_alloc.h"
#include "utils/utils.h"
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
    free(spd->src_block_idx_p);
    free(spd->src_block_idx);
    if (spd->work != NULL)
    {
        free_matrix((matrix *) spd->work);
    }
    free_CSR_matrix(spd->csr_cache); /* NULL-safe */
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

/* Per-block delegation: apply PD's index_alloc to each source block. The
   per-block result is a PD whose row_perm holds output positions
   (0..n_idxs-1) — disjoint across source blocks since source row_perms are
   disjoint. Empty per-block results (no requested row hits the block) are
   dropped; src_block_idx_* records which source blocks survived. */
static matrix *stacked_pd_vtable_index_alloc(matrix *self, const int *indices,
                                             int n_idxs)
{
    stacked_pd *src = (stacked_pd *) self;
    int nb = src->n_blocks;
    permuted_dense **tmp_blocks =
        (permuted_dense **) SP_MALLOC((size_t) nb * sizeof(permuted_dense *));
    int *tmp_src = (int *) SP_MALLOC((size_t) nb * sizeof(int));
    int out_nb = 0;

    for (int k = 0; k < nb; k++)
    {
        matrix *blk = (matrix *) src->blocks[k];
        matrix *out_blk = blk->index_alloc(blk, indices, n_idxs);
        permuted_dense *out_pd = (permuted_dense *) out_blk;
        if (out_pd->m0 > 0)
        {
            tmp_blocks[out_nb] = out_pd;
            tmp_src[out_nb] = k;
            out_nb++;
        }
        else
        {
            free_matrix(out_blk);
        }
    }

    int *src_p = (int *) SP_MALLOC((size_t) (out_nb + 1) * sizeof(int));
    for (int k = 0; k <= out_nb; k++)
    {
        src_p[k] = k;
    }

    matrix *out =
        new_stacked_pd(n_idxs, src->base.n, out_nb, tmp_blocks, src_p, tmp_src);
    free(tmp_blocks);
    free(tmp_src);
    free(src_p);
    return out;
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
static matrix *stacked_pd_vtable_promote_alloc(matrix *self, int size)
{
    stacked_pd *src = (stacked_pd *) self;
    int nb = src->n_blocks;
    permuted_dense **tmp_blocks =
        (permuted_dense **) SP_MALLOC((size_t) nb * sizeof(permuted_dense *));
    int *tmp_src = (int *) SP_MALLOC((size_t) nb * sizeof(int));
    int out_nb = 0;

    for (int k = 0; k < nb; k++)
    {
        matrix *blk = (matrix *) src->blocks[k];
        matrix *out_blk = blk->promote_alloc(blk, size);
        permuted_dense *out_pd = (permuted_dense *) out_blk;
        if (out_pd->m0 > 0)
        {
            tmp_blocks[out_nb] = out_pd;
            tmp_src[out_nb] = k;
            out_nb++;
        }
        else
        {
            free_matrix(out_blk);
        }
    }

    int *src_p = (int *) SP_MALLOC((size_t) (out_nb + 1) * sizeof(int));
    for (int k = 0; k <= out_nb; k++)
    {
        src_p[k] = k;
    }

    matrix *out =
        new_stacked_pd(size, src->base.n, out_nb, tmp_blocks, src_p, tmp_src);
    free(tmp_blocks);
    free(tmp_src);
    free(src_p);
    return out;
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
static matrix *stacked_pd_vtable_broadcast_alloc(matrix *self, broadcast_type type,
                                                 int d1, int d2)
{
    stacked_pd *src = (stacked_pd *) self;
    int nb = src->n_blocks;
    int out_m = d1 * d2;

    permuted_dense **tmp_blocks =
        (permuted_dense **) SP_MALLOC((size_t) nb * sizeof(permuted_dense *));
    int *tmp_src = (int *) SP_MALLOC((size_t) nb * sizeof(int));
    int out_nb = 0;

    for (int k = 0; k < nb; k++)
    {
        matrix *blk = (matrix *) src->blocks[k];
        matrix *out_blk = blk->broadcast_alloc(blk, type, d1, d2);
        permuted_dense *out_pd = (permuted_dense *) out_blk;
        if (out_pd->m0 > 0)
        {
            tmp_blocks[out_nb] = out_pd;
            tmp_src[out_nb] = k;
            out_nb++;
        }
        else
        {
            free_matrix(out_blk);
        }
    }

    int *src_p = (int *) SP_MALLOC((size_t) (out_nb + 1) * sizeof(int));
    for (int k = 0; k <= out_nb; k++)
    {
        src_p[k] = k;
    }

    matrix *out =
        new_stacked_pd(out_m, src->base.n, out_nb, tmp_blocks, src_p, tmp_src);
    free(tmp_blocks);
    free(tmp_src);
    free(src_p);
    return out;
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

    /* one source per output block: src_block_idx_p = identity prefix sum */
    int *tmp_src_p = (int *) SP_MALLOC((out_n + 1) * sizeof(int));
    for (int k = 0; k <= out_n; k++)
    {
        tmp_src_p[k] = k;
    }

    matrix *C =
        new_stacked_pd(B->base.m, A->n, out_n, tmp_blocks, tmp_src_p, tmp_src);
    free(tmp_blocks);
    free(tmp_src);
    free(tmp_src_p);
    return C;
}

void BA_spd_csc_fill_values(const stacked_pd *B, const CSC_matrix *A, stacked_pd *C)
{
    for (int k = 0; k < C->n_blocks; k++)
    {
        int src = C->src_block_idx[C->src_block_idx_p[k]];
        const permuted_dense *Bk = B->blocks[src];
        BA_pd_csc_fill_values(Bk->X, Bk->n0, Bk->col_inv, A, C->blocks[k]);
    }
}

/* ------------------------------------------------------------------ */
/* Coalesce                                                            */
/* ------------------------------------------------------------------ */

/* K-way merge of sorted int arrays into `out` (sorted, deduplicated).
   `out` is reset and then populated; on return `out->data[0..out->len)`
   is the sorted union. Implementation: repeatedly pick the smallest
   head across all arrays, append if it differs from the previous
   appended value, advance the chosen array's cursor. O(total_len *
   n_arrs) which is fine for small n_arrs. */
static void sorted_union_int_arrays(const int *const *arrs, const int *lens,
                                    int n_arrs, iVec *out)
{
    iVec_clear_no_resize(out);
    int *cursor = (int *) SP_CALLOC(n_arrs, sizeof(int));
    while (1)
    {
        int min_val = 0;
        int min_arr = -1;
        for (int a = 0; a < n_arrs; a++)
        {
            if (cursor[a] >= lens[a])
            {
                continue;
            }
            int v = arrs[a][cursor[a]];
            if (min_arr == -1 || v < min_val)
            {
                min_val = v;
                min_arr = a;
            }
        }
        if (min_arr == -1)
        {
            break;
        }
        if (out->len == 0 || out->data[out->len - 1] != min_val)
        {
            iVec_append(out, min_val);
        }
        cursor[min_arr]++;
    }
    free(cursor);
}

#ifndef NDEBUG
/* Assert pairwise-disjoint cells: any two source blocks sharing a row
   must have disjoint col_perms. We only need to check pairs within the
   same signature (others share no row by construction). */
static void assert_disjoint_cells(const stacked_pd *src, const int *cat_sig_p,
                                  const int *cat_sig_data, int n_sigs)
{
    for (int s = 0; s < n_sigs; s++)
    {
        int sig_lo = cat_sig_p[s];
        int sig_hi = cat_sig_p[s + 1];
        for (int u = sig_lo; u < sig_hi; u++)
        {
            for (int v = u + 1; v < sig_hi; v++)
            {
                permuted_dense *bu = src->blocks[cat_sig_data[u]];
                permuted_dense *bv = src->blocks[cat_sig_data[v]];
                assert(!has_overlap(bu->col_perm, bu->n0, bv->col_perm, bv->n0, 0) &&
                       "coalesce: source blocks share both a row and a col");
            }
        }
    }
}
#endif

static matrix *coalesce_spd_alloc_impl(const stacked_pd *src,
                                       bool check_disjoint_cells)
{
    int m = src->base.m;
    int n = src->base.n;
    int n_blocks = src->n_blocks;

    /* Pass 1: row -> blocks CSR. */
    int *row_blocks_p = (int *) SP_CALLOC(m + 1, sizeof(int));
    for (int k = 0; k < n_blocks; k++)
    {
        permuted_dense *blk = src->blocks[k];
        for (int i = 0; i < blk->m0; i++)
        {
            row_blocks_p[blk->row_perm[i] + 1]++;
        }
    }
    for (int r = 0; r < m; r++)
    {
        row_blocks_p[r + 1] += row_blocks_p[r];
    }
    int row_blocks_total = row_blocks_p[m];
    int *row_blocks_data = (int *) SP_MALLOC(row_blocks_total * sizeof(int));
    /* If row_blocks_total == 0 we still need a valid pointer for free; SP_MALLOC
       handles that. */
    int *cursor = (int *) SP_CALLOC(m, sizeof(int));
    for (int k = 0; k < n_blocks; k++)
    {
        permuted_dense *blk = src->blocks[k];
        for (int i = 0; i < blk->m0; i++)
        {
            int r = blk->row_perm[i];
            row_blocks_data[row_blocks_p[r] + cursor[r]++] = k;
        }
    }
    free(cursor);

    /* Pass 2: catalog unique signatures. */
    iVec *cat_sig_p = iVec_new(8);
    iVec_append(cat_sig_p, 0);
    iVec *cat_sig_data = iVec_new(16);
    int *row_to_sig = (int *) SP_MALLOC(m * sizeof(int));

    for (int r = 0; r < m; r++)
    {
        int sig_start = row_blocks_p[r];
        int sig_len = row_blocks_p[r + 1] - sig_start;
        if (sig_len == 0)
        {
            row_to_sig[r] = -1;
            continue;
        }
        const int *sig = row_blocks_data + sig_start;
        int n_sigs = cat_sig_p->len - 1;
        int found = -1;
        for (int s = 0; s < n_sigs; s++)
        {
            int s_lo = cat_sig_p->data[s];
            int s_hi = cat_sig_p->data[s + 1];
            if (s_hi - s_lo != sig_len)
            {
                continue;
            }
            if (memcmp(cat_sig_data->data + s_lo, sig,
                       (size_t) sig_len * sizeof(int)) == 0)
            {
                found = s;
                break;
            }
        }
        if (found == -1)
        {
            iVec_append_array(cat_sig_data, sig, sig_len);
            iVec_append(cat_sig_p, cat_sig_data->len);
            found = n_sigs;
        }
        row_to_sig[r] = found;
    }

    int n_sigs = cat_sig_p->len - 1;

#ifndef NDEBUG
    if (check_disjoint_cells)
    {
        assert_disjoint_cells(src, cat_sig_p->data, cat_sig_data->data, n_sigs);
    }
#else
    (void) check_disjoint_cells;
#endif

    /* Pass 3: group rows by sig id. */
    int *sig_rows_p = (int *) SP_CALLOC(n_sigs + 1, sizeof(int));
    for (int r = 0; r < m; r++)
    {
        if (row_to_sig[r] >= 0)
        {
            sig_rows_p[row_to_sig[r] + 1]++;
        }
    }
    for (int s = 0; s < n_sigs; s++)
    {
        sig_rows_p[s + 1] += sig_rows_p[s];
    }
    int sig_rows_total = sig_rows_p[n_sigs];
    int *sig_rows_data = (int *) SP_MALLOC(sig_rows_total * sizeof(int));
    int *cursor2 = (int *) SP_CALLOC(n_sigs > 0 ? n_sigs : 1, sizeof(int));
    for (int r = 0; r < m; r++)
    {
        int s = row_to_sig[r];
        if (s >= 0)
        {
            sig_rows_data[sig_rows_p[s] + cursor2[s]++] = r;
        }
    }
    free(cursor2);

    /* Pass 4: allocate one output PD per signature (unordered for now). */
    permuted_dense **sig_blocks = (permuted_dense **) SP_MALLOC(
        (n_sigs > 0 ? n_sigs : 1) * sizeof(permuted_dense *));
    /* scratch for the k-way col_perm merge */
    iVec *col_union = iVec_new(8);
    /* scratch arrays passed to sorted_union_int_arrays */
    for (int s = 0; s < n_sigs; s++)
    {
        int sig_lo = cat_sig_p->data[s];
        int sig_hi = cat_sig_p->data[s + 1];
        int sig_sz = sig_hi - sig_lo;

        const int **col_arrs =
            (const int **) SP_MALLOC((size_t) sig_sz * sizeof(int *));
        int *col_lens = (int *) SP_MALLOC((size_t) sig_sz * sizeof(int));
        for (int t = 0; t < sig_sz; t++)
        {
            permuted_dense *blk = src->blocks[cat_sig_data->data[sig_lo + t]];
            col_arrs[t] = blk->col_perm;
            col_lens[t] = blk->n0;
        }
        sorted_union_int_arrays(col_arrs, col_lens, sig_sz, col_union);
        free(col_arrs);
        free(col_lens);

        int *row_perm = sig_rows_data + sig_rows_p[s];
        int m0 = sig_rows_p[s + 1] - sig_rows_p[s];
        int n0 = col_union->len;
        sig_blocks[s] = (permuted_dense *) new_permuted_dense(m, n, m0, n0, row_perm,
                                                              col_union->data, NULL);
    }
    iVec_free(col_union);

    /* Pass 5: order signatures by min row index (insertion sort). */
    int *order = (int *) SP_MALLOC((n_sigs > 0 ? n_sigs : 1) * sizeof(int));
    for (int s = 0; s < n_sigs; s++)
    {
        order[s] = s;
    }
    for (int i = 1; i < n_sigs; i++)
    {
        int key = order[i];
        int key_min_row = sig_rows_data[sig_rows_p[key]];
        int j = i - 1;
        while (j >= 0 && sig_rows_data[sig_rows_p[order[j]]] > key_min_row)
        {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }

    /* Pass 6: build final block array and CSR-style src_block_idx in
       output order; hand off to new_stacked_pd. */
    permuted_dense **out_blocks = (permuted_dense **) SP_MALLOC(
        (n_sigs > 0 ? n_sigs : 1) * sizeof(permuted_dense *));
    int *out_src_p = (int *) SP_MALLOC((n_sigs + 1) * sizeof(int));
    out_src_p[0] = 0;
    /* sum of sig sizes in output order = cat_sig_data->len (same as sum
       in catalog order, just permuted). */
    int *out_src_data = (int *) SP_MALLOC(
        (cat_sig_data->len > 0 ? cat_sig_data->len : 1) * sizeof(int));
    int cursor3 = 0;
    for (int k = 0; k < n_sigs; k++)
    {
        int s = order[k];
        out_blocks[k] = sig_blocks[s];
        int sig_lo = cat_sig_p->data[s];
        int sig_hi = cat_sig_p->data[s + 1];
        int sig_sz = sig_hi - sig_lo;
        memcpy(out_src_data + cursor3, cat_sig_data->data + sig_lo,
               (size_t) sig_sz * sizeof(int));
        cursor3 += sig_sz;
        out_src_p[k + 1] = cursor3;
    }

    matrix *out = new_stacked_pd(m, n, n_sigs, out_blocks, out_src_p, out_src_data);

    free(out_blocks);
    free(out_src_p);
    free(out_src_data);
    free(sig_blocks);
    free(order);
    free(sig_rows_p);
    free(sig_rows_data);
    free(row_to_sig);
    iVec_free(cat_sig_p);
    iVec_free(cat_sig_data);
    free(row_blocks_p);
    free(row_blocks_data);

    return out;
}

matrix *coalesce_spd_alloc(const stacked_pd *src)
{
    return coalesce_spd_alloc_impl(src, true);
}

/* Internal: same as coalesce_spd_alloc but skips the
   pairwise-disjoint-cells precondition assert. Used by ATA_spd_alloc,
   which intentionally feeds a cell-overlapping pseudo-spd through the
   coalesce grouping algorithm. */
static matrix *coalesce_spd_alloc_unchecked(const stacked_pd *src)
{
    return coalesce_spd_alloc_impl(src, false);
}

void coalesce_spd_fill_values(const stacked_pd *src, stacked_pd *out)
{
    for (int k = 0; k < out->n_blocks; k++)
    {
        permuted_dense *out_k = out->blocks[k];
        int s_lo = out->src_block_idx_p[k];
        int s_hi = out->src_block_idx_p[k + 1];
        for (int t = s_lo; t < s_hi; t++)
        {
            const permuted_dense *src_k = src->blocks[out->src_block_idx[t]];
            for (int i = 0; i < src_k->m0; i++)
            {
                int out_i = out_k->row_inv[src_k->row_perm[i]];
                if (out_i < 0)
                {
                    continue; /* row not in this output PD */
                }
                for (int j = 0; j < src_k->n0; j++)
                {
                    int out_j = out_k->col_inv[src_k->col_perm[j]];
                    out_k->X[out_i * out_k->n0 + out_j] =
                        src_k->X[i * src_k->n0 + j];
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Transpose                                                           */
/* ------------------------------------------------------------------ */

matrix *transpose_spd_alloc(const stacked_pd *src)
{
    int n_blocks = src->n_blocks;
    permuted_dense **raw_blocks = (permuted_dense **) SP_MALLOC(
        (n_blocks > 0 ? n_blocks : 1) * sizeof(permuted_dense *));
    for (int k = 0; k < n_blocks; k++)
    {
        matrix *blk = (matrix *) src->blocks[k];
        raw_blocks[k] = (permuted_dense *) blk->transpose_alloc(blk);
    }

    /* Raw spd: dimensions swapped, identity src_block_idx_*; rows may
       overlap so use the unchecked constructor. */
    matrix *raw = new_stacked_pd_unchecked(src->base.n, src->base.m, n_blocks,
                                           raw_blocks, NULL, NULL);
    free(raw_blocks);

    matrix *out = coalesce_spd_alloc((stacked_pd *) raw);
    ((stacked_pd *) out)->work = (stacked_pd *) raw;
    return out;
}

void transpose_spd_fill_values(const stacked_pd *src, stacked_pd *out)
{
    stacked_pd *raw = out->work;
    for (int k = 0; k < src->n_blocks; k++)
    {
        matrix *src_blk = (matrix *) src->blocks[k];
        matrix *raw_blk = (matrix *) raw->blocks[k];
        src_blk->transpose_fill_values(src_blk, raw_blk);
    }
    coalesce_spd_fill_values(raw, out);
}

/* ------------------------------------------------------------------ */
/* copy_sparsity and DA                                                */
/* ------------------------------------------------------------------ */

matrix *copy_sparsity_spd_alloc(const stacked_pd *src)
{
    int n_blocks = src->n_blocks;
    permuted_dense **out_blocks = (permuted_dense **) SP_MALLOC(
        (n_blocks > 0 ? n_blocks : 1) * sizeof(permuted_dense *));
    for (int k = 0; k < n_blocks; k++)
    {
        matrix *src_blk = (matrix *) src->blocks[k];
        out_blocks[k] = (permuted_dense *) src_blk->copy_sparsity(src_blk);
    }
    matrix *out =
        new_stacked_pd(src->base.m, src->base.n, n_blocks, out_blocks, NULL, NULL);
    free(out_blocks);
    return out;
}

void DA_spd_fill_values(const double *d, const stacked_pd *A, stacked_pd *C)
{
    for (int k = 0; k < A->n_blocks; k++)
    {
        DA_pd_fill_values(d, A->blocks[k], C->blocks[k]);
    }
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
    permuted_dense **scratch_blocks = (permuted_dense **) SP_MALLOC(
        (n_blocks > 0 ? n_blocks : 1) * sizeof(permuted_dense *));
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

void ATDA_spd_fill_values(const stacked_pd *A, const double *d, stacked_pd *C)
{
    stacked_pd *scratch = C->work;

    /* Step 1: per-source-block ATDA into the scratch PDs. */
    for (int k = 0; k < A->n_blocks; k++)
    {
        ATDA_pd_fill_values(A->blocks[k], d, scratch->blocks[k]);
    }

    /* Step 2: zero output buffers (we're += accumulating). */
    for (int k = 0; k < C->n_blocks; k++)
    {
        permuted_dense *Ck = C->blocks[k];
        memset(Ck->X, 0, (size_t) Ck->m0 * Ck->n0 * sizeof(double));
    }

    /* Step 3: accumulate from scratch into output via row_inv/col_inv. */
    for (int k = 0; k < C->n_blocks; k++)
    {
        permuted_dense *out_k = C->blocks[k];
        int s_lo = C->src_block_idx_p[k];
        int s_hi = C->src_block_idx_p[k + 1];
        for (int t = s_lo; t < s_hi; t++)
        {
            const permuted_dense *src_k = scratch->blocks[C->src_block_idx[t]];
            for (int i = 0; i < src_k->m0; i++)
            {
                int out_i = out_k->row_inv[src_k->row_perm[i]];
                if (out_i < 0) continue; /* row not in this output PD */
                for (int j = 0; j < src_k->n0; j++)
                {
                    int out_j = out_k->col_inv[src_k->col_perm[j]];
                    /* out_j >= 0: out_k->col_perm contains src_k's
                       col_perm because src_k contributes to out_k. */
                    out_k->X[out_i * out_k->n0 + out_j] +=
                        src_k->X[i * src_k->n0 + j];
                }
            }
        }
    }
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

/* ------------------------------------------------------------------ */
/* BA_spd_spd: C = B @ A where both B and A are stacked_pds.           */
/* Thin loop over B's blocks, each call producing a PD via             */
/* BA_pd_spd_alloc; PDs assemble into output spd with disjoint rows.   */
/* ------------------------------------------------------------------ */

matrix *BA_spd_spd_alloc(const stacked_pd *B, const stacked_pd *A)
{
    permuted_dense **tmp_blocks = NULL;
    int *tmp_src = NULL;
    int *tmp_src_p = NULL;
    if (B->n_blocks > 0)
    {
        tmp_blocks =
            (permuted_dense **) SP_MALLOC(B->n_blocks * sizeof(permuted_dense *));
        tmp_src = (int *) SP_MALLOC(B->n_blocks * sizeof(int));
    }

    int out_n = 0;
    for (int k = 0; k < B->n_blocks; k++)
    {
        matrix *Ck = BA_pd_spd_alloc(B->blocks[k], A);
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

    /* one source per output block: identity prefix sum. Only allocate if
       B was non-empty; otherwise pass NULL/NULL to new_stacked_pd to
       satisfy its "both NULL or both non-NULL" invariant. */
    if (B->n_blocks > 0)
    {
        tmp_src_p = (int *) SP_MALLOC((out_n + 1) * sizeof(int));
        for (int k = 0; k <= out_n; k++)
        {
            tmp_src_p[k] = k;
        }
    }

    matrix *C =
        new_stacked_pd(B->base.m, A->base.n, out_n, tmp_blocks, tmp_src_p, tmp_src);
    free(tmp_blocks);
    free(tmp_src);
    free(tmp_src_p);
    return C;
}

void BA_spd_spd_fill_values(const stacked_pd *B, const stacked_pd *A, stacked_pd *C)
{
    for (int k = 0; k < C->n_blocks; k++)
    {
        int src = C->src_block_idx[C->src_block_idx_p[k]];
        BA_pd_spd_fill_values(B->blocks[src], A, C->blocks[k]);
    }
}
