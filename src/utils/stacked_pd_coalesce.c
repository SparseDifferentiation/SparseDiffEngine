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

#include "utils/iVec.h"
#include "utils/permuted_dense.h"
#include "utils/tracked_alloc.h"
#include "utils/utils.h"
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/* A CSR-style ragged 2D int array, indexed by an integer key:
     csr->data[csr->p[k]..csr->p[k+1])  — the int list for key k
   Invariants: csr->p has length n+1, csr->p[0] == 0, csr->data has
   length csr->p[n]. Read-only after construction. */
typedef struct
{
    int *p;
    int *data;
    int n;
} int_csr;

static int_csr *int_csr_create(int *p, int *data, int n)
{
    int_csr *csr = (int_csr *) sp_malloc(sizeof(int_csr));
    csr->p = p;
    csr->data = data;
    csr->n = n;
    return csr;
}

static void int_csr_free(int_csr *csr)
{
    free(csr->p);
    free(csr->data);
    free(csr);
}

/* Slice of csr->data for key k. Sets *out_len. */
static inline const int *int_csr_get(const int_csr *csr, int k, int *out_len)
{
    int lo = csr->p[k];
    int hi = csr->p[k + 1];
    *out_len = hi - lo;
    return csr->data + lo;
}

/* Look up 'sig' in the catalog represented as the CSR pair (cat_p, cat_data). Return
   the existing signature index if found; otherwise append it and return the new
   index. */
static int lookup_or_append_signature(iVec *cat_p, iVec *cat_data, const int *sig,
                                      int sig_len)
{
    int n_sigs = cat_p->len - 1;

    /* compare the signature 'sig' with all the existing signatures */
    for (int s = 0; s < n_sigs; s++)
    {
        int lo = cat_p->data[s];
        int hi = cat_p->data[s + 1];
        if (hi - lo == sig_len &&
            memcmp(cat_data->data + lo, sig, sig_len * sizeof(int)) == 0)
        {
            return s;
        }
    }

    /* signature not found, append it */
    iVec_append_array(cat_data, sig, sig_len);
    iVec_append(cat_p, cat_data->len);
    return n_sigs;
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

static int_csr *build_row_to_blocks_csr(const stacked_pd *A, int *scratch)
{
    int m = A->base.m;
    int n_blocks = A->n_blocks;

    int *p = (int *) sp_calloc(m + 1, sizeof(int));
    for (int k = 0; k < n_blocks; k++)
    {
        permuted_dense *Ak = A->blocks[k];
        for (int i = 0; i < Ak->m0; i++)
        {
            p[Ak->row_perm[i] + 1]++;
        }
    }
    cumsum(p, m);

    int *data = (int *) sp_malloc(p[m] * sizeof(int));
    int *blocks_added_per_row = scratch;
    for (int k = 0; k < n_blocks; k++)
    {
        permuted_dense *Ak = A->blocks[k];
        for (int i = 0; i < Ak->m0; i++)
        {
            int row = Ak->row_perm[i];
            data[p[row] + blocks_added_per_row[row]++] = k;
        }
    }

    return int_csr_create(p, data, m);
}

static void catalog_signatures(const int_csr *row_to_blocks, iVec *catalog_sig_p,
                               iVec *catalog_sig_data, int **row_to_sig_out)
{
    int m = row_to_blocks->n;
    int *row_to_sig = (int *) sp_malloc(m * sizeof(int));
    for (int i = 0; i < m; i++)
    {
        int signature_len;
        const int *sig = int_csr_get(row_to_blocks, i, &signature_len);

        /* no block contains row i */
        if (signature_len == 0)
        {
            row_to_sig[i] = -1;
            continue;
        }
        row_to_sig[i] = lookup_or_append_signature(catalog_sig_p, catalog_sig_data,
                                                   sig, signature_len);
    }
    *row_to_sig_out = row_to_sig;
}

static int_csr *group_rows_by_signature(int m, int n_sigs, const int *row_to_sig,
                                        int *scratch)
{
    int *p = (int *) sp_calloc(n_sigs + 1, sizeof(int));
    for (int i = 0; i < m; i++)
    {
        if (row_to_sig[i] >= 0)
        {
            p[row_to_sig[i] + 1]++;
        }
    }
    cumsum(p, n_sigs);

    int *data = (int *) sp_malloc(p[n_sigs] * sizeof(int));
    int *rows_added_per_sig = scratch;
    for (int i = 0; i < m; i++)
    {
        int s = row_to_sig[i];
        if (s >= 0)
        {
            data[p[s] + rows_added_per_sig[s]++] = i;
        }
    }

    return int_csr_create(p, data, n_sigs);
}

/* Allocate one output PD per signature. The PD for signature s has
   row_perm = sig_to_rows[s] and col_perm = sorted union of
   A->blocks[k]->col_perm for k ranging over the signature's source-block
   IDs. Returns a newly-allocated array of n_sigs PD pointers; caller
   owns the array (PDs themselves are taken over by the eventual
   stacked_pd). */
static permuted_dense **build_output_blocks(const stacked_pd *A,
                                            const iVec *catalog_sig_p,
                                            const iVec *catalog_sig_data,
                                            const int_csr *sig_to_rows)
{
    int m = A->base.m;
    int n = A->base.n;
    int n_sigs = catalog_sig_p->len - 1;

    permuted_dense **out_blocks = (permuted_dense **) sp_malloc(
        (n_sigs > 0 ? n_sigs : 1) * sizeof(permuted_dense *));
    iVec *col_union = iVec_new(8);
    /* col_arrs / col_lens: scratch passed to sorted_union_int_arrays.
       Max sig size is bounded by A->n_blocks, so size once outside the loop. */
    const int **col_arrs = (const int **) sp_malloc(A->n_blocks * sizeof(int *));
    int *col_lens = (int *) sp_malloc(A->n_blocks * sizeof(int));

    for (int s = 0; s < n_sigs; s++)
    {
        int sig_lo = catalog_sig_p->data[s];
        int sig_sz = catalog_sig_p->data[s + 1] - sig_lo;
        const int *sig_ids = catalog_sig_data->data + sig_lo;

        for (int t = 0; t < sig_sz; t++)
        {
            permuted_dense *blk = A->blocks[sig_ids[t]];
            col_arrs[t] = blk->col_perm;
            col_lens[t] = blk->n0;
        }
        sorted_union_int_arrays(col_arrs, col_lens, sig_sz, col_union);

        int m0;
        const int *row_perm = int_csr_get(sig_to_rows, s, &m0);
        out_blocks[s] = (permuted_dense *) new_permuted_dense(
            m, n, m0, col_union->len, row_perm, col_union->data, NULL);
    }

    iVec_free(col_union);
    free(col_arrs);
    free(col_lens);
    return out_blocks;
}

static matrix *coalesce_spd_alloc_impl(const stacked_pd *A,
                                       bool check_disjoint_cells)
{
    int m = A->base.m;
    int n = A->base.n;
    int *iwork = (int *) sp_calloc(m, sizeof(int));

    // ---------------------------------------------------------------------------
    // For each global row in A, catalog which blocks contain it.
    // ---------------------------------------------------------------------------
    int_csr *row_to_blocks = build_row_to_blocks_csr(A, iwork);

    // ------------------------------------------------------------------------------
    // Catalog unique signatures (sets of source blocks sharing each row), and assign
    // each row to a signature id. Each unique signature will become one output
    // block. The array 'row_to_sig' maps each row to its signature id; rows with the
    // same signature have the same set of source blocks and thus land in the same
    // output block.
    // ------------------------------------------------------------------------------
    iVec *catalog_sig_p = iVec_new(8);
    iVec_append(catalog_sig_p, 0);
    iVec *catalog_sig_data = iVec_new(16);
    int *row_to_sig;
    catalog_signatures(row_to_blocks, catalog_sig_p, catalog_sig_data, &row_to_sig);
    int n_sigs = catalog_sig_p->len - 1;

#ifndef NDEBUG
    if (check_disjoint_cells)
    {
        assert_disjoint_cells(A, catalog_sig_p->data, catalog_sig_data->data,
                              n_sigs);
    }
#else
    (void) check_disjoint_cells;
#endif

    // -----------------------------------------------------------------------------
    //  Group rows by signature. For each unique signature s, which rows have it?
    // -----------------------------------------------------------------------------
    memset(iwork, 0, n_sigs * sizeof(int));
    int_csr *sig_to_rows = group_rows_by_signature(m, n_sigs, row_to_sig, iwork);

    // -----------------------------------------------------------------------------
    //  Build one output PD per signature, then hand off to new_stacked_pd. Catalog
    //  order is already sorted by min row (pass 2 introduces signatures in row
    //  scan order), so the output uses catalog order directly — no reorder needed.
    //  new_stacked_pd memcpy's src_block_idx_p/_data, so we pass the catalog's
    //  arrays straight through.
    // -----------------------------------------------------------------------------
    permuted_dense **out_blocks =
        build_output_blocks(A, catalog_sig_p, catalog_sig_data, sig_to_rows);

    matrix *C = new_stacked_pd(m, n, n_sigs, out_blocks, catalog_sig_p->data,
                               catalog_sig_data->data);

    free(out_blocks);
    int_csr_free(sig_to_rows);
    free(row_to_sig);
    iVec_free(catalog_sig_p);
    iVec_free(catalog_sig_data);
    int_csr_free(row_to_blocks);
    free(iwork);

    return C;
}

matrix *coalesce_spd_alloc(const stacked_pd *src)
{
    return coalesce_spd_alloc_impl(src, true);
}

matrix *coalesce_spd_alloc_unchecked(const stacked_pd *src)
{
    return coalesce_spd_alloc_impl(src, false);
}

/* Scatter src's blocks into out's blocks, following out's src_block_idx_*
   map. If accumulate is true, += into out (caller must zero out->base.x
   first); otherwise = (precondition: source cells are pairwise disjoint
   across blocks targeting the same out block). `accumulate` is a
   compile-time constant at every call site, so the inner-loop branch is
   folded away when this is inlined into the public wrappers. */
static inline void coalesce_spd_scatter(const stacked_pd *src, stacked_pd *out,
                                        bool accumulate)
{
    /* for each block out_k of out */
    for (int k = 0; k < out->n_blocks; k++)
    {
        permuted_dense *out_k = out->blocks[k];
        int s_lo = out->src_block_idx_p[k];
        int s_hi = out->src_block_idx_p[k + 1];

        /* for each block src_k in src contributing to out_k */
        for (int t = s_lo; t < s_hi; t++)
        {
            const permuted_dense *src_k = src->blocks[out->src_block_idx[t]];

            /* for each row in src_k */
            for (int i = 0; i < src_k->m0; i++)
            {
                int out_i = out_k->row_inv[src_k->row_perm[i]];
                if (out_i < 0)
                {
                    /* this row in src_k does not contribute to out_k */
                    continue;
                }

                double *out_row = out_k->X + out_i * out_k->n0;
                const double *src_row = src_k->X + i * src_k->n0;

                /* place each col in src_row at its correct position in out_row */
                for (int j = 0; j < src_k->n0; j++)
                {
                    int out_j = out_k->col_inv[src_k->col_perm[j]];
                    assert(out_j >= 0);
                    if (accumulate)
                    {
                        out_row[out_j] += src_row[j];
                    }
                    else
                    {
                        out_row[out_j] = src_row[j];
                    }
                }
            }
        }
    }
}

void coalesce_spd_fill_values(const stacked_pd *src, stacked_pd *out)
{
    coalesce_spd_scatter(src, out, false);
}

void coalesce_spd_fill_values_accumulate(const stacked_pd *src, stacked_pd *out)
{
    coalesce_spd_scatter(src, out, true);
}
