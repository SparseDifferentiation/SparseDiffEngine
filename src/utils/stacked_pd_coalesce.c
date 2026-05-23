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
    int_csr *csr = (int_csr *) SP_MALLOC(sizeof(int_csr));
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
static void assert_disjoint_cells(const spd_per_block_shape *shapes,
                                  const int *cat_sig_p, const int *cat_sig_data,
                                  int n_sigs)
{
    for (int s = 0; s < n_sigs; s++)
    {
        int sig_lo = cat_sig_p[s];
        int sig_hi = cat_sig_p[s + 1];
        for (int u = sig_lo; u < sig_hi; u++)
        {
            for (int v = u + 1; v < sig_hi; v++)
            {
                const spd_per_block_shape *bu = &shapes[cat_sig_data[u]];
                const spd_per_block_shape *bv = &shapes[cat_sig_data[v]];
                assert(!has_overlap(bu->col_perm, bu->n0, bv->col_perm, bv->n0, 0) &&
                       "coalesce: source blocks share both a row and a col");
            }
        }
    }
}
#endif

static int_csr *build_row_to_blocks_csr(const spd_per_block_shape *shapes,
                                        int n_blocks, int m, int *scratch)
{
    int *p = (int *) SP_CALLOC(m + 1, sizeof(int));
    for (int k = 0; k < n_blocks; k++)
    {
        const spd_per_block_shape *sk = &shapes[k];
        for (int i = 0; i < sk->m0; i++)
        {
            p[sk->row_perm[i] + 1]++;
        }
    }
    cumsum(p, m);

    int *data = (int *) SP_MALLOC(p[m] * sizeof(int));
    int *blocks_added_per_row = scratch;
    for (int k = 0; k < n_blocks; k++)
    {
        const spd_per_block_shape *sk = &shapes[k];
        for (int i = 0; i < sk->m0; i++)
        {
            int row = sk->row_perm[i];
            data[p[row] + blocks_added_per_row[row]++] = k;
        }
    }

    return int_csr_create(p, data, m);
}

static void catalog_signatures(const int_csr *row_to_blocks, iVec *catalog_sig_p,
                               iVec *catalog_sig_data, int **row_to_sig_out)
{
    int m = row_to_blocks->n;
    int *row_to_sig = (int *) SP_MALLOC(m * sizeof(int));
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
    int *p = (int *) SP_CALLOC(n_sigs + 1, sizeof(int));
    for (int i = 0; i < m; i++)
    {
        if (row_to_sig[i] >= 0)
        {
            p[row_to_sig[i] + 1]++;
        }
    }
    cumsum(p, n_sigs);

    int *data = (int *) SP_MALLOC(p[n_sigs] * sizeof(int));
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

/* Allocate one output PD per signature, all sharing a single contiguous X
   buffer. The PD for signature s has row_perm = sig_to_rows[s] and
   col_perm = sorted union of shapes[k].col_perm for k ranging over the
   signature's source-block IDs. Output blocks are created via
   new_permuted_dense_view (owns_X = false) pointing into *shared_x_out
   at the right offsets. *shared_x_out is freshly SP_MALLOC'd here; the
   eventual stacked_pd takes ownership of it. Returns a newly-allocated
   array of n_sigs PD pointers; caller owns the array (PDs themselves are
   taken over by the eventual stacked_pd). */
static permuted_dense **
build_output_blocks(const spd_per_block_shape *shapes, int n_input_blocks, int m,
                    int n, const iVec *catalog_sig_p, const iVec *catalog_sig_data,
                    const int_csr *sig_to_rows, double **shared_x_out)
{
    int n_sigs = catalog_sig_p->len - 1;

    /* Two passes over the sigs: pass 1 sums output nnz so we can allocate
       the shared X buffer once; pass 2 recomputes each sig's col_union and
       wires the view PDs at sequential offsets into shared X. The
       col_union work is symbolic (no X allocation) and small, so we trade
       the recompute for simpler bookkeeping. */
    iVec *col_union = iVec_new(8);
    const int **col_arrs = (const int **) SP_MALLOC(n_input_blocks * sizeof(int *));
    int *col_lens = (int *) SP_MALLOC(n_input_blocks * sizeof(int));

    /* Pass 1: total nnz. */
    size_t total_nnz = 0;
    for (int s = 0; s < n_sigs; s++)
    {
        int sig_lo = catalog_sig_p->data[s];
        int sig_sz = catalog_sig_p->data[s + 1] - sig_lo;
        const int *sig_ids = catalog_sig_data->data + sig_lo;
        for (int t = 0; t < sig_sz; t++)
        {
            col_arrs[t] = shapes[sig_ids[t]].col_perm;
            col_lens[t] = shapes[sig_ids[t]].n0;
        }
        sorted_union_int_arrays(col_arrs, col_lens, sig_sz, col_union);

        int m0;
        (void) int_csr_get(sig_to_rows, s, &m0);
        total_nnz += (size_t) m0 * (size_t) col_union->len;
    }

    double *shared_x = (double *) SP_MALLOC(total_nnz * sizeof(double));

    /* Pass 2: build view PDs at sequential offsets. */
    permuted_dense **out_blocks = (permuted_dense **) SP_MALLOC(
        (n_sigs > 0 ? n_sigs : 1) * sizeof(permuted_dense *));
    size_t cursor = 0;
    for (int s = 0; s < n_sigs; s++)
    {
        int sig_lo = catalog_sig_p->data[s];
        int sig_sz = catalog_sig_p->data[s + 1] - sig_lo;
        const int *sig_ids = catalog_sig_data->data + sig_lo;
        for (int t = 0; t < sig_sz; t++)
        {
            col_arrs[t] = shapes[sig_ids[t]].col_perm;
            col_lens[t] = shapes[sig_ids[t]].n0;
        }
        sorted_union_int_arrays(col_arrs, col_lens, sig_sz, col_union);

        int m0;
        const int *row_perm = int_csr_get(sig_to_rows, s, &m0);
        out_blocks[s] = (permuted_dense *) new_permuted_dense_view(
            m, n, m0, col_union->len, row_perm, col_union->data, shared_x + cursor);
        cursor += (size_t) m0 * (size_t) col_union->len;
    }

    iVec_free(col_union);
    free(col_arrs);
    free(col_lens);

    *shared_x_out = shared_x;
    return out_blocks;
}

/* Symbolic coalesce driven by an array of per-block shapes (no actual
   stacked_pd or per-block X buffers needed). Returns an output stacked_pd
   whose blocks are views into one shared X buffer that the output owns.
   The output's `src_block_idx_p` / `src_block_idx` already hold the
   forward map, so callers that need to invert it (input block → output
   blocks) can read those fields directly. */
static matrix *coalesce_alloc_from_shapes(const spd_per_block_shape *shapes,
                                          int n_input_blocks, int m, int n,
                                          bool check_disjoint_cells)
{
    int *iwork = (int *) SP_CALLOC(m, sizeof(int));

    int_csr *row_to_blocks =
        build_row_to_blocks_csr(shapes, n_input_blocks, m, iwork);

    iVec *catalog_sig_p = iVec_new(8);
    iVec_append(catalog_sig_p, 0);
    iVec *catalog_sig_data = iVec_new(16);
    int *row_to_sig;
    catalog_signatures(row_to_blocks, catalog_sig_p, catalog_sig_data, &row_to_sig);
    int n_sigs = catalog_sig_p->len - 1;

#ifndef NDEBUG
    if (check_disjoint_cells)
    {
        assert_disjoint_cells(shapes, catalog_sig_p->data, catalog_sig_data->data,
                              n_sigs);
    }
#else
    (void) check_disjoint_cells;
#endif

    memset(iwork, 0, n_sigs * sizeof(int));
    int_csr *sig_to_rows = group_rows_by_signature(m, n_sigs, row_to_sig, iwork);

    double *shared_x = NULL;
    permuted_dense **out_blocks =
        build_output_blocks(shapes, n_input_blocks, m, n, catalog_sig_p,
                            catalog_sig_data, sig_to_rows, &shared_x);

    matrix *C =
        new_stacked_pd_borrowed_x(m, n, n_sigs, out_blocks, catalog_sig_p->data,
                                  catalog_sig_data->data, shared_x);

    free(out_blocks);
    int_csr_free(sig_to_rows);
    free(row_to_sig);
    iVec_free(catalog_sig_p);
    iVec_free(catalog_sig_data);
    int_csr_free(row_to_blocks);
    free(iwork);

    return C;
}

/* Bridge: build a shapes array from a real stacked_pd and delegate. */
static matrix *coalesce_spd_alloc_impl(const stacked_pd *A,
                                       bool check_disjoint_cells)
{
    int n_blocks = A->n_blocks;
    spd_per_block_shape *shapes = (spd_per_block_shape *) SP_MALLOC(
        (n_blocks > 0 ? n_blocks : 1) * sizeof(spd_per_block_shape));
    for (int k = 0; k < n_blocks; k++)
    {
        const permuted_dense *Ak = A->blocks[k];
        shapes[k].m0 = Ak->m0;
        shapes[k].n0 = Ak->n0;
        shapes[k].row_perm = Ak->row_perm;
        shapes[k].col_perm = Ak->col_perm;
        shapes[k].scratch_dwork_needed = 0;
        shapes[k].scratch_iwork_needed = 0;
    }

    matrix *C = coalesce_alloc_from_shapes(shapes, n_blocks, A->base.m, A->base.n,
                                           check_disjoint_cells);

    free(shapes);
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

/* Public entry for stacked_pd_linalg's blockwise alloc: same symbolic
   coalesce, but driven by a caller-supplied shapes array (no per-block
   PDs needed). The output's src_block_idx_p / src_block_idx hold the
   forward map; callers that need the inverse read those fields directly. */
matrix *coalesce_spd_alloc_from_shapes_unchecked(const spd_per_block_shape *shapes,
                                                 int n_input_blocks, int m, int n)
{
    return coalesce_alloc_from_shapes(shapes, n_input_blocks, m, n, false);
}

/* Scatter the cells of one source PD into one output PD. If accumulate is
   true, += into out_k (caller must zero out_k->X before the first call);
   otherwise =. `accumulate` is a compile-time constant at every call site,
   so the inner-loop branch is folded away when this is inlined. */
static inline void scatter_one_source_into_one_output(const permuted_dense *src_k,
                                                      permuted_dense *out_k,
                                                      bool accumulate)
{
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

/* Scatter src's blocks into out's blocks, following out's src_block_idx_*
   map. */
static inline void coalesce_spd_scatter(const stacked_pd *src, stacked_pd *out,
                                        bool accumulate)
{
    /* for each block out_k of out */
    for (int k = 0; k < out->n_blocks; k++)
    {
        permuted_dense *out_k = out->blocks[k];
        int s_lo = out->src_block_idx_p[k];
        int s_hi = out->src_block_idx_p[k + 1];

        for (int t = s_lo; t < s_hi; t++)
        {
            const permuted_dense *src_k = src->blocks[out->src_block_idx[t]];
            scatter_one_source_into_one_output(src_k, out_k, accumulate);
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

void scatter_one_source_into_one_output_accumulate(const permuted_dense *src_k,
                                                   permuted_dense *out_k)
{
    scatter_one_source_into_one_output(src_k, out_k, true);
}
