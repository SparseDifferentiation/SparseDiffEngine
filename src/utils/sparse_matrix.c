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
#include "utils/sparse_matrix.h"

#include "utils/CSC_matrix.h"
#include "utils/CSR_sum.h"
#include "utils/linalg_sparse_matmuls.h"
#include "utils/matrix.h"
#include "utils/mini_numpy.h"
#include "utils/tracked_alloc.h"
#include "utils/utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void sparse_block_left_mult_vec(const matrix *self, const double *x,
                                       double *y, int p)
{
    const sparse_matrix *sm = (const sparse_matrix *) self;
    block_left_multiply_vec(sm->csr, x, y, p);
}

static CSC_matrix *sparse_block_left_mult_sparsity(const matrix *self,
                                                   const CSC_matrix *J, int p)
{
    const sparse_matrix *sm = (const sparse_matrix *) self;
    return block_left_multiply_fill_sparsity(sm->csr, J, p);
}

static void sparse_block_left_mult_values(const matrix *self, const CSC_matrix *J,
                                          CSC_matrix *C)
{
    const sparse_matrix *sm = (const sparse_matrix *) self;
    block_left_multiply_fill_values(sm->csr, J, C);
}

static void sparse_free(matrix *self)
{
    sparse_matrix *sm = (sparse_matrix *) self;
    free_CSR_matrix(sm->csr);
    free_CSC_matrix(sm->csc_cache);
    sp_free(sm->csc_iwork);
    sp_free(sm->bound_iwork);
    sp_free(sm);
}

/* Forward decl: ctor is referenced by copy_sparsity below. */
matrix *new_sparse_matrix(CSR_matrix *A);

/* Build the CSC_matrix cache structure if absent. Values are NOT filled here; call
   refresh_csc_values before consuming (it no-ops when the cache is already
   fresh). ATA_alloc only needs structure, so it's safe to call without a
   subsequent refresh. */
void sparse_matrix_ensure_csc_cache(sparse_matrix *sm)
{
    if (sm->csc_cache != NULL) return;
    sm->csc_iwork = (int *) sp_malloc(sm->csr->n * sizeof(int));
    sm->csc_cache = csr_to_csc_alloc(sm->csr, sm->csc_iwork);

    /* Deliberately stale: the structure is built during the symbolic phase,
       before values are meaningful, so the first refresh must fill. */
    sm->csc_seen = sm->base.values_version - 1;
}

static matrix *sparse_copy_sparsity(const matrix *self)
{
    const sparse_matrix *sm = (const sparse_matrix *) self;
    return new_sparse_matrix(new_csr_copy_sparsity(sm->csr));
}

static void sparse_DA_fill_values(const double *d, const matrix *self, matrix *out)
{
    const sparse_matrix *sm = (const sparse_matrix *) self;
    sparse_matrix *sm_out = (sparse_matrix *) out;
    DA_fill_values(d, sm->csr, sm_out->csr);
}

static matrix *sparse_ATA_alloc(matrix *self)
{
    sparse_matrix *sm = (sparse_matrix *) self;
    sparse_matrix_ensure_csc_cache(sm);
    return new_sparse_matrix(ATA_alloc(sm->csc_cache));
}

/* Call refresh_csc_values before consuming; it no-ops when the cache is
   already fresh. */
static void sparse_ATDA_fill_values(const matrix *self, const double *d, matrix *out)
{
    const sparse_matrix *sm = (const sparse_matrix *) self;
    sparse_matrix *sm_out = (sparse_matrix *) out;
    ATDA_fill_values(sm->csc_cache, d, sm_out->csr);
}

static CSR_matrix *sparse_to_csr(matrix *self)
{
    return ((sparse_matrix *) self)->csr;
}

static matrix *sparse_transpose_alloc(const matrix *self)
{
    const sparse_matrix *sm = (const sparse_matrix *) self;
    int *iwork = (int *) sp_malloc(sm->csr->n * sizeof(int));
    CSR_matrix *AT = AT_alloc(sm->csr, iwork);
    sparse_matrix *out = (sparse_matrix *) new_sparse_matrix(AT);
    out->bound_iwork = iwork;
    return &out->base;
}

static void sparse_transpose_fill_values(const matrix *self, matrix *out)
{
    const sparse_matrix *sm_in = (const sparse_matrix *) self;
    sparse_matrix *sm_out = (sparse_matrix *) out;
    AT_fill_values(sm_in->csr, sm_out->csr, sm_out->bound_iwork);
}

static matrix *sparse_row_gather_alloc(const matrix *self, const int *map, int m_out)
{
    const CSR_matrix *Jx = ((const sparse_matrix *) self)->csr;

    /* Exact output nnz: sum the selected rows' nnz. Jx->nnz is NOT an upper
       bound — repeated map entries select the same source row more than once
       (cvxpy#3442). Repeated gathers of dense rows can push the true count
       past INT_MAX, which a CSR cannot represent, so fail before wrapping. */
    int nnz = 0;
    for (int i = 0; i < m_out; i++)
    {
        int row = map[i];
        int len = Jx->p[row + 1] - Jx->p[row];
        if (len > INT_MAX - nnz)
        {
            fprintf(stderr, "Error in sparse_row_gather_alloc: gathered nnz "
                            "exceeds INT_MAX.\n");
            exit(1);
        }
        nnz += len;
    }
    CSR_matrix *J = new_CSR_matrix(m_out, self->n, nnz);

    J->p[0] = 0;
    for (int i = 0; i < m_out; i++)
    {
        int row = map[i];
        int len = Jx->p[row + 1] - Jx->p[row];
        memcpy(J->i + J->p[i], Jx->i + Jx->p[row], len * sizeof(int));
        J->p[i + 1] = J->p[i] + len;
    }
    J->nnz = J->p[m_out];

    sparse_matrix *out = (sparse_matrix *) new_sparse_matrix(J);
    out->bound_iwork = (int *) sp_malloc(m_out * sizeof(int));
    if (m_out > 0)
    {
        memcpy(out->bound_iwork, map, m_out * sizeof(int));
    }
    return &out->base;
}

static void sparse_row_gather_fill_values(const matrix *self, matrix *out)
{
    const CSR_matrix *Jx = ((const sparse_matrix *) self)->csr;
    sparse_matrix *sm_out = (sparse_matrix *) out;
    CSR_matrix *J = sm_out->csr;
    const int *map = sm_out->bound_iwork;
    assert(map != NULL && self->n == out->n);
    for (int i = 0; i < J->m; i++)
    {
        int len = J->p[i + 1] - J->p[i];
        memcpy(J->x + J->p[i], Jx->x + Jx->p[map[i]], len * sizeof(double));
    }
}

static matrix *sparse_diag_vec_alloc(matrix *self)
{
    CSR_matrix *Jx = ((sparse_matrix *) self)->csr;
    int n = self->m;
    int out_m = n * n;
    CSR_matrix *J = new_CSR_matrix(out_m, self->n, Jx->nnz);

    int nnz = 0;
    int next_diag = 0;
    for (int row = 0; row < out_m; row++)
    {
        J->p[row] = nnz;
        if (row == next_diag)
        {
            int child_row = row / (n + 1);
            int len = Jx->p[child_row + 1] - Jx->p[child_row];
            memcpy(J->i + nnz, Jx->i + Jx->p[child_row], len * sizeof(int));
            nnz += len;
            next_diag += n + 1;
        }
    }
    J->p[out_m] = nnz;
    J->nnz = nnz;
    return new_sparse_matrix(J);
}

static void sparse_diag_vec_fill_values(matrix *self, matrix *out)
{
    CSR_matrix *Jx = ((sparse_matrix *) self)->csr;
    CSR_matrix *J = ((sparse_matrix *) out)->csr;
    int n = self->m;
    for (int i = 0; i < n; i++)
    {
        int out_row = i * (n + 1);
        int len = J->p[out_row + 1] - J->p[out_row];
        memcpy(J->x + J->p[out_row], Jx->x + Jx->p[i], len * sizeof(double));
    }
}

/* Build CSC_matrix structure on first call; refill values from csr->x iff they
   changed since the last refresh (tracked via base.values_version). */
static void sparse_refresh_csc_values(matrix *self)
{
    sparse_matrix *sm = (sparse_matrix *) self;
    sparse_matrix_ensure_csc_cache(sm);
    if (sm->csc_seen == sm->base.values_version) return;
    csr_to_csc_fill_values(sm->csr, sm->csc_cache, sm->csc_iwork);
    sm->csc_seen = sm->base.values_version;
}

static matrix *sparse_row_reduce_alloc(const matrix *self, const int *group,
                                       int m_out)
{
    const CSR_matrix *A = ((const sparse_matrix *) self)->csr;
    int m = A->m;
    int n = A->n;

    /* Inverse of group by counting sort: rows[start[k] .. start[k+1]) lists the
       source rows of output row k. */
    int *start = (int *) sp_calloc(m_out + 1, sizeof(int));
    int *fill = (int *) sp_malloc(m_out * sizeof(int));
    int *rows = (int *) sp_malloc(m * sizeof(int));
    for (int i = 0; i < m; i++)
    {
        assert(group[i] >= 0 && group[i] < m_out);
        start[group[i] + 1]++;
    }
    for (int k = 0; k < m_out; k++)
    {
        start[k + 1] += start[k];
    }
    memcpy(fill, start, m_out * sizeof(int));
    for (int i = 0; i < m; i++)
    {
        rows[fill[group[i]]++] = i;
    }

    /* Output row k is the sorted union of its source rows' columns. Summing can
       only merge entries, so A->nnz bounds the output nnz. pos_of[j] is the
       position in J of column j within the output row being built; anything
       below row_start is left over from an earlier row and means "not yet in
       this row". */
    int cap = MIN(A->nnz, sat_mul_int(m_out, n));
    CSR_matrix *J = new_CSR_matrix(m_out, n, cap);
    int *map = (int *) sp_malloc(A->nnz * sizeof(int));
    int *pos_of = (int *) sp_malloc(n * sizeof(int));
    for (int j = 0; j < n; j++)
    {
        pos_of[j] = -1;
    }

    int nnz = 0;
    J->p[0] = 0;
    for (int k = 0; k < m_out; k++)
    {
        int row_start = nnz;
        for (int ii = start[k]; ii < start[k + 1]; ii++)
        {
            int i = rows[ii];
            for (int jj = A->p[i]; jj < A->p[i + 1]; jj++)
            {
                int j = A->i[jj];
                if (pos_of[j] < row_start)
                {
                    pos_of[j] = nnz;
                    J->i[nnz++] = j;
                }
            }
        }
        if (nnz > row_start)
        {
            sort_int_array(J->i + row_start, nnz - row_start);
        }
        J->p[k + 1] = nnz;

        /* sorting moved the columns: record final positions, then map every
           source entry of this output row to its output position */
        for (int jj = row_start; jj < nnz; jj++)
        {
            pos_of[J->i[jj]] = jj;
        }
        for (int ii = start[k]; ii < start[k + 1]; ii++)
        {
            int i = rows[ii];
            for (int jj = A->p[i]; jj < A->p[i + 1]; jj++)
            {
                map[jj] = pos_of[A->i[jj]];
            }
        }
    }
    J->nnz = nnz;
    CSR_trim(J);

    sp_free(pos_of);
    sp_free(rows);
    sp_free(fill);
    sp_free(start);

    sparse_matrix *out = (sparse_matrix *) new_sparse_matrix(J);
    out->bound_iwork = map;
    return &out->base;
}

static void sparse_row_reduce_fill_values(const matrix *self, matrix *out)
{
    if (out->nnz == 0) return;
    const CSR_matrix *A = ((const sparse_matrix *) self)->csr;
    sparse_matrix *sm_out = (sparse_matrix *) out;
    assert(sm_out->bound_iwork != NULL && self->n == out->n);
    memset(out->x, 0, out->nnz * sizeof(double));
    accumulator(A->x, A->nnz, sm_out->bound_iwork, out->x);
}

static void wire_vtable(sparse_matrix *sm)
{
    sm->base.block_left_mult_vec = sparse_block_left_mult_vec;
    sm->base.block_left_mult_sparsity = sparse_block_left_mult_sparsity;
    sm->base.block_left_mult_values = sparse_block_left_mult_values;
    sm->base.copy_sparsity = sparse_copy_sparsity;
    sm->base.DA_fill_values = sparse_DA_fill_values;
    sm->base.ATA_alloc = sparse_ATA_alloc;
    sm->base.ATDA_fill_values = sparse_ATDA_fill_values;
    sm->base.to_csr = sparse_to_csr;
    sm->base.transpose_alloc = sparse_transpose_alloc;
    sm->base.transpose_fill_values = sparse_transpose_fill_values;
    sm->base.row_gather_alloc = sparse_row_gather_alloc;
    sm->base.row_gather_fill_values = sparse_row_gather_fill_values;
    sm->base.diag_vec_alloc = sparse_diag_vec_alloc;
    sm->base.diag_vec_fill_values = sparse_diag_vec_fill_values;
    sm->base.row_reduce_alloc = sparse_row_reduce_alloc;
    sm->base.row_reduce_fill_values = sparse_row_reduce_fill_values;
    sm->base.refresh_csc_values = sparse_refresh_csc_values;
    sm->base.free_fn = sparse_free;
}

matrix *new_sparse_matrix(CSR_matrix *A)
{
    sparse_matrix *sm = (sparse_matrix *) sp_calloc(1, sizeof(sparse_matrix));
    sm->base.m = A->m;
    sm->base.n = A->n;
    sm->base.nnz = A->nnz;
    sm->base.x = A->x;
    wire_vtable(sm);
    sm->csr = A;
    return &sm->base;
}

matrix *new_sparse_matrix_alloc(int m, int n, int nnz)
{
    return new_sparse_matrix(new_CSR_matrix(m, n, nnz));
}

void sparse_matrix_trim(matrix *M)
{
    CSR_matrix *csr = ((sparse_matrix *) M)->csr;
    CSR_trim(csr);
    M->x = csr->x;
    M->nnz = csr->nnz;
}

matrix *sparse_matrix_trans(const sparse_matrix *self, int *iwork)
{
    CSR_matrix *AT = transpose(self->csr, iwork);
    sparse_matrix *sm = (sparse_matrix *) sp_calloc(1, sizeof(sparse_matrix));
    sm->base.m = AT->m;
    sm->base.n = AT->n;
    sm->base.nnz = AT->nnz;
    sm->base.x = AT->x;
    wire_vtable(sm);
    sm->csr = AT;
    return &sm->base;
}
