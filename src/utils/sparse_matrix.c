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
    free(sm->csc_iwork);
    free(sm->transpose_iwork);
    free(sm);
}

/* Forward decl: ctor is referenced by copy_sparsity below. */
matrix *new_sparse_matrix(CSR_matrix *A);

/* Build the CSC_matrix cache structure if absent. Values are NOT filled here; caller
   must call refresh_csc_values before consuming. ATA_alloc only needs structure,
   so it's safe to call without a subsequent refresh. */
void sparse_matrix_ensure_csc_cache(sparse_matrix *sm)
{
    if (sm->csc_cache != NULL) return;
    sm->csc_iwork = (int *) SP_MALLOC(sm->csr->n * sizeof(int));
    sm->csc_cache = csr_to_csc_alloc(sm->csr, sm->csc_iwork);
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

/* Caller must have called refresh_csc_values since the last change to csr->x. */
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
    int *iwork = (int *) SP_MALLOC(sm->csr->n * sizeof(int));
    CSR_matrix *AT = AT_alloc(sm->csr, iwork);
    sparse_matrix *out = (sparse_matrix *) new_sparse_matrix(AT);
    out->transpose_iwork = iwork;
    return &out->base;
}

static void sparse_transpose_fill_values(const matrix *self, matrix *out)
{
    const sparse_matrix *sm_in = (const sparse_matrix *) self;
    sparse_matrix *sm_out = (sparse_matrix *) out;
    AT_fill_values(sm_in->csr, sm_out->csr, sm_out->transpose_iwork);
}

static matrix *sparse_index_alloc(matrix *self, const int *indices, int n_idxs)
{
    CSR_matrix *Jx = ((sparse_matrix *) self)->csr;
    CSR_matrix *J = new_CSR_matrix(n_idxs, self->n, Jx->nnz);

    J->p[0] = 0;
    for (int i = 0; i < n_idxs; i++)
    {
        int row = indices[i];
        int len = Jx->p[row + 1] - Jx->p[row];
        memcpy(J->i + J->p[i], Jx->i + Jx->p[row], len * sizeof(int));
        J->p[i + 1] = J->p[i] + len;
    }
    J->nnz = J->p[n_idxs];
    return new_sparse_matrix(J);
}

static void sparse_index_fill_values(matrix *self, const int *indices, int n_idxs,
                                     matrix *out)
{
    CSR_matrix *Jx = ((sparse_matrix *) self)->csr;
    CSR_matrix *J = ((sparse_matrix *) out)->csr;
    for (int i = 0; i < n_idxs; i++)
    {
        int len = J->p[i + 1] - J->p[i];
        memcpy(J->x + J->p[i], Jx->x + Jx->p[indices[i]], len * sizeof(double));
    }
}

static matrix *sparse_promote_alloc(matrix *self, int size)
{
    CSR_matrix *Jx = ((sparse_matrix *) self)->csr;
    int row_nnz = Jx->nnz;
    CSR_matrix *J = new_CSR_matrix(size, self->n, size * row_nnz);

    for (int row = 0; row < size; row++)
    {
        J->p[row] = row * row_nnz;
        memcpy(J->i + row * row_nnz, Jx->i, row_nnz * sizeof(int));
    }
    J->p[size] = size * row_nnz;
    J->nnz = size * row_nnz;
    return new_sparse_matrix(J);
}

static void sparse_promote_fill_values(matrix *self, matrix *out)
{
    CSR_matrix *Jx = ((sparse_matrix *) self)->csr;
    int row_nnz = Jx->nnz;
    for (int row = 0; row < out->m; row++)
    {
        memcpy(out->x + row * row_nnz, Jx->x, row_nnz * sizeof(double));
    }
}

static matrix *sparse_broadcast_alloc(matrix *self, broadcast_type type, int d1,
                                      int d2)
{
    CSR_matrix *Jx = ((sparse_matrix *) self)->csr;
    int out_m = d1 * d2;
    int total_nnz;
    if (type == BROADCAST_ROW)
    {
        total_nnz = Jx->nnz * d1;
    }
    else if (type == BROADCAST_COL)
    {
        total_nnz = Jx->nnz * d2;
    }
    else /* BROADCAST_SCALAR */
    {
        total_nnz = Jx->nnz * out_m;
    }

    CSR_matrix *J = new_CSR_matrix(out_m, self->n, total_nnz);

    if (type == BROADCAST_ROW)
    {
        int acc = 0;
        for (int i = 0; i < d2; i++)
        {
            int nnz_in_row = Jx->p[i + 1] - Jx->p[i];
            tile_int(J->i + acc, Jx->i + Jx->p[i], nnz_in_row, d1);
            for (int rep = 0; rep < d1; rep++)
            {
                J->p[i * d1 + rep] = acc;
                acc += nnz_in_row;
            }
        }
        J->p[out_m] = total_nnz;
    }
    else if (type == BROADCAST_COL)
    {
        tile_int(J->i, Jx->i, Jx->nnz, d2);
        int offset = 0;
        for (int i = 0; i < d2; i++)
        {
            for (int j = 0; j < d1; j++)
            {
                int nnz_in_row = Jx->p[j + 1] - Jx->p[j];
                J->p[i * d1 + j] = offset;
                offset += nnz_in_row;
            }
        }
        J->p[out_m] = total_nnz;
    }
    else /* BROADCAST_SCALAR */
    {
        tile_int(J->i, Jx->i, Jx->nnz, out_m);
        int row_nnz = Jx->nnz;
        for (int i = 0; i < out_m; i++)
        {
            J->p[i] = i * row_nnz;
        }
        J->p[out_m] = total_nnz;
    }
    return new_sparse_matrix(J);
}

static void sparse_broadcast_fill_values(matrix *self, broadcast_type type, int d1,
                                         int d2, matrix *out)
{
    CSR_matrix *Jx = ((sparse_matrix *) self)->csr;
    if (type == BROADCAST_ROW)
    {
        int acc = 0;
        for (int i = 0; i < d2; i++)
        {
            int nnz_in_row = Jx->p[i + 1] - Jx->p[i];
            tile_double(out->x + acc, Jx->x + Jx->p[i], nnz_in_row, d1);
            acc += nnz_in_row * d1;
        }
    }
    else if (type == BROADCAST_COL)
    {
        tile_double(out->x, Jx->x, Jx->nnz, d2);
    }
    else /* BROADCAST_SCALAR */
    {
        tile_double(out->x, Jx->x, Jx->nnz, d1 * d2);
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

/* C = sum over all rows of self. Output is a single-row sparse_matrix
   with the union of self's column indices. idx_map[j] (j in [0, self->nnz))
   gives the position in C's CSR x-array for each of self's CSR cells. */
static matrix *sparse_sum_all_rows_alloc(matrix *self, int *idx_map)
{
    CSR_matrix *A = self->to_csr(self);
    int max_out_nnz = (int) MIN((size_t) A->nnz, (size_t) A->n);
    CSR_matrix *out = new_CSR_matrix(1, A->n, max_out_nnz);
    int *iwork = (int *) SP_MALLOC(MAX(A->n, A->nnz) * sizeof(int));
    sum_all_rows_csr_alloc(A, out, iwork, idx_map);
    free(iwork);
    return new_sparse_matrix(out);
}

/* Build CSC_matrix structure on first call; refill values from csr->x on every call.
 */
static void sparse_refresh_csc_values(matrix *self)
{
    sparse_matrix *sm = (sparse_matrix *) self;
    sparse_matrix_ensure_csc_cache(sm);
    csr_to_csc_fill_values(sm->csr, sm->csc_cache, sm->csc_iwork);
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
    sm->base.index_alloc = sparse_index_alloc;
    sm->base.index_fill_values = sparse_index_fill_values;
    sm->base.promote_alloc = sparse_promote_alloc;
    sm->base.promote_fill_values = sparse_promote_fill_values;
    sm->base.broadcast_alloc = sparse_broadcast_alloc;
    sm->base.broadcast_fill_values = sparse_broadcast_fill_values;
    sm->base.diag_vec_alloc = sparse_diag_vec_alloc;
    sm->base.diag_vec_fill_values = sparse_diag_vec_fill_values;
    sm->base.sum_all_rows_alloc = sparse_sum_all_rows_alloc;
    sm->base.refresh_csc_values = sparse_refresh_csc_values;
    sm->base.free_fn = sparse_free;
}

matrix *new_sparse_matrix(CSR_matrix *A)
{
    sparse_matrix *sm = (sparse_matrix *) SP_CALLOC(1, sizeof(sparse_matrix));
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

matrix *sparse_matrix_trans(const sparse_matrix *self, int *iwork)
{
    CSR_matrix *AT = transpose(self->csr, iwork);
    sparse_matrix *sm = (sparse_matrix *) SP_CALLOC(1, sizeof(sparse_matrix));
    sm->base.m = AT->m;
    sm->base.n = AT->n;
    sm->base.nnz = AT->nnz;
    sm->base.x = AT->x;
    wire_vtable(sm);
    sm->csr = AT;
    return &sm->base;
}
