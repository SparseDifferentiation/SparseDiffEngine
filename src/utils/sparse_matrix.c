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
#include "utils/CSC_Matrix.h"
#include "utils/linalg_sparse_matmuls.h"
#include "utils/matrix.h"
#include "utils/tracked_alloc.h"
#include <stdlib.h>
#include <string.h>

static void sparse_block_left_mult_vec(const Matrix *self, const double *x,
                                       double *y, int p)
{
    const Sparse_Matrix *sm = (const Sparse_Matrix *) self;
    block_left_multiply_vec(sm->csr, x, y, p);
}

static CSC_Matrix *sparse_block_left_mult_sparsity(const Matrix *self,
                                                   const CSC_Matrix *J, int p)
{
    const Sparse_Matrix *sm = (const Sparse_Matrix *) self;
    return block_left_multiply_fill_sparsity(sm->csr, J, p);
}

static void sparse_block_left_mult_values(const Matrix *self, const CSC_Matrix *J,
                                          CSC_Matrix *C)
{
    const Sparse_Matrix *sm = (const Sparse_Matrix *) self;
    block_left_multiply_fill_values(sm->csr, J, C);
}

static void sparse_update_values(Matrix *self, const double *new_values)
{
    Sparse_Matrix *sm = (Sparse_Matrix *) self;
    memcpy(sm->csr->x, new_values, sm->csr->nnz * sizeof(double));
}

static void sparse_free(Matrix *self)
{
    Sparse_Matrix *sm = (Sparse_Matrix *) self;
    free_csr_matrix(sm->csr);
    free_csc_matrix(sm->csc_cache);
    free(sm->csc_iwork);
    free(sm);
}

/* Forward decl: ctor is referenced by copy_sparsity below. */
Matrix *new_sparse_matrix(CSR_Matrix *A);

/* Build the CSC cache structure if absent. Values are NOT filled here; caller
   must call refresh_csc_values before consuming. ATA_alloc only needs structure,
   so it's safe to call after build_csc_structure alone. */
static void build_csc_structure_if_absent(Sparse_Matrix *sm)
{
    if (sm->csc_cache != NULL) return;
    sm->csc_iwork = (int *) SP_MALLOC(sm->csr->n * sizeof(int));
    sm->csc_cache = csr_to_csc_alloc(sm->csr, sm->csc_iwork);
}

static Matrix *sparse_copy_sparsity(const Matrix *self)
{
    const Sparse_Matrix *sm = (const Sparse_Matrix *) self;
    return new_sparse_matrix(new_csr_copy_sparsity(sm->csr));
}

static void sparse_DA_fill_values(const double *d, const Matrix *self, Matrix *out)
{
    const Sparse_Matrix *sm = (const Sparse_Matrix *) self;
    Sparse_Matrix *sm_out = (Sparse_Matrix *) out;
    DA_fill_values(d, sm->csr, sm_out->csr);
}

static Matrix *sparse_ATA_alloc(Matrix *self)
{
    Sparse_Matrix *sm = (Sparse_Matrix *) self;
    build_csc_structure_if_absent(sm);
    return new_sparse_matrix(ATA_alloc(sm->csc_cache));
}

/* Caller must have called refresh_csc_values since the last change to csr->x. */
static void sparse_ATDA_fill_values(const Matrix *self, const double *d, Matrix *out)
{
    const Sparse_Matrix *sm = (const Sparse_Matrix *) self;
    Sparse_Matrix *sm_out = (Sparse_Matrix *) out;
    ATDA_fill_values(sm->csc_cache, d, sm_out->csr);
}

static CSR_Matrix *sparse_to_csr(Matrix *self)
{
    return ((Sparse_Matrix *) self)->csr;
}

/* Build CSC structure on first call; refill values from csr->x on every call. */
static void sparse_refresh_csc_values(Matrix *self)
{
    Sparse_Matrix *sm = (Sparse_Matrix *) self;
    build_csc_structure_if_absent(sm);
    csr_to_csc_fill_values(sm->csr, sm->csc_cache, sm->csc_iwork);
}

static void wire_vtable(Sparse_Matrix *sm)
{
    sm->base.block_left_mult_vec = sparse_block_left_mult_vec;
    sm->base.block_left_mult_sparsity = sparse_block_left_mult_sparsity;
    sm->base.block_left_mult_values = sparse_block_left_mult_values;
    sm->base.update_values = sparse_update_values;
    sm->base.copy_sparsity = sparse_copy_sparsity;
    sm->base.DA_fill_values = sparse_DA_fill_values;
    sm->base.ATA_alloc = sparse_ATA_alloc;
    sm->base.ATDA_fill_values = sparse_ATDA_fill_values;
    sm->base.to_csr = sparse_to_csr;
    sm->base.refresh_csc_values = sparse_refresh_csc_values;
    sm->base.free_fn = sparse_free;
}

Matrix *new_sparse_matrix(CSR_Matrix *A)
{
    Sparse_Matrix *sm = (Sparse_Matrix *) SP_CALLOC(1, sizeof(Sparse_Matrix));
    sm->base.m = A->m;
    sm->base.n = A->n;
    wire_vtable(sm);
    sm->csr = A;
    return &sm->base;
}

Matrix *sparse_matrix_trans(const Sparse_Matrix *self, int *iwork)
{
    CSR_Matrix *AT = transpose(self->csr, iwork);
    Sparse_Matrix *sm = (Sparse_Matrix *) SP_CALLOC(1, sizeof(Sparse_Matrix));
    sm->base.m = AT->m;
    sm->base.n = AT->n;
    wire_vtable(sm);
    sm->csr = AT;
    return &sm->base;
}
