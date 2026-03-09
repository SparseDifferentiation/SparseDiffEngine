/*
 * Copyright 2026 Daniel Cederberg and William Zhang
 *
 * This file is part of the DNLP-differentiation-engine project.
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
#include "utils/linalg_sparse_matmuls.h"
#include "utils/matrix.h"
#include <stdlib.h>

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

static void sparse_free(Matrix *self)
{
    Sparse_Matrix *sm = (Sparse_Matrix *) self;
    free_csr_matrix(sm->csr);
    free(sm);
}

Matrix *new_sparse_matrix(const CSR_Matrix *A)
{
    Sparse_Matrix *sm = (Sparse_Matrix *) calloc(1, sizeof(Sparse_Matrix));
    sm->base.m = A->m;
    sm->base.n = A->n;
    sm->base.block_left_mult_vec = sparse_block_left_mult_vec;
    sm->base.block_left_mult_sparsity = sparse_block_left_mult_sparsity;
    sm->base.block_left_mult_values = sparse_block_left_mult_values;
    sm->base.free_fn = sparse_free;
    sm->csr = new_csr(A);
    return &sm->base;
}

Matrix *sparse_matrix_trans(const Sparse_Matrix *self, int *iwork)
{
    CSR_Matrix *AT = transpose(self->csr, iwork);
    return new_sparse_matrix(AT);
}
