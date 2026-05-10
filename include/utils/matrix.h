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
#ifndef MATRIX_H
#define MATRIX_H

#include "CSC_Matrix.h"
#include "CSR_Matrix.h"

/* We implement three different types of matrices.

    1. 'sparse_matrix' represents a generic CSR matrix.
    2. 'permuted_dense' represents a matrix that only consists of a dense block
        (potentially after permuting columns).
    3. 'blkdiag_dense' represents a block diagonal matrix with a constant dense
        block.

    Each of these types implements its own functionality for common matrix operations
    such as DA_fill_values etc. The return type of most of these operations are the
    same as the type of the input. For example, DA_fill_values for permuted_dense
    fills the values of a new permuted_dense object.

    2, 'permuted_dense':
       * DA_fill_values just scales the rows. It does not affect the permutation
         indices.
       * ATA_alloc
       * ATDA_fill_values
       * to_csr_sparsity
       * to_csr_values
       *

   1. sparse_matrix: generic CSR matrix.
   2. permuted_dense:


*/

/* Base matrix type with function pointers for polymorphic dispatch */
typedef struct Matrix
{
    int m, n;
    void (*block_left_mult_vec)(const struct Matrix *self, const double *x,
                                double *y, int p);
    CSC_Matrix *(*block_left_mult_sparsity)(const struct Matrix *self,
                                            const CSC_Matrix *J, int p);
    void (*block_left_mult_values)(const struct Matrix *self, const CSC_Matrix *J,
                                   CSC_Matrix *C);
    void (*update_values)(struct Matrix *self, const double *new_values);
    void (*free_fn)(struct Matrix *self);
} Matrix;

/* Sparse matrix wrapping CSR */
typedef struct Sparse_Matrix
{
    Matrix base;
    CSR_Matrix *csr;
} Sparse_Matrix;

/* Constructors */
Matrix *new_sparse_matrix(const CSR_Matrix *A);

/* Transpose helper */
Matrix *sparse_matrix_trans(const Sparse_Matrix *self, int *iwork);

/* Free helper */
static inline void free_matrix(Matrix *m)
{
    if (m)
    {
        m->free_fn(m);
    }
}

#endif /* MATRIX_H */
