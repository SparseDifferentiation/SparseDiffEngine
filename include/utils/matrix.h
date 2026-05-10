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

    /* Operators for the left-multiply matrix in left_matmul. */
    void (*block_left_mult_vec)(const struct Matrix *self, const double *x,
                                double *y, int p);
    CSC_Matrix *(*block_left_mult_sparsity)(const struct Matrix *self,
                                            const CSC_Matrix *J, int p);
    void (*block_left_mult_values)(const struct Matrix *self, const CSC_Matrix *J,
                                   CSC_Matrix *C);
    void (*update_values)(struct Matrix *self, const double *new_values);

    /* Chain-rule operations used by transformer atoms (elementwise, etc.).
       All chain-rule outputs are the same concrete type as self (uniform
       polymorphism). copy_sparsity returns a matrix of same shape and type as
       self; DA_fill_values writes diag(d) * self into out; ATA_alloc allocates
       a matrix with sparsity of self^T * self; ATDA_fill_values fills out with
       self^T * diag(d) * self; to_csr returns a CSR view of self (constant-time
       for Sparse_Matrix, lazily built/refreshed for other types). */
    struct Matrix *(*copy_sparsity)(const struct Matrix *self);
    void (*DA_fill_values)(const double *d, const struct Matrix *self,
                           struct Matrix *out);
    struct Matrix *(*ATA_alloc)(struct Matrix *self);
    void (*ATDA_fill_values)(const struct Matrix *self, const double *d,
                             struct Matrix *out);
    CSR_Matrix *(*to_csr)(struct Matrix *self);

    /* Refresh any internal caches (e.g. a CSC mirror) so subsequent ATA / ATDA
       calls reflect the current values. Atoms whose child Jacobian is affine
       can skip this on iterations after the first; non-affine children must
       call it before every chain-rule call. No-op for types that don't have
       a cache (e.g. permuted_dense). */
    void (*refresh_csc_values)(struct Matrix *self);

    /* Lifecycle. */
    void (*free_fn)(struct Matrix *self);
} Matrix;

/* Sparse matrix wrapping CSR. csc_cache is a lazily-built CSC mirror used by
   the chain-rule ATA / ATDA paths; it's allocated on first need and refilled
   by refresh_csc_values. csc_iwork is the workspace for csr_to_csc. */
typedef struct Sparse_Matrix
{
    Matrix base;
    CSR_Matrix *csr;
    CSC_Matrix *csc_cache;
    int *csc_iwork;
} Sparse_Matrix;

/* Constructor. Takes ownership of A; the caller must not free A separately
   (free_matrix on the returned Matrix frees A). */
Matrix *new_sparse_matrix(CSR_Matrix *A);

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
