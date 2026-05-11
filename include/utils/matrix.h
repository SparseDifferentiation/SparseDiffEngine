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

/* Broadcast shape used by the broadcast atom and its vtable methods. */
typedef enum
{
    BROADCAST_ROW,   /* (1, n) -> (m, n) */
    BROADCAST_COL,   /* (m, 1) -> (m, n) */
    BROADCAST_SCALAR /* (1, 1) -> (m, n) */
} broadcast_type;

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

/* Base matrix type with function pointers for polymorphic dispatch. There are
   two types of matrices: 'sparse_matrix' and 'permuted_dense'. Each type
   implements the same set of operations, but with different algorithms.
   The following operations are implemented: TODO
*/
typedef struct Matrix
{
    /* Dimensions and number of explicitly stored entries. For Sparse_Matrix
       nnz is the CSR nnz; for Dense_Matrix it is m * n; for Permuted_Dense it
       is dense_m * dense_n (the size of the stored dense block). */
    int m, n, nnz;

    /* Non-owning pointer to the value buffer. Sparse_Matrix: csr->x.
       Permuted_Dense: pd->X (also aliased as pd->csr_cache->x). Dense_Matrix:
       dm->x. Sparse and Permuted_Dense share row-major layout for equal
       sparsity patterns, so memcpy via M->x is valid between same-shape
       Sparse/PD pairs. */
    double *x;

    /* Operators for the left-multiply matrix in left_matmul. */
    void (*block_left_mult_vec)(const struct Matrix *self, const double *x,
                                double *y, int p);
    CSC_Matrix *(*block_left_mult_sparsity)(const struct Matrix *self,
                                            const CSC_Matrix *J, int p);
    void (*block_left_mult_values)(const struct Matrix *self, const CSC_Matrix *J,
                                   CSC_Matrix *C);

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

    /* Row-selection / indexing: returns a new Matrix that selects rows
       indices[0..n_idxs) of self. Output shape is (n_idxs, self->n). The
       returned type matches self's concrete type. index_alloc sets up
       sparsity (values uninitialized); index_fill_values fills values into
       out, which must have been produced by a prior index_alloc with the
       same indices/n_idxs. */
    struct Matrix *(*index_alloc)(struct Matrix *self, const int *indices,
                                  int n_idxs);
    void (*index_fill_values)(struct Matrix *self, const int *indices, int n_idxs,
                              struct Matrix *out);

    /* Row-tiling for the promote atom: self must be a 1-row matrix; returns
       a new Matrix of shape (size, self->n) where every row is a copy of
       self's single row. Output type matches self's concrete type.
       promote_alloc sets sparsity; promote_fill_values fills values. */
    struct Matrix *(*promote_alloc)(struct Matrix *self, int size);
    void (*promote_fill_values)(struct Matrix *self, struct Matrix *out);

    /* Broadcast: lift the child Jacobian of a broadcast atom into the output
       Jacobian. `type` is the broadcast variant; (d1, d2) is the output shape.
       Output type matches self's concrete type. broadcast_alloc sets sparsity;
       broadcast_fill_values fills values into out. */
    struct Matrix *(*broadcast_alloc)(struct Matrix *self, broadcast_type type,
                                      int d1, int d2);
    void (*broadcast_fill_values)(struct Matrix *self, broadcast_type type, int d1,
                                  int d2, struct Matrix *out);

    /* diag_vec: child is an (n, self->n) Jacobian for a length-n vector;
       output is (n*n, self->n) where child row i lands at output row
       i*(n+1) (column-major diagonal positions). Other output rows are
       structurally zero. Output type matches self's concrete type. */
    struct Matrix *(*diag_vec_alloc)(struct Matrix *self);
    void (*diag_vec_fill_values)(struct Matrix *self, struct Matrix *out);

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

/* Convenience: allocate a Sparse_Matrix of shape (m, n) with capacity for
   nnz entries. Equivalent to new_sparse_matrix(new_csr_matrix(m, n, nnz)).
   Sparsity pattern and values are uninitialized. */
Matrix *new_sparse_matrix_alloc(int m, int n, int nnz);

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
