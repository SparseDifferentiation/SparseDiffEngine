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

#include "CSC_matrix.h"
#include "CSR_matrix.h"

/* Forward declaration; full definition in permuted_dense.h. Used by the
   as_permuted_dense vtable getter. */
struct permuted_dense;

/* Broadcast shape used by the broadcast atom and its vtable methods. */
typedef enum
{
    BROADCAST_ROW,   /* (1, n) -> (m, n) */
    BROADCAST_COL,   /* (m, 1) -> (m, n) */
    BROADCAST_SCALAR /* (1, 1) -> (m, n) */
} broadcast_type;

/* We implement three different types of matrices.

    1. 'sparse_matrix' represents a generic CSR_matrix matrix.
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

   1. sparse_matrix: generic CSR_matrix matrix.
   2. permuted_dense:


*/

/* Base matrix type with function pointers for polymorphic dispatch. There are two
   types of matrices: 'sparse_matrix' and 'permuted_dense'. Each type implements the
   same set of operations, but with different algorithms. The following operations
   are implemented: TODO
*/
typedef struct matrix
{
    int m, n, nnz; /* shape and nnz*/
    double *x;     /* non-owning pointer to the value buffer */

    /* Operators for the left-multiply matrix in left_matmul. */
    void (*block_left_mult_vec)(const struct matrix *self, const double *x,
                                double *y, int p);
    CSC_matrix *(*block_left_mult_sparsity)(const struct matrix *self,
                                            const CSC_matrix *J, int p);
    void (*block_left_mult_values)(const struct matrix *self, const CSC_matrix *J,
                                   CSC_matrix *C);

    /* Chain-rule operations used by transformer atoms (elementwise, etc.).
       All chain-rule outputs are the same concrete type as self (uniform
       polymorphism). copy_sparsity returns a matrix of same shape and type as
       self; DA_fill_values writes diag(d) * self into out; ATA_alloc allocates
       a matrix with sparsity of self^T * self; ATDA_fill_values fills out with
       self^T * diag(d) * self; to_csr returns a CSR_matrix view of self
       (constant-time for sparse_matrix, lazily built/refreshed for other types). */
    struct matrix *(*copy_sparsity)(const struct matrix *self);
    void (*DA_fill_values)(const double *d, const struct matrix *self,
                           struct matrix *out);
    struct matrix *(*ATA_alloc)(struct matrix *self);
    void (*ATDA_fill_values)(const struct matrix *self, const double *d,
                             struct matrix *out);
    CSR_matrix *(*to_csr)(struct matrix *self);

    /* Returns self downcast to permuted_dense if self is PD-backed, NULL
       otherwise. Used by bivariate dispatchers to route to type-specialized
       kernels. */
    struct permuted_dense *(*as_permuted_dense)(struct matrix *self);

    /* Row-selection / indexing: returns a new matrix that selects rows
       indices[0..n_idxs) of self. Output shape is (n_idxs, self->n). The
       returned type matches self's concrete type. index_alloc sets up
       sparsity (values uninitialized); index_fill_values fills values into
       out, which must have been produced by a prior index_alloc with the
       same indices/n_idxs. */
    struct matrix *(*index_alloc)(struct matrix *self, const int *indices,
                                  int n_idxs);
    void (*index_fill_values)(struct matrix *self, const int *indices, int n_idxs,
                              struct matrix *out);

    /* Row-tiling for the promote atom: self must be a 1-row matrix; returns
       a new matrix of shape (size, self->n) where every row is a copy of
       self's single row. Output type matches self's concrete type.
       promote_alloc sets sparsity; promote_fill_values fills values. */
    struct matrix *(*promote_alloc)(struct matrix *self, int size);
    void (*promote_fill_values)(struct matrix *self, struct matrix *out);

    /* Broadcast: lift the child Jacobian of a broadcast atom into the output
       Jacobian. `type` is the broadcast variant; (d1, d2) is the output shape.
       Output type matches self's concrete type. broadcast_alloc sets sparsity;
       broadcast_fill_values fills values into out. */
    struct matrix *(*broadcast_alloc)(struct matrix *self, broadcast_type type,
                                      int d1, int d2);
    void (*broadcast_fill_values)(struct matrix *self, broadcast_type type, int d1,
                                  int d2, struct matrix *out);

    /* diag_vec: child is an (n, self->n) Jacobian for a length-n vector;
       output is (n*n, self->n) where child row i lands at output row
       i*(n+1) (column-major diagonal positions). Other output rows are
       structurally zero. Output type matches self's concrete type. */
    struct matrix *(*diag_vec_alloc)(struct matrix *self);
    void (*diag_vec_fill_values)(struct matrix *self, struct matrix *out);

    /* Refresh any internal caches (e.g. a CSC_matrix mirror) so subsequent ATA /
       ATDA calls reflect the current values. Atoms whose child Jacobian is affine
       can skip this on iterations after the first; non-affine children must
       call it before every chain-rule call. No-op for types that don't have
       a cache (e.g. permuted_dense). */
    void (*refresh_csc_values)(struct matrix *self);

    /* Lifecycle. */
    void (*free_fn)(struct matrix *self);
} matrix;

/* Free helper */
static inline void free_matrix(matrix *m)
{
    if (m)
    {
        m->free_fn(m);
    }
}

#endif /* MATRIX_H */
