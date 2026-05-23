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
#include <stdbool.h>

/* Broadcast shape used by the broadcast atom and its vtable methods. */
typedef enum
{
    BROADCAST_ROW,   /* (1, n) -> (m, n) */
    BROADCAST_COL,   /* (m, 1) -> (m, n) */
    BROADCAST_SCALAR /* (1, 1) -> (m, n) */
} broadcast_type;

/* Polymorphic matrix base. Concrete types embed `matrix` as their first
   member and implement the vtable slots below. Currently implemented:
       1. sparse_matrix  — generic CSR_matrix-backed matrix.
       2. permuted_dense — matrix whose nonzeros lie in a single dense block
                           located at chosen rows and columns of the global
                           index space.
   A third type is potentially planned. */

typedef struct matrix matrix;

/* y = kron(I_p, A) @ x */
typedef void (*matrix_block_left_mult_vec_fn)(const matrix *A, const double *x,
                                              double *y, int p);

/* Allocate sparsity of C = kron(I_p, A) @ J */
typedef CSC_matrix *(*matrix_block_left_mult_sparsity_fn)(const matrix *A,
                                                          const CSC_matrix *J,
                                                          int p);

/* Fill values of C = kron(I_p, A) @ J */
typedef void (*matrix_block_left_mult_values_fn)(const matrix *A,
                                                 const CSC_matrix *J, CSC_matrix *C);

/* Allocate a new matrix with the same sparsity as A */
typedef matrix *(*matrix_copy_sparsity_fn)(const matrix *A);

/* Fill values of C = diag(d) @ A */
typedef void (*matrix_DA_fill_values_fn)(const double *d, const matrix *A,
                                         matrix *C);

/* Allocate C = AT @ A */
typedef matrix *(*matrix_ATA_alloc_fn)(matrix *A);

/* Fill values of C = AT @ diag(d) @ A */
typedef void (*matrix_ATDA_fill_values_fn)(const matrix *A, const double *d,
                                           matrix *C);

/* Allocate AT = transpose(A) */
typedef matrix *(*matrix_transpose_alloc_fn)(const matrix *A);

/* Fill values of AT = transpose(A) */
typedef void (*matrix_transpose_fill_values_fn)(const matrix *A, matrix *AT);

/* Returns a CSR_matrix view of A */
typedef CSR_matrix *(*matrix_to_csr_fn)(matrix *A);

/* Refresh any internal caches (e.g. a CSC_matrix mirror) so subsequent ATA /
   ATDA calls reflect the current values. */
typedef void (*matrix_refresh_csc_values_fn)(matrix *A);

/* Allocate C = A[indices, :] */
typedef matrix *(*matrix_index_alloc_fn)(matrix *A, const int *indices, int n_idxs);

/* Fill values of C = A[indices, :] */
typedef void (*matrix_index_fill_values_fn)(matrix *A, const int *indices,
                                            int n_idxs, matrix *C);

/* Row-tiling for the promote atom: A must be a 1-row matrix; returns
   a new matrix of shape (size, A->n) where every row is a copy of A's
   single row. */
typedef matrix *(*matrix_promote_alloc_fn)(matrix *A, int size);
typedef void (*matrix_promote_fill_values_fn)(matrix *A, matrix *out);

/* Broadcast: lift the child Jacobian of a broadcast atom into the output
   Jacobian. `type` is the broadcast variant; (d1, d2) is the output shape. */
typedef matrix *(*matrix_broadcast_alloc_fn)(matrix *A, broadcast_type type, int d1,
                                             int d2);
typedef void (*matrix_broadcast_fill_values_fn)(matrix *A, broadcast_type type,
                                                int d1, int d2, matrix *out);

/* diag_vec: A is an (n, A->n) Jacobian for a length-n vector; output is
   (n*n, A->n) where row i lands at output row i*(n+1) (column-major
   diagonal positions). Other output rows are structurally zero. */
typedef matrix *(*matrix_diag_vec_alloc_fn)(matrix *A);
typedef void (*matrix_diag_vec_fill_values_fn)(matrix *A, matrix *out);

/* Allocate C as a row-wise reduction of A. The reduction pattern is chosen by
   axis:
     - axis = -1: sum all rows of A. C has shape (1, A->n).
                  C[0, k] = sum_{i in [0, A->m)} A[i, k].
     - axis = 0:  block-sum rows in consecutive groups of d1 (requires
                  A->m % d1 == 0). C has shape (A->m / d1, A->n).
                  C[j, k] = sum_{i in [j*d1, (j+1)*d1)} A[i, k].
     - axis = 1:  stride-sum rows at modular spacing d1 (requires
                  A->m % d1 == 0). C has shape (d1, A->n).
                  C[j, k] = sum_{i : i % d1 == j} A[i, k].
   d1 is ignored when axis == -1.

   Caller pre-allocates idx_map of size A->nnz. On return idx_map[ii] holds the
   position in C's base.x where A's base.x[ii] scatter-adds. A values-fill pass
   can compute C->x by zeroing it and accumulating A->x[ii] into
   C->x[idx_map[ii]] for ii in [0, A->nnz). */
typedef matrix *(*matrix_sum_alloc_fn)(matrix *A, int axis, int d1, int *idx_map);

typedef void (*matrix_free_fn)(matrix *self);

struct matrix
{
    int m, n, nnz;
    double *x; /* non-owning pointer to the value buffer */
    bool is_permuted_dense;
    bool is_stacked_pd;

    /* Operator ops */
    matrix_block_left_mult_vec_fn block_left_mult_vec;
    matrix_block_left_mult_sparsity_fn block_left_mult_sparsity;
    matrix_block_left_mult_values_fn block_left_mult_values;

    /* Chain-rule ops */
    matrix_copy_sparsity_fn copy_sparsity;
    matrix_DA_fill_values_fn DA_fill_values;
    matrix_ATA_alloc_fn ATA_alloc;
    matrix_ATDA_fill_values_fn ATDA_fill_values;
    matrix_transpose_alloc_fn transpose_alloc;
    matrix_transpose_fill_values_fn transpose_fill_values;

    /* Views and cache */
    matrix_to_csr_fn to_csr;
    matrix_refresh_csc_values_fn refresh_csc_values;

    /* Atom-specific ops */
    matrix_index_alloc_fn index_alloc;
    matrix_index_fill_values_fn index_fill_values;
    matrix_promote_alloc_fn promote_alloc;
    matrix_promote_fill_values_fn promote_fill_values;
    matrix_broadcast_alloc_fn broadcast_alloc;
    matrix_broadcast_fill_values_fn broadcast_fill_values;
    matrix_diag_vec_alloc_fn diag_vec_alloc;
    matrix_diag_vec_fill_values_fn diag_vec_fill_values;
    matrix_sum_alloc_fn sum_alloc;

    /* Lifecycle */
    matrix_free_fn free_fn;
};

/* Free helper */
static inline void free_matrix(matrix *m)
{
    if (m)
    {
        m->free_fn(m);
    }
}

#endif /* MATRIX_H */
