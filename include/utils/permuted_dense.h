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
#ifndef PERMUTED_DENSE_H
#define PERMUTED_DENSE_H

#include "matrix.h"
#include <stdbool.h>
#include <stddef.h>

/* permuted_dense represents a matrix whose only nonzeros lie in a dense block
   of size m0 x n0, after rows and columns are restricted to chosen subsets of
   the global index space. For local indices (ii, jj) with 0 <= ii < m0 and 0 <=
   jj < n0,

       M[row_perm[ii], col_perm[jj]] = X[ii, jj].

   All other entries of M are zero. row_perm and col_perm are stored in strictly
   increasing order; the constructor asserts this. */
typedef struct permuted_dense
{
    matrix base;
    int m0;
    int n0;
    int *row_perm;
    int *col_perm;
    int *col_inv; /* col_inv[col_perm[jj]] = jj, otherwise -1 */
    int *row_inv; /* row_inv[row_perm[ii]] = ii, otherwise -1 */

    /* Row-major block of size m0 x n0. Owned by this PD when owns_X == true,
       and otherwise X is a view into a buffer (eg., stacked_pd's shared values
       buffer) and must not be freed by this PD. */
    double *X;
    bool owns_X;

    /* Cached CSR_matrix view of the permuted dense, built on demand by to_csr.
       The structure (p/i arrays) is allocated on the first call to to_csr and
       reused thereafter; the values array aliases self->X so is always live with
       no separate fill needed. */
    CSR_matrix *csr_cache;

    /* Workspace used for BLAS operations. Allocated lazily by the first kernel
       that needs it, and resizes if another kernel needs more space than
       currently allocated (that's why we store dwork_size) */
    double *kernel_dwork;
    size_t kernel_dwork_size;

    /* Private buffers owned by the kernel that produced this PD. The
       producing kernel may use them for per-call scratch and/or small
       persistent state (e.g. cached config the fill_values needs to
       read). Allocated lazily. */
    int *kernel_iwork;
    size_t kernel_iwork_size;
} permuted_dense;

/* Constructor. row_perm and col_perm must be strictly increasing in their
   respective ranges. If X_data is NULL the value buffer is allocated but left
   uninitialized. Otherwise m0 * n0 entries are copied. */
matrix *new_permuted_dense(int m, int n, int m0, int n0, const int *row_perm,
                           const int *col_perm, const double *X_data);

/* Convenience constructor for the trivial-perm case: row_perm = [0..m-1],
   col_perm = [0..n-1], dense block fills the full (m, n) shape. */
matrix *new_permuted_dense_full(int m, int n, const double *data);

/* Ensure A->kernel_dwork is sized at least 'size' doubles. Grows in
   place; contents are NOT preserved. */
void permuted_dense_ensure_kernel_dwork(const permuted_dense *A, size_t size);

/* Allocate C = broadcast(A, type, d1, d2), where A and C are permuted dense. */
matrix *broadcast_pd_alloc(const permuted_dense *A, broadcast_type type, int d1,
                           int d2);

/* Fill values of C = broadcast(A, type, d1, d2). */
void broadcast_pd_fill_values(const permuted_dense *A, broadcast_type type, int d1,
                              int d2, permuted_dense *C);

/* Allocate C = A[indices, :], where A and C are permuted dense. */
matrix *index_pd_alloc(const permuted_dense *A, const int *indices, int n_idxs);

/* Fill values of C = A[indices, :]. */
void index_pd_fill_values(const permuted_dense *A, const int *indices, int n_idxs,
                          permuted_dense *C);

/* Allocate C = promote(A, size), where A and C are permuted dense. */
matrix *promote_pd_alloc(const permuted_dense *A, int size);

/* Fill values of C = promote(A, size). */
void promote_pd_fill_values(const permuted_dense *A, permuted_dense *C);

/* Allocate C = diag_vec(A), where A and C are permuted dense. */
matrix *diag_vec_pd_alloc(const permuted_dense *A);

/* Fill values of C = diag_vec(A). */
void diag_vec_pd_fill_values(const permuted_dense *A, permuted_dense *C);

#endif /* PERMUTED_DENSE_H */
