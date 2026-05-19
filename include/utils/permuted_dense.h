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

/* permuted_dense represents a matrix whose only nonzeros lie in a dense
   block, after rows and columns are restricted to chosen subsets of the
   global index space. For local indices (ii, jj) with 0 <= ii < m0
   and 0 <= jj < n0,

       M[row_perm[ii], col_perm[jj]] = X[ii, jj].

   All other entries of M are zero. row_perm and col_perm are stored in
   strictly increasing order; the constructor asserts this. */
typedef struct permuted_dense
{
    matrix base;
    int m0;                /* rows of dense block (= len(row_perm))      */
    int n0;                /* cols of dense block (= len(col_perm))      */
    int *row_perm;         /* row_perm[ii] in [0, base.m), sorted        */
    int *col_perm;         /* col_perm[jj] in [0, base.n), sorted        */
    double *X;             /* m0 * n0, row-major. Owned by this PD when */
                           /* owns_X == true; otherwise X is a view into */
                           /* an enclosing buffer (e.g. a stacked_pd's   */
                           /* shared values buffer) and must not be      */
                           /* freed by this PD's free fn.                */
    bool owns_X;           /* see X above; true by default               */
    int *col_inv;          /* length base.n: col_inv[col_perm[jj]] = jj, */
                           /* otherwise -1; used by `x CSC_matrix` allocation.  */
    int *row_inv;          /* length base.m: row_inv[row_perm[ii]] = ii, */
                           /* otherwise -1; used by index_alloc.         */
    CSR_matrix *csr_cache; /* lazy CSR_matrix view built by to_csr; structure */
                           /* allocated on first call, values refilled */
                           /* on every call. NULL until first call.    */
    /* Mutable double-precision BLAS scratch shared across kernels that
       operate on this PD. Two non-overlapping roles (a given fill call uses
       at most one):
         - Y-buffer: holds diag(d_perm) X for ATDA / BTDA_pd_pd (size m0*n0).
         - transpose: holds (diag(d) X)^T for the BA_pd_csc-based BTDA
                     kernels (BTDA_pd_csc and, transitively, BTDA_csc_pd
                     via its delegate). Size m0*n0 doubles.
       Allocated lazily on the first kernel that needs it; grown in place
       (free + SP_MALLOC, contents not preserved) if a later kernel needs
       more. `dwork == NULL` and `dwork_size == 0` before first use.
       Functions taking a const permuted_dense * may still mutate `dwork`. */
    double *dwork;
    size_t dwork_size;

    /* Mutable int scratch. Currently only used to hold the row-intersection
       index arrays idx_A / idx_B in BTA_pd_pd_fill_values and the
       slow path of BTDA_pd_pd_fill_values; allocated by
       BTA_pd_pd_alloc for those outputs (NULL on PDs from other
       allocators). Fill kernels fall back to a per-call SP_MALLOC if
       iwork_size is too small. */
    int *iwork;
    size_t iwork_size;

    /* CONTRACT: `dwork` and `iwork` are freely overwritten by every kernel
       that takes this PD as input or output — contents do NOT survive
       across calls. Do not use them to cache precomputed factors or carry
       state between kernel invocations: any subsequent call (ATDA, BTDA,
       BTA gather, …) may clobber them without warning. If you need
       persistence, add a dedicated field. */
} permuted_dense;

/* Constructor. row_perm and col_perm must be strictly increasing in their
   respective ranges. If X_data is NULL the value buffer is allocated but
   left uninitialized; otherwise m0 * n0 entries are copied. */
matrix *new_permuted_dense(int m, int n, int m0, int n0, const int *row_perm,
                           const int *col_perm, const double *X_data);

/* Convenience constructor for the trivial-perm case: row_perm = [0..m-1],
   col_perm = [0..n-1], dense block fills the full (m, n) shape. */
matrix *new_permuted_dense_full(int m, int n, const double *data);

/* CSR_matrix view: callers should use the vtable, i.e. base.to_csr(base). The PD
   owns and caches the returned CSR_matrix; its value array aliases self->X,
   so values are always live with no separate fill needed. Callers must not
   free the returned CSR_matrix — it's released by free_matrix on the PD. */

/* Ensure pd->dwork is sized at least `size` doubles. Grows in place;
   contents are NOT preserved. Called from allocator functions so that the
   corresponding fill kernels never need to allocate. Takes a const pointer
   and casts internally — this matches the dwork contract above that
   dwork is mutable through a const permuted_dense *. Used by the linalg
   allocators in permuted_dense_linalg.c and by the operator-role
   block_left_mult adapter in permuted_dense.c. */
void permuted_dense_ensure_dwork(const permuted_dense *pd_const, size_t size);

/* Allocate C = broadcast(A, type, d1, d2). Output is a PD whose row_perm
   layout depends on `type` (SCALAR / ROW / COL). Same as A's broadcast
   vtable adapter; exposed so stacked_pd can call it directly per-block. */
matrix *broadcast_pd_alloc(const permuted_dense *A, broadcast_type type, int d1,
                           int d2);

/* Fill values of C = broadcast(A, type, d1, d2). */
void broadcast_pd_fill_values(const permuted_dense *A, broadcast_type type, int d1,
                              int d2, permuted_dense *C);

/* Allocate C = A[indices, :]. Output is a PD whose row_perm holds the
   output positions i (in [0, n_idxs)) at which `indices[i]` lands in A's
   row_perm. */
matrix *index_pd_alloc(const permuted_dense *A, const int *indices, int n_idxs);

/* Fill values of C = A[indices, :]. */
void index_pd_fill_values(const permuted_dense *A, const int *indices, int n_idxs,
                          permuted_dense *C);

/* Allocate C = promote(A, size): broadcast A's single (or zero) row up to
   `size` rows. Precondition: A->m0 <= 1. */
matrix *promote_pd_alloc(const permuted_dense *A, int size);

/* Fill values of C = promote(A, size). */
void promote_pd_fill_values(const permuted_dense *A, permuted_dense *C);

/* Allocate C = diag_vec(A): A's row_perm entries r are remapped to
   r * (n + 1) where n = A->base.m, producing the column vector of the
   diagonal of A interpreted as an n x n matrix. */
matrix *diag_vec_pd_alloc(const permuted_dense *A);

/* Fill values of C = diag_vec(A). */
void diag_vec_pd_fill_values(const permuted_dense *A, permuted_dense *C);

#endif /* PERMUTED_DENSE_H */
