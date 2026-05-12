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
    matrix base;           /* base.m, base.n = global ambient dimensions */
    int m0;                /* rows of dense block (= len(row_perm))      */
    int n0;                /* cols of dense block (= len(col_perm))      */
    int *row_perm;         /* row_perm[ii] in [0, base.m), sorted        */
    int *col_perm;         /* col_perm[jj] in [0, base.n), sorted        */
    double *X;             /* m0 * n0, row-major               */
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
         - gather:   holds densified CSR rows for BTA/BTDA_csr_pd /
                     _pd_csr (size depends on the input PD's dimensions).
       Sized at alloc time for the largest role this PD could play. Functions
       taking a const permuted_dense * may still mutate `dwork`. */
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

/* CSR_matrix view: callers should use the vtable, i.e. base.to_csr(base). The PD
   owns and caches the returned CSR_matrix; its value array aliases self->X,
   so values are always live with no separate fill needed. Callers must not
   free the returned CSR_matrix — it's released by free_matrix on the PD. */

/* Fill values of C = diag(d) @ A where len(d) = number of (global) rows of A */
void permuted_dense_DA_fill_values(const double *d, const permuted_dense *A,
                                   permuted_dense *C);

/* Allocate new permuted dense for C = AT @ A */
matrix *permuted_dense_ATA_alloc(const permuted_dense *A);

/* Fill values of C = AT @ diag(d) @ A */
void permuted_dense_ATDA_fill_values(const permuted_dense *A, const double *d,
                                     permuted_dense *C);

/* Allocate new permuted dense forC = BT @ A where A and B are both permuted_dense.
   (If B and A have no overlapping rows, then C is empty) */
matrix *BTA_pd_pd_alloc(const permuted_dense *B, const permuted_dense *A);

/* Fill values of C = BT @ A where A and B are both permuted dense. */
void BTA_pd_pd_fill_values(const permuted_dense *B, const permuted_dense *A,
                           permuted_dense *C);

/* Allocate new permuted dense for C = B @ A where B is PD and A is CSC */
matrix *BA_pd_csc_alloc(const permuted_dense *B, const CSC_matrix *A);

/* Fill values of C = B @ A where B is value buffer to permuted dense and A is CSC.

   The raw-buffer signature for B lets callers pass a transposed dense block
   (e.g. (diag(d) B)^T stored in B->dwork) without needing to build a transposed
   permuted dense. */
void BA_pd_csc_fill_values(const double *B, int n0_B, const int *inv,
                           const CSC_matrix *A, permuted_dense *C);

// ------------------- OK SO FAR

/* Allocate a new permuted_dense for C = B^T @ A where B is Sparse (CSR_matrix)
   and A is PD. Output is PD with row_perm = the sorted union of columns
   appearing in B's rows at positions row_perm_A, and col_perm = col_perm_A.
   Dense block size = (|row_active|, n0_A). Values uninitialized. */
matrix *BTA_csr_pd_alloc(const CSR_matrix *B_csr, const permuted_dense *A);

/* Fill C->X = B_sub^T @ X_A, where B_sub is B's rows at positions
   row_perm_A, columns restricted to C's row_perm, scattered to a dense
   buffer. C must have the structure produced by BTA_csr_pd_alloc(B, A). */
void BTA_csr_pd_fill_values(const CSR_matrix *B_csr, const permuted_dense *A,
                            permuted_dense *C);

/* BTDA variants — fold a diagonal d into the BTA computation. Each fills
   C->X = B^T diag(d) A (d may be NULL for plain B^T A). C must have the
   structure produced by the corresponding BTA *_alloc function. */
void BTDA_csr_pd_fill_values(const CSR_matrix *B_csr, const double *d,
                             const permuted_dense *A, permuted_dense *C);
void BTDA_pd_pd_fill_values(const permuted_dense *B, const double *d,
                            const permuted_dense *A, permuted_dense *C);

/* CSC-based (PD, Sparse) BTA / BTDA kernels — production path. Alloc is a
   single pass over A's CSC columns; the fill kernel transposes B's dense
   block into B->dwork (folding d in) and delegates to
   BA_pd_csc_fill_values. d MUST be non-NULL — production callers always
   supply chain-rule weights; for plain B^T A pass d = {1, 1, …, 1}.
   See include/old-code/old_permuted_dense.h for the legacy CSR equivalents
   kept as reference implementations. */
matrix *BTA_pd_csc_alloc(const permuted_dense *B, const CSC_matrix *A);
void BTDA_pd_csc_fill_values(const permuted_dense *B, const double *d,
                             const CSC_matrix *A, permuted_dense *C);

#endif /* PERMUTED_DENSE_H */
