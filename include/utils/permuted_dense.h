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
    matrix base;
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

/* Fill values of C = diag(d) @ A where len(d) = number of (global) rows of A */
void DA_pd_fill_values(const double *d, const permuted_dense *A, permuted_dense *C);

/* Allocate new permuted dense for C = AT @ A */
matrix *ATA_pd_alloc(const permuted_dense *A);

/* Fill values of C = AT @ diag(d) @ A */
void ATDA_pd_fill_values(const permuted_dense *A, const double *d,
                         permuted_dense *C);

/* Allocate new permuted dense forC = BT @ A where A and B are both permuted_dense.
   (If B and A have no overlapping rows, then C is empty) */
matrix *BTA_pd_pd_alloc(const permuted_dense *B, const permuted_dense *A);

/* Fill values of C = BT @ A where A and B are both permuted dense. */
void BTA_pd_pd_fill_values(const permuted_dense *B, const permuted_dense *A,
                           permuted_dense *C);

/* Fill values of C = BT @ diag(d) @ A where A and B are both permuted dense. */
void BTDA_pd_pd_fill_values(const permuted_dense *B, const double *d,
                            const permuted_dense *A, permuted_dense *C);

/* Allocate new permuted dense for C = B @ A where B is PD and A is CSC. */
matrix *BA_pd_csc_alloc(const permuted_dense *B, const CSC_matrix *A);

/* Fill values of C = B @ A where B is value buffer to permuted dense and A is CSC.

   The raw-buffer signature for B lets callers pass a transposed dense block
   (e.g. (diag(d) B)^T stored in B->dwork) without needing to build a transposed
   permuted dense. */
void BA_pd_csc_fill_values(const double *B, int n0_B, const int *inv,
                           const CSC_matrix *A, permuted_dense *C);

/* Allocate new permuted dense for C = B @ A where B and A are both PD. Both
   may have arbitrary (sorted) row_perm / col_perm; no full-block assumption.
   If B->col_perm and A->row_perm have no overlap C is structurally empty;
   otherwise C has row_perm = B->row_perm, col_perm = A->col_perm. */
matrix *BA_pd_pd_alloc(const permuted_dense *B, const permuted_dense *A);

/* Fill values of C = B @ A for two PDs (general row_perm / col_perm).
   Intersects B->col_perm with A->row_perm, gathers the matching column
   slice of B and row slice of A into the operands' dwork scratch, and
   computes one cblas_dgemm. */
void BA_pd_pd_fill_values(const permuted_dense *B, const permuted_dense *A,
                          permuted_dense *C);

/* Allocate new permuted dense for C = B^T @ A where B is PD and A is CSC */
matrix *BTA_pd_csc_alloc(const permuted_dense *B, const CSC_matrix *A);

/* Fill values of C = B^T @ diag(d) @ A where B is PD and A is CSC */
void BTDA_pd_csc_fill_values(const permuted_dense *B, const double *d,
                             const CSC_matrix *A, permuted_dense *C);

/* Allocate new permuted_dense for C = B^T @ A where B is Sparse CSC and A is PD. */
matrix *BTA_csc_pd_alloc(const CSC_matrix *B, const permuted_dense *A);

/* Fill values of C = B^T @ diag(d) @ A where B is CSC and A is PD */
void BTDA_csc_pd_fill_values(const CSC_matrix *B, const double *d,
                             const permuted_dense *A, permuted_dense *C);

#endif /* PERMUTED_DENSE_H */
