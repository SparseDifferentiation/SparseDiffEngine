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

/* Permuted_Dense represents a matrix whose only nonzeros lie in a dense
   block, after rows and columns are restricted to chosen subsets of the
   global index space. For local indices (ii, jj) with 0 <= ii < dense_m
   and 0 <= jj < dense_n,

       M[row_perm[ii], col_perm[jj]] = X[ii, jj].

   All other entries of M are zero. row_perm and col_perm are stored in
   strictly increasing order; the constructor asserts this. */
typedef struct Permuted_Dense
{
    Matrix base;           /* base.m, base.n = global ambient dimensions */
    int dense_m;           /* rows of dense block (= len(row_perm))      */
    int dense_n;           /* cols of dense block (= len(col_perm))      */
    int *row_perm;         /* row_perm[ii] in [0, base.m), sorted        */
    int *col_perm;         /* col_perm[jj] in [0, base.n), sorted        */
    double *X;             /* dense_m * dense_n, row-major               */
    double *Y_scratch;     /* dense_m * dense_n, used by ATDA            */
    int *col_inv;          /* length base.n: col_inv[col_perm[jj]] = jj, */
                           /* otherwise -1; used by `x CSC` allocation.  */
    int *row_inv;          /* length base.m: row_inv[row_perm[ii]] = ii, */
                           /* otherwise -1; used by index_alloc.         */
    CSR_Matrix *csr_cache; /* lazy CSR view built by to_csr; structure */
                           /* allocated on first call, values refilled */
                           /* on every call. NULL until first call.    */
    /* Scratch buffer for BTA_csr_pd / BTA_pd_csr fill kernels. Owned by the
       output PD, allocated by the corresponding BTA *_alloc so per-call
       BTA / BTDA fill kernels can reuse it across solver iterations
       (avoids malloc/free of large dense buffers per Hessian eval). Sized
       at alloc time; NULL on PDs not produced by those allocators. */
    double *gather_X_scratch;
    size_t gather_X_size;
} Permuted_Dense;

/* Constructor. row_perm and col_perm must be strictly increasing in their
   respective ranges. If X_data is NULL the value buffer is allocated but
   left uninitialized; otherwise dense_m * dense_n entries are copied. */
Matrix *new_permuted_dense(int m, int n, int dense_m, int dense_n,
                           const int *row_perm, const int *col_perm,
                           const double *X_data);

/* CSR view: callers should use the vtable, i.e. base.to_csr(base). The PD
   owns and caches the returned CSR_Matrix; its value array aliases self->X,
   so values are always live with no separate fill needed. Callers must not
   free the returned CSR — it's released by free_matrix on the PD. */

/* Fill out = diag(d) * self, where d has length self->base.m. out must have
   the same structure as self (same dimensions and same row_perm/col_perm). */
void permuted_dense_DA_fill_values(const double *d, const Permuted_Dense *self,
                                   Permuted_Dense *out);

/* Allocate a new Permuted_Dense for C = self^T @ self. C is square of global
   size self->base.n, with dense block self->dense_n x self->dense_n and both
   permutations equal to self->col_perm. Values are uninitialized; the caller
   is expected to fill them via permuted_dense_ATDA_fill_values. */
Matrix *permuted_dense_ATA_alloc(const Permuted_Dense *self);

/* Fill out.X = self.X^T diag(d) self.X, where d has length self->base.m.
   out must have the structure produced by permuted_dense_ATA_alloc(self).
   Uses self->Y_scratch as workspace; const-correctness is preserved because
   only the buffer pointed to by Y_scratch is mutated. */
void permuted_dense_ATDA_fill_values(const Permuted_Dense *self, const double *d,
                                     Permuted_Dense *out);

/* Allocate a new Permuted_Dense for C = B^T @ A where A and B are both PD.
   Output shape (B->base.n, A->base.n) with dense block (B->dense_n,
   A->dense_n), row_perm = B->col_perm, col_perm = A->col_perm. Values
   uninitialized. The output structure does not depend on row_perm_A or
   row_perm_B (only the values do; see permuted_dense_BTA_fill_values). */
Matrix *permuted_dense_BTA_alloc(const Permuted_Dense *A, const Permuted_Dense *B);

/* Fill out->X = B->X^T @ A->X restricted to rows in row_perm_A ∩ row_perm_B.
   out must have the structure produced by permuted_dense_BTA_alloc(A, B).
   For matching row_perms, this is a single cblas_dgemm; otherwise the
   intersecting rows are first gathered into contiguous scratch buffers. */
void permuted_dense_BTA_fill_values(const Permuted_Dense *A, const Permuted_Dense *B,
                                    Permuted_Dense *out);

/* Allocate a new Permuted_Dense for C = B^T @ A where A is Sparse (CSR)
   and B is PD. Output is PD with row_perm = B->col_perm and col_perm = the
   sorted union of columns appearing in A's rows at positions row_perm_B.
   Dense block size = (B->dense_n, |col_active|). Values uninitialized. */
Matrix *BTA_csr_pd_alloc(const CSR_Matrix *A_csr, const Permuted_Dense *B);

/* Fill out->X = X_B^T @ A_sub_dense, where A_sub_dense is A's rows at
   positions row_perm_B, columns restricted to out's col_perm, scattered
   to a dense buffer. out must have the structure produced by
   BTA_csr_pd_alloc(A_csr, B). */
void BTA_csr_pd_fill_values(const CSR_Matrix *A_csr, const Permuted_Dense *B,
                            Permuted_Dense *out);

/* Allocate a new Permuted_Dense for C = B^T @ A where A is PD and B is
   Sparse (CSR). Output is PD with row_perm = the sorted union of columns
   appearing in B's rows at positions row_perm_A, and col_perm = col_perm_A.
   Dense block size = (|row_active|, dense_n_A). Values uninitialized. */
Matrix *BTA_pd_csr_alloc(const Permuted_Dense *A, const CSR_Matrix *B_csr);

/* Fill out->X = B_sub^T @ X_A, where B_sub is B's rows at positions
   row_perm_A, columns restricted to out's row_perm, scattered to a dense
   buffer. out must have the structure produced by BTA_pd_csr_alloc(A, B). */
void BTA_pd_csr_fill_values(const Permuted_Dense *A, const CSR_Matrix *B_csr,
                            Permuted_Dense *out);

/* BTDA variants — fold a diagonal d into the BTA computation. Each fills
   out->X = B^T diag(d) A (d may be NULL for plain B^T A). out must have
   the structure produced by the corresponding BTA *_alloc function. */
void BTDA_csr_pd_fill_values(const CSR_Matrix *A_csr, const double *d,
                             const Permuted_Dense *B, Permuted_Dense *out);
void BTDA_pd_csr_fill_values(const Permuted_Dense *A, const double *d,
                             const CSR_Matrix *B_csr, Permuted_Dense *out);
void BTDA_pd_pd_fill_values(const Permuted_Dense *A, const double *d,
                            const Permuted_Dense *B, Permuted_Dense *out);

/* Allocate a new Permuted_Dense for C = self @ J. C has global shape
   (self->base.m, J->n) with row_perm = self->row_perm and col_perm equal
   to the sorted list of columns of J that have at least one structural
   nonzero in some row in self->col_perm. Values are uninitialized. */
Matrix *permuted_dense_times_csc_alloc(const Permuted_Dense *self,
                                       const CSC_Matrix *J);

/* Fill out.X[ii, jj] = sum_kk self.X[ii, kk] * J[col_perm_self[kk],
   col_perm_out[jj]]. out must have the structure produced by
   permuted_dense_times_csc_alloc(self, J). */
void permuted_dense_times_csc_fill_values(const Permuted_Dense *self,
                                          const CSC_Matrix *J, Permuted_Dense *out);

#endif /* PERMUTED_DENSE_H */
