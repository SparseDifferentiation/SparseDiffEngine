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
    CSR_Matrix *csr_cache; /* lazy CSR view built by to_csr; structure */
                           /* allocated on first call, values refilled */
                           /* on every call. NULL until first call.    */
} Permuted_Dense;

/* Constructor. row_perm and col_perm must be strictly increasing in their
   respective ranges. If X_data is NULL the value buffer is allocated but
   left uninitialized; otherwise dense_m * dense_n entries are copied. */
Matrix *new_permuted_dense(int m, int n, int dense_m, int dense_n,
                           const int *row_perm, const int *col_perm,
                           const double *X_data);

/* Convert to CSR. The output has dense_m * dense_n nonzeros. */
CSR_Matrix *permuted_dense_to_csr_alloc(const Permuted_Dense *self);
void permuted_dense_to_csr_fill_values(const Permuted_Dense *self, CSR_Matrix *out);

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
