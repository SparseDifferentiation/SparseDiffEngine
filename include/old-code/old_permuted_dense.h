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
#ifndef OLD_PERMUTED_DENSE_H
#define OLD_PERMUTED_DENSE_H

#include "utils/CSR_matrix.h"
#include "utils/permuted_dense.h"

/* Legacy CSR-based (PD, Sparse) BTA / BTDA kernels.

   Mathematically equivalent to BTA_pd_csc_alloc / BTDA_pd_csc_fill_values
   in src/utils/permuted_dense.c — they all compute C = B^T (diag(d)) A
   for B PD and A sparse. The matrix_BTA dispatcher used to choose between
   the CSR-here and CSC-in-utils variants; after a benchmark on
   trimmed_log_reg-shaped workloads we committed to CSC and moved these
   kernels out of production paths.

   Kept here as a reference implementation, as cross-comparison fodder for
   tests (test_BTA_pd_csc_matches_csr), and as the CSR side of the
   profile_BTA_pd_csr_vs_csc microbenchmark. */

/* Allocate a new permuted_dense for C = B^T A where B is PD and A is
   CSR-sparse. Output is PD with row_perm = B->col_perm and col_perm = the
   sorted union of columns appearing in A's rows at positions row_perm_B.
   Dense block size = (B->n0, |col_active|). Values uninitialized. */
matrix *BTA_pd_csr_alloc(const permuted_dense *B, const CSR_matrix *A);

/* Fill C->X = X_B^T @ A_sub_dense, where A_sub_dense is A's rows at
   positions row_perm_B, columns restricted to C's col_perm, scattered to a
   dense buffer. C must have the structure produced by BTA_pd_csr_alloc. */
void BTA_pd_csr_fill_values(const permuted_dense *B, const CSR_matrix *A,
                            permuted_dense *C);

/* BTDA variant: C->X = X_B^T diag(d) A_sub_dense. d may be NULL (treated
   as identity scaling). C must have the structure produced by
   BTA_pd_csr_alloc. */
void BTDA_pd_csr_fill_values(const permuted_dense *B, const double *d,
                             const CSR_matrix *A, permuted_dense *C);

/* Legacy CSR-pd kernels (B=CSR, A=PD), formerly in src/utils/permuted_dense.c.
   Production now dispatches the (PD A, sparse B) branch through CSC-pd
   kernels (BTA_csc_pd_alloc / BTDA_csc_pd_fill_values in utils/permuted_dense.h),
   so these CSR variants live here as reference implementations and as
   targets for the direct unit tests in tests/old-code. */

/* Allocate a new permuted_dense for C = B^T A where B is CSR-sparse and A
   is PD. Output is PD with row_perm = the sorted union of columns appearing
   in B's rows at positions row_perm_A, and col_perm = A->col_perm. */
matrix *BTA_csr_pd_alloc(const CSR_matrix *B_csr, const permuted_dense *A);

/* No-d BTA fill. C must have the structure produced by BTA_csr_pd_alloc. */
void BTA_csr_pd_fill_values(const CSR_matrix *B_csr, const permuted_dense *A,
                            permuted_dense *C);

/* BTDA variant: C->X = B_sub_dense^T diag(d) X_A. d may be NULL (treated
   as identity scaling). C must have the structure produced by
   BTA_csr_pd_alloc. */
void BTDA_csr_pd_fill_values(const CSR_matrix *B_csr, const double *d,
                             const permuted_dense *A, permuted_dense *C);

#endif /* OLD_PERMUTED_DENSE_H */
