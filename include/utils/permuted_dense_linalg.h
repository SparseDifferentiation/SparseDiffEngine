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
#ifndef PERMUTED_DENSE_LINALG_H
#define PERMUTED_DENSE_LINALG_H

#include "CSC_matrix.h"
#include "permuted_dense.h"

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

/* Allocate a new PD C with the same sparsity (m, n, m0, n0, row_perm,
   col_perm) as A; values uninitialized. */
matrix *copy_sparsity_pd_alloc(const permuted_dense *A);

/* Allocate C = A^T (transpose). */
matrix *transpose_pd_alloc(const permuted_dense *A);

/* Fill values of C = A^T. */
void transpose_pd_fill_values(const permuted_dense *A, permuted_dense *C);

#endif /* PERMUTED_DENSE_LINALG_H */
