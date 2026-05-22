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
#ifndef STACKED_PD_LINALG_H
#define STACKED_PD_LINALG_H

#include "permuted_dense.h"
#include "stacked_pd.h"

/* Allocate a new spd C with the same sparsity as A.
   TODO: should this be publically available? */
matrix *copy_sparsity_spd_alloc(const stacked_pd *A);

/* Fill values of C = diag(d) @ A, where 'd' is 'global' with length A->m. */
void DA_spd_fill_values(const double *d, const stacked_pd *A, stacked_pd *C);

/* Allocate sparsity for C = B @ A where B is a permuted_dense and A is spd.
   The output C is permuted_dense. */
matrix *BA_pd_spd_alloc(const permuted_dense *B, const stacked_pd *A);

/* Fill values of C = B @ A where C and B are permuted_dense and A is spd */
void BA_pd_spd_fill_values(const permuted_dense *B, const stacked_pd *A,
                           permuted_dense *C);

/* Allocate sparsity for C = B^T @ A where B is a permuted_dense and A is spd.
   The output C is permuted_dense. */
matrix *BTA_pd_spd_alloc(const permuted_dense *B, const stacked_pd *A);

/* Fill values of C = B^T @ A where B is permuted_dense, A is spd and C is
   permuted_dense. */
void BTA_pd_spd_fill_values(const permuted_dense *B, const stacked_pd *A,
                            permuted_dense *C);

/* Fill values of C = B^T @ diag(d) @ A where B is permuted_dense, A is spd,
   d is a global vector of length A->m, and C is permuted_dense. */
void BTDA_pd_spd_fill_values(const permuted_dense *B, const double *d,
                             const stacked_pd *A, permuted_dense *C);

/* Allocate sparsity for C = B^T @ A where B is stacked_pd and A is
   permuted_dense. The output C is stacked_pd. */
matrix *BTA_spd_pd_alloc(const stacked_pd *B, const permuted_dense *A);

/* Fill values of C = B^T @ diag(d) @ A where B is stacked_pd, A is
   permuted_dense, d is a global vector of length B->m, and C is stacked_pd. */
void BTDA_spd_pd_fill_values(const stacked_pd *B, const double *d,
                             const permuted_dense *A, stacked_pd *C);

/* Allocate sparsity for a new stacked_pd C = A^T A. */
matrix *ATA_spd_alloc(const stacked_pd *A);

/* Fill values of C = A^T diag(d) A. */
void ATDA_spd_fill_values(const stacked_pd *A, const double *d, stacked_pd *C);

/* Allocate a new stacked pd C = transpose(A). C is guaranteed to be a valid
   stacked_pd (ie. it has disjoint row permutations across blocks). The raw spd
   is stored on `C->pre_coalesce` for reuse by `transpose_spd_fill_values`. */
matrix *transpose_spd_alloc(const stacked_pd *A);

/* Fill values of C = transpose(A). */
void transpose_spd_fill_values(const stacked_pd *A, stacked_pd *C);

#endif /* STACKED_PD_LINALG_H */
