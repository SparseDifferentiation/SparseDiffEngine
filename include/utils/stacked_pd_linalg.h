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

#include "CSC_matrix.h"
#include "permuted_dense.h"
#include "stacked_pd.h"

/* Allocate C = B @ A where B is spd, A is CSC. */
matrix *BA_spd_csc_alloc(const stacked_pd *B, const CSC_matrix *A);

/* Fill values of C = B @ A where B is spd and A is CSC. */
void BA_spd_csc_fill_values(const stacked_pd *B, const CSC_matrix *A, stacked_pd *C);

/* Allocate a new spd C with the same sparsity as A.
   TODO: should this be publically available? */
matrix *copy_sparsity_spd_alloc(const stacked_pd *A);

/* Fill values of C = diag(d) @ A, where 'd' is 'global' with length A->m. */
void DA_spd_fill_values(const double *d, const stacked_pd *A, stacked_pd *C);

/* Allocate sparsity for C = B @ A where B spd and A is a permuted_dense. */
matrix *BA_spd_pd_alloc(const stacked_pd *B, const permuted_dense *A);

/* Fill values of C = B @ A where B is pd and A is a permuted_dense. */
void BA_spd_pd_fill_values(const stacked_pd *B, const permuted_dense *A,
                           stacked_pd *C);

/* Allocate sparsity for C = B @ A where B is a permuted_dense and A is spd */
matrix *BA_pd_spd_alloc(const permuted_dense *B, const stacked_pd *A);

/* Fill values of C = B @ A where B is a permuted_dense and A is spd */
void BA_pd_spd_fill_values(const permuted_dense *B, const stacked_pd *A,
                           permuted_dense *C);

/* Allocate sparsity for C = B @ A where both B and A are spds */
matrix *BA_spd_spd_alloc(const stacked_pd *B, const stacked_pd *A);

/* Fill values of C = B @ A where both B and A are spds */
void BA_spd_spd_fill_values(const stacked_pd *B, const stacked_pd *A, stacked_pd *C);

/* Allocate sparsity for C = A^T A. A^T A decomposes as Σ_k B_k^T B_k,
   where summands with overlapping col_perms (C_k) share cells. The
   output groups cols of A by signature sig_C(c) = {k : c ∈ C_k}; each
   unique signature becomes one output PD with row_perm = group cols
   and col_perm = ⋃ C_k for k in the signature. No structural zeros.
   The output's `work` slot holds per-source scratch PDs (one symmetric
   PD per source block, row_perm = col_perm = C_k) that
   ATDA_spd_fill_values writes into. */
matrix *ATA_spd_alloc(const stacked_pd *A);

/* Fill values of C = A^T diag(d) A. */
void ATDA_spd_fill_values(const stacked_pd *A, const double *d, stacked_pd *C);

/* Allocate out = transpose(src). Implementation: transpose each source
   PD block individually (yielding a raw spd that may have overlapping
   row perms but pairwise-disjoint cells, by the spd invariant), then
   coalesce. The raw spd is stored on `out->work` for reuse by
   `transpose_spd_fill_values`. */
matrix *transpose_spd_alloc(const stacked_pd *A);

/* Fill values of out = transpose(src). The caller must have produced
   `out` via transpose_spd_alloc on a `src` with the same structure
   (block count, row_perms, col_perms) as the one passed now. */
void transpose_spd_fill_values(const stacked_pd *A, stacked_pd *C);

#endif /* STACKED_PD_LINALG_H */
