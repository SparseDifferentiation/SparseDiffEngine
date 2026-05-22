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
#ifndef STACKED_PD_KRON_LINALG_H
#define STACKED_PD_KRON_LINALG_H

#include "CSC_matrix.h"
#include "matrix.h"
#include "permuted_dense.h"
#include "stacked_pd.h"

/* All three families compute C = kron(I_p, A) @ J as a stacked_pd, without
   materializing kron(I_p, A).

   Contract on A: A must be a "full" permuted_dense — m0 == base.m,
   n0 == base.n, and row_perm / col_perm are the identity — i.e., a
   local dense matrix wrapped in the permuted_dense struct, typically
   constructed via new_permuted_dense_full. The kron helpers index into
   A by local (i, j); a non-full A would produce silently wrong sparsity.
   The public dispatcher BA_dense_kron_matrices_alloc asserts this in
   Debug builds. */

/* Allocate sparsity for stacked_pd C = kron(I_p, A) @ J where J is CSC. */
matrix *BA_dense_kron_csc_alloc(const permuted_dense *A, int p, const CSC_matrix *J);

/* Fill values of stacked_pd C = kron(I_p, A) @ J where J is CSC. */
void BA_dense_kron_csc_fill_values(const permuted_dense *A, int p,
                                   const CSC_matrix *J, stacked_pd *C);

/* Allocate sparsity for stacked_pd C = kron(I_p, A) @ J where J is PD. */
matrix *BA_dense_kron_pd_alloc(const permuted_dense *A, int p,
                               const permuted_dense *J);

/* Fill values of stacked_pd C = kron(I_p, A) @ J where J is permuted_dense. */
void BA_dense_kron_pd_fill_values(const permuted_dense *A, int p,
                                  const permuted_dense *J, stacked_pd *C);

/* Allocate sparsity for stacked_pd C = kron(I_p, A) @ J where J is stacked_pd. */
matrix *BA_dense_kron_spd_alloc(const permuted_dense *A, int p, const stacked_pd *J);

/* Fill values of stacked_pd C = kron(I_p, A) @ J where J is stacked_pd. */
void BA_dense_kron_spd_fill_values(const permuted_dense *A, int p,
                                   const stacked_pd *J, stacked_pd *C);

#endif /* STACKED_PD_KRON_LINALG_H */
