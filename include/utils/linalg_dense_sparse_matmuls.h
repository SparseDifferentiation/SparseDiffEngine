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
#ifndef LINALG_DENSE_SPARSE_H
#define LINALG_DENSE_SPARSE_H

#include "CSC_matrix.h"
#include "CSR_matrix.h"
#include "matrix.h"

/* C = (I_p kron A) @ J via the polymorphic matrix interface.
 * A is dense m x n, J is (n*p) x k in CSC_matrix, C is (m*p) x k in CSC_matrix. */
// TODO: maybe we can replace these with I_kron_X functionality?
CSC_matrix *I_kron_A_alloc(const matrix *A, const CSC_matrix *J, int p);
void I_kron_A_fill_values(const matrix *A, const CSC_matrix *J, CSC_matrix *C);

/* Sparsity and values of C = (Y^T kron I_m) @ J where Y is k x n, J is (m*k) x p,
   and C is (m*n) x p. Y is given in column-major dense format. */
CSR_matrix *YT_kron_I_alloc(int m, int k, int n, const CSC_matrix *J);
void YT_kron_I_fill_values(int m, int k, int n, const double *Y, const CSC_matrix *J,
                           CSR_matrix *C);

/* Sparsity and values of C = (I_n kron X) @ J where X is m x k (col-major dense),
   J is (k*n) x p, and C is (m*n) x p. */
CSR_matrix *I_kron_X_alloc(int m, int k, int n, const CSC_matrix *J);
void I_kron_X_fill_values(int m, int k, int n, const double *X, const CSC_matrix *J,
                          CSR_matrix *C);

#endif /* LINALG_DENSE_SPARSE_H */
