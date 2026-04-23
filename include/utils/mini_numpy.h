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
#ifndef MINI_NUMPY_H
#define MINI_NUMPY_H

#include "utils/CSR_Matrix.h"

/* Example: a = [1, 2], len = 2, repeats = 3, result = [1, 1, 1, 2, 2, 2] */
void repeat(double *result, const double *a, int len, int repeats);

/* Example: a = [1, 2], len = 2, tiles = 3, result = [1, 2, 1, 2, 1, 2] */
void tile_double(double *result, const double *a, int len, int tiles);
void tile_int(int *result, const int *a, int len, int tiles);

/* Example: size = 5, value = 3.0, result = [3.0, 3.0, 3.0, 3.0, 3.0] */
void scaled_ones(double *result, int size, double value);

/* Naive implementation of Z = X @ Y, X is m x k, Y is k x n, Z is m x n */
void mat_mat_mult(const double *X, const double *Y, double *Z, int m, int k, int n);

/* Compute v = (Y kron I_m) @ w where Y is k x n (col-major), len(w) = m * n, and
   len(v) = m * k.  Equivalently, reshape w as the m x n matrix W (col-major) and
   compute v = vec(W @ Y^T). */
void Y_kron_I_vec(int m, int k, int n, const double *Y, const double *w, double *v);

/* Compute v = (I_n kron X^T) @ w where X is m x k (col-major), len(w) = m * n, and
   len(v) = k * n.  Equivalently, reshape w as the m x n matrix W (col-major) and
   compute v = vec(X^T @ W). */
void I_kron_XT_vec(int m, int k, int n, const double *X, const double *w, double *v);

/* Fill T_csr's row pointers and column indices for the 1D full-convolution
   Toeplitz matrix T(a), sized (m+n-1) x n with m*n nonzeros. Values (x) are
   not written; call conv_matrix_fill_values to populate them. */
void conv_matrix_fill_sparsity(CSR_Matrix *T_csr, int m, int n);

/* Overwrite T_csr->x from kernel a, using the sparsity already written by
   conv_matrix_fill_sparsity. T[r, col] = a[r - col]. */
void conv_matrix_fill_values(CSR_Matrix *T_csr, const double *a);

#endif /* MINI_NUMPY_H */
