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
#ifndef AFFINE_H
#define AFFINE_H

#include "expr.h"
#include "subexpr.h"
#include "utils/CSR_Matrix.h"

expr *new_add(expr *left, expr *right);
expr *new_neg(expr *child);

expr *new_sum(expr *child, int axis);
expr *new_hstack(expr **args, int n_args, int n_vars);
expr *new_vstack(expr **args, int n_args, int n_vars);
expr *new_promote(expr *child, int d1, int d2);
expr *new_trace(expr *child);

expr *new_parameter(int d1, int d2, int param_id, int n_vars, const double *values);
expr *new_variable(int d1, int d2, int var_id, int n_vars);

expr *new_index(expr *child, int d1, int d2, const int *indices, int n_idxs);
expr *new_reshape(expr *child, int d1, int d2);
expr *new_broadcast(expr *child, int target_d1, int target_d2);
expr *new_diag_vec(expr *child);
expr *new_diag_mat(expr *child);
expr *new_upper_tri(expr *child);
expr *new_transpose(expr *child);

/* Left matrix multiplication: A @ f(x) where A is a constant or parameter
 * sparse matrix. param_node is NULL for fixed constants. */
expr *new_left_matmul(expr *param_node, expr *u, const CSR_Matrix *A);

/* Left matrix multiplication: A @ f(x) where A is a constant or parameter
 * dense matrix (row-major, m x n). Uses CBLAS for efficient computation. */
expr *new_left_matmul_dense(expr *param_node, expr *u, int m, int n,
                            const double *data);

/* Right matrix multiplication: f(x) @ A where A is a constant or parameter
 * matrix. */
expr *new_right_matmul(expr *param_node, expr *u, const CSR_Matrix *A);

expr *new_right_matmul_dense(expr *param_node, expr *u, int m, int n,
                             const double *data);

/* Kronecker product with constant on the left: Z = kron(C, u) where C is a
 * constant sparse matrix and u is a (p x q) expression. Output shape
 * (C->m * p, C->n * q). param_node must be NULL; the parameter path is
 * reserved for a future change. */
expr *new_kron_left(expr *param_node, expr *u, const CSR_Matrix *C, int p, int q);

/* Scalar multiplication: a * f(x) where a comes from param_node */
expr *new_scalar_mult(expr *param_node, expr *child);

/* Vector elementwise multiplication: a . f(x) where a comes from
 * param_node */
expr *new_vector_mult(expr *param_node, expr *child);

#endif /* AFFINE_H */
