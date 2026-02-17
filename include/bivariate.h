/*
 * Copyright 2026 Daniel Cederberg and William Zhang
 *
 * This file is part of the DNLP-differentiation-engine project.
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
#ifndef BIVARIATE_H
#define BIVARIATE_H

#include "expr.h"

expr *new_elementwise_mult(expr *left, expr *right);
expr *new_rel_entr_vector_args(expr *left, expr *right);
expr *new_quad_over_lin(expr *left, expr *right);

expr *new_rel_entr_first_arg_scalar(expr *left, expr *right);
expr *new_rel_entr_second_arg_scalar(expr *left, expr *right);

/* Matrix multiplication: Z = X @ Y */
expr *new_matmul(expr *x, expr *y);

/* Left matrix multiplication: A @ f(x) where A comes from a parameter node.
   Only the forward pass possibly updates the parameter. */
expr *new_left_matmul(expr *param_node, expr *child, const CSR_Matrix *A);

/* Right matrix multiplication: f(x) @ A where A is a fixed parameter matrix */
expr *new_right_matmul(expr *u, const CSR_Matrix *A);

/* Scalar multiplication: a * f(x) where a comes from a parameter node */
expr *new_scalar_mult(expr *param_node, expr *child);

/* Vector elementwise multiplication: a ∘ f(x) where a comes from a parameter node */
expr *new_vector_mult(expr *param_node, expr *child);

#endif /* BIVARIATE_H */
