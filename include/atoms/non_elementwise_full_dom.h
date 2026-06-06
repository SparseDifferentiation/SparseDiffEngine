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
#ifndef NON_ELEMENTWISE_FULL_DOM_H
#define NON_ELEMENTWISE_FULL_DOM_H

#include "expr.h"
#include "subexpr.h"
#include "utils/CSR_matrix.h"

expr *new_quad_form_sparse(expr *child, CSR_matrix *Q);

/* Dense / parametric quadratic form y = x' P x over a vector expression x (a
 * leaf variable, or a composition x = f(u) handled via the chain rule).
 *
 * P is n x n, row-major, and assumed symmetric (matching the new_quad_form_sparse
 * convention where the Hessian of x'Qx is taken to be 2Q). For a leaf x the
 * Hessian is materialized as a dense permuted_dense block.
 *
 *   - constant P:    P_data points to n*n doubles, param_source == NULL.
 *   - parametric P:  P_data == NULL, param_source is the parameter node that
 *                    supplies P (n*n doubles) and is refreshed each solve.
 */
expr *new_quad_form_dense(expr *child, int n, const double *P_data,
                          expr *param_source);

/* product of all entries, without axis argument */
expr *new_prod(expr *child);

/* product of entries along axis=0 (columnwise products) */
expr *new_prod_axis_zero(expr *child);

/* product of entries along axis=1 (rowwise products) */
expr *new_prod_axis_one(expr *child);

#endif /* NON_ELEMENTWISE_FULL_DOM_H */
