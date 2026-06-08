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

/* quad-form with sparse constant matrix Q */
expr *new_quad_form_sparse(expr *child, CSR_matrix *Q);

/* quad-form with dense constant or parametric matrix Q (assumed to be
   symmetric). For constant Q, Q_data should point to the values of Q in
   row-major order. For parametric Q, Q_data should be NULL and param_source
   should point to the parameter node. */
expr *new_quad_form_dense(expr *child, int n, const double *Q_data,
                          expr *param_source);

/* product of all entries, without axis argument */
expr *new_prod(expr *child);

/* product of entries along axis=0 (columnwise products) */
expr *new_prod_axis_zero(expr *child);

/* product of entries along axis=1 (rowwise products) */
expr *new_prod_axis_one(expr *child);

#endif /* NON_ELEMENTWISE_FULL_DOM_H */
