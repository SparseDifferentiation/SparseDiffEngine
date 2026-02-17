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
#include "affine.h"
#include "bivariate.h"
#include "subexpr.h"
#include "utils/CSR_Matrix.h"
#include "utils/linalg_sparse_matmuls.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* Refresh AT and A values from param_source for right matmul.
   param_source stores values in CSR order for the original A. */
static void refresh_param_values(left_matmul_expr *lin_node)
{
    parameter_expr *param = (parameter_expr *) lin_node->param_source;

    if (!param || param->has_been_refreshed) return;
    param->has_been_refreshed = true;

    assert(param->param_id != PARAM_FIXED);

    /* update values of original A (stored in lin_node->AT) */
    memcpy(lin_node->AT->x, lin_node->param_source->value,
           lin_node->AT->nnz * sizeof(double));

    /* update values of A^T (stored in lin_node->A) */
    AT_fill_values(lin_node->AT, lin_node->A, lin_node->AT_iwork);
}

/* This file implements the atom 'right_matmul' corresponding to the operation y =
    f(x) @ A, where A is a given matrix and f(x) is an arbitrary expression.
    We implement this by expressing right matmul in terms of left matmul and
    transpose: f(x) @ A = (A^T @ f(x)^T)^T. */
expr *new_right_matmul(expr *param_node, expr *u, const CSR_Matrix *A)
{
    /* We can express right matmul using left matmul and transpose:
       u @ A = (A^T @ u^T)^T. */
    int *work_transpose = (int *) malloc(A->n * sizeof(int));
    CSR_Matrix *AT = transpose(A, work_transpose);
    expr *u_transpose = new_transpose(u);
    expr *left_matmul_node = new_left_matmul(param_node, u_transpose, AT);
    expr *node = new_transpose(left_matmul_node);

    /* functionality for parameter */
    left_matmul_expr *left_matmul_data = (left_matmul_expr *) left_matmul_node;
    free(left_matmul_data->AT_iwork);
    left_matmul_data->AT_iwork = work_transpose;
    left_matmul_data->refresh_param_values = refresh_param_values;

    free_csr_matrix(AT);
    return node;
}
