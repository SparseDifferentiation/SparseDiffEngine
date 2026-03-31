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
#include "atoms/affine.h"
#include "subexpr.h"
#include "utils/CSR_Matrix.h"
#include "utils/dense_matrix.h"
#include <stdlib.h>

/* This file implements the atom 'right_matmul' corresponding to the operation y =
   f(x) @ A, where A is a given matrix and f(x) is an arbitrary expression.
    We implement this by expressing right matmul in terms of left matmul and
    transpose: f(x) @ A = (A^T @ f(x)^T)^T.

   For the parameter case:
     - param_source stores A values in CSR data order
     - inner left_matmul stores AT as its A-matrix and A as its AT-matrix
     - on refresh: update AT (inner's AT, the original A) from param_source,
       then recompute A^T (inner's A) from the updated A. */

/* Refresh for sparse right_matmul: param stores A in CSR data order.
   Inner left_matmul: lnode->A = AT (transposed), lnode->AT = A (original).
   So: update lnode->AT from param values, then recompute lnode->A. */
static void refresh_sparse_right(left_matmul_expr *lnode)
{
    Sparse_Matrix *sm_AT_inner = (Sparse_Matrix *) lnode->A;
    Sparse_Matrix *sm_A_inner = (Sparse_Matrix *) lnode->AT;
    /* lnode->AT holds the original A; update its values from param */
    lnode->AT->update_values(lnode->AT, lnode->param_source->value);
    /* Recompute A^T (lnode->A) from A (lnode->AT) */
    AT_fill_values(sm_A_inner->csr, sm_AT_inner->csr, lnode->base.work->iwork);
}

static void refresh_dense_right(left_matmul_expr *lnode)
{
    Dense_Matrix *dm_AT_inner = (Dense_Matrix *) lnode->A;
    Dense_Matrix *dm_A_inner = (Dense_Matrix *) lnode->AT;
    int m_orig = dm_A_inner->base.m; /* original A is m x n */
    int n_orig = dm_A_inner->base.n;
    /* Update original A (inner's AT) from param values */
    lnode->AT->update_values(lnode->AT, lnode->param_source->value);
    /* Recompute A^T (inner's A) from A */
    for (int i = 0; i < m_orig; i++)
    {
        for (int j = 0; j < n_orig; j++)
        {
            dm_AT_inner->x[j * m_orig + i] = dm_A_inner->x[i * n_orig + j];
        }
    }
}

expr *new_right_matmul(expr *param_node, expr *u, const CSR_Matrix *A)
{
    /* We can express right matmul using left matmul and transpose:
       u @ A = (A^T @ u^T)^T. */
    int *work_transpose = (int *) malloc(A->n * sizeof(int));
    CSR_Matrix *AT = transpose(A, work_transpose);

    expr *u_transpose = new_transpose(u);
    expr *left_matmul = new_left_matmul(NULL, u_transpose, AT);

    /* If parameterized, attach param_source and custom refresh to inner
       left_matmul */
    if (param_node != NULL)
    {
        left_matmul_expr *lnode = (left_matmul_expr *) left_matmul;
        lnode->param_source = param_node;
        expr_retain(param_node);
        lnode->refresh_param_values = refresh_sparse_right;
    }

    expr *node = new_transpose(left_matmul);

    free_csr_matrix(AT);
    free(work_transpose);
    return node;
}

expr *new_right_matmul_dense(expr *param_node, expr *u, int m, int n,
                             const double *data)
{
    /* We express: u @ A = (A^T @ u^T)^T
       A is m x n, so A^T is n x m. */
    double *AT = (double *) malloc(n * m * sizeof(double));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            AT[j * m + i] = data[i * n + j];
        }
    }

    expr *u_transpose = new_transpose(u);
    expr *left_matmul_node = new_left_matmul_dense(NULL, u_transpose, n, m, AT);

    /* If parameterized, attach param_source and custom refresh */
    if (param_node != NULL)
    {
        left_matmul_expr *lnode = (left_matmul_expr *) left_matmul_node;
        lnode->param_source = param_node;
        expr_retain(param_node);
        lnode->refresh_param_values = refresh_dense_right;
    }

    expr *node = new_transpose(left_matmul_node);

    free(AT);
    return node;
}
