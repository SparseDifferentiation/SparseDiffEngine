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
#include "old-code/old_affine.h"
#include "subexpr.h"
#include "utils/CSR_Matrix.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    linear_op_expr *lin_node = (linear_op_expr *) node;

    /* child's forward pass */
    node->left->forward(node->left, u);

    /* y = A * x (A is stored as node->jacobian) */
    Ax_csr(node->jacobian, x->value, node->value, x->var_id);

    /* y += b (if offset exists) */
    if (lin_node->b != NULL)
    {
        for (int i = 0; i < node->size; i++)
        {
            node->value[i] += lin_node->b[i];
        }
    }
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    linear_op_expr *lin_node = (linear_op_expr *) node;
    if (lin_node->b != NULL)
    {
        free(lin_node->b);
        lin_node->b = NULL;
    }
}

static void jacobian_init_impl(expr *node)
{
    /* jacobian is set at construction time — nothing to do */
    (void) node;
}

static void eval_jacobian(expr *node)
{
    /* Linear operator jacobian never changes - nothing to evaluate */
    (void) node;
}

static void wsum_hess_init_impl(expr *node)
{
    /* Linear operator Hessian is always zero */
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, 0, NULL);
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* Linear operator Hessian is always zero - nothing to evaluate */
    (void) node;
    (void) w;
}

expr *new_linear(expr *u, const CSR_Matrix *A, const double *b)
{
    assert(u->d2 == 1);
    /* Allocate the type-specific struct */
    linear_op_expr *lin_node = (linear_op_expr *) calloc(1, sizeof(linear_op_expr));
    expr *node = &lin_node->base;
    init_expr(node, A->m, 1, u->n_vars, forward, jacobian_init_impl, eval_jacobian,
              is_affine, wsum_hess_init_impl, eval_wsum_hess, free_type_data);
    node->left = u;
    expr_retain(u);

    /* Store A directly as the jacobian (linear op jacobian is constant) */
    node->jacobian = new_csr_matrix(A->m, A->n, A->nnz, NULL);
    copy_csr_matrix(A, node->jacobian);

    /* Initialize offset (copy b if provided, otherwise NULL) */
    if (b != NULL)
    {
        lin_node->b = (double *) malloc(A->m * sizeof(double));
        memcpy(lin_node->b, b, A->m * sizeof(double));
    }
    else
    {
        lin_node->b = NULL;
    }

    return node;
}
