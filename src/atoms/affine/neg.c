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
#include "utils/tracked_alloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    /* child's forward pass */
    node->left->forward(node->left, u);

    /* negate values */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = -node->left->value[i];
    }
}

static void jacobian_init_impl(expr *node)
{
    expr *x = node->left;
    /* initialize child's jacobian */
    jacobian_init(x);

    /* same sparsity pattern as child */
    node->jacobian = new_csr_copy_sparsity(x->jacobian);
}

static void eval_jacobian(expr *node)
{
    /* evaluate child's jacobian */
    node->left->eval_jacobian(node->left);

    /* negate values only (sparsity pattern set in jacobian_init_impl) */
    CSR_Matrix *child_jac = node->left->jacobian;
    for (int k = 0; k < child_jac->nnz; k++)
    {
        node->jacobian->x[k] = -child_jac->x[k];
    }
}

static void wsum_hess_init_impl(expr *node)
{
    expr *x = node->left;

    /* initialize child's wsum_hess */
    wsum_hess_init(x);

    /* same sparsity pattern as child */
    CSR_Matrix *child_hess = x->wsum_hess;
    node->wsum_hess = new_csr_copy_sparsity(child_hess);
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* For f(x) = -g(x): d²f/dx² = -d²g/dx² */
    node->left->eval_wsum_hess(node->left, w);

    /* negate values (sparsity pattern set in wsum_hess_init_impl) */
    CSR_Matrix *child_hess = node->left->wsum_hess;
    for (int k = 0; k < child_hess->nnz; k++)
    {
        node->wsum_hess->x[k] = -child_hess->x[k];
    }
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_neg(expr *child)
{
    expr *node = (expr *) SP_CALLOC(1, sizeof(expr));
    init_expr(node, child->d1, child->d2, child->n_vars, forward, jacobian_init_impl,
              eval_jacobian, is_affine, wsum_hess_init_impl, eval_wsum_hess, NULL);
    node->left = child;
    expr_retain(child);

    return node;
}
