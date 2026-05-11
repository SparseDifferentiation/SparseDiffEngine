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
#include "atoms/affine.h"
#include "utils/tracked_alloc.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* Promote broadcasts a scalar expression to a vector/matrix shape.
 * This matches CVXPY's promote atom which only handles scalars. */

static void forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);

    /* broadcast scalar value to all output elements */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = node->left->value[0];
    }
}

static void jacobian_init_impl(expr *node)
{
    expr *x = node->left;
    jacobian_init(x);

    /* allocate sparsity for an (node->size, n_vars) matrix whose rows are all
       copies of the child's single row; output type matches child's type. */
    node->jacobian = x->jacobian->promote_alloc(x->jacobian, node->size);
}

static void eval_jacobian(expr *node)
{
    node->left->eval_jacobian(node->left);

    /* tile the child's single row into the preallocated output. */
    node->left->jacobian->promote_fill_values(node->left->jacobian, node->jacobian);
}

static void wsum_hess_init_impl(expr *node)
{
    wsum_hess_init(node->left);
    node->wsum_hess = node->left->wsum_hess->copy_sparsity(node->left->wsum_hess);
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* Sum all weights (they all correspond to the same scalar child) */
    double sum_w = 0.0;
    for (int i = 0; i < node->size; i++)
    {
        sum_w += w[i];
    }

    /* evaluate child's wsum_hess with summed weight */
    node->left->eval_wsum_hess(node->left, &sum_w);

    /* copy values */
    memcpy(node->wsum_hess->x, node->left->wsum_hess->x,
           node->left->wsum_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_promote(expr *child, int d1, int d2)
{
    assert(child->size == 1);
    expr *node = (expr *) SP_CALLOC(1, sizeof(expr));
    init_expr(node, d1, d2, child->n_vars, forward, jacobian_init_impl,
              eval_jacobian, is_affine, wsum_hess_init_impl, eval_wsum_hess, NULL);
    node->left = child;
    expr_retain(child);

    return node;
}
