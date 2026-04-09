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
#include "utils/tracked_alloc.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Scalar multiplication: y = a * child where a comes from param_source */

static void forward(expr *node, const double *u)
{
    expr *child = node->left;
    scalar_mult_expr *snode = (scalar_mult_expr *) node;

    /* call forward for param_source expr tree
     ex: broadcast(param) or promote(const)*/
    snode->param_source->forward(snode->param_source, NULL);

    double a = snode->param_source->value[0];

    /* child's forward pass */
    child->forward(child, u);

    /* local forward pass: multiply each element by scalar a */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = a * child->value[i];
    }
}

static void jacobian_init_impl(expr *node)
{
    expr *x = node->left;

    /* initialize child jacobian */
    jacobian_init(x);

    /* same sparsity as child */
    node->jacobian = new_csr_copy_sparsity(x->jacobian);
}

static void eval_jacobian(expr *node)
{
    expr *child = node->left;
    double a = ((scalar_mult_expr *) node)->param_source->value[0];

    /* evaluate child */
    child->eval_jacobian(child);

    /* scale child's jacobian */
    for (int j = 0; j < child->jacobian->nnz; j++)
    {
        node->jacobian->x[j] = a * child->jacobian->x[j];
    }
}

static void wsum_hess_init_impl(expr *node)
{
    expr *x = node->left;

    /* initialize child's weighted Hessian */
    wsum_hess_init(x);

    /* same sparsity as child */
    node->wsum_hess = new_csr_copy_sparsity(x->wsum_hess);
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    x->eval_wsum_hess(x, w);

    double a = ((scalar_mult_expr *) node)->param_source->value[0];
    for (int j = 0; j < x->wsum_hess->nnz; j++)
    {
        node->wsum_hess->x[j] = a * x->wsum_hess->x[j];
    }
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    scalar_mult_expr *snode = (scalar_mult_expr *) node;
    if (snode->param_source != NULL)
    {
        free_expr(snode->param_source);
        snode->param_source = NULL;
    }
}

expr *new_scalar_mult(expr *param_node, expr *child)
{
    scalar_mult_expr *mult_node =
        (scalar_mult_expr *) SP_CALLOC(1, sizeof(scalar_mult_expr));
    expr *node = &mult_node->base;

    init_expr(node, child->d1, child->d2, child->n_vars, forward, jacobian_init_impl,
              eval_jacobian, is_affine, wsum_hess_init_impl, eval_wsum_hess,
              free_type_data);
    node->left = child;
    expr_retain(child);

    mult_node->param_source = param_node;
    expr_retain(param_node);

    return node;
}
