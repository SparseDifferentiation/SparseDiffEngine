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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Constant vector elementwise multiplication: y = a \circ child */

static void forward(expr *node, const double *u)
{
    expr *child = node->left;
    const double *a = ((const_vector_mult_expr *) node)->a;

    /* child's forward pass */
    child->forward(child, u);

    /* local forward pass: elementwise multiply by a */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = a[i] * child->value[i];
    }
}

static void jacobian_init_impl(expr *node)
{
    expr *x = node->left;

    /* initialize child jacobian */
    jacobian_init(x);

    /* same sparsity as child */
    node->jacobian = new_csr_copy_sparsity(x->jacobian, &node->memory_bytes);
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    const double *a = ((const_vector_mult_expr *) node)->a;

    /* evaluate x */
    x->eval_jacobian(x);

    /* row-wise scale child's jacobian */
    for (int i = 0; i < node->size; i++)
    {
        for (int j = x->jacobian->p[i]; j < x->jacobian->p[i + 1]; j++)
        {
            node->jacobian->x[j] = a[i] * x->jacobian->x[j];
        }
    }
}

static void wsum_hess_init_impl(expr *node)
{
    expr *x = node->left;

    /* initialize child's weighted Hessian */
    wsum_hess_init(x);

    /* same sparsity as child */
    node->wsum_hess = new_csr_copy_sparsity(x->wsum_hess, &node->memory_bytes);

    node->work->dwork = (double *) malloc(node->size * sizeof(double));
    node->memory_bytes += node->size * sizeof(double);
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    const double *a = ((const_vector_mult_expr *) node)->a;

    /* scale weights w by a */
    for (int i = 0; i < node->size; i++)
    {
        node->work->dwork[i] = a[i] * w[i];
    }

    x->eval_wsum_hess(x, node->work->dwork);

    /* copy values from child to this node */
    memcpy(node->wsum_hess->x, x->wsum_hess->x, x->wsum_hess->nnz * sizeof(double));
}

static void free_type_data(expr *node)
{
    const_vector_mult_expr *vnode = (const_vector_mult_expr *) node;
    free(vnode->a);
}

static bool is_affine(const expr *node)
{
    /* Affine iff the child is affine */
    return node->left->is_affine(node->left);
}

expr *new_const_vector_mult(const double *a, expr *child)
{
    const_vector_mult_expr *vnode =
        (const_vector_mult_expr *) calloc(1, sizeof(const_vector_mult_expr));
    expr *node = &vnode->base;

    init_expr(node, child->d1, child->d2, child->n_vars, forward, jacobian_init_impl,
              eval_jacobian, is_affine, wsum_hess_init_impl, eval_wsum_hess,
              free_type_data);
    node->left = child;
    expr_retain(child);

    /* copy a vector */
    vnode->a = (double *) malloc(child->size * sizeof(double));
    node->memory_bytes += child->size * sizeof(double);
    memcpy(vnode->a, a, child->size * sizeof(double));

    return node;
}
