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
#include "sparsediff_linalg/matrix.h"
#include "sparsediff_linalg/matrix_sum.h"
#include "sparsediff_linalg/matrix_vstack.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include "subexpr.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    hstack_expr *hnode = (hstack_expr *) node;

    /* children's forward passes */
    for (int i = 0; i < hnode->n_args; i++)
    {
        hnode->args[i]->forward(hnode->args[i], u);
    }

    /* concatenate values horizontally */
    int offset = 0;
    for (int i = 0; i < hnode->n_args; i++)
    {
        expr *child = hnode->args[i];
        memcpy(node->value + offset, child->value, child->size * sizeof(double));
        offset += child->size;
    }
}

static void jacobian_init_impl(expr *node)
{
    hstack_expr *hnode = (hstack_expr *) node;

    /* initialize children's jacobians; this node's is their vertical stack */
    hnode->child_jac = (matrix **) sp_malloc(hnode->n_args * sizeof(matrix *));
    for (int i = 0; i < hnode->n_args; i++)
    {
        assert(hnode->args[i] != NULL);
        jacobian_init(hnode->args[i]);
        hnode->child_jac[i] = hnode->args[i]->jacobian;
    }
    node->jacobian = vstack_matrices_alloc(hnode->child_jac, hnode->n_args);
}

static void eval_jacobian_impl(expr *node)
{
    hstack_expr *hnode = (hstack_expr *) node;
    for (int i = 0; i < hnode->n_args; i++)
    {
        eval_jacobian(hnode->args[i]);
    }

    /* copies only the children whose jacobian values changed */
    vstack_matrices_fill_values(hnode->child_jac, hnode->n_args, node->jacobian);
}

static void wsum_hess_init_impl(expr *node)
{
    /* initialize children's hessians; this node's is their sum */
    hstack_expr *hnode = (hstack_expr *) node;
    hnode->child_hess = (matrix **) sp_malloc(hnode->n_args * sizeof(matrix *));
    for (int i = 0; i < hnode->n_args; i++)
    {
        wsum_hess_init(hnode->args[i]);
        hnode->child_hess[i] = hnode->args[i]->wsum_hess;
    }
    node->wsum_hess = sum_n_matrices_alloc(hnode->child_hess, hnode->n_args);
}

static void wsum_hess_eval(expr *node, const double *w)
{
    hstack_expr *hnode = (hstack_expr *) node;
    int row_offset = 0;

    /* add each child's hessian right after evaluating it: args may share a
       subexpression, whose later evaluation would overwrite earlier values */
    sum_n_matrices_fill_zero(node->wsum_hess);
    for (int i = 0; i < hnode->n_args; i++)
    {
        expr *child = hnode->args[i];
        eval_wsum_hess(child, w + row_offset);
        sum_n_matrices_fill_add(child->wsum_hess, i, node->wsum_hess);
        row_offset += child->size;
    }
}

static bool is_affine(const expr *node)
{
    const hstack_expr *hnode = (const hstack_expr *) node;

    for (int i = 0; i < hnode->n_args; i++)
    {
        if (!hnode->args[i]->is_affine(hnode->args[i]))
        {
            return false;
        }
    }
    return true;
}

/* Children live in args[], not left/right, so the parameter-refresh walk
   needs this hook to reach them. */
static bool set_needs_refresh_children(expr *node)
{
    hstack_expr *hnode = (hstack_expr *) node;
    bool child_has_params = false;
    for (int i = 0; i < hnode->n_args; i++)
    {
        child_has_params |= expr_set_needs_refresh(hnode->args[i]);
    }
    return child_has_params;
}

static void free_type_data(expr *node)
{
    hstack_expr *hnode = (hstack_expr *) node;
    for (int i = 0; i < hnode->n_args; i++)
    {
        free_expr(hnode->args[i]);
        hnode->args[i] = NULL;
    }

    sp_free(hnode->child_hess);
    hnode->child_hess = NULL;
    sp_free(hnode->child_jac);
    hnode->child_jac = NULL;
    sp_free(hnode->args);
    hnode->args = NULL;
}

expr *new_hstack(expr **args, int n_args, int n_vars)
{
    /* compute second dimension */
    int d2 = 0;
    for (int i = 0; i < n_args; i++)
    {
        assert(args[i]->d1 == args[0]->d1); /* all args must have same d1 */
        d2 += args[i]->d2;
    }

    /* Allocate the type-specific struct */
    hstack_expr *hnode = (hstack_expr *) sp_calloc(1, sizeof(hstack_expr));
    expr *node = &hnode->base;
    init_expr(node, args[0]->d1, d2, n_vars, forward, jacobian_init_impl,
              eval_jacobian_impl, is_affine, wsum_hess_init_impl, wsum_hess_eval,
              free_type_data);

    node->set_needs_refresh_children = set_needs_refresh_children;

    /* Set type-specific fields (deep copy args array) */
    hnode->args = (expr **) sp_calloc(n_args, sizeof(expr *));
    hnode->n_args = n_args;
    for (int i = 0; i < n_args; i++)
    {
        hnode->args[i] = args[i];
        expr_retain(args[i]);
    }

    return node;
}
