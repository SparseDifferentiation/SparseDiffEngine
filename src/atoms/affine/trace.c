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
#include "sparsediff_linalg/tracked_alloc.h"
#include "subexpr.h"
#include <assert.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    expr *x = node->left;

    /* child's forward pass */
    x->forward(x, u);

    /* local forward pass */
    double sum = 0.0;
    int row_spacing = x->d1 + 1;
    for (int idx = 0; idx < x->size; idx += row_spacing)
    {
        sum += x->value[idx];
    }

    node->value[0] = sum;
}

static void jacobian_init_impl(expr *node)
{
    trace_expr *tnode = (trace_expr *) node;
    expr *x = node->left;
    assert(x->d1 == x->d2);

    /* initialize child's jacobian */
    jacobian_init(x);

    /* J = sum of the rows of J_X at X's diagonal (column-major index i*(d1+1)):
       gather those d1 rows, then reduce them into one. */
    int d1 = x->d1;
    int *map = (int *) sp_malloc(d1 * sizeof(int));
    for (int i = 0; i < d1; i++)
    {
        map[i] = i * (d1 + 1);
    }
    tnode->diag_rows = x->jacobian->row_gather_alloc(x->jacobian, map, d1);
    memset(map, 0, d1 * sizeof(int));
    node->jacobian = tnode->diag_rows->row_reduce_alloc(tnode->diag_rows, map, 1);
    sp_free(map);
}

static void eval_jacobian_impl(expr *node)
{
    trace_expr *tnode = (trace_expr *) node;
    expr *x = node->left;

    eval_jacobian(x);
    x->jacobian->row_gather_fill_values(x->jacobian, tnode->diag_rows);
    tnode->diag_rows->row_reduce_fill_values(tnode->diag_rows, node->jacobian);
}

/* Placeholders for Hessian-related functions */
static void wsum_hess_init_impl(expr *node)
{
    expr *x = node->left;

    /* initialize child's hessian */
    wsum_hess_init(x);

    node->work->dwork = (double *) sp_calloc(x->size, sizeof(double));

    /* We copy over the sparsity pattern from the child. This also includes the
       contribution to wsum_hess of entries of the child that will always have
       zero weight in eval_wsum_hess. We do this for simplicity. But the Hessian
       can for sure be made more sophisticated. */
    node->wsum_hess = copy_sparsity_matrices_alloc(x->wsum_hess);
}

static void eval_wsum_hess_impl(expr *node, const double *w)
{
    expr *x = node->left;

    int row_spacing = x->d1 + 1;
    for (int i = 0; i < x->size; i += row_spacing)
    {
        node->work->dwork[i] = w[0];
    }

    eval_wsum_hess(x, node->work->dwork);

    memcpy(node->wsum_hess->x, x->wsum_hess->x,
           node->wsum_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    if (node)
    {
        trace_expr *tnode = (trace_expr *) node;
        free_matrix(tnode->diag_rows);
    }
}

expr *new_trace(expr *child)
{
    trace_expr *tnode = (trace_expr *) sp_calloc(1, sizeof(trace_expr));
    expr *node = &tnode->base;
    init_expr(node, 1, 1, child->n_vars, forward, jacobian_init_impl,
              eval_jacobian_impl, is_affine, wsum_hess_init_impl,
              eval_wsum_hess_impl, free_type_data);
    node->left = child;
    expr_retain(child);

    return node;
}
