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
#include "subexpr.h"
#include "utils/mini_numpy.h"
#include "utils/tracked_alloc.h"
#include <assert.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    int i, j, end;
    double sum;
    expr *x = node->left;
    sum_expr *snode = (sum_expr *) node;
    int axis = snode->axis;

    /* child's forward pass */
    x->forward(x, u);

    if (axis == -1)
    {
        sum = 0.0;
        end = x->d1 * x->d2;

        /* sum all elements */
        for (i = 0; i < end; i++)
        {
            sum += x->value[i];
        }
        node->value[0] = sum;
    }
    else if (axis == 0)
    {
        /* sum rows together */
        for (j = 0; j < x->d2; j++)
        {
            sum = 0.0;
            end = (j + 1) * x->d1;
            for (i = j * x->d1; i < end; i++)
            {
                sum += x->value[i];
            }
            node->value[j] = sum;
        }
    }
    else if (axis == 1)
    {
        memset(node->value, 0, node->size * sizeof(double));

        /* sum columns together */
        for (j = 0; j < x->d2; j++)
        {
            int offset = j * x->d1;
            for (i = 0; i < x->d1; i++)
            {
                node->value[i] += x->value[offset + i];
            }
        }
    }
}

static void jacobian_init_impl(expr *node)
{
    expr *x = node->left;
    sum_expr *snode = (sum_expr *) node;
    jacobian_init(x);
    assert(x->jacobian->m == x->size);

    /* Child rows are column-major: row i = r + c * x->d1. Row i of the child
       Jacobian is summed into output row 0 (axis -1), c (axis 0) or r (axis 1).
       The reduction map is bound to node->jacobian, so group is not kept. */
    int d1 = x->d1;
    int m_out;
    int *group;
    if (snode->axis == -1)
    {
        m_out = 1;
        group = (int *) sp_calloc(x->size, sizeof(int));
    }
    else if (snode->axis == 0)
    {
        m_out = x->d2;
        group = (int *) sp_malloc(x->size * sizeof(int));
        for (int i = 0; i < x->size; i++)
        {
            group[i] = i / d1;
        }
    }
    else
    {
        m_out = d1;
        group = (int *) sp_malloc(x->size * sizeof(int));
        for (int i = 0; i < x->size; i++)
        {
            group[i] = i % d1;
        }
    }
    node->jacobian = x->jacobian->row_reduce_alloc(x->jacobian, group, m_out);
    sp_free(group);
}

static void eval_jacobian_impl(expr *node)
{
    expr *child = node->left;
    eval_jacobian(child);
    child->jacobian->row_reduce_fill_values(child->jacobian, node->jacobian);
}

static void wsum_hess_init_impl(expr *node)
{
    expr *child = node->left;
    /* initialize child's wsum_hess */
    wsum_hess_init(child);

    /* we never have to store more than the child's nnz */
    node->wsum_hess = child->wsum_hess->copy_sparsity(child->wsum_hess);
    node->work->dwork = sp_malloc(child->size * sizeof(double));
}

static void eval_wsum_hess_impl(expr *node, const double *w)
{
    expr *child = node->left;
    sum_expr *snode = (sum_expr *) node;
    int axis = snode->axis;

    if (axis == -1)
    {
        scaled_ones(node->work->dwork, child->size, *w);
    }
    else if (axis == 0)
    {
        repeat(node->work->dwork, w, child->d2, child->d1);
    }
    else if (axis == 1)
    {
        tile_double(node->work->dwork, w, child->d1, child->d2);
    }

    eval_wsum_hess(child, node->work->dwork);

    memcpy(node->wsum_hess->x, child->wsum_hess->x,
           node->wsum_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_sum(expr *child, int axis)
{
    int d2 = 0;

    switch (axis)
    {
        case -1:
            /* no axis specified */
            d2 = 1;
            break;
        case 0:
            /* sum rows together */
            d2 = child->d2;
            break;
        case 1:
            /* sum columns together */
            d2 = child->d1;
            break;
    }

    /* Allocate the type-specific struct */
    sum_expr *snode = (sum_expr *) sp_calloc(1, sizeof(sum_expr));
    expr *node = &snode->base;

    /* to be consistent with CVXPY and NumPy we treat the result from
       sum with an axis argument as a row vector */
    init_expr(node, 1, d2, child->n_vars, forward, jacobian_init_impl,
              eval_jacobian_impl, is_affine, wsum_hess_init_impl,
              eval_wsum_hess_impl, NULL);
    node->left = child;
    expr_retain(child);

    /* Set type-specific fields */
    snode->axis = axis;

    return node;
}
