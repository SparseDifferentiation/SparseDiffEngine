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
#include "utils/CSR_sum.h"
#include "utils/int_double_pair.h"
#include "utils/mini_numpy.h"
#include "utils/tracked_alloc.h"
#include "utils/utils.h"
#include <assert.h>
#include <stdlib.h>
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
    int axis = snode->axis;

    /* initialize child's jacobian */
    jacobian_init(x);

    /* we never have to store more than the child's nnz */
    node->jacobian = new_csr_matrix(node->size, node->n_vars, x->jacobian->nnz);
    node->work->iwork =
        SP_MALLOC(MAX(node->jacobian->n, x->jacobian->nnz) * sizeof(int));
    snode->idx_map = SP_MALLOC(x->jacobian->nnz * sizeof(int));

    /* the idx_map array maps each nonzero entry j in x->jacobian
       to the corresponding index in the output row matrix C. Specifically, for
       each nonzero entry j in A, idx_map[j] gives the position in C->x where
       the value from x->jacobian->x[j] should be accumulated. */

    if (axis == -1)
    {
        sum_all_rows_csr_alloc(x->jacobian, node->jacobian, node->work->iwork,
                               snode->idx_map);
    }
    else if (axis == 0)
    {
        sum_block_of_rows_csr_alloc(x->jacobian, node->jacobian, x->d1,
                                    node->work->iwork, snode->idx_map);
    }
    else if (axis == 1)
    {
        sum_evenly_spaced_rows_csr_alloc(x->jacobian, node->jacobian, node->size,
                                         node->work->iwork, snode->idx_map);
    }
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;

    /* evaluate child's jacobian */
    x->eval_jacobian(x);

    /* we have precomputed an idx map between the nonzeros of the child's jacobian
       and this node's jacobian, so we just accumulate accordingly */
    memset(node->jacobian->x, 0, node->jacobian->nnz * sizeof(double));
    accumulator(x->jacobian, ((sum_expr *) node)->idx_map, node->jacobian->x);
}

static void wsum_hess_init_impl(expr *node)
{
    expr *x = node->left;
    /* initialize child's wsum_hess */
    wsum_hess_init(x);

    /* we never have to store more than the child's nnz */
    node->wsum_hess = new_csr_copy_sparsity(x->wsum_hess);
    node->work->dwork = SP_MALLOC(x->size * sizeof(double));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    sum_expr *snode = (sum_expr *) node;
    int axis = snode->axis;

    if (axis == -1)
    {
        scaled_ones(node->work->dwork, x->size, *w);
    }
    else if (axis == 0)
    {
        repeat(node->work->dwork, w, x->d2, x->d1);
    }
    else if (axis == 1)
    {
        tile_double(node->work->dwork, w, x->d1, x->d2);
    }

    x->eval_wsum_hess(x, node->work->dwork);

    /* copy values */
    memcpy(node->wsum_hess->x, x->wsum_hess->x, x->wsum_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    sum_expr *snode = (sum_expr *) node;
    free(snode->idx_map);
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
    sum_expr *snode = (sum_expr *) SP_CALLOC(1, sizeof(sum_expr));
    expr *node = &snode->base;

    /* to be consistent with CVXPY and NumPy we treat the result from
       sum with an axis argument as a row vector */
    init_expr(node, 1, d2, child->n_vars, forward, jacobian_init_impl, eval_jacobian,
              is_affine, wsum_hess_init_impl, eval_wsum_hess, free_type_data);
    node->left = child;
    expr_retain(child);

    /* Set type-specific fields */
    snode->axis = axis;

    return node;
}
