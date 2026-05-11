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
#include "utils/tracked_alloc.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Index/slicing: y = child[indices] where indices is a list of flat positions */

/* Check if indices array contains duplicates using a bitmap.
 * Returns true if duplicates exist, false otherwise. */
static bool check_for_duplicates(const int *indices, int n_idxs, int max_idx)
{
    bool *seen = (bool *) SP_CALLOC(max_idx, sizeof(bool));
    bool has_dup = false;
    for (int i = 0; i < n_idxs && !has_dup; i++)
    {
        if (seen[indices[i]])
        {
            has_dup = true;
        }
        seen[indices[i]] = true;
    }
    free(seen);
    return has_dup;
}

static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    index_expr *idx = (index_expr *) node;

    /* child's forward pass */
    x->forward(x, u);

    /* gather selected elements */
    for (int i = 0; i < idx->n_idxs; i++)
    {
        node->value[i] = x->value[idx->indices[i]];
    }
}

static void jacobian_init_impl(expr *node)
{
    expr *x = node->left;
    index_expr *idx = (index_expr *) node;
    jacobian_init(x);

    /* allocate sparsity pattern for the matrix consisting of rows
       'idx->indices' of the child's Jacobian */
    node->jacobian =
        x->jacobian->index_alloc(x->jacobian, idx->indices, idx->n_idxs);
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    index_expr *idx = (index_expr *) node;
    x->eval_jacobian(x);

    /* copy values of the selected rows into the preallocated output */
    x->jacobian->index_fill_values(x->jacobian, idx->indices, idx->n_idxs,
                                   node->jacobian);
}

static void wsum_hess_init_impl(expr *node)
{
    expr *x = node->left;

    /* initialize child's wsum_hess */
    wsum_hess_init(x);

    /* for setting weight vector to evaluate hessian of child */
    node->work->dwork = (double *) SP_CALLOC(x->size, sizeof(double));

    /* in the implementation of eval_wsum_hess we evaluate the
       child's hessian with a weight vector that has w[i] = 0
       if i is not included in idx->indices. This can lead to
       many numerical zeros in child->wsum_hess that are actually
       structural zeros, but we do not try to exploit that sparsity
       right now. */
    node->wsum_hess = x->wsum_hess->copy_sparsity(x->wsum_hess);
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    index_expr *idx = (index_expr *) node;

    if (idx->has_duplicates)
    {
        /* zero and accumulate for repeated indices */
        memset(node->work->dwork, 0, x->size * sizeof(double));
        for (int i = 0; i < idx->n_idxs; i++)
        {
            node->work->dwork[idx->indices[i]] += w[i];
        }
    }
    else
    {
        /* direct write (no memset needed, no accumulation) */
        for (int i = 0; i < idx->n_idxs; i++)
        {
            node->work->dwork[idx->indices[i]] = w[i];
        }
    }

    /* evalute hessian of child */
    x->eval_wsum_hess(x, node->work->dwork);
    memcpy(node->wsum_hess->x, x->wsum_hess->x,
           node->wsum_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    index_expr *idx = (index_expr *) node;
    if (idx->indices)
    {
        free(idx->indices);
        idx->indices = NULL;
    }
}

expr *new_index(expr *child, int d1, int d2, const int *indices, int n_idxs)
{
    assert(d1 * d2 == n_idxs);
    /* allocate type-specific struct */
    index_expr *idx = (index_expr *) SP_CALLOC(1, sizeof(index_expr));
    expr *node = &idx->base;

    init_expr(node, d1, d2, child->n_vars, forward, jacobian_init_impl,
              eval_jacobian, is_affine, wsum_hess_init_impl, eval_wsum_hess,
              free_type_data);

    node->left = child;
    expr_retain(child);

    /* copy indices */
    idx->indices = (int *) SP_MALLOC(n_idxs * sizeof(int));
    memcpy(idx->indices, indices, n_idxs * sizeof(int));
    idx->n_idxs = n_idxs;

    /* detect duplicates for Hessian optimization */
    idx->has_duplicates = check_for_duplicates(indices, n_idxs, child->size);
    return node;
}
