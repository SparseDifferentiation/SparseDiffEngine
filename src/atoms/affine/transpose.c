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
#include <stdlib.h>
#include <string.h>

// Forward pass for transpose atom
static void forward(expr *node, const double *u)
{
    /* forward pass for child */
    node->left->forward(node->left, u);

    /* local forward pass */
    int d1 = node->d1;
    int d2 = node->d2;
    double *X = node->left->value;
    double *XT = node->value;
    for (int i = 0; i < d1; ++i)
    {
        for (int j = 0; j < d2; ++j)
        {
            XT[j * d1 + i] = X[i * d2 + j];
        }
    }
}

static void jacobian_init_impl(expr *node)
{
    expr *child = node->left;
    jacobian_init(child);

    int n_out = node->size;
    int d1 = node->d1;
    int d2 = node->d2;

    /* The transpose's Jacobian is a row permutation of the child's:
       J_node[r, :] = J_child[k(r), :] where k(r) = (r/d1) + (r%d1)*d2. */
    int *indices = (int *) SP_MALLOC(n_out * sizeof(int));
    for (int r = 0; r < n_out; r++)
    {
        indices[r] = (r / d1) + (r % d1) * d2;
    }

    node->jacobian = child->jacobian->index_alloc(child->jacobian, indices, n_out);

    /* save indices for eval_jacobian */
    node->work->iwork = indices;
}

static void eval_jacobian(expr *node)
{
    expr *child = node->left;
    child->eval_jacobian(child);
    child->jacobian->index_fill_values(child->jacobian, node->work->iwork,
                                       node->size, node->jacobian);
}

static void wsum_hess_init_impl(expr *node)
{
    /* initialize child */
    expr *x = node->left;
    wsum_hess_init(x);

    /* same sparsity pattern as child */
    node->wsum_hess = x->wsum_hess->copy_sparsity(x->wsum_hess);

    /* for computing Kw where K is the commutation matrix */
    node->work->dwork = (double *) SP_MALLOC(node->size * sizeof(double));
}
static void eval_wsum_hess(expr *node, const double *w)
{
    int d2 = node->d2;
    int d1 = node->d1;

    /* evaluate hessian of child at Kw */
    for (int i = 0; i < d2; ++i)
    {
        for (int j = 0; j < d1; ++j)
        {
            node->work->dwork[j * d2 + i] = w[i * d1 + j];
        }
    }

    node->left->eval_wsum_hess(node->left, node->work->dwork);

    /* copy to this node's hessian */
    memcpy(node->wsum_hess->x, node->left->wsum_hess->x,
           node->wsum_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_transpose(expr *child)
{
    expr *node = (expr *) SP_CALLOC(1, sizeof(expr));
    init_expr(node, child->d2, child->d1, child->n_vars, forward, jacobian_init_impl,
              eval_jacobian, is_affine, wsum_hess_init_impl, eval_wsum_hess, NULL);
    node->left = child;
    expr_retain(child);

    return node;
}
