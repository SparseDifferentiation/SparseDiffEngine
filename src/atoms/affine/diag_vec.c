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
// SPDX-License-Identifier: Apache-2.0

#include "atoms/affine.h"
#include "utils/tracked_alloc.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* diag_vec: converts a vector of size n into an n×n diagonal matrix.
 * In Fortran (column-major) order, element i of the input maps to
 * position i*(n+1) in the flattened output (the diagonal positions). */

static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    int n = x->size;

    /* child's forward pass */
    x->forward(x, u);

    /* zero-initialize output, TODO: do we need to do this? */
    memset(node->value, 0, node->size * sizeof(double));

    /* place input elements on the diagonal */
    for (int i = 0; i < n; i++)
    {
        node->value[i * (n + 1)] = x->value[i];
    }
}

static void jacobian_init_impl(expr *node)
{
    expr *x = node->left;
    jacobian_init(x);

    /* output type matches child's; rows i*(n+1) hold child row i, others zero. */
    node->jacobian = x->jacobian->diag_vec_alloc(x->jacobian);
}

static void eval_jacobian(expr *node)
{
    node->left->eval_jacobian(node->left);

    /* fill the diagonal rows of the preallocated output. */
    node->left->jacobian->diag_vec_fill_values(node->left->jacobian, node->jacobian);
}

static void wsum_hess_init_impl(expr *node)
{
    expr *x = node->left;

    /* initialize child's wsum_hess */
    wsum_hess_init(x);

    /* workspace for extracting diagonal weights */
    node->work->dwork = (double *) SP_CALLOC(x->size, sizeof(double));

    /* Copy child's Hessian structure (diag_vec is linear, so its own Hessian is
     * zero) */
    node->wsum_hess = x->wsum_hess->copy_sparsity(x->wsum_hess);
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    int n = x->size;

    /* Extract weights from diagonal positions of w (which has n^2 elements) */
    for (int i = 0; i < n; i++)
    {
        node->work->dwork[i] = w[i * (n + 1)];
    }

    /* Evaluate child's Hessian with extracted weights */
    x->eval_wsum_hess(x, node->work->dwork);
    memcpy(node->wsum_hess->x, x->wsum_hess->x,
           node->wsum_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_diag_vec(expr *child)
{
    /* child must be a vector: either column (n, 1) or row (1, n) */
    assert(child->d1 == 1 || child->d2 == 1);

    /* n is the number of elements (works for both row and column vectors) */
    int n = child->size;
    expr *node = (expr *) SP_CALLOC(1, sizeof(expr));
    init_expr(node, n, n, child->n_vars, forward, jacobian_init_impl, eval_jacobian,
              is_affine, wsum_hess_init_impl, eval_wsum_hess, NULL);
    node->left = child;
    expr_retain(child);

    return node;
}
