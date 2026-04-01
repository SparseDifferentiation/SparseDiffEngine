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
#include <assert.h>
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
    CSR_Matrix *Jc = child->jacobian;
    node->jacobian = new_csr_matrix(node->size, node->n_vars, Jc->nnz, &node->bytes);

    /* fill sparsity */
    CSR_Matrix *J = node->jacobian;
    int d1 = node->d1;
    int d2 = node->d2;
    int nnz = 0;
    J->p[0] = 0;

    /* 'k' is the old row that gets swapped to 'row'*/
    int k, len;
    for (int row = 0; row < J->m; ++row)
    {
        k = (row / d1) + (row % d1) * d2;
        len = Jc->p[k + 1] - Jc->p[k];
        memcpy(J->i + nnz, Jc->i + Jc->p[k], len * sizeof(int));
        nnz += len;
        J->p[row + 1] = nnz;
    }
}

static void eval_jacobian(expr *node)
{
    expr *child = node->left;
    child->eval_jacobian(child);
    CSR_Matrix *Jc = child->jacobian;
    CSR_Matrix *J = node->jacobian;

    int d1 = node->d1;
    int d2 = node->d2;
    int nnz = 0;
    for (int row = 0; row < J->m; ++row)
    {
        int k = (row / d1) + (row % d1) * d2;
        int len = Jc->p[k + 1] - Jc->p[k];
        memcpy(J->x + nnz, Jc->x + Jc->p[k], len * sizeof(double));
        nnz += len;
    }
}

static void wsum_hess_init_impl(expr *node)
{
    /* initialize child */
    expr *x = node->left;
    wsum_hess_init(x);

    /* same sparsity pattern as child */
    node->wsum_hess = new_csr_copy_sparsity(x->wsum_hess, &node->bytes);

    /* for computing Kw where K is the commutation matrix */
    node->work->dwork = (double *) malloc(node->size * sizeof(double));
    node->bytes += node->size * sizeof(double);
}
static void eval_wsum_hess(expr *node, const double *w)
{
    int d2 = node->d2;
    int d1 = node->d1;
    // TODO: meaybe more efficient to do this with memcpy first

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
    expr *node = (expr *) calloc(1, sizeof(expr));
    init_expr(node, child->d2, child->d1, child->n_vars, forward, jacobian_init_impl,
              eval_jacobian, is_affine, wsum_hess_init_impl, eval_wsum_hess, NULL);
    node->left = child;
    expr_retain(child);

    return node;
}
