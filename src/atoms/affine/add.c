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
#include "utils/CSR_sum.h"
#include "utils/tracked_alloc.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static void forward(expr *node, const double *u)
{
    /* children's forward passes */
    node->left->forward(node->left, u);
    node->right->forward(node->right, u);

    /* add left and right values */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = node->left->value[i] + node->right->value[i];
    }
}

static void jacobian_init_impl(expr *node)
{
    /* initialize children's jacobians */
    jacobian_init(node->left);
    jacobian_init(node->right);

    CSR_Matrix *Jl = node->left->jacobian->to_csr(node->left->jacobian);
    CSR_Matrix *Jr = node->right->jacobian->to_csr(node->right->jacobian);

    /* we never have to store more than the sum of children's nnz */
    int nnz_max = Jl->nnz + Jr->nnz;
    CSR_Matrix *jac = new_csr_matrix(node->size, node->n_vars, nnz_max);

    /* fill sparsity pattern  */
    sum_csr_alloc(Jl, Jr, jac);
    node->jacobian = new_sparse_matrix(jac);
}

static void eval_jacobian(expr *node)
{
    /* evaluate children's jacobians */
    node->left->eval_jacobian(node->left);
    node->right->eval_jacobian(node->right);

    /* sum children's jacobians */
    sum_csr_fill_values(node->left->jacobian->to_csr(node->left->jacobian), node->right->jacobian->to_csr(node->right->jacobian),
                        node->jacobian->to_csr(node->jacobian));
}

static void wsum_hess_init_impl(expr *node)
{
    /* initialize children's wsum_hess */
    wsum_hess_init(node->left);
    wsum_hess_init(node->right);

    CSR_Matrix *Hl = node->left->wsum_hess->to_csr(node->left->wsum_hess);
    CSR_Matrix *Hr = node->right->wsum_hess->to_csr(node->right->wsum_hess);

    /* we never have to store more than the sum of children's nnz */
    int nnz_max = Hl->nnz + Hr->nnz;
    CSR_Matrix *hess = new_csr_matrix(node->n_vars, node->n_vars, nnz_max);

    /* fill sparsity pattern of hessian */
    sum_csr_alloc(Hl, Hr, hess);
    node->wsum_hess = new_sparse_matrix(hess);
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* evaluate children's wsum_hess */
    node->left->eval_wsum_hess(node->left, w);
    node->right->eval_wsum_hess(node->right, w);

    /* sum children's wsum_hess */
    sum_csr_fill_values(node->left->wsum_hess->to_csr(node->left->wsum_hess),
                        node->right->wsum_hess->to_csr(node->right->wsum_hess),
                        node->wsum_hess->to_csr(node->wsum_hess));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left) && node->right->is_affine(node->right);
}

expr *new_add(expr *left, expr *right)
{
    assert(left->d1 == right->d1 && left->d2 == right->d2);
    expr *node = (expr *) SP_CALLOC(1, sizeof(expr));
    init_expr(node, left->d1, left->d2, left->n_vars, forward, jacobian_init_impl,
              eval_jacobian, is_affine, wsum_hess_init_impl, eval_wsum_hess, NULL);
    node->left = left;
    node->right = right;
    expr_retain(left);
    expr_retain(right);

    return node;
}
