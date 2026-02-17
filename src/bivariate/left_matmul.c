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
#include "bivariate.h"
#include "subexpr.h"
#include "utils/Timer.h"
#include "utils/linalg_sparse_matmuls.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* This file implements the atom 'left_matmul' corresponding to the operation y =
   A @ f(x), where A is a given matrix (from a parameter node) and f(x) is an
   arbitrary expression. Here, f(x) can be a vector-valued expression and a
   matrix-valued expression. The dimensions are A - m x n, f(x) - n x p, y - m x p.
   Note that here A does not have global column indices but it is a local matrix.
   This is an important distinction compared to linear_op_expr.

   * To compute the forward pass: vec(y) = A_kron @ vec(f(x)),
     where A_kron = I_p kron A is a Kronecker product of size (m*p) x (n*p),
     or more specificely, a block-diagonal matrix with p blocks of A along the
     diagonal. In the refactored implementation we don't form A_kron explicitly,
     only conceptually. This led to a 100x speedup in the initialization of the
     Jacobian sparsity pattern.

   * To compute the Jacobian: J_y = A_kron @ J_f(x), where J_f(x) is the
     Jacobian of f(x) of size (n*p) x n_vars.

    * To compute the contribution to the Lagrange Hessian: we form
    w = A_kron^T @ lambda and then evaluate the hessian of f(x).

    Working in terms of A_kron unifies the implementation of f(x) being
    vector-valued or matrix-valued.
*/

#include "utils/utils.h"
#include <assert.h>
#include <string.h>

/* Refresh A and AT values from param_source.
   A is the small m x n matrix (NOT block-diagonal).
   No-op when param_source is NULL (fixed constant — values already in A). */
static void refresh_param_values(left_matmul_expr *lin_node)
{
    parameter_expr *param = (parameter_expr *) lin_node->param_source;

    if (!param || param->has_been_refreshed) return;
    param->has_been_refreshed = true;

    assert(param->param_id != PARAM_FIXED);

    /* update values of A */
    memcpy(lin_node->A->x, lin_node->param_source->value,
           lin_node->A->nnz * sizeof(double));

    /* update values of AT */
    AT_fill_values(lin_node->A, lin_node->AT, lin_node->AT_iwork);
}

static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    left_matmul_expr *lin_node = (left_matmul_expr *) node;

    /* possibly refresh A and AT */
    if (lin_node->refresh_param_values) lin_node->refresh_param_values(lin_node);

    /* child's forward pass */
    node->left->forward(node->left, u);

    /* y = A_kron @ vec(f(x)) */
    block_left_multiply_vec(lin_node->A, x->value, node->value, lin_node->n_blocks);
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    left_matmul_expr *lin_node = (left_matmul_expr *) node;
    free_csr_matrix(lin_node->A);
    free_csr_matrix(lin_node->AT);
    free_csc_matrix(lin_node->Jchild_CSC);
    free_csc_matrix(lin_node->J_CSC);
    free(lin_node->csc_to_csr_workspace);
    free(lin_node->AT_iwork);
    free_expr(lin_node->param_source);
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    left_matmul_expr *lin_node = (left_matmul_expr *) node;

    /* initialize child's jacobian and precompute sparsity of its CSC */
    x->jacobian_init(x);
    lin_node->Jchild_CSC = csr_to_csc_fill_sparsity(x->jacobian, node->iwork);

    /* precompute sparsity of this node's jacobian in CSC and CSR */
    lin_node->J_CSC = block_left_multiply_fill_sparsity(
        lin_node->A, lin_node->Jchild_CSC, lin_node->n_blocks);
    node->jacobian =
        csc_to_csr_fill_sparsity(lin_node->J_CSC, lin_node->csc_to_csr_workspace);
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    left_matmul_expr *lnode = (left_matmul_expr *) node;

    CSC_Matrix *Jchild_CSC = lnode->Jchild_CSC;
    CSC_Matrix *J_CSC = lnode->J_CSC;

    /* evaluate child's jacobian and convert to CSC */
    x->eval_jacobian(x);
    csr_to_csc_fill_values(x->jacobian, Jchild_CSC, node->iwork);

    /* compute this node's jacobian: */
    block_left_multiply_fill_values(lnode->A, Jchild_CSC, J_CSC);
    csc_to_csr_fill_values(J_CSC, node->jacobian, lnode->csc_to_csr_workspace);
}

static void wsum_hess_init(expr *node)
{
    /* initialize child's hessian */
    expr *x = node->left;
    x->wsum_hess_init(x);

    /* allocate this node's hessian with the same sparsity as child's */
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, x->wsum_hess->nnz);
    memcpy(node->wsum_hess->p, x->wsum_hess->p, (node->n_vars + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, x->wsum_hess->i, x->wsum_hess->nnz * sizeof(int));

    /* work for computing A^T w*/
    int n_blocks = ((left_matmul_expr *) node)->n_blocks;
    int dim = ((left_matmul_expr *) node)->A->n * n_blocks;
    node->dwork = (double *) malloc(dim * sizeof(double));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* compute A^T w */
    CSR_Matrix *AT = ((left_matmul_expr *) node)->AT;
    int n_blocks = ((left_matmul_expr *) node)->n_blocks;
    block_left_multiply_vec(AT, w, node->dwork, n_blocks);

    node->left->eval_wsum_hess(node->left, node->dwork);
    memcpy(node->wsum_hess->x, node->left->wsum_hess->x,
           node->wsum_hess->nnz * sizeof(double));
}

expr *new_left_matmul(expr *param_node, expr *child, const CSR_Matrix *A)
{
    /* Dimension logic: handle numpy broadcasting (1, n) as (n, )/
       We expect u->d1 == A->n. However, numpy's broadcasting rules allow users
       to do A @ u where u is (n, ) which in C is actually (1, n). In that case
       the result of A @ u is (m, ), which is (1, m) according to broadcasting
       rules. We therefore check if this is the case. */
    int d1, d2, n_blocks;
    if (child->d1 == A->n)
    {
        d1 = A->m;
        d2 = child->d2;
        n_blocks = child->d2;
    }
    else if (child->d2 == A->n && child->d1 == 1)
    {
        d1 = 1;
        d2 = A->m;
        n_blocks = 1;
    }
    else
    {
        fprintf(stderr, "Error in new_left_matmul: dimension mismatch\n");
        exit(1);
    }

    /* Allocate the type-specific struct */
    left_matmul_expr *lin_node =
        (left_matmul_expr *) calloc(1, sizeof(left_matmul_expr));
    expr *node = &lin_node->base;
    init_expr(node, d1, d2, child->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, free_type_data);
    node->left = child;
    expr_retain(child);

    /* Store small A (NOT block-diagonal) — block functions handle the rest
       Allocate workspace. iwork is used for converting J_child csr to csc
       (requring size node->n_vars). csc_to_csr_workspace is used for
       converting J_CSC to CSR (requring node->size) */
    node->iwork = (int *) malloc(node->n_vars * sizeof(int));
    lin_node->AT_iwork = (int *) malloc(A->n * sizeof(int));
    lin_node->csc_to_csr_workspace = (int *) malloc(node->size * sizeof(int));
    lin_node->n_blocks = n_blocks;
    lin_node->A = new_csr(A);
    lin_node->AT = transpose(lin_node->A, lin_node->AT_iwork);
    lin_node->param_source = param_node;
    lin_node->refresh_param_values = refresh_param_values;

    if (param_node) expr_retain(param_node);

    return node;
}
