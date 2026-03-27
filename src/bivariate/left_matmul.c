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
#include "utils/matrix.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* This file implement the atom 'left_matmul' corresponding to the operation y =
   A @ f(x), where A is a given matrix and f(x) is an arbitrary expression.
   Here, f(x) can be a vector-valued expression and a matrix-valued
   expression. The dimensions are A - m x n, f(x) - n x p, y - m x p.
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

static void refresh_param_values(left_matmul_expr *lnode)
{
    if (lnode->param_source == NULL)
    {
        return;
    }
    parameter_expr *param = (parameter_expr *) lnode->param_source;
    if (param->has_been_refreshed)
    {
        return;
    }
    param->has_been_refreshed = true;
    lnode->refresh_param_values(lnode);
}

static void forward(expr *node, const double *u)
{
    left_matmul_expr *lnode = (left_matmul_expr *) node;
    refresh_param_values(lnode);

    expr *x = node->left;

    /* child's forward pass */
    node->left->forward(node->left, u);

    /* y = A_kron @ vec(f(x)) */
    Matrix *A = lnode->A;
    int n_blocks = lnode->n_blocks;
    A->block_left_mult_vec(A, x->value, node->value, n_blocks);
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    left_matmul_expr *lnode = (left_matmul_expr *) node;
    free_matrix(lnode->A);
    free_matrix(lnode->AT);
    free_csc_matrix(lnode->Jchild_CSC);
    free_csc_matrix(lnode->J_CSC);
    free(lnode->csc_to_csr_work);
    if (lnode->param_source != NULL)
    {
        free_expr(lnode->param_source);
    }
    lnode->A = NULL;
    lnode->AT = NULL;
    lnode->Jchild_CSC = NULL;
    lnode->J_CSC = NULL;
    lnode->csc_to_csr_work = NULL;
    lnode->param_source = NULL;
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    left_matmul_expr *lnode = (left_matmul_expr *) node;

    /* initialize child's jacobian and precompute sparsity of its CSC */
    x->jacobian_init(x);
    lnode->Jchild_CSC = csr_to_csc_fill_sparsity(x->jacobian, node->work->iwork);

    /* precompute sparsity of this node's jacobian in CSC and CSR */
    lnode->J_CSC = lnode->A->block_left_mult_sparsity(lnode->A, lnode->Jchild_CSC,
                                                      lnode->n_blocks);
    node->jacobian = csc_to_csr_fill_sparsity(lnode->J_CSC, lnode->csc_to_csr_work);
}

static void eval_jacobian(expr *node)
{
    left_matmul_expr *lnode = (left_matmul_expr *) node;
    expr *x = node->left;

    CSC_Matrix *Jchild_CSC = lnode->Jchild_CSC;
    CSC_Matrix *J_CSC = lnode->J_CSC;

    /* evaluate child's jacobian and convert to CSC */
    x->eval_jacobian(x);
    csr_to_csc_fill_values(x->jacobian, Jchild_CSC, node->work->iwork);

    /* compute this node's jacobian: */
    lnode->A->block_left_mult_values(lnode->A, Jchild_CSC, J_CSC);
    csc_to_csr_fill_values(J_CSC, node->jacobian, lnode->csc_to_csr_work);
}

static void wsum_hess_init(expr *node)
{
    /* initialize child's hessian */
    expr *x = node->left;
    x->wsum_hess_init(x);

    /* allocate this node's hessian with the same sparsity as child's */
    node->wsum_hess = new_csr_copy_sparsity(x->wsum_hess);

    /* work for computing A^T w*/
    int n_blocks = ((left_matmul_expr *) node)->n_blocks;
    int dim = ((left_matmul_expr *) node)->AT->m * n_blocks;
    node->work->dwork = (double *) malloc(dim * sizeof(double));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    left_matmul_expr *lnode = (left_matmul_expr *) node;

    /* compute A^T w*/
    Matrix *AT = lnode->AT;
    int n_blocks = lnode->n_blocks;
    AT->block_left_mult_vec(AT, w, node->work->dwork, n_blocks);

    node->left->eval_wsum_hess(node->left, node->work->dwork);
    memcpy(node->wsum_hess->x, node->left->wsum_hess->x,
           node->wsum_hess->nnz * sizeof(double));
}

static void refresh_sparse_left(left_matmul_expr *lnode)
{
    Sparse_Matrix *sm_A = (Sparse_Matrix *) lnode->A;
    Sparse_Matrix *sm_AT = (Sparse_Matrix *) lnode->AT;
    lnode->A->update_values(lnode->A, lnode->param_source->value);
    /* Recompute AT values from A */
    AT_fill_values(sm_A->csr, sm_AT->csr, lnode->base.work->iwork);
}

static void refresh_dense_left(left_matmul_expr *lnode)
{
    Dense_Matrix *dm_A = (Dense_Matrix *) lnode->A;
    int m = dm_A->base.m;
    int n = dm_A->base.n;
    lnode->A->update_values(lnode->A, lnode->param_source->value);
    /* Recompute AT data (transpose of row-major A) */
    Dense_Matrix *dm_AT = (Dense_Matrix *) lnode->AT;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            dm_AT->x[j * m + i] = dm_A->x[i * n + j];
        }
    }
}

expr *new_left_matmul(expr *param_node, expr *u, const CSR_Matrix *A)
{
    /* We expect u->d1 == A->n. However, numpy's broadcasting rules allow users
       to do A @ u where u is (n, ) which in C is actually (1, n). In that case
       the result of A @ u is (m, ), which is (1, m) according to broadcasting
       rules. We therefore check if this is the case. */
    int d1, d2, n_blocks;
    if (u->d1 == A->n)
    {
        d1 = A->m;
        d2 = u->d2;
        n_blocks = u->d2;
    }
    else if (u->d2 == A->n && u->d1 == 1)
    {
        d1 = 1;
        d2 = A->m;
        n_blocks = 1;
    }
    else
    {
        fprintf(stderr, "Error in new_left_matmul: dimension mismatch \n");
        exit(1);
    }

    /* Allocate the type-specific struct */
    left_matmul_expr *lnode =
        (left_matmul_expr *) calloc(1, sizeof(left_matmul_expr));
    expr *node = &lnode->base;
    init_expr(node, d1, d2, u->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, free_type_data);
    node->left = u;
    expr_retain(u);

    /* allocate workspace. iwork is used for converting J_child csr to csc
       (requiring size node->n_vars) and for transposing A (requiring size A->n).
       csc_to_csr_work is used for converting J_CSC to CSR (requiring
       node->size) */
    node->work->iwork = (int *) malloc(MAX(A->n, node->n_vars) * sizeof(int));
    lnode->csc_to_csr_work = (int *) malloc(node->size * sizeof(int));
    lnode->n_blocks = n_blocks;

    /* store A and AT */
    lnode->A = new_sparse_matrix(A);
    lnode->AT =
        sparse_matrix_trans((const Sparse_Matrix *) lnode->A, node->work->iwork);

    /* parameter support */
    lnode->param_source = param_node;
    if (param_node != NULL)
    {
        expr_retain(param_node);
        lnode->refresh_param_values = refresh_sparse_left;
    }

    return node;
}

expr *new_left_matmul_dense(expr *param_node, expr *u, int m, int n,
                            const double *data)
{
    int d1, d2, n_blocks;
    if (u->d1 == n)
    {
        d1 = m;
        d2 = u->d2;
        n_blocks = u->d2;
    }
    else if (u->d2 == n && u->d1 == 1)
    {
        d1 = 1;
        d2 = m;
        n_blocks = 1;
    }
    else
    {
        fprintf(stderr, "Error in new_left_matmul_dense: dimension mismatch\n");
        exit(1);
    }

    left_matmul_expr *lnode =
        (left_matmul_expr *) calloc(1, sizeof(left_matmul_expr));
    expr *node = &lnode->base;
    init_expr(node, d1, d2, u->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, free_type_data);
    node->left = u;
    expr_retain(u);

    node->work->iwork = (int *) malloc(MAX(n, node->n_vars) * sizeof(int));
    lnode->csc_to_csr_work = (int *) malloc(node->size * sizeof(int));
    lnode->n_blocks = n_blocks;

    lnode->A = new_dense_matrix(m, n, data);
    lnode->AT = dense_matrix_trans((const Dense_Matrix *) lnode->A);

    /* parameter support */
    lnode->param_source = param_node;
    if (param_node != NULL)
    {
        expr_retain(param_node);
        lnode->refresh_param_values = refresh_dense_left;
    }

    return node;
}
