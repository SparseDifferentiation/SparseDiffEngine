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
#include "sparsediff_linalg/dense.h"
#include "sparsediff_linalg/matmul_kron.h"
#include "sparsediff_linalg/matrix.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include "subexpr.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* This file implement the atom 'left_matmul' corresponding to the operation y =
   A @ f(x), where A is a given matrix and f(x) is an arbitrary expression.
   Here, f(x) can be a vector-valued expression and a matrix-valued
   expression. The dimensions are A - m x n, f(x) - n x p, y - m x p.
   Note that A has local column indices (f(x)'s entries), not global variable
   indices.

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

static void refresh_param_values(left_matmul_expr *lnode)
{
    if (lnode->param_source == NULL || !lnode->base.needs_parameter_refresh)
    {
        return;
    }

    lnode->base.needs_parameter_refresh = false;
    lnode->refresh_param_values(lnode);
}

static void forward(expr *node, const double *u)
{
    left_matmul_expr *lnode = (left_matmul_expr *) node;

    /* call forward on param_source if it exists and needs refresh */
    if (lnode->param_source != NULL && lnode->base.needs_parameter_refresh)
    {
        lnode->param_source->forward(lnode->param_source, NULL);
    }

    refresh_param_values(lnode);

    expr *x = node->left;

    /* child's forward pass */
    node->left->forward(node->left, u);

    /* y = A_kron @ vec(f(x)) */
    matrix *A = lnode->A;
    int n_blocks = lnode->n_blocks;
    BA_kron_vec_matrices(A, x->value, node->value, n_blocks);
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

/* param_source lives outside left/right, so the refresh walk reaches it
   here -- and reports whether it actually holds an updatable parameter. */
static bool set_needs_refresh_param_source(expr *node)
{
    return expr_set_needs_refresh(((left_matmul_expr *) node)->param_source);
}

static void free_type_data(expr *node)
{
    left_matmul_expr *lnode = (left_matmul_expr *) node;
    free_matrix(lnode->A);
    free_matrix(lnode->AT);
    free_expr(lnode->param_source);

    lnode->A = NULL;
    lnode->AT = NULL;
    lnode->param_source = NULL;
}

/* J_y = kron(I_p, A) J_f; the dispatcher routes on A's kind and on p == 1. */
static void jacobian_init_impl(expr *node)
{
    /* initialize jacobian of child */
    expr *x = node->left;
    left_matmul_expr *lnode = (left_matmul_expr *) node;
    jacobian_init(x);

    node->jacobian = BA_kron_matrices_alloc(lnode->A, lnode->n_blocks, x->jacobian);
}

static void eval_jacobian_impl(expr *node)
{
    /* evaluate jacobian of child */
    left_matmul_expr *lnode = (left_matmul_expr *) node;
    expr *x = node->left;
    eval_jacobian(x);

    BA_kron_matrices_fill_values(lnode->A, lnode->n_blocks, x->jacobian,
                                 node->jacobian);
}

static void wsum_hess_init_impl(expr *node)
{
    /* initialize child's hessian */
    expr *x = node->left;
    wsum_hess_init(x);

    /* allocate this node's hessian with the same sparsity as child's */
    node->wsum_hess = copy_sparsity_matrices_alloc(x->wsum_hess);

    /* work for computing A^T w*/
    int n_blocks = ((left_matmul_expr *) node)->n_blocks;
    int dim = ((left_matmul_expr *) node)->AT->m * n_blocks;
    node->work->dwork = (double *) sp_malloc(dim * sizeof(double));
}

static void eval_wsum_hess_impl(expr *node, const double *w)
{
    left_matmul_expr *lnode = (left_matmul_expr *) node;

    /* compute A^T w*/
    matrix *AT = lnode->AT;
    int n_blocks = lnode->n_blocks;
    BA_kron_vec_matrices(AT, w, node->work->dwork, n_blocks);

    eval_wsum_hess(node->left, node->work->dwork);
    memcpy(node->wsum_hess->x, node->left->wsum_hess->x,
           node->wsum_hess->nnz * sizeof(double));
}

static void refresh_dense_left(left_matmul_expr *lnode)
{
    int m = lnode->A->m;
    int n = lnode->A->n;

    /* The parameter represents the A in left_matmul_dense(A, x) in column-major.
       In this diffengine, we store A in row-major order. Hence, param->vals
       actually corresponds to the transpose of A, and we transpose AT to get A. */
    memcpy(lnode->AT->x, lnode->param_source->value, m * n * sizeof(double));
    A_transpose(lnode->A->x, lnode->AT->x, n, m);
    matrix_values_changed(lnode->AT);
    matrix_values_changed(lnode->A);
}

/* We expect u->d1 == A->n. However, numpy's broadcasting rules allow users to
   do A @ u where u is (n, ) which in C is actually (1, n). In that case the
   result of A @ u is (m, ), which is (1, m) according to broadcasting
   rules. We therefore check if this is the case. */
static void compute_dims(const expr *u, int A_rows, int A_cols, int *d1, int *d2,
                         int *n_blocks)
{
    if (u->d1 == A_cols)
    {
        *d1 = A_rows;
        *d2 = u->d2;
        *n_blocks = u->d2;
    }
    else if (u->d2 == A_cols && u->d1 == 1)
    {
        *d1 = 1;
        *d2 = A_rows;
        *n_blocks = 1;
    }
    else
    {
        fprintf(stderr, "Error in new_left_matmul: dimension mismatch\n");
        exit(1);
    }
}

expr *new_left_matmul(expr *param_node, expr *u, const CSR_matrix *A)
{
    int d1, d2, n_blocks;
    compute_dims(u, A->m, A->n, &d1, &d2, &n_blocks);

    /* Allocate the type-specific struct */
    left_matmul_expr *lnode =
        (left_matmul_expr *) sp_calloc(1, sizeof(left_matmul_expr));
    expr *node = &lnode->base;
    init_expr(node, d1, d2, u->n_vars, forward, jacobian_init_impl,
              eval_jacobian_impl, is_affine, wsum_hess_init_impl,
              eval_wsum_hess_impl, free_type_data);
    node->left = u;
    expr_retain(u);

    lnode->n_blocks = n_blocks;

    /* store A and AT. new_sparse_matrix takes ownership, so clone first. */
    lnode->A = new_sparse_matrix(new_csr(A));
    lnode->AT = transpose_matrices_alloc(lnode->A);
    transpose_matrices_fill_values(lnode->A, lnode->AT);

    /* parameter support */
    lnode->param_source = param_node;
    if (param_node != NULL)
    {
        fprintf(stderr, "Error in new_left_matmul: parameter for a sparse matrix "
                        "not supported \n");
        exit(1);
    }

    return node;
}

expr *new_left_matmul_dense(expr *param_node, expr *u, int m, int n,
                            const double *data)
{
    int d1, d2, n_blocks;
    compute_dims(u, m, n, &d1, &d2, &n_blocks);

    left_matmul_expr *lnode =
        (left_matmul_expr *) sp_calloc(1, sizeof(left_matmul_expr));
    expr *node = &lnode->base;
    init_expr(node, d1, d2, u->n_vars, forward, jacobian_init_impl,
              eval_jacobian_impl, is_affine, wsum_hess_init_impl,
              eval_wsum_hess_impl, free_type_data);
    node->left = u;
    expr_retain(u);

    lnode->n_blocks = n_blocks;

    /* parameter support */
    lnode->param_source = param_node;
    if (param_node != NULL)
    {
        if (data != NULL)
        {
            fprintf(stderr, "Error in new_left_matmul_dense with ptr convention \n");
            exit(1);
        }

        expr_retain(param_node);
        lnode->refresh_param_values = refresh_dense_left;

        /* A and AT buffers are filled by refresh_dense_left from the parameter. */
        lnode->A = new_permuted_dense_full(m, n, NULL);
        lnode->AT = new_permuted_dense_full(n, m, NULL);
        node->needs_parameter_refresh = true;
        node->set_needs_refresh_children = set_needs_refresh_param_source;
    }
    /* constant matrix case */
    else
    {
        if (data == NULL)
        {
            fprintf(stderr, "Error in new_left_matmul_dense with ptr convention \n");
            exit(1);
        }

        lnode->A = new_permuted_dense_full(m, n, data);
        lnode->AT = transpose_matrices_alloc(lnode->A);
        transpose_matrices_fill_values(lnode->A, lnode->AT);
    }

    return node;
}
