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
#include "sparsediff_linalg/matmul.h"
#include "atoms/bivariate_full_dom.h"
#include "int_utils.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/cblas_wrapper.h"
#include "sparsediff_linalg/dense.h"
#include "sparsediff_linalg/matmul_kron.h"
#include "sparsediff_linalg/matrix.h"
#include "sparsediff_linalg/matrix_sum.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include "subexpr.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------------------
// Helpers for the cross-Hessian B(w) of the bilinear map X @ Y.
// B(w) is the mk x kn weighted cross-Hessian of the bilinear map Z = XY.
// It captures d^2(w^T vec(Z)) / d(vec(X)) d(vec(Y)).
//
// Entry: B[row + j*m, j + col*k] = w[row + col*m].
// Each row has exactly n nonzeros. All k block-rows (one per j) have
// the same values (columns of W = reshape(w, m, n)), but at different
// column positions (offset by j in the Y-variable indexing).
// ------------------------------------------------------------------------------

static CSR_matrix *build_cross_hessian_sparsity(int m, int k, int n)
{
    int total_nnz = m * k * n;
    CSR_matrix *B = new_CSR_matrix(m * k, k * n, total_nnz);
    int idx = 0;

    for (int j = 0; j < k; j++)
    {
        for (int row = 0; row < m; row++)
        {
            B->p[row + j * m] = idx;
            for (int col = 0; col < n; col++)
            {
                B->i[idx++] = j + col * k;
            }
        }
    }
    B->p[m * k] = idx;
    assert(idx == total_nnz);
    return B;
}

static void fill_cross_hessian_values(int m, int k, int n, const double *w,
                                      CSR_matrix *B)
{
    int idx = 0;
    for (int j = 0; j < k; j++)
    {
        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < n; col++)
            {
                B->x[idx++] = w[row + col * m];
            }
        }
    }
}

// ------------------------------------------------------------------------------
// Implementation of matrix multiplication: Z = X @ Y
// where X is m x k and Y is k x n, producing Z which is m x n
// All matrices are stored in column-major order
// ------------------------------------------------------------------------------

static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    expr *y = node->right;

    /* children's forward passes */
    x->forward(x, u);
    y->forward(y, u);

    /* local forward pass */
    int m = x->d1, k = x->d2, n = y->d2;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, x->value, m,
                y->value, k, 0.0, node->value, m);
}

static void free_matmul_data(expr *node)
{
    matmul_expr *mnode = (matmul_expr *) node;
    /* Jacobian workspace */
    free_matrix(mnode->YT);
    free_matrix(mnode->X);
    free_matrix(mnode->term1);
    free_matrix(mnode->term2);
    /* Hessian workspace */
    free_matrix(mnode->B);
    free_matrix(mnode->BJg);
    free_matrix(mnode->C);
    free_matrix(mnode->CT);
}

static bool is_affine(const expr *node)
{
    (void) node;
    return false;
}

// --------------------------------------------------------------------------------
//  Jacobian initialization and evaluation for the case when no chain rule is needed,
//  ie., when both children are different leaf variables.
// --------------------------------------------------------------------------------
static void jacobian_init_no_chain_rule(expr *node)
{
    assert(node->left->var_id != NOT_A_VARIABLE &&
           node->right->var_id != NOT_A_VARIABLE &&
           node->left->var_id != node->right->var_id);

    expr *x = node->left;
    expr *y = node->right;
    int m = x->d1;
    int k = x->d2;
    int n = y->d2;
    int nnz = m * n * 2 * k;
    CSR_matrix *jac = new_CSR_matrix(node->size, node->n_vars, nnz);

    int nnz_idx = 0;
    for (int i = 0; i < node->size; i++)
    {
        int row = i % m;
        int col = i / m;

        jac->p[i] = nnz_idx;

        if (x->var_id < y->var_id)
        {
            for (int j = 0; j < k; j++)
            {
                jac->i[nnz_idx++] = x->var_id + row + j * m;
            }
            for (int j = 0; j < k; j++)
            {
                jac->i[nnz_idx++] = y->var_id + col * k + j;
            }
        }
        else
        {
            for (int j = 0; j < k; j++)
            {
                jac->i[nnz_idx++] = y->var_id + col * k + j;
            }
            for (int j = 0; j < k; j++)
            {
                jac->i[nnz_idx++] = x->var_id + row + j * m;
            }
        }
    }
    jac->p[node->size] = nnz_idx;
    assert(nnz_idx == nnz);
    node->jacobian = new_sparse_matrix(jac);
}

static void eval_jacobian_no_chain_rule(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;
    int m = x->d1;
    int k = x->d2;
    CSR_matrix *jac = node->jacobian->to_csr(node->jacobian);
    double *Jx = jac->x;

    for (int i = 0; i < node->size; i++)
    {
        int row = i % m;
        int col = i / m;
        int pos = jac->p[i];

        if (x->var_id < y->var_id)
        {
            memcpy(Jx + pos, y->value + col * k, k * sizeof(double));
            for (int j = 0; j < k; j++)
            {
                Jx[pos + k + j] = x->value[row + j * m];
            }
        }
        else
        {
            for (int j = 0; j < k; j++)
            {
                Jx[pos + j] = x->value[row + j * m];
            }
            memcpy(Jx + pos + k, y->value + col * k, k * sizeof(double));
        }
    }
}

// ------------------------------------------------------------------------------------
//  Jacobian initialization and evaluation for the case where chain rule is needed,
//  ie., when at least one child is composite, or same variable. The jacobian of h(u)
//  = f(u) @ g(u) where f is m x k and g is k x n, is given by
//      Jh = (g^T kron I_m) Jf + (I_n kron f) Jg.
// ------------------------------------------------------------------------------------
static void jacobian_init_chain_rule(expr *node)
{
    expr *f = node->left;
    expr *g = node->right;
    matmul_expr *mnode = (matmul_expr *) node;
    int m = f->d1;
    int k = f->d2;
    int n = g->d2;

    /* initialize Jacobian of children */
    jacobian_init(f);
    jacobian_init(g);

    /* initialize term1, term2, and their sum */
    mnode->YT = new_permuted_dense_full(n, k, NULL);
    mnode->X = new_permuted_dense_full(m, k, NULL);
    mnode->term1 = BA_dense_kron_I_matrices_alloc(mnode->YT, m, f->jacobian);
    mnode->term2 = BA_kron_matrices_alloc(mnode->X, n, g->jacobian);
    int max_nnz = mnode->term1->nnz + mnode->term2->nnz;
    max_nnz = MIN(max_nnz, sat_mul_int(node->size, node->n_vars));
    node->jacobian = new_sparse_matrix_alloc(node->size, node->n_vars, max_nnz);
    sum_matrices_alloc(mnode->term1, mnode->term2, node->jacobian);
}

static void eval_jacobian_chain_rule(expr *node)
{
    expr *f = node->left;
    expr *g = node->right;
    matmul_expr *mnode = (matmul_expr *) node;
    int m = f->d1;
    int k = f->d2;
    int n = g->d2;

    /* evaluate Jacobians of children */
    eval_jacobian(f);
    eval_jacobian(g);

    /* refresh the values of Y^T and X */
    memcpy(mnode->YT->x, g->value, (size_t) k * n * sizeof(double));
    matrix_values_changed(mnode->YT);
    A_transpose(mnode->X->x, f->value, k, m);
    matrix_values_changed(mnode->X);

    /* Jh = term1 + term2 */
    BA_dense_kron_I_matrices_fill_values(mnode->YT, m, f->jacobian, mnode->term1);
    BA_kron_matrices_fill_values(mnode->X, n, g->jacobian, mnode->term2);
    sum_matrices_fill_values(mnode->term1, mnode->term2, node->jacobian);
}

// ------------------------------------------------------------------------------------
// Hessian initialization and evaluation for the case where no chain rule is needed,
// ie., when both children are different leaf variables.
// ------------------------------------------------------------------------------------
static void wsum_hess_init_no_chain_rule(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;
    int m = x->d1;
    int k = x->d2;
    int n = y->d2;
    int total_nnz = 2 * m * k * n;
    CSR_matrix *hess = new_CSR_matrix(node->n_vars, node->n_vars, total_nnz);
    int nnz = 0;
    int *Hi = hess->i;
    int *Hp = hess->p;
    int start, i;

    if (x->var_id < y->var_id)
    {
        for (i = 0; i < x->size; i++)
        {
            Hp[x->var_id + i] = nnz;
            start = y->var_id + (i / m);
            for (int col = 0; col < n; col++)
            {
                Hi[nnz++] = start + col * k;
            }
        }
        for (i = x->var_id + x->size; i < y->var_id; i++)
        {
            Hp[i] = nnz;
        }
        for (i = 0; i < y->size; i++)
        {
            Hp[y->var_id + i] = nnz;
            start = x->var_id + (i % k) * m;
            for (int row = 0; row < m; row++)
            {
                Hi[nnz++] = start + row;
            }
        }
        for (i = y->var_id + y->size; i <= node->n_vars; i++)
        {
            Hp[i] = nnz;
        }
    }
    else
    {
        for (i = 0; i < y->size; i++)
        {
            Hp[y->var_id + i] = nnz;
            start = x->var_id + (i % k) * m;
            for (int row = 0; row < m; row++)
            {
                Hi[nnz++] = start + row;
            }
        }
        for (i = y->var_id + y->size; i < x->var_id; i++)
        {
            Hp[i] = nnz;
        }
        for (i = 0; i < x->size; i++)
        {
            Hp[x->var_id + i] = nnz;
            start = y->var_id + (i / m);
            for (int col = 0; col < n; col++)
            {
                Hi[nnz++] = start + col * k;
            }
        }
        for (i = x->var_id + x->size; i <= node->n_vars; i++)
        {
            Hp[i] = nnz;
        }
    }
    Hp[node->n_vars] = nnz;
    assert(nnz == total_nnz);
    node->wsum_hess = new_sparse_matrix(hess);
}

static void eval_wsum_hess_no_chain_rule(expr *node, const double *w)
{
    expr *x = node->left;
    expr *y = node->right;
    int m = x->d1;
    int k = x->d2;
    int n = y->d2;
    int offset = 0;
    double *Hx = node->wsum_hess->x;
    const double *w_temp;

    if (x->var_id < y->var_id)
    {
        for (int k_idx = 0; k_idx < k; k_idx++)
        {
            for (int row = 0; row < m; row++)
            {
                for (int col = 0; col < n; col++)
                {
                    Hx[offset++] = w[row + col * m];
                }
            }
        }
        for (int col = 0; col < n; col++)
        {
            w_temp = w + col * m;
            for (int k_idx = 0; k_idx < k; k_idx++)
            {
                memcpy(Hx + offset, w_temp, m * sizeof(double));
                offset += m;
            }
        }
    }
    else
    {
        for (int col = 0; col < n; col++)
        {
            w_temp = w + col * m;
            for (int k_idx = 0; k_idx < k; k_idx++)
            {
                memcpy(Hx + offset, w_temp, m * sizeof(double));
                offset += m;
            }
        }
        for (int k_idx = 0; k_idx < k; k_idx++)
        {
            for (int row = 0; row < m; row++)
            {
                for (int col = 0; col < n; col++)
                {
                    Hx[offset++] = w[row + col * m];
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------
// Hessian chain rule for Z = f(u) @ g(u).
// H = C + C^T + H_f(v_f) + H_g(v_g)  where:
//
//   C = J_f^T B(w) J_g     cross term (B is the weighted cross-Hessian)
//   v_f = (Y kron I_m) w   backpropagated weights for left child
//   v_g = (I_n kron X^T) w backpropagated weights for right child
//   H_f(v_f), H_g(v_g)     child Hessians evaluated with transformed weights
// ------------------------------------------------------------------------------------
static void wsum_hess_init_chain_rule(expr *node)
{
    expr *f = node->left;
    expr *g = node->right;
    matmul_expr *mnode = (matmul_expr *) node;
    int m = f->d1;
    int k = f->d2;
    int n = g->d2;

    /* initialize C = Jf^T @ B @ Jg = Jf^T @ (B @ Jg) and C^T */
    mnode->B = new_sparse_matrix(build_cross_hessian_sparsity(m, k, n));
    mnode->BJg = BA_matrices_alloc(mnode->B, g->jacobian);
    mnode->C = BTA_matrices_alloc(mnode->BJg, f->jacobian);
    mnode->CT = transpose_matrices_alloc(mnode->C);

    /* initialize Hessians of children */
    wsum_hess_init(f);
    wsum_hess_init(g);

    /* H = C + C^T + H_f + H_g */
    matrix *ops[4] = {mnode->C, mnode->CT, f->wsum_hess, g->wsum_hess};
    node->wsum_hess = sum_n_matrices_alloc(ops, 4);

    /* allocate weight backprop workspace */
    if (!f->is_affine(f) || !g->is_affine(g))
    {
        node->work->dwork =
            (double *) sp_malloc(MAX(f->size, g->size) * sizeof(double));
    }
}

static void eval_wsum_hess_chain_rule(expr *node, const double *w)
{
    expr *f = node->left;
    expr *g = node->right;
    matmul_expr *mnode = (matmul_expr *) node;
    int m = f->d1;
    int k = f->d2;
    int n = g->d2;
    bool is_f_affine = f->is_affine(f);
    bool is_g_affine = g->is_affine(g);

    /* compute C = J_f^T @ B(w) @ J_g and C^T */
    fill_cross_hessian_values(m, k, n, w, mnode->B->to_csr(mnode->B));
    matrix_values_changed(mnode->B);
    BA_matrices_fill_values(mnode->B, g->jacobian, mnode->BJg);
    BTA_matrices_fill_values(mnode->BJg, f->jacobian, mnode->C);
    transpose_matrices_fill_values(mnode->C, mnode->CT);

    /* H = C + C^T + H_f + H_g, adding each term right after it is evaluated */
    sum_n_matrices_fill_zero(node->wsum_hess);
    sum_n_matrices_fill_add(mnode->C, 0, node->wsum_hess);
    sum_n_matrices_fill_add(mnode->CT, 1, node->wsum_hess);

    /* compute Hessian of f */
    if (!is_f_affine)
    {
        Y_kron_I_vec(m, k, n, g->value, w, node->work->dwork);
        eval_wsum_hess(f, node->work->dwork);
    }
    sum_n_matrices_fill_add(f->wsum_hess, 2, node->wsum_hess);

    /* compute Hessian of g */
    if (!is_g_affine)
    {
        I_kron_XT_vec(m, k, n, f->value, w, node->work->dwork);
        eval_wsum_hess(g, node->work->dwork);
    }
    sum_n_matrices_fill_add(g->wsum_hess, 3, node->wsum_hess);
}

expr *new_matmul(expr *x, expr *y)
{
    /* Verify dimensions: x->d2 must equal y->d1 */
    if (x->d2 != y->d1)
    {
        fprintf(stderr,
                "Error in new_matmul: dimension mismatch. "
                "X is %d x %d, Y is %d x %d. X.d2 (%d) must equal Y.d1 (%d)\n",
                x->d1, x->d2, y->d1, y->d2, x->d2, y->d1);
        exit(1);
    }

    /* Allocate the expression node */
    expr *node = (expr *) sp_calloc(1, sizeof(matmul_expr));

    /* Choose no-chain-rule or chain-rule function pointers */
    bool use_chain_rule = !(x->var_id != NOT_A_VARIABLE &&
                            y->var_id != NOT_A_VARIABLE && x->var_id != y->var_id);

    if (!use_chain_rule)
    {
        init_expr(node, x->d1, y->d2, x->n_vars, forward,
                  jacobian_init_no_chain_rule, eval_jacobian_no_chain_rule,
                  is_affine, wsum_hess_init_no_chain_rule,
                  eval_wsum_hess_no_chain_rule, free_matmul_data);
    }
    else
    {
        init_expr(node, x->d1, y->d2, x->n_vars, forward, jacobian_init_chain_rule,
                  eval_jacobian_chain_rule, is_affine, wsum_hess_init_chain_rule,
                  eval_wsum_hess_chain_rule, free_matmul_data);
    }

    /* Set children */
    node->left = x;
    node->right = y;
    expr_retain(x);
    expr_retain(y);

    return node;
}
