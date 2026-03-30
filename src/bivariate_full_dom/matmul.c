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
#include "bivariate_full_dom.h"
#include "subexpr.h"
#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"
#include "utils/CSR_sum.h"
#include "utils/linalg_dense_sparse_matmuls.h"
#include "utils/linalg_sparse_matmuls.h"
#include "utils/mini_numpy.h"
#include "utils/utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------------------
// Helpers for the cross-Hessian B(w) of the bilinear map X @ Y.
// B is mk x kn with B[row + j*m, j + col*k] = w[row + col*m].
// Each row has exactly n nonzeros.
// ------------------------------------------------------------------------------

static CSR_Matrix *build_cross_hessian_sparsity(int m, int k, int n)
{
    int total_nnz = m * k * n;
    CSR_Matrix *B = new_csr_matrix(m * k, k * n, total_nnz);
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
                                      CSR_Matrix *B)
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

static void accumulate_mapped(double *dest, const CSR_Matrix *src,
                              const int *idx_map)
{
    for (int j = 0; j < src->nnz; j++)
    {
        dest[idx_map[j]] += src->x[j];
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
    mat_mat_mult(x->value, y->value, node->value, x->d1, x->d2, y->d2);
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
    node->jacobian = new_csr_matrix(node->size, node->n_vars, nnz);

    int nnz_idx = 0;
    for (int i = 0; i < node->size; i++)
    {
        int row = i % m;
        int col = i / m;

        node->jacobian->p[i] = nnz_idx;

        if (x->var_id < y->var_id)
        {
            for (int j = 0; j < k; j++)
            {
                node->jacobian->i[nnz_idx++] = x->var_id + row + j * m;
            }
            for (int j = 0; j < k; j++)
            {
                node->jacobian->i[nnz_idx++] = y->var_id + col * k + j;
            }
        }
        else
        {
            for (int j = 0; j < k; j++)
            {
                node->jacobian->i[nnz_idx++] = y->var_id + col * k + j;
            }
            for (int j = 0; j < k; j++)
            {
                node->jacobian->i[nnz_idx++] = x->var_id + row + j * m;
            }
        }
    }
    node->jacobian->p[node->size] = nnz_idx;
    assert(nnz_idx == nnz);
}

static void eval_jacobian_no_chain_rule(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;
    int m = x->d1;
    int k = x->d2;
    double *Jx = node->jacobian->x;

    for (int i = 0; i < node->size; i++)
    {
        int row = i % m;
        int col = i / m;
        int pos = node->jacobian->p[i];

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
//  = f(u) @ g(u) where f is  m x k and g is k x n, is given by Jh = (g^T kron I_m)
//  Jf + (I_n kron f) Jg . */
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
    jacobian_csc_init(f);
    jacobian_csc_init(g);

    /* initialize term1, term2, and their sum */
    mnode->term1_CSR = YT_kron_I_alloc(m, k, n, f->work->jacobian_csc);
    mnode->term2_CSR = I_kron_X_alloc(m, k, n, g->work->jacobian_csc);
    int max_nnz = mnode->term1_CSR->nnz + mnode->term2_CSR->nnz;
    node->jacobian = new_csr_matrix(node->size, node->n_vars, max_nnz);
    sum_csr_alloc(mnode->term1_CSR, mnode->term2_CSR, node->jacobian);
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
    f->eval_jacobian(f);
    g->eval_jacobian(g);
    csr_to_csc_fill_vals(f->jacobian, f->work->jacobian_csc, f->work->csc_work);
    csr_to_csc_fill_vals(g->jacobian, g->work->jacobian_csc, g->work->csc_work);

    /* evaluate term1, term2, and their sum */
    YT_kron_I_fill_vals(m, k, n, g->value, f->work->jacobian_csc, mnode->term1_CSR);
    I_kron_X_fill_vals(m, k, n, f->value, g->work->jacobian_csc, mnode->term2_CSR);
    sum_csr_fill_vals(mnode->term1_CSR, mnode->term2_CSR, node->jacobian);
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
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, total_nnz);
    int nnz = 0;
    int *Hi = node->wsum_hess->i;
    int *Hp = node->wsum_hess->p;
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

static void wsum_hess_init_chain_rule(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;
    matmul_expr *mnode = (matmul_expr *) node;
    int m = x->d1;
    int k = x->d2;
    int n = y->d2;

    jacobian_csc_init(x);
    jacobian_csc_init(y);
    CSC_Matrix *Jf = x->work->jacobian_csc;
    CSC_Matrix *Jg = y->work->jacobian_csc;

    /* build cross-Hessian B sparsity */
    mnode->B = build_cross_hessian_sparsity(m, k, n);

    /* C = J_f^T @ B @ J_g:
     * step 1: BJ_g = B @ J_g */
    mnode->BJ_g = csr_csc_matmul_alloc(mnode->B, Jg);
    mnode->BJ_g_csc_work =
        (int *) malloc(MAX(mnode->BJ_g->m, mnode->BJ_g->n) * sizeof(int));
    mnode->BJ_g_CSC = csr_to_csc_alloc(mnode->BJ_g, mnode->BJ_g_csc_work);

    /* step 2: C = J_f^T @ BJ_g via BTA (B^T D A with D=I) */
    mnode->C = BTA_alloc(mnode->BJ_g_CSC, Jf);

    /* C^T */
    node->work->iwork = (int *) malloc(mnode->C->m * sizeof(int));
    mnode->CT = AT_alloc(mnode->C, node->work->iwork);

    /* allocate weight backprop workspace */
    if (!x->is_affine(x) || !y->is_affine(y))
    {
        node->work->dwork =
            (double *) malloc(MAX(x->size, y->size) * sizeof(double));
    }

    /* init child Hessians */
    wsum_hess_init(x);
    wsum_hess_init(y);

    /* merge 4 sparsity patterns */
    int *maps[4];
    node->wsum_hess = sum_4_csr_fill_sparsity_and_idx_maps(
        mnode->C, mnode->CT, x->wsum_hess, y->wsum_hess, maps);
    mnode->idx_map_C = maps[0];
    mnode->idx_map_CT = maps[1];
    mnode->idx_map_Hf = maps[2];
    mnode->idx_map_Hg = maps[3];
}

static void eval_wsum_hess_chain_rule(expr *node, const double *w)
{
    expr *x = node->left;
    expr *y = node->right;
    matmul_expr *mnode = (matmul_expr *) node;
    int m = x->d1;
    int k = x->d2;
    int n = y->d2;
    bool is_x_affine = x->is_affine(x);
    bool is_y_affine = y->is_affine(y);

    /* refresh child Jacobian CSC values (cache if affine) */
    if (!x->work->jacobian_csc_filled)
    {
        csr_to_csc_fill_vals(x->jacobian, x->work->jacobian_csc, x->work->csc_work);
        if (is_x_affine)
        {
            x->work->jacobian_csc_filled = true;
        }
    }
    if (!y->work->jacobian_csc_filled)
    {
        csr_to_csc_fill_vals(y->jacobian, y->work->jacobian_csc, y->work->csc_work);
        if (is_y_affine)
        {
            y->work->jacobian_csc_filled = true;
        }
    }

    CSC_Matrix *Jf = x->work->jacobian_csc;
    CSC_Matrix *Jg = y->work->jacobian_csc;

    /* compute C = J_f^T @ B(w) @ J_g */
    fill_cross_hessian_values(m, k, n, w, mnode->B);
    csr_csc_matmul_fill_vals(mnode->B, Jg, mnode->BJ_g);
    csr_to_csc_fill_vals(mnode->BJ_g, mnode->BJ_g_CSC, mnode->BJ_g_csc_work);
    BTDA_fill_vals(mnode->BJ_g_CSC, Jf, NULL, mnode->C);

    /* C^T */
    AT_fill_vals(mnode->C, mnode->CT, node->work->iwork);

    /* backpropagate weights and recurse into children */
    if (!is_x_affine)
    {
        /* v_f = vec(W @ Y^T):
         * v_f[row + j*m] = sum_col Y[j,col] * w[row + col*m] */
        double *v_f = node->work->dwork;
        for (int j = 0; j < k; j++)
        {
            for (int row = 0; row < m; row++)
            {
                double sum = 0.0;
                for (int col = 0; col < n; col++)
                {
                    sum += y->value[j + col * k] * w[row + col * m];
                }
                v_f[row + j * m] = sum;
            }
        }
        x->eval_wsum_hess(x, v_f);
    }

    if (!is_y_affine)
    {
        /* v_g = vec(X^T @ W):
         * v_g[j + col*k] = sum_row X[row,j] * w[row + col*m] */
        double *v_g = node->work->dwork;
        for (int col = 0; col < n; col++)
        {
            for (int j = 0; j < k; j++)
            {
                double sum = 0.0;
                for (int row = 0; row < m; row++)
                {
                    sum += x->value[row + j * m] * w[row + col * m];
                }
                v_g[j + col * k] = sum;
            }
        }
        y->eval_wsum_hess(y, v_g);
    }

    /* accumulate H = C + C^T + H_f + H_g */
    memset(node->wsum_hess->x, 0, node->wsum_hess->nnz * sizeof(double));
    accumulate_mapped(node->wsum_hess->x, mnode->C, mnode->idx_map_C);
    accumulate_mapped(node->wsum_hess->x, mnode->CT, mnode->idx_map_CT);
    accumulate_mapped(node->wsum_hess->x, x->wsum_hess, mnode->idx_map_Hf);
    accumulate_mapped(node->wsum_hess->x, y->wsum_hess, mnode->idx_map_Hg);
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
    expr *node = (expr *) calloc(1, sizeof(matmul_expr));

    /* Choose no-chain-rule or chain-rule function pointers */
    bool use_chain_rule = !(x->var_id != NOT_A_VARIABLE &&
                            y->var_id != NOT_A_VARIABLE && x->var_id != y->var_id);

    jacobian_init_fn jac_init =
        use_chain_rule ? jacobian_init_chain_rule : jacobian_init_no_chain_rule;
    eval_jacobian_fn jac_eval =
        use_chain_rule ? eval_jacobian_chain_rule : eval_jacobian_no_chain_rule;
    wsum_hess_init_fn hess_init =
        use_chain_rule ? wsum_hess_init_chain_rule : wsum_hess_init_no_chain_rule;
    wsum_hess_fn hess_eval =
        use_chain_rule ? eval_wsum_hess_chain_rule : eval_wsum_hess_no_chain_rule;

    init_expr(node, x->d1, y->d2, x->n_vars, forward, jac_init, jac_eval, is_affine,
              hess_init, hess_eval, NULL);

    /* Set children */
    node->left = x;
    node->right = y;
    expr_retain(x);
    expr_retain(y);

    return node;
}
