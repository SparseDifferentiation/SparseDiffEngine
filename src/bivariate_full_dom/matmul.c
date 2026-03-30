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
#include "utils/CSR_sum.h"
#include "utils/linalg_dense_sparse_matmuls.h"
#include "utils/mini_numpy.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void jacobian_init_impl(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* dimensions: X is m x k, Y is k x n, Z is m x n */
    int m = x->d1;
    int k = x->d2;
    int n = y->d2;

    if (x->var_id != NOT_A_VARIABLE && y->var_id != NOT_A_VARIABLE &&
        x->var_id != y->var_id)
    {
        /* both children are differentleaf variables */
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
    else
    {
        /* chain rule: the jacobian of f(u) @ g(u) with f(u) and g(u) matrices
           is term1 + term2 where term1 = (g(u)^T kron I) @ J_f and
           term2 = (I kron f(u)) @ J_g. */
        matmul_expr *mnode = (matmul_expr *) node;

        jacobian_init(x);
        jacobian_init(y);
        jacobian_csc_init(x);
        jacobian_csc_init(y);

        mnode->term1_CSR = YT_kron_I_alloc(m, k, n, x->work->jacobian_csc);
        mnode->term2_CSR = I_kron_X_alloc(m, k, n, y->work->jacobian_csc);

        int max_nnz = mnode->term1_CSR->nnz + mnode->term2_CSR->nnz;
        node->jacobian = new_csr_matrix(node->size, node->n_vars, max_nnz);
        sum_csr_matrices_fill_sparsity(mnode->term1_CSR, mnode->term2_CSR,
                                       node->jacobian);
    }
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* dimensions: X is m x k, Y is k x n, Z is m x n */
    int m = x->d1;
    int k = x->d2;
    int n = y->d2;

    if (x->var_id != NOT_A_VARIABLE && y->var_id != NOT_A_VARIABLE &&
        x->var_id != y->var_id)
    {
        /* both children are different leaf variables */
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
    else
    {
        /* composite case */
        matmul_expr *mnode = (matmul_expr *) node;

        x->eval_jacobian(x);
        y->eval_jacobian(y);

        CSC_Matrix *Jx_csc = x->work->jacobian_csc;
        CSC_Matrix *Jy_csc = y->work->jacobian_csc;

        /* refresh children's CSC values */
        csr_to_csc_fill_values(x->jacobian, Jx_csc, x->work->csc_work);
        csr_to_csc_fill_values(y->jacobian, Jy_csc, y->work->csc_work);

        /* compute term1, term2, and sum */
        YT_kron_I_fill_values(m, k, n, y->value, Jx_csc, mnode->term1_CSR);
        I_kron_X_fill_values(m, k, n, x->value, Jy_csc, mnode->term2_CSR);
        sum_csr_matrices_fill_values(mnode->term1_CSR, mnode->term2_CSR,
                                     node->jacobian);
    }
}

static void wsum_hess_init_impl(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* dimensions: X is m x k, Y is k x n, Z is m x n */
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
        /* fill rows corresponding to x */
        for (i = 0; i < x->size; i++)
        {
            Hp[x->var_id + i] = nnz;
            start = y->var_id + (i / m);
            for (int col = 0; col < n; col++)
            {
                Hi[nnz++] = start + col * k;
            }
        }

        /* fill rows between x and y */
        for (i = x->var_id + x->size; i < y->var_id; i++)
        {
            Hp[i] = nnz;
        }

        /* fill rows corresponding to y */
        for (i = 0; i < y->size; i++)
        {
            Hp[y->var_id + i] = nnz;
            start = x->var_id + (i % k) * m;
            for (int row = 0; row < m; row++)
            {
                Hi[nnz++] = start + row;
            }
        }

        /* fill rows after y */
        for (i = y->var_id + y->size; i <= node->n_vars; i++)
        {
            Hp[i] = nnz;
        }
    }
    else
    {
        /* Y has lower var_id than X */
        /* fill rows corresponding to y */
        for (i = 0; i < y->size; i++)
        {
            Hp[y->var_id + i] = nnz;
            start = x->var_id + (i % k) * m;
            for (int row = 0; row < m; row++)
            {
                Hi[nnz++] = start + row;
            }
        }

        /* fill rows between y and x */
        for (i = y->var_id + y->size; i < x->var_id; i++)
        {
            Hp[i] = nnz;
        }

        /* fill rows corresponding to x */
        for (i = 0; i < x->size; i++)
        {
            Hp[x->var_id + i] = nnz;
            start = y->var_id + (i / m);
            for (int col = 0; col < n; col++)
            {
                Hi[nnz++] = start + col * k;
            }
        }

        /* fill rows after x */
        for (i = x->var_id + x->size; i <= node->n_vars; i++)
        {
            Hp[i] = nnz;
        }
    }

    Hp[node->n_vars] = nnz;
    assert(nnz == total_nnz);
}

static void eval_wsum_hess(expr *node, const double *w)
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
        /* rows corresponding to x */
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

        /* rows corresponding to y */
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
        /* rows corresponding to y */
        for (int col = 0; col < n; col++)
        {
            w_temp = w + col * m;
            for (int k_idx = 0; k_idx < k; k_idx++)
            {
                memcpy(Hx + offset, w_temp, m * sizeof(double));
                offset += m;
            }
        }

        /* rows corresponding to x */
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

    /* Initialize with d1 = x->d1, d2 = y->d2 (result is m x n) */
    init_expr(node, x->d1, y->d2, x->n_vars, forward, jacobian_init_impl,
              eval_jacobian, is_affine, wsum_hess_init_impl, eval_wsum_hess, NULL);

    /* Set children */
    node->left = x;
    node->right = y;
    expr_retain(x);
    expr_retain(y);

    return node;
}
