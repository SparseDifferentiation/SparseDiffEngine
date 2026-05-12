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
#include "atoms/bivariate_full_dom.h"
#include "subexpr.h"
#include "utils/CSR_sum.h"
#include "utils/matrix_BTA.h"
#include "utils/matrix_sum.h"
#include "utils/sparse_matrix.h"
#include "utils/tracked_alloc.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------------------
// Implementation of elementwise multiplication when both arguments are vectors.
// If one argument is a scalar variable, the broadcasting should be represented
// as a linear operator child node? How to treat if one variable is a constant?
// ------------------------------------------------------------------------------
static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    expr *y = node->right;

    /* children's forward passes */
    x->forward(x, u);
    y->forward(y, u);

    /* local forward pass */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = x->value[i] * y->value[i];
    }
}

static void jacobian_init_impl(expr *node)
{
    jacobian_init(node->left);
    jacobian_init(node->right);
    int nnz_max = node->left->jacobian->nnz + node->right->jacobian->nnz;
    node->jacobian = new_sparse_matrix_alloc(node->size, node->n_vars, nnz_max);

    /* fill sparsity pattern */
    sum_matrices_alloc(node->left->jacobian, node->right->jacobian, node->jacobian);
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    x->eval_jacobian(x);
    y->eval_jacobian(y);

    /* chain rule: the jacobian of h(x) = f(g1(x), g2(x))) is Jh = J_{f, 1} J_{g1} +
     * J_{f, 2} J_{g2} */
    sum_scaled_matrices_fill_values(x->jacobian, y->jacobian, node->jacobian,
                                    y->value, x->value);
}

static void wsum_hess_init_impl(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* both x and y are variables, and not the same */
    if (x->var_id != NOT_A_VARIABLE && y->var_id != NOT_A_VARIABLE &&
        x->var_id != y->var_id)
    {
        assert(y->var_id != NOT_A_VARIABLE);
        CSR_matrix *hess =
            new_CSR_matrix(node->n_vars, node->n_vars, 2 * node->size);

        int i, var1_id, var2_id;

        if (x->var_id < y->var_id)
        {
            var1_id = x->var_id;
            var2_id = y->var_id;
        }
        else
        {
            var1_id = y->var_id;
            var2_id = x->var_id;
        }

        /* var1 rows of Hessian */
        for (i = 0; i < node->size; i++)
        {
            hess->p[var1_id + i] = i;
            hess->i[i] = var2_id + i;
        }

        int nnz = node->size;

        /* rows between var1 and var2 */
        for (i = var1_id + node->size; i < var2_id; i++)
        {
            hess->p[i] = nnz;
        }

        /* var2 rows of Hessian */
        for (i = 0; i < node->size; i++)
        {
            hess->p[var2_id + i] = nnz + i;
            hess->i[nnz + i] = var1_id + i;
        }

        /* remaining rows */
        nnz += node->size;
        for (i = var2_id + node->size; i <= node->n_vars; i++)
        {
            hess->p[i] = nnz;
        }
        node->wsum_hess = new_sparse_matrix(hess);
    }
    else
    {
        /* chain rule: the Hessian is in this case given by
           wsum_hess = C + C^T + term2 + term3 where

           * C = J_{g2}^T diag(w) J_{g1}
           * term2 = sum_k w_k g2_k H_{g1_k}
           * term3 = sum_k w_k g1_k H_{g2_k}

           The two last terms are nonzero only if g1 and g2 are nonlinear.
           Here, we view multiply as the composition h(x) = f(g1(x), g2(x)) where f
           is the elementwise multiplication operator, and g1 and g2 are the left and
           right child nodes.
        */

        /* used for computing weights to wsum_hess of children */
        if (!x->is_affine(x) || !y->is_affine(y))
        {
            node->work->dwork = (double *) SP_MALLOC(node->size * sizeof(double));
        }

        /* CSC_matrix scaffolding is still needed for the (Sparse, Sparse) fast path
           through BTA_matrices_* / BTDA_matrices_* — those route through
           sparse_matrix's csc_cache. For PD operands, refresh_csc_values is
           a no-op so the call is harmless. */
        jacobian_csc_init(x);
        jacobian_csc_init(y);

        /* initialize wsum_hessians of children */
        wsum_hess_init(x);
        wsum_hess_init(y);

        elementwise_mult_expr *mul_node = (elementwise_mult_expr *) node;

        /* compute sparsity of C polymorphically (Sparse, PD-CSR_matrix, CSR_matrix-PD, PD-PD). */
        mul_node->cross_C = BTA_matrices_alloc(x->jacobian, y->jacobian);

        /* CT structure is always CSR_matrix (via AT on C's CSR_matrix view). */
        CSR_matrix *C_csr = mul_node->cross_C->to_csr(mul_node->cross_C);
        node->work->iwork = (int *) SP_MALLOC(C_csr->m * sizeof(int));
        CSR_matrix *CT = AT_alloc(C_csr, node->work->iwork);
        mul_node->CSR_work2 = CT;

        /* compute sparsity pattern of H = C + C^T + term2 + term3 (we also
           fill index maps telling us where to accumulate each element of each
           matrix in the sum) */
        int *maps[4];
        CSR_matrix *hess = sum_4_csr_alloc(C_csr, CT,
                                           x->wsum_hess->to_csr(x->wsum_hess),
                                           y->wsum_hess->to_csr(y->wsum_hess), maps);
        node->wsum_hess = new_sparse_matrix(hess);
        mul_node->idx_map_C = maps[0];
        mul_node->idx_map_CT = maps[1];
        mul_node->idx_map_Hx = maps[2];
        mul_node->idx_map_Hy = maps[3];
    }
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    expr *y = node->right;

    /* both x and y are variables, and not the same */
    if (x->var_id != NOT_A_VARIABLE && y->var_id != NOT_A_VARIABLE &&
        x->var_id != y->var_id)
    {
        memcpy(node->wsum_hess->x, w, node->size * sizeof(double));
        memcpy(node->wsum_hess->x + node->size, w, node->size * sizeof(double));
    }
    else
    {
        bool is_x_affine = x->is_affine(x);
        bool is_y_affine = y->is_affine(y);
        // ----------------------------------------------------------------------
        //  Refresh each operand's CSC_matrix cache as needed for the (Sparse, Sparse)
        //  dispatch path. For PD operands, refresh_csc_values is a no-op. The
        //  jacobian_csc_filled flag preserves the affine optimization: we only
        //  refresh on the first eval for affine children.
        // ----------------------------------------------------------------------
        if (!x->work->jacobian_csc_filled)
        {
            x->jacobian->refresh_csc_values(x->jacobian);
            if (is_x_affine)
            {
                x->work->jacobian_csc_filled = true;
            }
        }
        if (!y->work->jacobian_csc_filled)
        {
            y->jacobian->refresh_csc_values(y->jacobian);
            if (is_y_affine)
            {
                y->work->jacobian_csc_filled = true;
            }
        }

        // ---------------------------------------------------------------
        //                    compute C and CT
        // ---------------------------------------------------------------
        elementwise_mult_expr *mul_node = (elementwise_mult_expr *) node;
        CSR_matrix *CT = mul_node->CSR_work2;
        BTDA_matrices_fill_values(x->jacobian, w, y->jacobian, mul_node->cross_C);
        AT_fill_values(mul_node->cross_C->to_csr(mul_node->cross_C), CT,
                       node->work->iwork);

        // ---------------------------------------------------------------
        //              compute term2 and term 3
        // ---------------------------------------------------------------
        if (!is_x_affine)
        {
            for (int i = 0; i < node->size; i++)
            {
                node->work->dwork[i] = w[i] * y->value[i];
            }
            x->eval_wsum_hess(x, node->work->dwork);
        }

        if (!is_y_affine)
        {
            for (int i = 0; i < node->size; i++)
            {
                node->work->dwork[i] = w[i] * x->value[i];
            }
            y->eval_wsum_hess(y, node->work->dwork);
        }

        // ---------------------------------------------------------------
        //        compute H = C + C^T + term2 + term3
        // ---------------------------------------------------------------
        memset(node->wsum_hess->x, 0, node->wsum_hess->nnz * sizeof(double));
        accumulator(mul_node->cross_C->x, mul_node->cross_C->nnz,
                    mul_node->idx_map_C, node->wsum_hess->x);
        accumulator(CT->x, CT->nnz, mul_node->idx_map_CT, node->wsum_hess->x);
        accumulator(x->wsum_hess->x, x->wsum_hess->nnz, mul_node->idx_map_Hx,
                    node->wsum_hess->x);
        accumulator(y->wsum_hess->x, y->wsum_hess->nnz, mul_node->idx_map_Hy,
                    node->wsum_hess->x);
    }
}

static void free_type_data(expr *node)
{
    elementwise_mult_expr *mul_node = (elementwise_mult_expr *) node;
    free_matrix(mul_node->cross_C);
    free_CSR_matrix(mul_node->CSR_work2);
    free(mul_node->idx_map_C);
    free(mul_node->idx_map_CT);
    free(mul_node->idx_map_Hx);
    free(mul_node->idx_map_Hy);
}

static bool is_affine(const expr *node)
{
    (void) node;
    return false;
}

expr *new_elementwise_mult(expr *left, expr *right)
{
    elementwise_mult_expr *mul_node =
        (elementwise_mult_expr *) SP_CALLOC(1, sizeof(elementwise_mult_expr));
    expr *node = &mul_node->base;

    init_expr(node, left->d1, left->d2, left->n_vars, forward, jacobian_init_impl,
              eval_jacobian, is_affine, wsum_hess_init_impl, eval_wsum_hess,
              free_type_data);
    node->left = left;
    node->right = right;
    expr_retain(left);
    expr_retain(right);

    return node;
}
