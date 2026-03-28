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
#include "utils/CSR_sum.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Scatter-add src->x into dest using precomputed index map */
static void accumulate_mapped(double *dest, const CSR_Matrix *src,
                              const int *idx_map)
{
    for (int j = 0; j < src->nnz; j++)
    {
        dest[idx_map[j]] += src->x[j];
    }
}

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
    node->jacobian = new_csr_matrix(node->size, node->n_vars, nnz_max);

    /* fill sparsity pattern */
    sum_csr_matrices_fill_sparsity(node->left->jacobian, node->right->jacobian,
                                   node->jacobian);
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    x->eval_jacobian(x);
    y->eval_jacobian(y);

    /* chain rule: the jacobian of h(x) = f(g1(x), g2(x))) is Jh = J_{f, 1} J_{g1} +
     * J_{f, 2} J_{g2} */
    sum_scaled_csr_matrices_fill_values(x->jacobian, y->jacobian, node->jacobian,
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
        node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, 2 * node->size);

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
            node->wsum_hess->p[var1_id + i] = i;
            node->wsum_hess->i[i] = var2_id + i;
        }

        int nnz = node->size;

        /* rows between var1 and var2 */
        for (i = var1_id + node->size; i < var2_id; i++)
        {
            node->wsum_hess->p[i] = nnz;
        }

        /* var2 rows of Hessian */
        for (i = 0; i < node->size; i++)
        {
            node->wsum_hess->p[var2_id + i] = nnz + i;
            node->wsum_hess->i[nnz + i] = var1_id + i;
        }

        /* remaining rows */
        nnz += node->size;
        for (i = var2_id + node->size; i <= node->n_vars; i++)
        {
            node->wsum_hess->p[i] = nnz;
        }
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
            node->work->dwork = (double *) malloc(node->size * sizeof(double));
        }

        /* prepare sparsity pattern of csc conversion */
        jacobian_csc_init(x);
        jacobian_csc_init(y);
        CSC_Matrix *Jg1 = x->work->jacobian_csc;
        CSC_Matrix *Jg2 = y->work->jacobian_csc;

        /* compute sparsity of C and prepare CT */
        CSR_Matrix *C = BTA_alloc(Jg1, Jg2);
        node->work->iwork = (int *) malloc(C->m * sizeof(int));
        CSR_Matrix *CT = AT_alloc(C, node->work->iwork);

        /* initialize wsum_hessians of children */
        wsum_hess_init(x);
        wsum_hess_init(y);

        elementwise_mult_expr *mul_node = (elementwise_mult_expr *) node;
        mul_node->CSR_work1 = C;
        mul_node->CSR_work2 = CT;

        /* compute sparsity pattern of H = C + C^T + term2 + term3 (we also
           fill index maps telling us where to accumulate each element of each
           matrix in the sum) */
        int *maps[4];
        node->wsum_hess = sum_4_csr_fill_sparsity_and_idx_maps(C, CT, x->wsum_hess,
                                                               y->wsum_hess, maps);
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
        //            convert Jacobians of children to CSC format
        //      (we only need to do this once if the child is affine)
        //      TODO: what if we have parameters? Should we set jacobian_csc_filled
        //      to false whenever parameters change value?
        // ----------------------------------------------------------------------
        if (!x->work->jacobian_csc_filled)
        {
            csr_to_csc_fill_values(x->jacobian, x->work->jacobian_csc,
                                   x->work->csc_work);

            if (is_x_affine)
            {
                x->work->jacobian_csc_filled = true;
            }
        }

        if (!y->work->jacobian_csc_filled)
        {
            csr_to_csc_fill_values(y->jacobian, y->work->jacobian_csc,
                                   y->work->csc_work);

            if (is_y_affine)
            {
                y->work->jacobian_csc_filled = true;
            }
        }

        CSC_Matrix *Jg1 = x->work->jacobian_csc;
        CSC_Matrix *Jg2 = y->work->jacobian_csc;

        // ---------------------------------------------------------------
        //                    compute C and CT
        // ---------------------------------------------------------------
        elementwise_mult_expr *mul_node = (elementwise_mult_expr *) node;
        CSR_Matrix *C = mul_node->CSR_work1;
        CSR_Matrix *CT = mul_node->CSR_work2;
        BTDA_fill_values(Jg1, Jg2, w, C);
        AT_fill_values(C, CT, node->work->iwork);

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
        accumulate_mapped(node->wsum_hess->x, C, mul_node->idx_map_C);
        accumulate_mapped(node->wsum_hess->x, CT, mul_node->idx_map_CT);
        accumulate_mapped(node->wsum_hess->x, x->wsum_hess, mul_node->idx_map_Hx);
        accumulate_mapped(node->wsum_hess->x, y->wsum_hess, mul_node->idx_map_Hy);
    }
}

static void free_type_data(expr *node)
{
    elementwise_mult_expr *mul_node = (elementwise_mult_expr *) node;
    free_csr_matrix(mul_node->CSR_work1);
    free_csr_matrix(mul_node->CSR_work2);
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
        (elementwise_mult_expr *) calloc(1, sizeof(elementwise_mult_expr));
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
