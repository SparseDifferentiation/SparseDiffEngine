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
#include "atoms/non_elementwise_full_dom.h"
#include "subexpr.h"
#include "utils/CSC_matrix.h"
#include "utils/matrix_sum.h"
#include "utils/cblas_wrapper.h"
#include "utils/sparse_matrix.h"
#include "utils/tracked_alloc.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    expr *x = node->left;

    /* child's forward pass */
    x->forward(x, u);

    /* local forward pass  */
    CSR_matrix *Q = ((quad_form_expr *) node)->Q;
    Ax_csr(Q, x->value, node->work->dwork, 0);
    node->value[0] = 0.0;

    for (int i = 0; i < x->size; i++)
    {
        node->value[0] += x->value[i] * node->work->dwork[i];
    }
}

static void jacobian_init_impl(expr *node)
{
    expr *x = node->left;

    if (x->var_id != NOT_A_VARIABLE)
    {
        CSR_matrix *jac = new_CSR_matrix(1, node->n_vars, x->size);
        jac->p[0] = 0;
        jac->p[1] = x->size;

        for (int j = 0; j < x->size; j++)
        {
            jac->i[j] = x->var_id + j;
        }
        node->jacobian = new_sparse_matrix(jac);
    }
    else
    {
        /* chain rule: J = 2 * (Q @ f(x))^T * J_f */
        jacobian_init(x);
        jacobian_csc_init(x);
        CSC_matrix *J_csc = x->work->jacobian_csc;

        /* allocate the right number of nnz */
        int nnz = count_nonzero_cols_csc(J_csc);
        CSR_matrix *jac = new_CSR_matrix(1, node->n_vars, nnz);
        jac->p[0] = 0;
        jac->p[1] = nnz;

        /* fill sparsity pattern */
        int idx = 0;
        for (int j = 0; j < J_csc->n; j++)
        {
            if (J_csc->p[j + 1] > J_csc->p[j])
            {
                jac->i[idx++] = j;
            }
        }
        node->jacobian = new_sparse_matrix(jac);
    }
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    CSR_matrix *Q = ((quad_form_expr *) node)->Q;
    CSR_matrix *jac = node->jacobian->to_csr(node->jacobian);

    if (x->var_id != NOT_A_VARIABLE)
    {
        /* jacobian = 2 * (Q @ x)^T */
        Ax_csr(Q, x->value, jac->x, 0);
        cblas_dscal(x->size, 2.0, jac->x, 1);
    }
    else
    {
        /* jacobian = 2 * (Q @ f(x))^T @ J_f */
        x->eval_jacobian(x);

        if (!x->work->jacobian_csc_filled)
        {
            csr_to_csc_fill_values(x->jacobian->to_csr(x->jacobian), x->work->jacobian_csc,
                                   x->work->csc_work);

            if (x->is_affine(x))
            {
                x->work->jacobian_csc_filled = true;
            }
        }

        /* The jacobian has same values as the gradient, which is
           J_f^T (Q @ f(x)). Here, dwork stores Q @ f(x) from forward */
        yTA_fill_values(x->work->jacobian_csc, node->work->dwork, jac);

        cblas_dscal(jac->nnz, 2.0, jac->x, 1);
    }
}

static void wsum_hess_init_impl(expr *node)
{
    CSR_matrix *Q = ((quad_form_expr *) node)->Q;
    expr *x = node->left;

    if (x->var_id != NOT_A_VARIABLE)
    {
        CSR_matrix *H = new_CSR_matrix(node->n_vars, node->n_vars, Q->nnz);

        /* set global row pointers */
        memcpy(H->p + x->var_id, Q->p, (x->size + 1) * sizeof(int));
        for (int i = x->var_id + x->size + 1; i <= node->n_vars; i++)
        {
            H->p[i] = Q->nnz;
        }

        /* set global column indices */
        for (int i = 0; i < Q->nnz; i++)
        {
            H->i[i] = Q->i[i] + x->var_id;
        }

        node->wsum_hess = new_sparse_matrix(H);
    }
    else
    {
        /* The hessian of h(x) = f(x)^T Q f(x) is term1 + term2 where

            * term1 = J_f^T Q J_f
            * term2 = sum_i (Qf(x))_i nabla^2 f_i.

            To compute term1, we first compute B = Q J_f and then compute term1
            = J_f^T B.
        */

        /* jacobian_csc_init(x) already called in jacobian_init */
        quad_form_expr *qnode = (quad_form_expr *) node;
        CSC_matrix *Jf = x->work->jacobian_csc;

        /* term1 = Jf^T W Jf = Jf^T B*/
        CSC_matrix *B = symBA_alloc(Q, Jf);
        qnode->QJf = B;
        node->work->hess_term1 = new_sparse_matrix(BTA_alloc(Jf, B));

        /* term2 = sum_i (Qf(x))_i nabla^2 f_i */
        wsum_hess_init(x);
        node->work->hess_term2 = x->wsum_hess->copy_sparsity(x->wsum_hess);

        /* hess = term1 + term2 */
        int max_nnz =
            node->work->hess_term1->nnz + node->work->hess_term2->nnz;
        node->wsum_hess =
            new_sparse_matrix_alloc(node->n_vars, node->n_vars, max_nnz);
        sum_matrices_alloc(node->work->hess_term1, node->work->hess_term2,
                           node->wsum_hess);
    }
}

static void eval_wsum_hess(expr *node, const double *w)
{
    CSR_matrix *Q = ((quad_form_expr *) node)->Q;
    expr *x = node->left;
    double two_w = 2.0 * w[0];

    if (x->var_id != NOT_A_VARIABLE)
    {
        /* TODO: do we want to compute this hessian only once (up to a scaling)?
         * Maybe unnecessary optimization. */
        memcpy(node->wsum_hess->x, Q->x, Q->nnz * sizeof(double));
        cblas_dscal(Q->nnz, two_w, node->wsum_hess->x, 1);
    }
    else
    {
        /* fill the CSC_matrix representation of the Jacobian of the child */
        CSC_matrix *Jf = x->work->jacobian_csc;
        if (!x->work->jacobian_csc_filled)
        {
            csr_to_csc_fill_values(x->jacobian->to_csr(x->jacobian), Jf, x->work->csc_work);

            if (x->is_affine(x))
            {
                x->work->jacobian_csc_filled = true;
            }
        }

        CSC_matrix *QJf = ((quad_form_expr *) node)->QJf;
        CSR_matrix *term1 = node->work->hess_term1->to_csr(node->work->hess_term1);

        /* term1 = J_f^T Q J_f = J_f^T B  */
        BA_fill_values(Q, Jf, QJf);
        BTDA_fill_values(Jf, QJf, NULL, term1);

        /* term2 */
        x->eval_wsum_hess(x, node->work->dwork);
        memcpy(node->work->hess_term2->x, x->wsum_hess->x,
               x->wsum_hess->nnz * sizeof(double));

        /* scale both terms by 2w */
        cblas_dscal(node->work->hess_term1->nnz, two_w, node->work->hess_term1->x, 1);
        cblas_dscal(node->work->hess_term2->nnz, two_w, node->work->hess_term2->x, 1);

        /* sum the two terms */
        sum_matrices_fill_values(node->work->hess_term1, node->work->hess_term2,
                                 node->wsum_hess);
    }
}

static void free_type_data(expr *node)
{
    quad_form_expr *qnode = (quad_form_expr *) node;
    free_CSR_matrix(qnode->Q);
    qnode->Q = NULL;
    if (qnode->QJf != NULL)
    {
        free_CSC_matrix(qnode->QJf);
        qnode->QJf = NULL;
    }
}

static bool is_affine(const expr *node)
{
    (void) node;
    /* TODO: it is affine (constant) if both children are constant */
    return false;
}

expr *new_quad_form(expr *left, CSR_matrix *Q)
{
    assert(left->d1 == 1 || left->d2 == 1); /* left must be a vector */
    quad_form_expr *qnode = (quad_form_expr *) SP_CALLOC(1, sizeof(quad_form_expr));
    expr *node = &qnode->base;

    init_expr(node, 1, 1, left->n_vars, forward, jacobian_init_impl, eval_jacobian,
              is_affine, wsum_hess_init_impl, eval_wsum_hess, free_type_data);
    node->left = left;
    expr_retain(left);

    /* Set type-specific field */
    qnode->Q = new_CSR_matrix(Q->m, Q->n, Q->nnz);
    copy_CSR_matrix(Q, qnode->Q);

    /* dwork stores the result of Q @ f(x) in the forward pass */
    node->work->dwork = (double *) SP_MALLOC(left->size * sizeof(double));
    return node;
}
