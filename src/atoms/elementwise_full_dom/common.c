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
#include "atoms/elementwise_full_dom.h"
#include "subexpr.h"
#include "utils/CSC_matrix.h"
#include "utils/CSR_matrix.h"
#include "utils/matrix_sum.h"
#include "utils/sparse_matrix.h"
#include "utils/tracked_alloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void jacobian_init_elementwise(expr *node)
{
    expr *child = node->left;

    /* if the variable is a child */
    if (child->var_id != NOT_A_VARIABLE)
    {
        CSR_matrix *jac = new_csr_matrix(node->size, node->n_vars, node->size);
        for (int j = 0; j < node->size; j++)
        {
            jac->p[j] = j;
            jac->i[j] = j + child->var_id;
        }
        jac->p[node->size] = node->size;
        node->jacobian = new_sparse_matrix(jac);
    }
    else
    {
        /* jacobian of h(x) = f(g(x)) is Jf @ Jg, and here Jf is diagonal */
        jacobian_init(child);
        node->jacobian = child->jacobian->copy_sparsity(child->jacobian);
        node->work->dwork = (double *) SP_MALLOC(node->size * sizeof(double));
        node->work->local_jac_diag =
            (double *) SP_MALLOC(node->size * sizeof(double));
    }
}

void eval_jacobian_elementwise(expr *node)
{
    expr *child = node->left;

    if (child->var_id != NOT_A_VARIABLE)
    {
        node->local_jacobian(node, node->jacobian->x);
    }
    else
    {
        /* jacobian of h(x) = f(g(x)) is Jf @ Jg, and here Jf is diagonal */
        child->eval_jacobian(child);
        node->local_jacobian(node, node->work->local_jac_diag);
        memcpy(node->work->dwork, node->work->local_jac_diag,
               node->size * sizeof(double));
        child->jacobian->DA_fill_values(node->work->dwork, child->jacobian,
                                        node->jacobian);
    }
}

void wsum_hess_init_elementwise(expr *node)
{
    expr *child = node->left;
    int id = child->var_id;
    int i;

    /* if the variable is a child */
    if (id != NOT_A_VARIABLE)
    {
        CSR_matrix *hess = new_csr_matrix(node->n_vars, node->n_vars, node->size);

        for (i = 0; i < node->size; i++)
        {
            hess->p[id + i] = i;
            hess->i[i] = id + i;
        }

        for (i = id + node->size; i <= node->n_vars; i++)
        {
            hess->p[i] = node->size;
        }
        node->wsum_hess = new_sparse_matrix(hess);
    }
    else
    {
        /* Hessian of h(x) = w^T f(g(x) is term1 + term 2 where
            term1 = J_g^T @ D @ J_g with D = sum_i w_i Hf_i,
            term2 = sum_i (J_f^T w)_i^T Hg_i.

            For elementwise functions, D is diagonal. */
        if (child->is_affine(child))
        {
            node->wsum_hess = child->jacobian->ATA_alloc(child->jacobian);
        }
        else
        {
            /* term1: Jg^T @ D @ Jg */
            node->work->hess_term1 = child->jacobian->ATA_alloc(child->jacobian);

            /* term2: child's Hessian (mirror its sparsity polymorphically) */
            wsum_hess_init(child);
            node->work->hess_term2 =
                child->wsum_hess->copy_sparsity(child->wsum_hess);

            /* wsum_hess = term1 + term2 */
            int max_nnz =
                node->work->hess_term1->nnz + node->work->hess_term2->nnz;
            node->wsum_hess =
                new_sparse_matrix_alloc(node->n_vars, node->n_vars, max_nnz);
            sum_matrices_alloc(node->work->hess_term1, node->work->hess_term2,
                               node->wsum_hess);
        }
    }
}

void eval_wsum_hess_elementwise(expr *node, const double *w)
{
    expr *child = node->left;

    if (child->var_id != NOT_A_VARIABLE)
    {
        node->local_wsum_hess(node, node->wsum_hess->x, w);
    }
    else
    {
        if (child->is_affine(child))
        {
            /* Refresh the child Jacobian's CSC_matrix mirror once; subsequent calls
               skip since the affine child's values don't change. */
            if (!child->work->jacobian_csc_filled)
            {
                child->jacobian->refresh_csc_values(child->jacobian);
                child->work->jacobian_csc_filled = true;
            }

            node->local_wsum_hess(node, node->work->dwork, w);
            child->jacobian->ATDA_fill_values(child->jacobian, node->work->dwork,
                                              node->wsum_hess);
        }
        else
        {
            /* Non-affine child: values change every iteration, must refresh. */
            child->jacobian->refresh_csc_values(child->jacobian);

            /* term1: Jg^T @ D @ Jg */
            node->local_wsum_hess(node, node->work->dwork, w);
            child->jacobian->ATDA_fill_values(child->jacobian, node->work->dwork,
                                              node->work->hess_term1);

            /* term2: child Hessian with weight Jf^T w */
            memcpy(node->work->dwork, node->work->local_jac_diag,
                   node->size * sizeof(double));
            for (int k = 0; k < node->size; k++)
            {
                node->work->dwork[k] *= w[k];
            }

            child->eval_wsum_hess(child, node->work->dwork);
            memcpy(node->work->hess_term2->x, child->wsum_hess->x,
                   child->wsum_hess->nnz * sizeof(double));

            /* wsum_hess = term1 + term2 */
            sum_matrices_fill_values(node->work->hess_term1,
                                     node->work->hess_term2, node->wsum_hess);
        }
    }
}

bool is_affine_elementwise(const expr *node)
{
    (void) node;
    return false;
}

void init_elementwise(expr *node, expr *child)
{
    /* Initialize base fields */
    init_expr(node, child->d1, child->d2, child->n_vars, NULL,
              jacobian_init_elementwise, eval_jacobian_elementwise,
              is_affine_elementwise, wsum_hess_init_elementwise,
              eval_wsum_hess_elementwise, NULL);

    /* Set left child */
    node->left = child;
    expr_retain(child);
}

expr *new_elementwise(expr *child)
{
    expr *node = (expr *) SP_CALLOC(1, sizeof(expr));
    if (!node) return NULL;

    init_elementwise(node, child);
    return node;
}
