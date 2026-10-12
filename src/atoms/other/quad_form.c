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
#include "sparsediff_linalg/cblas_wrapper.h"
#include "sparsediff_linalg/matmul.h"
#include "sparsediff_linalg/matrix.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include "subexpr.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Quadratic form y = x'Qx. Sparse path: Q is a CSR matrix. Dense path: Q is an
   n x n permuted_dense (optionally parameter-fed). For a leaf x the Hessian 2Q is
   materialized directly; for a composition x = f(u) the Jacobian 2 (Q f(u))^T J_f
   and the chain rule J_f^T Q J_f are formed through the matrix dispatchers, with
   the child's Jacobian read as a handle of any kind. Q is assumed symmetric. */

/* Refresh Q from the parameter once per solve (no-op when Q is constant).
   Q is symmetric, so column-major == row-major and the copy is verbatim. */
static void refresh_param_values_qf(quad_form_expr *qnode)
{
    if (qnode->param_source == NULL || !qnode->base.needs_parameter_refresh)
    {
        return;
    }
    qnode->base.needs_parameter_refresh = false;
    memcpy(qnode->Q->x, qnode->param_source->value,
           (size_t) qnode->n * qnode->n * sizeof(double));
    matrix_values_changed(qnode->Q);
}

static void forward(expr *node, const double *u)
{
    quad_form_expr *qnode = (quad_form_expr *) node;
    expr *x = node->left;

    /* refresh Q from the parameter if needed (no-op on the constant/sparse path) */
    if (qnode->param_source != NULL && node->needs_parameter_refresh)
    {
        qnode->param_source->forward(qnode->param_source, NULL);
    }
    refresh_param_values_qf(qnode);

    /* child's forward pass */
    x->forward(x, u);

    /* dwork = Q @ x; value = x' (Q x) */
    matrix *Q = qnode->Q;
    Q->block_left_mult_vec(Q, x->value, node->work->dwork, 1);
    node->value[0] = cblas_ddot(qnode->n, x->value, 1, node->work->dwork, 1);
}

static void jacobian_init_impl(expr *node)
{
    quad_form_expr *qnode = (quad_form_expr *) node;
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
        /* chain rule: J = 2 (Q f(x))^T J_f, the dispatcher's B^T A with B = Q f(x)
           held as an n x 1 permuted_dense. The alloc derives the column pattern
           from J_f, whatever its kind; the result is a 1 x n_vars permuted_dense. */
        jacobian_init(x);
        qnode->Qx = new_permuted_dense_full(qnode->n, 1, NULL);
        node->jacobian = BTA_matrices_alloc(x->jacobian, qnode->Qx);
    }
}

static void eval_jacobian_impl(expr *node)
{
    quad_form_expr *qnode = (quad_form_expr *) node;
    expr *x = node->left;

    if (x->var_id != NOT_A_VARIABLE)
    {
        /* jacobian = 2 * (Q @ x)^T (leaf x: sparsity is the variable block) */
        CSR_matrix *jac = node->jacobian->to_csr(node->jacobian);
        matrix *Q = qnode->Q;
        Q->block_left_mult_vec(Q, x->value, jac->x, 1);
        cblas_dscal(qnode->n, 2.0, jac->x, 1);
    }
    else
    {
        /* jacobian = 2 (Q f(x))^T J_f; dwork holds Q f(x) from forward. The fill
           refreshes whatever cache it reads of J_f and bumps the jacobian. */
        eval_jacobian(x);
        memcpy(qnode->Qx->x, node->work->dwork, qnode->n * sizeof(double));
        matrix_values_changed(qnode->Qx);
        BTA_matrices_fill_values(x->jacobian, qnode->Qx, node->jacobian);
        cblas_dscal(node->jacobian->nnz, 2.0, node->jacobian->x, 1);
    }
}

/* Sparse-backend hessian. The non-leaf chain rule forms term1 = J_f^T Q J_f as
   (Q J_f)^T J_f through the matrix dispatchers, like the dense path. Q is
   symmetric, so Q J_f = Q^T J_f is the dispatcher's B^T A with Q as the
   transposed operand; the output takes J_f's kind. */
static void wsum_hess_init_sparse(expr *node)
{
    quad_form_expr *qnode = (quad_form_expr *) node;
    CSR_matrix *Q = qnode->Q->to_csr(qnode->Q);
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

        /* QJf = Q J_f = Q^T J_f; term1 = (Q J_f)^T J_f */
        qnode->QJf = BTA_matrices_alloc(x->jacobian, qnode->Q);
        node->work->hess_term1 = BTA_matrices_alloc(x->jacobian, qnode->QJf);

        /* term2 = sum_i (Qf(x))_i nabla^2 f_i */
        wsum_hess_init(x);
        node->work->hess_term2 = x->wsum_hess->copy_sparsity(x->wsum_hess);

        /* hess = term1 + term2 */
        int max_nnz = node->work->hess_term1->nnz + node->work->hess_term2->nnz;
        node->wsum_hess =
            new_sparse_matrix_alloc(node->n_vars, node->n_vars, max_nnz);
        sum_matrices_alloc(node->work->hess_term1, node->work->hess_term2,
                           node->wsum_hess);
    }
}

static void eval_wsum_hess_sparse(expr *node, const double *w)
{
    quad_form_expr *qnode = (quad_form_expr *) node;
    CSR_matrix *Q = qnode->Q->to_csr(qnode->Q);
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
        /* term1 = J_f^T Q J_f = (Q J_f)^T J_f, with Q J_f = Q^T J_f. The fills
           refresh the caches they read (J_f's each eval, Q's once) themselves. */
        BTA_matrices_fill_values(x->jacobian, qnode->Q, qnode->QJf);
        BTA_matrices_fill_values(x->jacobian, qnode->QJf, node->work->hess_term1);

        /* term2 */
        eval_wsum_hess(x, node->work->dwork);
        memcpy(node->work->hess_term2->x, x->wsum_hess->x,
               x->wsum_hess->nnz * sizeof(double));

        /* scale both terms by 2w */
        cblas_dscal(node->work->hess_term1->nnz, two_w, node->work->hess_term1->x,
                    1);
        matrix_values_changed(node->work->hess_term1);
        cblas_dscal(node->work->hess_term2->nnz, two_w, node->work->hess_term2->x,
                    1);
        matrix_values_changed(node->work->hess_term2);

        /* sum the two terms */
        sum_matrices_fill_values(node->work->hess_term1, node->work->hess_term2,
                                 node->wsum_hess);
    }
}

/* Dense-backend hessian. Leaf x: 2wQ materialized as a permuted_dense block (the
   fast common case). Composition x = f(u): the chain rule
       H = J_f^T (2w Q) J_f  +  sum_i (2w Q f(u))_i nabla^2 f_i  =  term1 + term2,
   with term1 formed via the PD matmul dispatchers (Q symmetric PD so QJf = Q J_f
   is PD; J_f^T Q J_f = (Q J_f)^T J_f keeps the PD operand on the dispatch key). */
static void wsum_hess_init_dense(expr *node)
{
    quad_form_expr *qnode = (quad_form_expr *) node;
    expr *x = node->left;
    int n = qnode->n;

    if (x->var_id != NOT_A_VARIABLE)
    {
        /* Hessian is the dense block 2Q over x's contiguous variable range. */
        int *perm = (int *) sp_malloc(n * sizeof(int));
        for (int i = 0; i < n; i++)
        {
            perm[i] = x->var_id + i;
        }
        node->wsum_hess =
            new_permuted_dense(node->n_vars, node->n_vars, n, n, perm, perm, NULL);
        sp_free(perm);
    }
    else
    {
        /* term1 = J_f^T Q J_f = (Q J_f)^T J_f. QJf is PD; passing it as the
           transposed operand B keeps the PD type on the dispatch key. */
        qnode->QJf = BA_pd_matrices_alloc(qnode->Q, x->jacobian);
        node->work->hess_term1 = BTA_matrices_alloc(x->jacobian, qnode->QJf);

        /* term2 = sum_i (Q f(x))_i nabla^2 f_i */
        wsum_hess_init(x);
        node->work->hess_term2 = x->wsum_hess->copy_sparsity(x->wsum_hess);

        /* hess = term1 + term2 (CSR-backed; sum_matrices is type-agnostic) */
        int max_nnz = node->work->hess_term1->nnz + node->work->hess_term2->nnz;
        node->wsum_hess =
            new_sparse_matrix_alloc(node->n_vars, node->n_vars, max_nnz);
        sum_matrices_alloc(node->work->hess_term1, node->work->hess_term2,
                           node->wsum_hess);
    }
}

static void eval_wsum_hess_dense(expr *node, const double *w)
{
    quad_form_expr *qnode = (quad_form_expr *) node;
    expr *x = node->left;
    double two_w = 2.0 * w[0];

    if (x->var_id != NOT_A_VARIABLE)
    {
        int nn = qnode->n * qnode->n;
        /* Hessian = 2 w Q (Q symmetric, constant up to the weight). The PD's value
           buffer (->x) aliases its dense block, so writing it updates to_csr too. */
        memcpy(node->wsum_hess->x, qnode->Q->x, nn * sizeof(double));
        cblas_dscal(nn, two_w, node->wsum_hess->x, 1);
    }
    else
    {
        /* term1 = 2w J_f^T Q J_f = 2w (Q J_f)^T J_f */
        BA_pd_matrices_fill_values(qnode->Q, x->jacobian, qnode->QJf);
        BTA_matrices_fill_values(x->jacobian, qnode->QJf, node->work->hess_term1);
        cblas_dscal(node->work->hess_term1->nnz, two_w, node->work->hess_term1->x,
                    1);
        matrix_values_changed(node->work->hess_term1);

        /* term2 = 2w sum_i (Q f(x))_i nabla^2 f_i (dwork = Q f(x) from forward) */
        eval_wsum_hess(x, node->work->dwork);
        memcpy(node->work->hess_term2->x, x->wsum_hess->x,
               x->wsum_hess->nnz * sizeof(double));
        cblas_dscal(node->work->hess_term2->nnz, two_w, node->work->hess_term2->x,
                    1);
        matrix_values_changed(node->work->hess_term2);

        sum_matrices_fill_values(node->work->hess_term1, node->work->hess_term2,
                                 node->wsum_hess);
    }
}

/* param_source lives outside left/right, so the refresh walk reaches it
   here -- and reports whether it actually holds an updatable parameter. */
static bool set_needs_refresh_param_source(expr *node)
{
    return expr_set_needs_refresh(((quad_form_expr *) node)->param_source);
}

static void free_type_data(expr *node)
{
    quad_form_expr *qnode = (quad_form_expr *) node;
    free_matrix(qnode->Q);
    qnode->Q = NULL;
    free_matrix(qnode->QJf);
    qnode->QJf = NULL;
    free_matrix(qnode->Qx);
    qnode->Qx = NULL;
    free_expr(qnode->param_source);
    qnode->param_source = NULL;
}

static bool is_affine(const expr *node)
{
    (void) node;
    /* TODO: it is affine (constant) if both children are constant */
    return false;
}

expr *new_quad_form_sparse(expr *left, CSR_matrix *Q)
{
    assert(left->d1 == 1 || left->d2 == 1); /* left must be a vector */
    quad_form_expr *qnode = (quad_form_expr *) sp_calloc(1, sizeof(quad_form_expr));
    expr *node = &qnode->base;

    init_expr(node, 1, 1, left->n_vars, forward, jacobian_init_impl,
              eval_jacobian_impl, is_affine, wsum_hess_init_sparse,
              eval_wsum_hess_sparse, free_type_data);
    node->left = left;
    expr_retain(left);

    /* Set type-specific field. new_sparse_matrix takes ownership, so clone. */
    qnode->Q = new_sparse_matrix(new_csr(Q));
    qnode->n = left->size; /* quadratic dimension; used by the shared forward */

    /* dwork stores the result of Q @ f(x) in the forward pass */
    node->work->dwork = (double *) sp_malloc(left->size * sizeof(double));
    return node;
}

expr *new_quad_form_dense(expr *child, int n, const double *P_data,
                          expr *param_source)
{
    assert(child->d1 == 1 || child->d2 == 1); /* child must be a vector */
    assert(child->size == n);

    quad_form_expr *qnode = (quad_form_expr *) sp_calloc(1, sizeof(quad_form_expr));
    expr *node = &qnode->base;

    init_expr(node, 1, 1, child->n_vars, forward, jacobian_init_impl,
              eval_jacobian_impl, is_affine, wsum_hess_init_dense,
              eval_wsum_hess_dense, free_type_data);
    node->left = child;
    expr_retain(child);

    qnode->n = n;
    /* dwork stores Q @ x in the forward pass */
    node->work->dwork = (double *) sp_malloc(n * sizeof(double));

    qnode->param_source = param_source;
    if (param_source != NULL)
    {
        if (P_data != NULL)
        {
            fprintf(stderr, "Error in new_quad_form_dense: param and data both "
                            "set\n");
            exit(1);
        }

        expr_retain(param_source);

        /* Q is filled from the parameter on the first forward pass. */
        qnode->Q = new_permuted_dense_full(n, n, NULL);
        node->needs_parameter_refresh = true;
        node->set_needs_refresh_children = set_needs_refresh_param_source;
    }
    else
    {
        if (P_data == NULL)
        {
            fprintf(stderr, "Error in new_quad_form_dense: need P data\n");
            exit(1);
        }

        qnode->Q = new_permuted_dense_full(n, n, P_data);
    }

    return node;
}
