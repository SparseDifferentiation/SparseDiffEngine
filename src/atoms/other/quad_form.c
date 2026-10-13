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
#include "sparsediff_linalg/matmul_kron.h"
#include "sparsediff_linalg/matrix.h"
#include "sparsediff_linalg/matrix_sum.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include "subexpr.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Quadratic form y = x'Qx, with Q a sparse_matrix (constant) or an n x n
   permuted_dense (constant or parameter-fed). Q is assumed symmetric. */

/* Refresh Q from the parameter (no-op when Q is constant). */
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

    /* refresh Q from the parameter if needed */
    if (qnode->param_source != NULL && node->needs_parameter_refresh)
    {
        qnode->param_source->forward(qnode->param_source, NULL);
    }
    refresh_param_values_qf(qnode);

    /* child's forward pass */
    x->forward(x, u);

    /* dwork = Q @ x; value = x' (Q x) */
    matrix *Q = qnode->Q;
    BA_kron_vec_matrices(Q, x->value, node->work->dwork, 1);
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
        /* chain rule: J = 2 (Q f(x))^T J_f (J is a 1 x n_vars permuted_dense.) */
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
        /* jacobian = 2 (Q x)^T; dwork holds Q x from forward */
        memcpy(node->jacobian->x, node->work->dwork, qnode->n * sizeof(double));
        cblas_dscal(qnode->n, 2.0, node->jacobian->x, 1);
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

/* Leaf x: the structure of Q placed at x's variable block, in Q's kind. The
   values are a verbatim copy of Q's either way (a CSR's value order and a
   permuted_dense block's row-major order coincide), so the eval only copies. */
static matrix *embed_Q_at_variable(const quad_form_expr *qnode, const expr *x)
{
    int n = qnode->n;
    int n_vars = qnode->base.n_vars;

    if (qnode->Q->kind == MATRIX_PERMUTED_DENSE)
    {
        int *perm = (int *) sp_malloc(n * sizeof(int));
        for (int i = 0; i < n; i++)
        {
            perm[i] = x->var_id + i;
        }
        matrix *H = new_permuted_dense(n_vars, n_vars, n, n, perm, perm, NULL);
        sp_free(perm);
        return H;
    }

    CSR_matrix *Q = qnode->Q->to_csr(qnode->Q);
    CSR_matrix *H = new_CSR_matrix(n_vars, n_vars, Q->nnz);

    /* rows before the block are empty (p[0..var_id] = 0 from calloc), the block
       reuses Q's row pointers, rows after it end at nnz */
    memcpy(H->p + x->var_id, Q->p, (n + 1) * sizeof(int));
    for (int i = x->var_id + n + 1; i <= n_vars; i++)
    {
        H->p[i] = Q->nnz;
    }
    for (int ii = 0; ii < Q->nnz; ii++)
    {
        H->i[ii] = Q->i[ii] + x->var_id;
    }
    return new_sparse_matrix(H);
}

static void wsum_hess_init_impl(expr *node)
{
    quad_form_expr *qnode = (quad_form_expr *) node;
    expr *x = node->left;

    if (x->var_id != NOT_A_VARIABLE)
    {
        node->wsum_hess = embed_Q_at_variable(qnode, x);
        return;
    }

    /* The hessian of h(x) = f(x)^T Q f(x) is term1 + term2 where
         term1 = J_f^T Q J_f = (Q J_f)^T J_f = (Q^T J_f)^T J_f
         term2 = sum_i (Q f(x))_i nabla^2 f_i. */
    qnode->QJf = BTA_matrices_alloc(x->jacobian, qnode->Q);
    node->work->hess_term1 = BTA_matrices_alloc(x->jacobian, qnode->QJf);

    wsum_hess_init(x);
    node->work->hess_term2 = copy_sparsity_matrices_alloc(x->wsum_hess);

    /* hess = term1 + term2 */
    int max_nnz = node->work->hess_term1->nnz + node->work->hess_term2->nnz;
    node->wsum_hess = new_sparse_matrix_alloc(node->n_vars, node->n_vars, max_nnz);
    sum_matrices_alloc(node->work->hess_term1, node->work->hess_term2,
                       node->wsum_hess);
}

static void eval_wsum_hess_impl(expr *node, const double *w)
{
    quad_form_expr *qnode = (quad_form_expr *) node;
    expr *x = node->left;
    double two_w = 2.0 * w[0];

    if (x->var_id != NOT_A_VARIABLE)
    {
        /* hessian = 2wQ  */
        matrix *Q = qnode->Q;
        memcpy(node->wsum_hess->x, Q->x, Q->nnz * sizeof(double));
        cblas_dscal(Q->nnz, two_w, node->wsum_hess->x, 1);
        return;
    }

    matrix *H1 = node->work->hess_term1;
    matrix *H2 = node->work->hess_term2;

    /* term1 = 2w (Q J_f)^T J_f. The fills refresh the caches they read (J_f's
       each eval, Q's only when Q changes) themselves. */
    BTA_matrices_fill_values(x->jacobian, qnode->Q, qnode->QJf);
    BTA_matrices_fill_values(x->jacobian, qnode->QJf, H1);
    cblas_dscal(H1->nnz, two_w, H1->x, 1);
    matrix_values_changed(H1);

    /* term2 = 2w sum_i (Q f(x))_i nabla^2 f_i (dwork = Q f(x) from forward) */
    eval_wsum_hess(x, node->work->dwork);
    memcpy(H2->x, x->wsum_hess->x, x->wsum_hess->nnz * sizeof(double));
    cblas_dscal(H2->nnz, two_w, H2->x, 1);
    matrix_values_changed(H2);

    sum_matrices_fill_values(H1, H2, node->wsum_hess);
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
    return false;
}

/* Shared constructor body. Takes ownership of Q; param_source may be NULL. */
static expr *init_quad_form(expr *child, matrix *Q, expr *param_source)
{
    assert(child->d1 == 1 || child->d2 == 1); /* child must be a vector */
    quad_form_expr *qnode = (quad_form_expr *) sp_calloc(1, sizeof(quad_form_expr));
    expr *node = &qnode->base;

    init_expr(node, 1, 1, child->n_vars, forward, jacobian_init_impl,
              eval_jacobian_impl, is_affine, wsum_hess_init_impl,
              eval_wsum_hess_impl, free_type_data);
    node->left = child;
    expr_retain(child);

    qnode->Q = Q;
    qnode->n = child->size; /* quadratic dimension */

    /* dwork stores Q @ x from the forward pass */
    node->work->dwork = (double *) sp_malloc(child->size * sizeof(double));

    qnode->param_source = param_source;
    if (param_source != NULL)
    {
        expr_retain(param_source);
        /* Q is filled from the parameter on the first forward pass. */
        node->needs_parameter_refresh = true;
        node->set_needs_refresh_children = set_needs_refresh_param_source;
    }
    return node;
}

expr *new_quad_form_sparse(expr *left, CSR_matrix *Q)
{
    /* new_sparse_matrix takes ownership, so clone. */
    return init_quad_form(left, new_sparse_matrix(new_csr(Q)), NULL);
}

expr *new_quad_form_dense(expr *child, int n, const double *P_data,
                          expr *param_source)
{
    assert(child->size == n);

    if (param_source != NULL && P_data != NULL)
    {
        fprintf(stderr, "Error in new_quad_form_dense: param and data both set\n");
        exit(1);
    }
    if (param_source == NULL && P_data == NULL)
    {
        fprintf(stderr, "Error in new_quad_form_dense: need P data\n");
        exit(1);
    }

    /* P_data == NULL leaves the block uninitialized until the first forward. */
    return init_quad_form(child, new_permuted_dense_full(n, n, P_data),
                          param_source);
}
