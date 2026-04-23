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
#include "subexpr.h"
#include "utils/CSR_Matrix.h"
#include "utils/linalg_sparse_matmuls.h"
#include "utils/mini_numpy.h"
#include "utils/tracked_alloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 1D full convolution: y = conv(a, child), where a is a length-m kernel
 * held by param_source and child is a length-n vector. Output length is
 * m + n - 1. In matrix form, y = T(a) @ child, where T(a) is the
 * (m+n-1) x n Toeplitz matrix with T(a)[k, j] = a[k-j] when 0 <= k-j < m.
 *
 * Forward and Hessian backprop (w' = T(a)^T w) are direct loops, avoiding
 * CSR traversal and AT materialization. Jacobian values go through the
 * existing block-left-mult path so composite children work correctly.
 *
 * When param_source is updatable, forward() refreshes T(a)'s CSR values
 * from the new kernel before running the convolution. The sparsity of
 * T(a) is kernel-independent (staircase), so only values are rewritten. */

static void forward(expr *node, const double *u)
{
    expr *child = node->left;
    convolve_expr *cnode = (convolve_expr *) node;

    if (cnode->base.needs_parameter_refresh)
    {
        cnode->param_source->forward(cnode->param_source, NULL);
        /* refresh the convolution matrix values if it exists (necessary to check
           for null in case someone calls forward before initializing the jacobian,
           which might happen when we expose Python bindings to SparseDiffEngine) */
        if (cnode->T != NULL)
        {
            conv_matrix_fill_values(cnode->T, cnode->param_source->value);
        }
        cnode->base.needs_parameter_refresh = false;
    }

    child->forward(child, u);

    const double *a = cnode->param_source->value;
    const double *x = child->value;
    double *y = node->value;

    memset(y, 0, node->size * sizeof(double));
    for (int j = 0; j < cnode->n; j++)
    {
        for (int i = 0; i < cnode->m; i++)
        {
            y[i + j] += a[i] * x[j];
        }
    }
}

static void jacobian_init_impl(expr *node)
{
    expr *child = node->left;
    convolve_expr *cnode = (convolve_expr *) node;
    int m = cnode->m;
    int n = cnode->n;
    const double *a = cnode->param_source->value;

    jacobian_init(child);

    /* Build convolution matrix of size (m+n-1) x n with m*n nonzeros */
    cnode->T = new_csr_matrix(m + n - 1, n, m * n);
    conv_matrix_fill_sparsity(cnode->T, m, n);
    conv_matrix_fill_values(cnode->T, a);

    /* J = T @ J_child */
    cnode->Jchild_CSC = csr_to_csc_alloc(child->jacobian, node->work->iwork);
    node->jacobian = csr_csc_matmul_alloc(cnode->T, cnode->Jchild_CSC);
}

static void eval_jacobian(expr *node)
{
    expr *child = node->left;
    convolve_expr *cnode = (convolve_expr *) node;

    child->eval_jacobian(child);

    /* J = T @ J_child */
    csr_to_csc_fill_values(child->jacobian, cnode->Jchild_CSC, node->work->iwork);
    csr_csc_matmul_fill_values(cnode->T, cnode->Jchild_CSC, node->jacobian);
}

static void wsum_hess_init_impl(expr *node)
{
    expr *child = node->left;
    convolve_expr *cnode = (convolve_expr *) node;

    wsum_hess_init(child);
    node->wsum_hess = new_csr_copy_sparsity(child->wsum_hess);
    node->work->dwork = (double *) SP_MALLOC(cnode->n * sizeof(double));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *child = node->left;
    convolve_expr *cnode = (convolve_expr *) node;
    int m = cnode->m;
    int n = cnode->n;
    const double *a = cnode->param_source->value;
    double *w_prime = node->work->dwork;

    /* w' = T(a)^T w. T(a)^T[j, k] = a[k-j] for j <= k < j+m, so
       w'[j] = sum_{i=0..m-1} a[i] * w[i + j]. */
    for (int j = 0; j < n; j++)
    {
        double sum = 0.0;
        for (int i = 0; i < m; i++)
        {
            sum += a[i] * w[i + j];
        }
        w_prime[j] = sum;
    }

    child->eval_wsum_hess(child, w_prime);
    memcpy(node->wsum_hess->x, child->wsum_hess->x,
           child->wsum_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    convolve_expr *cnode = (convolve_expr *) node;
    free_csr_matrix(cnode->T);
    free_csc_matrix(cnode->Jchild_CSC);
    free_expr(cnode->param_source);
}

expr *new_convolve(expr *param_node, expr *child)
{
    /* Accept both (n, 1) column and (1, n) row shapes — numpy-style 1D
       arrays reach us as (1, n) through the Python bindings, and the math
       is shape-agnostic (flat buffers of size m + n - 1). Output shape
       matches the input orientation. */
    int m = param_node->size;
    int n, out_d1, out_d2;
    if (child->d2 == 1)
    {
        n = child->d1;
        out_d1 = m + n - 1;
        out_d2 = 1;
    }
    else if (child->d1 == 1)
    {
        n = child->d2;
        out_d1 = 1;
        out_d2 = m + n - 1;
    }
    else
    {
        fprintf(stderr, "Error in new_convolve: child must be a vector "
                        "(shape (n, 1) or (1, n))\n");
        exit(1);
    }

    if (m < 1 || n < 1)
    {
        fprintf(stderr, "Error in new_convolve: m and n must be >= 1\n");
        exit(1);
    }

    convolve_expr *cnode = (convolve_expr *) SP_CALLOC(1, sizeof(convolve_expr));
    expr *node = &cnode->base;
    init_expr(node, out_d1, out_d2, child->n_vars, forward, jacobian_init_impl,
              eval_jacobian, is_affine, wsum_hess_init_impl, eval_wsum_hess,
              free_type_data);
    node->left = child;
    expr_retain(child);

    cnode->param_source = param_node;
    expr_retain(param_node);
    cnode->m = m;
    cnode->n = n;

    /* iwork is used for csr_to_csc of child's Jacobian (size n_vars). */
    node->work->iwork = (int *) SP_MALLOC(node->n_vars * sizeof(int));

    /* Ensure first forward() pulls current param values through any
       broadcast/promote wrappers and reflects them in T (once T is built). */
    cnode->base.needs_parameter_refresh = true;

    return node;
}
