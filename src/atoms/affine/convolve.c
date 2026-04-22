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
#include "utils/matrix.h"
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

/* Fill T(a)'s CSR row pointers, column indices, and values from kernel a.
   Row r holds T[r, col] = a[r - col] for col in
   [max(0, r-m+1), min(n-1, r)]. */
static void build_toeplitz(CSR_Matrix *T_csr, const double *a, int m, int n)
{
    int out_rows = m + n - 1;
    int nnz = 0;
    for (int r = 0; r < out_rows; r++)
    {
        T_csr->p[r] = nnz;
        int col_lo = r - m + 1 > 0 ? r - m + 1 : 0;
        int col_hi = r < n - 1 ? r : n - 1;
        for (int col = col_lo; col <= col_hi; col++)
        {
            T_csr->i[nnz] = col;
            T_csr->x[nnz] = a[r - col];
            nnz++;
        }
    }
    T_csr->p[out_rows] = nnz;
}

/* Values-only refresh analogous to refresh_dense_left in left_matmul.c:
   sparsity is kernel-independent so p/i stay intact; we just walk the
   existing CSR entries and overwrite x from the current kernel. */
static void refresh_toeplitz_values(convolve_expr *cnode)
{
    CSR_Matrix *T_csr = ((Sparse_Matrix *) cnode->T)->csr;
    const double *a = cnode->param_source->value;
    int out_rows = cnode->m + cnode->n - 1;
    for (int r = 0; r < out_rows; r++)
    {
        for (int k = T_csr->p[r]; k < T_csr->p[r + 1]; k++)
        {
            T_csr->x[k] = a[r - T_csr->i[k]];
        }
    }
}

static void forward(expr *node, const double *u)
{
    expr *child = node->left;
    convolve_expr *cnode = (convolve_expr *) node;

    if (cnode->base.needs_parameter_refresh)
    {
        cnode->param_source->forward(cnode->param_source, NULL);
        /* T is NULL until jacobian_init_impl runs; forward can be called
           first in standalone tests, in which case T is built later from
           the already-refreshed param values. */
        if (cnode->T != NULL)
        {
            refresh_toeplitz_values(cnode);
        }
        cnode->base.needs_parameter_refresh = false;
    }

    child->forward(child, u);

    const double *a = cnode->param_source->value;
    const double *x = child->value;
    double *y = node->value;
    int m = cnode->m;
    int n = cnode->n;

    memset(y, 0, node->size * sizeof(double));
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
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

    /* Build T(a) as CSR: (m+n-1) x n, m*n nonzeros, staircase pattern. */
    int out_rows = m + n - 1;
    CSR_Matrix *T_csr = new_csr_matrix(out_rows, n, m * n);
    build_toeplitz(T_csr, a, m, n);

    cnode->T = new_sparse_matrix(T_csr);
    free_csr_matrix(T_csr);

    /* Reuse left_matmul's sparsity precompute: J_node = T @ J_child via
       block_left_mult with n_blocks = 1. */
    cnode->Jchild_CSC = csr_to_csc_alloc(child->jacobian, node->work->iwork);
    cnode->J_CSC =
        cnode->T->block_left_mult_sparsity(cnode->T, cnode->Jchild_CSC, 1);
    node->jacobian = csc_to_csr_alloc(cnode->J_CSC, cnode->csc_to_csr_work);
}

static void eval_jacobian(expr *node)
{
    expr *child = node->left;
    convolve_expr *cnode = (convolve_expr *) node;

    child->eval_jacobian(child);

    /* T values have been refreshed by forward() if the kernel changed; here
       we just need to refresh child's Jacobian and rerun the block matmul. */
    csr_to_csc_fill_values(child->jacobian, cnode->Jchild_CSC, node->work->iwork);
    cnode->T->block_left_mult_values(cnode->T, cnode->Jchild_CSC, cnode->J_CSC);
    csc_to_csr_fill_values(cnode->J_CSC, node->jacobian, cnode->csc_to_csr_work);
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
    free_matrix(cnode->T);
    free_csc_matrix(cnode->Jchild_CSC);
    free_csc_matrix(cnode->J_CSC);
    free(cnode->csc_to_csr_work);
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
    cnode->csc_to_csr_work = (int *) SP_MALLOC(node->size * sizeof(int));

    /* Ensure first forward() pulls current param values through any
       broadcast/promote wrappers and reflects them in T (once T is built). */
    cnode->base.needs_parameter_refresh = true;

    return node;
}
