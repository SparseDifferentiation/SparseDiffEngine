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
#include "utils/CSR_matrix.h"
#include "utils/sparse_matrix.h"
#include "utils/tracked_alloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Kronecker product Z = kron(A, B), where exactly one operand is variable-free
 * (param_source) and the other (child = node->left) carries the variables.
 *
 * With column-major (Fortran) flattening, an output index OUT = I + J*(p*r)
 * decomposes as I = i*r + k and J = j*s + l (i in [0,p), k in [0,r), j in [0,q),
 * l in [0,s)). The output block (i, j) inner (k, l) equals A[i,j] * B[k,l], so
 * every output entry depends on a single child entry:
 *
 *   Z[OUT]        = coeff[OUT] * vec(child)[child_row[OUT]]
 *   J_kron[OUT,:] = coeff[OUT] * J_child[child_row[OUT], :]
 *
 * where coeff[OUT] = param_source->value[coeff_idx[OUT]]. child_row[] and
 * coeff_idx[] depend only on the shapes and are filled once in new_kron, so
 * forward, Jacobian and (affine) Hessian are all scaled gathers -- no
 * size_out x size_child coefficient matrix and no sparse matmul. */

static void forward(expr *node, const double *u)
{
    expr *child = node->left;
    kron_expr *knode = (kron_expr *) node;

    /* Pull current parameter values through any broadcast/promote wrappers. */
    if (knode->base.needs_parameter_refresh)
    {
        knode->param_source->forward(knode->param_source, NULL);
        knode->base.needs_parameter_refresh = false;
    }

    child->forward(child, u);

    const double *a = knode->param_source->value;
    const double *x = child->value;
    double *y = node->value;
    for (int out = 0; out < node->size; out++)
    {
        y[out] = a[knode->coeff_idx[out]] * x[knode->child_row[out]];
    }
}

static void jacobian_init_impl(expr *node)
{
    expr *child = node->left;
    kron_expr *knode = (kron_expr *) node;

    jacobian_init(child);

    /* Output row OUT shares the column set of child row child_row[OUT]. Build
       the result CSR sparsity by copying those child rows (with repetition). */
    CSR_matrix *Jc = child->jacobian->to_csr(child->jacobian);

    int total = 0;
    for (int out = 0; out < node->size; out++)
    {
        int cc = knode->child_row[out];
        total += Jc->p[cc + 1] - Jc->p[cc];
    }

    CSR_matrix *Jk = new_CSR_matrix(node->size, node->n_vars, total);
    int idx = 0;
    Jk->p[0] = 0;
    for (int out = 0; out < node->size; out++)
    {
        int cc = knode->child_row[out];
        for (int t = Jc->p[cc]; t < Jc->p[cc + 1]; t++)
        {
            Jk->i[idx++] = Jc->i[t];
        }
        Jk->p[out + 1] = idx;
    }
    node->jacobian = new_sparse_matrix(Jk);
}

static void eval_jacobian(expr *node)
{
    expr *child = node->left;
    kron_expr *knode = (kron_expr *) node;

    child->eval_jacobian(child);

    /* Child sparsity is fixed after jacobian_init, so the result row offsets
       still align; refill values as scale * child-row-values. */
    CSR_matrix *Jc = child->jacobian->to_csr(child->jacobian);
    CSR_matrix *Jk = node->jacobian->to_csr(node->jacobian);
    const double *a = knode->param_source->value;

    int idx = 0;
    for (int out = 0; out < node->size; out++)
    {
        int cc = knode->child_row[out];
        double scale = a[knode->coeff_idx[out]];
        for (int t = Jc->p[cc]; t < Jc->p[cc + 1]; t++)
        {
            Jk->x[idx++] = scale * Jc->x[t];
        }
    }
}

static void wsum_hess_init_impl(expr *node)
{
    expr *child = node->left;

    wsum_hess_init(child);
    node->wsum_hess = child->wsum_hess->copy_sparsity(child->wsum_hess);
    /* backprop workspace: one weight per child entry */
    node->work->dwork = (double *) sp_malloc(child->size * sizeof(double));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *child = node->left;
    kron_expr *knode = (kron_expr *) node;
    const double *a = knode->param_source->value;
    double *w_prime = node->work->dwork;

    /* kron is affine in child, so the Hessian is the child's with weights pushed
       back through the linear gather: w'[child_row] += coeff * w[OUT]. Many
       output rows map to one child entry, hence the accumulation. */
    memset(w_prime, 0, child->size * sizeof(double));
    for (int out = 0; out < node->size; out++)
    {
        w_prime[knode->child_row[out]] += a[knode->coeff_idx[out]] * w[out];
    }

    child->eval_wsum_hess(child, w_prime);
    memcpy(node->wsum_hess->x, child->wsum_hess->x,
           node->wsum_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    kron_expr *knode = (kron_expr *) node;
    sp_free(knode->child_row);
    sp_free(knode->coeff_idx);
    free_expr(knode->param_source);
}

expr *new_kron(expr *param_node, expr *child, int const_is_left, int p, int q,
               int r, int s)
{
    int d1 = p * r;
    int d2 = q * s;
    int size_out = d1 * d2;

    kron_expr *knode = (kron_expr *) sp_calloc(1, sizeof(kron_expr));
    expr *node = &knode->base;
    init_expr(node, d1, d2, child->n_vars, forward, jacobian_init_impl,
              eval_jacobian, is_affine, wsum_hess_init_impl, eval_wsum_hess,
              free_type_data);
    node->left = child;
    expr_retain(child);

    knode->param_source = param_node;
    expr_retain(param_node);
    knode->p = p;
    knode->q = q;
    knode->r = r;
    knode->s = s;
    knode->const_is_left = const_is_left;

    knode->child_row = (int *) sp_malloc(size_out * sizeof(int));
    knode->coeff_idx = (int *) sp_malloc(size_out * sizeof(int));

    int n_rows = p * r; /* number of output rows */
    for (int out = 0; out < size_out; out++)
    {
        int I = out % n_rows;
        int J = out / n_rows;
        int i = I / r, k = I % r;
        int j = J / s, l = J % s;
        if (const_is_left)
        {
            /* A = param (p x q), B = child (r x s) */
            knode->child_row[out] = k + l * r; /* col-major into B */
            knode->coeff_idx[out] = i + j * p; /* col-major into A */
        }
        else
        {
            /* A = child (p x q), B = param (r x s) */
            knode->child_row[out] = i + j * p; /* col-major into A */
            knode->coeff_idx[out] = k + l * r; /* col-major into B */
        }
    }

    knode->base.needs_parameter_refresh = true;
    return node;
}
