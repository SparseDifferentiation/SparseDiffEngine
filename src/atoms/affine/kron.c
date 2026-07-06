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
#include <assert.h>
#include <string.h>

/* Kronecker product Z = kron(A, B), where one operand is variable-free (held by
 * param_source) and the other (child = node->left) carries the variables.
 *
 * With column-major flattening, output entry OUT = (i*r + k) + (j*s + l)*(p*r)
 * equals A[i,j] * B[k,l], so each output entry depends on a single child entry:
 *
 *   Z[OUT]     = coeff[OUT] * vec(child)[child_row[OUT]]
 *   J[OUT, :]  = coeff[OUT] * J_child[child_row[OUT], :]
 *
 * where coeff[OUT] = param_source->value[coeff_idx[OUT]]. The constructors fill
 * child_row/coeff_idx only for the constant operand's active (nonzero) blocks;
 * the remaining rows keep child_row == -1 and are structurally zero. */

/* Pull current parameter values through any broadcast/promote wrappers. */
static void refresh_param_values(kron_expr *knode)
{
    if (!knode->base.needs_parameter_refresh)
    {
        return;
    }

    knode->param_source->forward(knode->param_source, NULL);
    knode->base.needs_parameter_refresh = false;
}

static void forward(expr *node, const double *u)
{
    expr *child = node->left;
    kron_expr *knode = (kron_expr *) node;

    refresh_param_values(knode);
    child->forward(child, u);

    const double *a = knode->param_source->value;
    const double *x = child->value;
    double *y = node->value;
    for (int out = 0; out < node->size; out++)
    {
        int cr = knode->child_row[out];
        y[out] = (cr < 0) ? 0.0 : a[knode->coeff_idx[out]] * x[cr];
    }
}

static void jacobian_init_impl(expr *node)
{
    expr *child = node->left;
    kron_expr *knode = (kron_expr *) node;

    jacobian_init(child);

    /* Row OUT of the result copies the sparsity of child row child_row[OUT]
       (with repetition); inactive rows are empty. */
    CSR_matrix *Jc = child->jacobian->to_csr(child->jacobian);

    int total = 0;
    for (int out = 0; out < node->size; out++)
    {
        int cr = knode->child_row[out];
        if (cr >= 0)
        {
            total += Jc->p[cr + 1] - Jc->p[cr];
        }
    }

    CSR_matrix *Jk = new_CSR_matrix(node->size, node->n_vars, total);
    int idx = 0;
    Jk->p[0] = 0;
    for (int out = 0; out < node->size; out++)
    {
        int cr = knode->child_row[out];
        if (cr >= 0)
        {
            for (int t = Jc->p[cr]; t < Jc->p[cr + 1]; t++)
            {
                Jk->i[idx++] = Jc->i[t];
            }
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

    /* Sparsity is fixed after jacobian_init, so the row offsets still align;
       refill active rows as scale * child-row-values. */
    CSR_matrix *Jc = child->jacobian->to_csr(child->jacobian);
    CSR_matrix *Jk = node->jacobian->to_csr(node->jacobian);
    const double *a = knode->param_source->value;

    int idx = 0;
    for (int out = 0; out < node->size; out++)
    {
        int cr = knode->child_row[out];
        if (cr < 0)
        {
            continue;
        }
        double scale = a[knode->coeff_idx[out]];
        for (int t = Jc->p[cr]; t < Jc->p[cr + 1]; t++)
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

    /* kron is affine in the child, so we only push the weights back through the
       gather: w'[child_row[OUT]] += coeff[OUT] * w[OUT]. Many output rows map to
       one child entry, hence the accumulation. */
    memset(w_prime, 0, child->size * sizeof(double));
    for (int out = 0; out < node->size; out++)
    {
        int cr = knode->child_row[out];
        if (cr >= 0)
        {
            w_prime[cr] += a[knode->coeff_idx[out]] * w[out];
        }
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

    knode->child_row = NULL;
    knode->coeff_idx = NULL;
    knode->param_source = NULL;
}

/* Allocate a kron node and its (all-inactive) index arrays. The left/right
   constructors then fill the active rows. */
static kron_expr *new_kron_common(expr *param_node, expr *child, int p, int q, int r,
                                  int s)
{
    int size_out = (p * r) * (q * s);

    kron_expr *knode = (kron_expr *) sp_calloc(1, sizeof(kron_expr));
    expr *node = &knode->base;
    init_expr(node, p * r, q * s, child->n_vars, forward, jacobian_init_impl,
              eval_jacobian, is_affine, wsum_hess_init_impl, eval_wsum_hess,
              free_type_data);
    node->left = child;
    expr_retain(child);

    knode->param_source = param_node;
    expr_retain(param_node);

    knode->child_row = (int *) sp_malloc(size_out * sizeof(int));
    knode->coeff_idx = (int *) sp_malloc(size_out * sizeof(int));
    for (int out = 0; out < size_out; out++)
    {
        knode->child_row[out] = -1; /* inactive until an active block fills it */
    }

    knode->base.needs_parameter_refresh = true;
    return knode;
}

/* Z = kron(A, B) with A = param_node (p x q) the constant, B = child (r x s) the
   variable. active_blocks holds column-major indices i + j*p of A's nonzeros. */
expr *new_left_kron(expr *param_node, expr *child, int p, int q, int r, int s,
                    const int *active_blocks, int n_active)
{
    kron_expr *knode = new_kron_common(param_node, child, p, q, r, s);
    int n_rows = p * r;
    for (int b = 0; b < n_active; b++)
    {
        int bidx = active_blocks[b]; /* = i + j*p into A */
        assert(0 <= bidx && bidx < p * q);
        int i = bidx % p;
        int j = bidx / p;
        for (int l = 0; l < s; l++)
        {
            for (int k = 0; k < r; k++)
            {
                int out = (i * r + k) + (j * s + l) * n_rows;
                knode->child_row[out] = k + l * r; /* col-major into B */
                knode->coeff_idx[out] = bidx;      /* col-major into A */
            }
        }
    }
    return &knode->base;
}

/* Z = kron(A, B) with A = child (p x q) the variable, B = param_node (r x s) the
   constant. active_blocks holds column-major indices k + l*r of B's nonzeros. */
expr *new_right_kron(expr *param_node, expr *child, int p, int q, int r, int s,
                     const int *active_blocks, int n_active)
{
    kron_expr *knode = new_kron_common(param_node, child, p, q, r, s);
    int n_rows = p * r;
    for (int b = 0; b < n_active; b++)
    {
        int bidx = active_blocks[b]; /* = k + l*r into B */
        assert(0 <= bidx && bidx < r * s);
        int k = bidx % r;
        int l = bidx / r;
        for (int j = 0; j < q; j++)
        {
            for (int i = 0; i < p; i++)
            {
                int out = (i * r + k) + (j * s + l) * n_rows;
                knode->child_row[out] = i + j * p; /* col-major into A */
                knode->coeff_idx[out] = bidx;      /* col-major into B */
            }
        }
    }
    return &knode->base;
}
