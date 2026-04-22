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
#include "utils/tracked_alloc.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Kronecker product with constant on the left: Z = kron(C, X) where
 *   C has shape (m, n) and is a constant sparse matrix,
 *   X has shape (p, q) and is an expression.
 * Output Z has shape (m*p, n*q), stored column-major as vec(Z) of length
 * m*p*n*q.
 *
 * Key identity: Z[i*p+k, j*q+l] = C[i,j] * X[k,l].
 * In column-major: vec(Z)[r] with r = (j*q+l)*(m*p) + i*p + k
 *   depends on vec(X)[s] with s = l*p + k and coefficient C[i,j].
 *
 * The atom is affine in X: each output row r (when C[i,j] != 0) is a
 * scaled copy of child row s of the child's Jacobian, and the weighted
 * Hessian inherits the child's sparsity with an adjoint accumulation
 * over the same index pattern.
 *
 * All inner loops iterate only over nonzeros of C (cached in the
 * active_i / active_j / active_idx tables at construction). No explicit
 * identity-detection is needed: for C = I_m, nnz_C == m and the work
 * naturally drops to O(m * p * q) without any special-case code. */

/* ------------------------------------------------------------------ */
/*                          Forward pass                              */
/* ------------------------------------------------------------------ */
static void forward(expr *node, const double *u)
{
    kron_left_expr *lnode = (kron_left_expr *) node;
    expr *child = node->left;
    CSR_Matrix *C = lnode->C;
    int p = lnode->p, q = lnode->q;
    int mp = C->m * p;

    child->forward(child, u);

    memset(node->value, 0, (size_t) node->size * sizeof(double));

    /* For each nonzero C[i,j], scatter the (p x q) block cij * X into
     * position Z[i*p .. i*p+p-1, j*q .. j*q+q-1]. */
    for (int t = 0; t < lnode->n_active; t++)
    {
        int i = lnode->active_i[t];
        int j = lnode->active_j[t];
        double cij = C->x[lnode->active_idx[t]];
        for (int l = 0; l < q; l++)
        {
            int z_col_start = (j * q + l) * mp + i * p;
            int x_col_start = l * p;
            for (int k = 0; k < p; k++)
            {
                node->value[z_col_start + k] = cij * child->value[x_col_start + k];
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*                          Affine check                              */
/* ------------------------------------------------------------------ */
static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

/* ------------------------------------------------------------------ */
/*                      Jacobian initialization                       */
/* ------------------------------------------------------------------ */
/* Two-pass construction over active C entries × (l, k):
 *   pass 1 fills row_nnz[r] for every active output row,
 *   pass 2 writes column indices into the already-allocated CSR.
 * Rows r that don't correspond to an active (i, j) stay at 0 nnz.
 *
 * Work: O(nnz_C * p * q * avg_nnz_per_Jchild_row). For C = I_m this is
 * O(m * p * q * avg_Jchild_row_nnz), i.e. a factor-of-n reduction vs a
 * naive iteration over every output row of Z. */
static void jacobian_init_impl(expr *node)
{
    kron_left_expr *lnode = (kron_left_expr *) node;
    expr *child = node->left;
    CSR_Matrix *C = lnode->C;
    int p = lnode->p, q = lnode->q;
    int mp = C->m * p;
    int out_size = node->size;

    jacobian_init(child);
    CSR_Matrix *Jchild = child->jacobian;

    /* Pass 1: row_nnz[r] = Jchild row-nnz for active r, else 0. */
    int *row_nnz = (int *) SP_CALLOC((size_t) out_size, sizeof(int));
    for (int t = 0; t < lnode->n_active; t++)
    {
        int i = lnode->active_i[t];
        int j = lnode->active_j[t];
        for (int l = 0; l < q; l++)
        {
            int r_col_base = (j * q + l) * mp + i * p;
            for (int k = 0; k < p; k++)
            {
                int s = l * p + k;
                row_nnz[r_col_base + k] = Jchild->p[s + 1] - Jchild->p[s];
            }
        }
    }

    /* Cumulative sum into a local buffer; we'll memcpy into the
     * Jacobian's p[] after allocation. */
    int *Jp = (int *) SP_MALLOC((size_t) (out_size + 1) * sizeof(int));
    int total_nnz = 0;
    for (int r = 0; r < out_size; r++)
    {
        Jp[r] = total_nnz;
        total_nnz += row_nnz[r];
    }
    Jp[out_size] = total_nnz;
    free(row_nnz);

    node->jacobian = new_csr_matrix(out_size, node->n_vars, total_nnz);
    memcpy(node->jacobian->p, Jp, (size_t) (out_size + 1) * sizeof(int));
    free(Jp);

    /* Pass 2: column indices are a copy of the corresponding Jchild row. */
    for (int t = 0; t < lnode->n_active; t++)
    {
        int i = lnode->active_i[t];
        int j = lnode->active_j[t];
        for (int l = 0; l < q; l++)
        {
            int r_col_base = (j * q + l) * mp + i * p;
            for (int k = 0; k < p; k++)
            {
                int s = l * p + k;
                int r = r_col_base + k;
                int cs = Jchild->p[s];
                int row_nnz_r = Jchild->p[s + 1] - cs;
                int row_start = node->jacobian->p[r];
                memcpy(node->jacobian->i + row_start, Jchild->i + cs,
                       (size_t) row_nnz_r * sizeof(int));
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*                      Jacobian evaluation                           */
/* ------------------------------------------------------------------ */
static void eval_jacobian(expr *node)
{
    kron_left_expr *lnode = (kron_left_expr *) node;
    expr *child = node->left;
    CSR_Matrix *C = lnode->C;
    CSR_Matrix *Jchild = child->jacobian;
    CSR_Matrix *J = node->jacobian;
    int p = lnode->p, q = lnode->q;
    int mp = C->m * p;

    child->eval_jacobian(child);

    for (int t = 0; t < lnode->n_active; t++)
    {
        int i = lnode->active_i[t];
        int j = lnode->active_j[t];
        double cij = C->x[lnode->active_idx[t]];
        for (int l = 0; l < q; l++)
        {
            int r_col_base = (j * q + l) * mp + i * p;
            for (int k = 0; k < p; k++)
            {
                int s = l * p + k;
                int r = r_col_base + k;
                int cs = Jchild->p[s];
                int row_nnz_r = Jchild->p[s + 1] - cs;
                int row_start = J->p[r];
                for (int u = 0; u < row_nnz_r; u++)
                {
                    J->x[row_start + u] = cij * Jchild->x[cs + u];
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*                  Weighted-sum Hessian initialization               */
/* ------------------------------------------------------------------ */
static void wsum_hess_init_impl(expr *node)
{
    expr *child = node->left;

    wsum_hess_init(child);

    /* Linear in X: Hessian sparsity equals the child's. */
    node->wsum_hess = new_csr_copy_sparsity(child->wsum_hess);

    /* Workspace for the reverse-mode weight vector passed down to child. */
    node->work->dwork = (double *) SP_MALLOC((size_t) child->size * sizeof(double));
}

/* ------------------------------------------------------------------ */
/*                  Weighted-sum Hessian evaluation                   */
/* ------------------------------------------------------------------ */
static void eval_wsum_hess(expr *node, const double *w)
{
    kron_left_expr *lnode = (kron_left_expr *) node;
    expr *child = node->left;
    CSR_Matrix *C = lnode->C;
    int p = lnode->p, q = lnode->q;
    int mp = C->m * p;
    int child_size = child->size;
    double *w_child = node->work->dwork;

    /* Adjoint of the forward pass: w_child[s] = sum_{(i,j,k,l): s=l*p+k}
     *   C[i,j] * w[(j*q+l)*mp + i*p + k]. */
    memset(w_child, 0, (size_t) child_size * sizeof(double));
    for (int t = 0; t < lnode->n_active; t++)
    {
        int i = lnode->active_i[t];
        int j = lnode->active_j[t];
        double cij = C->x[lnode->active_idx[t]];
        for (int l = 0; l < q; l++)
        {
            int r_col_base = (j * q + l) * mp + i * p;
            for (int k = 0; k < p; k++)
            {
                int s = l * p + k;
                w_child[s] += cij * w[r_col_base + k];
            }
        }
    }

    child->eval_wsum_hess(child, w_child);
    memcpy(node->wsum_hess->x, child->wsum_hess->x,
           (size_t) node->wsum_hess->nnz * sizeof(double));
}

/* ------------------------------------------------------------------ */
/*                          Cleanup                                   */
/* ------------------------------------------------------------------ */
static void free_type_data(expr *node)
{
    kron_left_expr *lnode = (kron_left_expr *) node;
    free_csr_matrix(lnode->C);
    free(lnode->active_i);
    free(lnode->active_j);
    free(lnode->active_idx);
    if (lnode->param_source != NULL)
    {
        free_expr(lnode->param_source);
    }
    lnode->C = NULL;
    lnode->active_i = NULL;
    lnode->active_j = NULL;
    lnode->active_idx = NULL;
    lnode->param_source = NULL;
}

/* ------------------------------------------------------------------ */
/*                        Constructor                                 */
/* ------------------------------------------------------------------ */
expr *new_kron_left(expr *param_node, expr *u, const CSR_Matrix *C, int p, int q)
{
    if (u->size != p * q)
    {
        fprintf(stderr,
                "Error in new_kron_left: child size %d != p*q = %d*%d = %d\n",
                u->size, p, q, p * q);
        exit(1);
    }

    int m = C->m;
    int n = C->n;

    kron_left_expr *lnode = (kron_left_expr *) SP_CALLOC(1, sizeof(kron_left_expr));
    expr *node = &lnode->base;
    init_expr(node, m * p, n * q, u->n_vars, forward, jacobian_init_impl,
              eval_jacobian, is_affine, wsum_hess_init_impl, eval_wsum_hess,
              free_type_data);
    node->left = u;
    expr_retain(u);

    lnode->p = p;
    lnode->q = q;
    lnode->C = new_csr(C);

    /* Precompute active (i, j) tuples and their offset into C->x. */
    lnode->n_active = C->nnz;
    lnode->active_i = (int *) SP_MALLOC((size_t) C->nnz * sizeof(int));
    lnode->active_j = (int *) SP_MALLOC((size_t) C->nnz * sizeof(int));
    lnode->active_idx = (int *) SP_MALLOC((size_t) C->nnz * sizeof(int));
    int t = 0;
    for (int i = 0; i < m; i++)
    {
        for (int idx = C->p[i]; idx < C->p[i + 1]; idx++)
        {
            lnode->active_i[t] = i;
            lnode->active_j[t] = C->i[idx];
            lnode->active_idx[t] = idx;
            t++;
        }
    }
    assert(t == C->nnz);

    /* Parameter slot is reserved but not yet wired up. */
    lnode->param_source = param_node;
    if (param_node != NULL)
    {
        fprintf(stderr, "Error in new_kron_left: parameter for kron C "
                        "not supported yet\n");
        exit(1);
    }

    return node;
}
