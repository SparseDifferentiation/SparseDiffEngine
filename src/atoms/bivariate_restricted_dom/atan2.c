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
#include "atoms/bivariate_restricted_dom.h"
#include "utils/sparse_matrix.h"
#include "utils/tracked_alloc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --------------------------------------------------------------------
// Elementwise atan2(y, x) (C argument order: left = y, right = x).
// Leaf-only: both arguments must be distinct variables of the same
// shape, so no chain rule is needed. The value has a branch cut on the
// negative x-axis, but the derivatives are smooth everywhere except at
// the origin:
//   d/dy = x / r2,  d/dx = -y / r2,          r2 = x^2 + y^2
//   d2/dy2 = -2xy / r4,  d2/dx2 = 2xy / r4,  d2/dxdy = (y^2 - x^2) / r4
// --------------------------------------------------------------------
static void forward(expr *node, const double *u)
{
    expr *y = node->left;
    expr *x = node->right;

    /* children's forward passes */
    y->forward(y, u);
    x->forward(x, u);

    /* local forward pass */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = atan2(y->value[i], x->value[i]);
    }
}

static void jacobian_init_impl(expr *node)
{
    CSR_matrix *jac = new_CSR_matrix(node->size, node->n_vars, 2 * node->size);

    expr *y = node->left;
    expr *x = node->right;

    /* the argument with the lower variable idx appears first in each row */
    if (y->var_id < x->var_id)
    {
        for (int j = 0; j < node->size; j++)
        {
            jac->i[2 * j] = j + y->var_id;
            jac->i[2 * j + 1] = j + x->var_id;
            jac->p[j] = 2 * j;
        }
    }
    else
    {
        for (int j = 0; j < node->size; j++)
        {
            jac->i[2 * j] = j + x->var_id;
            jac->i[2 * j + 1] = j + y->var_id;
            jac->p[j] = 2 * j;
        }
    }

    jac->p[node->size] = 2 * node->size;
    node->jacobian = new_sparse_matrix(jac);
}

static void eval_jacobian_impl(expr *node)
{
    double *y = node->left->value;
    double *x = node->right->value;
    double *jx = node->jacobian->x;

    if (node->left->var_id < node->right->var_id)
    {
        for (int i = 0; i < node->size; i++)
        {
            double r2 = x[i] * x[i] + y[i] * y[i];
            jx[2 * i] = x[i] / r2;
            jx[2 * i + 1] = -y[i] / r2;
        }
    }
    else
    {
        for (int i = 0; i < node->size; i++)
        {
            double r2 = x[i] * x[i] + y[i] * y[i];
            jx[2 * i] = -y[i] / r2;
            jx[2 * i + 1] = x[i] / r2;
        }
    }
}

static void wsum_hess_init_impl(expr *node)
{
    CSR_matrix *H = new_CSR_matrix(node->n_vars, node->n_vars, 4 * node->size);
    expr *y = node->left;
    expr *x = node->right;

    int i, var1_id, var2_id;

    if (y->var_id < x->var_id)
    {
        var1_id = y->var_id;
        var2_id = x->var_id;
    }
    else
    {
        var1_id = x->var_id;
        var2_id = y->var_id;
    }

    /* var1 rows of Hessian */
    for (i = 0; i < node->size; i++)
    {
        H->p[var1_id + i] = 2 * i;
        H->i[2 * i] = var1_id + i;
        H->i[2 * i + 1] = var2_id + i;
    }

    int nnz = 2 * node->size;

    /* rows between var1 and var2 */
    for (i = var1_id + node->size; i < var2_id; i++)
    {
        H->p[i] = nnz;
    }

    /* var2 rows of Hessian */
    for (i = 0; i < node->size; i++)
    {
        H->p[var2_id + i] = nnz + 2 * i;
    }
    memcpy(H->i + nnz, H->i, nnz * sizeof(int));

    /* remaining rows */
    for (i = var2_id + node->size; i <= node->n_vars; i++)
    {
        H->p[i] = 4 * node->size;
    }
    node->wsum_hess = new_sparse_matrix(H);
}

static void eval_wsum_hess_impl(expr *node, const double *w)
{
    double *y = node->left->value;
    double *x = node->right->value;
    double *hess = node->wsum_hess->x;
    int n = node->size;

    /* per element: hyy = -2xy/r4, hxx = 2xy/r4, hxy = (y^2 - x^2)/r4.
       Rows of var1 come first, then rows of var2; each row holds
       [H(var, var1), H(var, var2)]. */
    if (node->left->var_id < node->right->var_id)
    {
        /* var1 = y, var2 = x */
        for (int i = 0; i < n; i++)
        {
            double r2 = x[i] * x[i] + y[i] * y[i];
            double r4 = r2 * r2;
            double hxy = w[i] * (y[i] * y[i] - x[i] * x[i]) / r4;
            double hxy2 = w[i] * 2.0 * x[i] * y[i] / r4;
            hess[2 * i] = -hxy2;            /* hyy */
            hess[2 * i + 1] = hxy;          /* hyx */
            hess[2 * n + 2 * i] = hxy;      /* hxy */
            hess[2 * n + 2 * i + 1] = hxy2; /* hxx */
        }
    }
    else
    {
        /* var1 = x, var2 = y */
        for (int i = 0; i < n; i++)
        {
            double r2 = x[i] * x[i] + y[i] * y[i];
            double r4 = r2 * r2;
            double hxy = w[i] * (y[i] * y[i] - x[i] * x[i]) / r4;
            double hxy2 = w[i] * 2.0 * x[i] * y[i] / r4;
            hess[2 * i] = hxy2;              /* hxx */
            hess[2 * i + 1] = hxy;           /* hxy */
            hess[2 * n + 2 * i] = hxy;       /* hyx */
            hess[2 * n + 2 * i + 1] = -hxy2; /* hyy */
        }
    }
}

static bool is_affine(const expr *node)
{
    (void) node;
    return false;
}

expr *new_atan2(expr *y, expr *x)
{
    /* leaf-only: both arguments must be distinct variables of equal shape */
    if (y->var_id == NOT_A_VARIABLE || x->var_id == NOT_A_VARIABLE ||
        y->var_id == x->var_id || y->d1 != x->d1 || y->d2 != x->d2)
    {
        fprintf(stderr, "Error: Both arguments of atan2 must be distinct "
                        "variables of the same shape.\n");
        return NULL;
    }

    expr *node = (expr *) sp_calloc(1, sizeof(expr));
    init_expr(node, y->d1, y->d2, y->n_vars, forward, jacobian_init_impl,
              eval_jacobian_impl, is_affine, wsum_hess_init_impl,
              eval_wsum_hess_impl, NULL);
    node->left = y;
    node->right = x;
    expr_retain(y);
    expr_retain(x);
    return node;
}
