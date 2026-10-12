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
#include "sparsediff_linalg/matmul.h"
#include "sparsediff_linalg/matrix.h"
#include "sparsediff_linalg/matrix_sum.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include "subexpr.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------------------
// Implementation of quad-over-lin. The second argument will always be a variable
// that only appears in this node and as the left-hand side of another equality
// constraint.
// ------------------------------------------------------------------------------
static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    expr *y = node->right;

    /* children's forward passes */
    x->forward(x, u);
    y->forward(y, u);

    /* local forward pass */
    node->value[0] = 0.0;

    for (int i = 0; i < x->size; i++)
    {
        node->value[0] += x->value[i] * x->value[i];
    }

    node->value[0] /= y->value[0];
}

static void jacobian_init_impl(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* if left node is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        CSR_matrix *jac = new_CSR_matrix(1, node->n_vars, x->size + 1);
        jac->p[0] = 0;
        jac->p[1] = x->size + 1;

        /* if x has lower idx than y*/
        if (x->var_id < y->var_id)
        {
            for (int j = 0; j < x->size; j++)
            {
                jac->i[j] = x->var_id + j;
            }
            jac->i[x->size] = y->var_id;
        }
        else /* y has lower idx than x */
        {
            jac->i[0] = y->var_id;
            for (int j = 0; j < x->size; j++)
            {
                jac->i[j + 1] = x->var_id + j;
            }
        }
        node->jacobian = new_sparse_matrix(jac);
    }
    else /* left node is not a variable (an affine child x = f(u)) */
    {
        quad_over_lin_expr *qnode = (quad_over_lin_expr *) node;
        jacobian_init(x); /* no-op for a linear operator with a preset jacobian */

        /* x-part: (2 f(u)/y)^T J_f = d^T J_f through the dispatcher; the alloc
           derives the column pattern from J_f, whatever its kind. */
        qnode->d = new_permuted_dense_full(x->size, 1, NULL);
        qnode->xpart = BTA_matrices_alloc(x->jacobian, qnode->d);

        /* y-part: a single entry at y's column (y never appears in f(u)) */
        CSR_matrix *ey = new_CSR_matrix(1, node->n_vars, 1);
        ey->p[0] = 0;
        ey->p[1] = 1;
        ey->i[0] = y->var_id;
        qnode->ypart = new_sparse_matrix(ey);

        node->jacobian =
            new_sparse_matrix_alloc(1, node->n_vars, qnode->xpart->nnz + 1);
        sum_matrices_alloc(qnode->xpart, qnode->ypart, node->jacobian);
    }
}

static void eval_jacobian_impl(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        CSR_matrix *jac = node->jacobian->to_csr(node->jacobian);

        /* if x has lower idx than y*/
        if (x->var_id < y->var_id)
        {
            for (int j = 0; j < x->size; j++)
            {
                jac->x[j] = (2.0 * x->value[j]) / y->value[0];
            }
            jac->x[x->size] = -node->value[0] / y->value[0];
        }
        else /* y has lower idx than x */
        {
            jac->x[0] = -node->value[0] / y->value[0];
            for (int j = 0; j < x->size; j++)
            {
                jac->x[j + 1] = (2.0 * x->value[j]) / y->value[0];
            }
        }
    }
    else /* x is not a variable */
    {
        quad_over_lin_expr *qnode = (quad_over_lin_expr *) node;
        eval_jacobian(x); /* affine child: the wrapper short-circuits after once */

        /* x-part: d = 2 f(u)/y, then d^T J_f */
        double *d = qnode->d->x;
        for (int j = 0; j < x->size; j++)
        {
            d[j] = (2.0 * x->value[j]) / y->value[0];
        }
        matrix_values_changed(qnode->d);
        BTA_matrices_fill_values(x->jacobian, qnode->d, qnode->xpart);

        /* y-part: derivative wrt y (y does not appear in the numerator, since
           it is a new variable for the denominator) */
        qnode->ypart->x[0] = -node->value[0] / y->value[0];
        matrix_values_changed(qnode->ypart);

        sum_matrices_fill_values(qnode->xpart, qnode->ypart, node->jacobian);
    }
}

static void wsum_hess_init_impl(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;
    int var_id_x = x->var_id;
    int var_id_y = y->var_id;

    /* if left node is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        CSR_matrix *H = new_CSR_matrix(node->n_vars, node->n_vars, 3 * x->size + 1);
        node->wsum_hess = new_sparse_matrix(H);

        /* if x has lower idx than y*/
        if (var_id_x < var_id_y)
        {
            /* x rows: each row has 2 entries (diagonal element + element for y) */
            for (int i = 0; i < x->size; i++)
            {
                H->p[var_id_x + i] = 2 * i;
                H->i[2 * i] = var_id_x + i;
                H->i[2 * i + 1] = var_id_y;
            }

            /* rows between x and y are empty, all point to same offset */
            int offset = 2 * x->size;
            for (int i = var_id_x + x->size; i <= var_id_y; i++)
            {
                H->p[i] = offset;
            }

            /* y row has d1 + 1 entries */
            H->p[var_id_y + 1] = offset + x->size + 1;
            for (int i = 0; i < x->size; i++)
            {
                H->i[offset + i] = var_id_x + i;
            }
            H->i[offset + x->size] = var_id_y;

            /* remaining rows are empty */
            for (int i = var_id_y + 1; i <= node->n_vars; i++)
            {
                H->p[i] = 3 * x->size + 1;
            }
        }
        else /* y has lower idx than x */
        {
            /* y row has d1 + 1 entries */
            H->p[var_id_y + 1] = x->size + 1;
            H->i[0] = var_id_y;
            for (int i = 0; i < x->size; i++)
            {
                H->i[i + 1] = var_id_x + i;
            }

            /* rows between y and x are empty, all point to same offset */
            int offset = x->size + 1;
            for (int i = var_id_y + 1; i <= var_id_x; i++)
            {
                H->p[i] = offset;
            }

            /* x rows: each row has 2 entries */
            for (int i = 0; i < x->size; i++)
            {
                H->p[var_id_x + i] = offset + 2 * i;
                H->i[offset + 2 * i] = var_id_y;
                H->i[offset + 2 * i + 1] = var_id_x + i;
            }

            /* remaining rows are empty */
            for (int i = var_id_x + x->size; i <= node->n_vars; i++)
            {
                H->p[i] = 3 * x->size + 1;
            }
        }
    }
    else
    {
        /* TODO: implement */
        fprintf(stderr, "Error in quad_over_lin wsum_hess_init: non-variable child "
                        "not implemented\n");
        exit(1);
    }
}

static void eval_wsum_hess_impl(expr *node, const double *w)
{
    double *x = node->left->value;
    double y = node->right->value[0];
    double *H = node->wsum_hess->x;
    int var_id_x = node->left->var_id;
    int var_id_y = node->right->var_id;
    int x_size = node->left->size;
    double a = (2.0 * w[0]) / y;
    double b = -(2.0 * w[0]) / (y * y);

    /* if left node is a variable */
    if (var_id_x != NOT_A_VARIABLE)
    {
        /* if x has lower idx than y*/
        if (var_id_x < var_id_y)
        {
            /* x rows*/
            for (int i = 0; i < x_size; i++)
            {
                H[2 * i] = a;
                H[2 * i + 1] = b * x[i];
            }

            /* y row */
            int offset = 2 * x_size;
            for (int i = 0; i < x_size; i++)
            {
                H[offset + i] = b * x[i];
            }
            H[offset + x_size] = -b * node->value[0];
        }
        else /* y has lower idx than x */
        {
            /* y row */
            H[0] = -b * node->value[0];
            for (int i = 0; i < x_size; i++)
            {
                H[i + 1] = b * x[i];
            }

            /* x rows*/
            int offset = x_size + 1;
            for (int i = 0; i < x_size; i++)
            {
                H[offset + 2 * i] = b * x[i];
                H[offset + 2 * i + 1] = a;
            }
        }
    }
    else
    {
        /* TODO: implement */
        fprintf(stderr, "Error in quad_over_lin eval_wsum_hess: non-variable child "
                        "not implemented\n");
        exit(1);
    }
}

static bool is_affine(const expr *node)
{
    (void) node;
    return false;
}

static void free_type_data(expr *node)
{
    quad_over_lin_expr *qnode = (quad_over_lin_expr *) node;
    free_matrix(qnode->d);
    free_matrix(qnode->xpart);
    free_matrix(qnode->ypart);
}

expr *new_quad_over_lin(expr *left, expr *right)
{
    /* if right is not a scalar or a variable by itself, we raise an error */
    if (right->var_id == NOT_A_VARIABLE && !(right->d1 == 1 && right->d2 == 1))
    {
        fprintf(stderr,
                "Error: Denominator of quad-over-lin must be a scalar variable.\n");
        return NULL;
    }

    quad_over_lin_expr *qnode =
        (quad_over_lin_expr *) sp_calloc(1, sizeof(quad_over_lin_expr));
    expr *node = &qnode->base;
    init_expr(node, 1, 1, left->n_vars, forward, jacobian_init_impl,
              eval_jacobian_impl, is_affine, wsum_hess_init_impl,
              eval_wsum_hess_impl, free_type_data);
    node->left = left;
    node->right = right;
    expr_retain(left);
    expr_retain(right);
    return node;
}
