#include "bivariate.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// --------------------------------------------------------------------
// Implementation of relative entropy when both arguments are vectors.
// No chain rule is needed since both arguments must be variables.
// --------------------------------------------------------------------
static void forward_vector_args(expr *node, const double *u)
{
    expr *x = node->left;
    expr *y = node->right;

    /* children's forward passes */
    x->forward(x, u);
    y->forward(y, u);

    /* local forward pass */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = x->value[i] * log(x->value[i] / y->value[i]);
    }
}

/* TODO: this probably doesn't work for matrices because we use node->d1
   instead of node->size */
static void jacobian_init_vectors_args(expr *node)
{
    node->jacobian = new_csr_matrix(node->d1, node->n_vars, 2 * node->d1);

    expr *x = node->left;
    expr *y = node->right;
    assert(x->var_id != NOT_A_VARIABLE && y->var_id != NOT_A_VARIABLE);
    assert(x->var_id != y->var_id);

    /* if x has lower variable idx than y it should appear first */
    if (x->var_id < y->var_id)
    {
        for (int j = 0; j < node->d1; j++)
        {
            node->jacobian->i[2 * j] = j + x->var_id;
            node->jacobian->i[2 * j + 1] = j + y->var_id;
            node->jacobian->p[j] = 2 * j;
        }
    }
    else
    {
        for (int j = 0; j < node->d1; j++)
        {
            node->jacobian->i[2 * j] = j + y->var_id;
            node->jacobian->i[2 * j + 1] = j + x->var_id;
            node->jacobian->p[j] = 2 * j;
        }
    }

    node->jacobian->p[node->d1] = 2 * node->d1;
}

static void eval_jacobian_vector_args(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* if x has lower variable idx than y */
    if (x->var_id < y->var_id)
    {
        for (int i = 0; i < node->size; i++)
        {
            node->jacobian->x[2 * i] = log(x->value[i] / y->value[i]) + 1;
            node->jacobian->x[2 * i + 1] = -x->value[i] / y->value[i];
        }
    }
    else
    {
        for (int i = 0; i < node->size; i++)
        {
            node->jacobian->x[2 * i] = -x->value[i] / y->value[i];
            node->jacobian->x[2 * i + 1] = log(x->value[i] / y->value[i]) + 1;
        }
    }
}

/* TODO: this probably doesn't work for matrices because we use node->d1
   instead of node->size */
static void wsum_hess_init_vector_args(expr *node)
{
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, 4 * node->d1);
    expr *x = node->left;
    expr *y = node->right;

    int i, var1_id, var2_id;

    if (x->var_id < y->var_id)
    {
        var1_id = x->var_id;
        var2_id = y->var_id;
    }
    else
    {
        var1_id = y->var_id;
        var2_id = x->var_id;
    }

    /* var1 rows of Hessian */
    for (i = 0; i < node->d1; i++)
    {
        node->wsum_hess->p[var1_id + i] = 2 * i;
        node->wsum_hess->i[2 * i] = var1_id + i;
        node->wsum_hess->i[2 * i + 1] = var2_id + i;
    }

    int nnz = 2 * node->d1;

    /* rows between var1 and var2 */
    for (i = var1_id + node->d1; i < var2_id; i++)
    {
        node->wsum_hess->p[i] = nnz;
    }

    /* var2 rows of Hessian */
    for (i = 0; i < node->d1; i++)
    {
        node->wsum_hess->p[var2_id + i] = nnz + 2 * i;
    }
    memcpy(node->wsum_hess->i + nnz, node->wsum_hess->i, nnz * sizeof(int));

    /* remaining rows */
    for (i = var2_id + node->d1; i <= node->n_vars; i++)
    {
        node->wsum_hess->p[i] = 4 * node->d1;
    }
}

static void eval_wsum_hess_vector_args(expr *node, const double *w)
{
    double *x = node->left->value;
    double *y = node->right->value;
    double *hess = node->wsum_hess->x;

    if (node->left->var_id < node->right->var_id)
    {
        for (int i = 0; i < node->d1; i++)
        {
            hess[2 * i] = w[i] / x[i];
            hess[2 * i + 1] = -w[i] / y[i];
        }

        hess += 2 * node->d1;

        for (int i = 0; i < node->d1; i++)
        {
            hess[2 * i] = -w[i] / y[i];
            hess[2 * i + 1] = w[i] * x[i] / (y[i] * y[i]);
        }
    }
    else
    {
        for (int i = 0; i < node->d1; i++)
        {
            hess[2 * i] = w[i] * x[i] / (y[i] * y[i]);
            hess[2 * i + 1] = -w[i] / y[i];
        }

        hess += 2 * node->d1;

        for (int i = 0; i < node->d1; i++)
        {
            hess[2 * i] = -w[i] / y[i];
            hess[2 * i + 1] = w[i] / x[i];
        }
    }
}

expr *new_rel_entr_vector_args(expr *left, expr *right)
{
    expr *node = new_expr(left->d1, 1, left->n_vars);
    node->left = left;
    node->right = right;
    expr_retain(left);
    expr_retain(right);
    node->forward = forward_vector_args;
    node->jacobian_init = jacobian_init_vectors_args;
    node->eval_jacobian = eval_jacobian_vector_args;
    node->wsum_hess_init = wsum_hess_init_vector_args;
    node->eval_wsum_hess = eval_wsum_hess_vector_args;
    // node->is_affine = is_affine_elementwise;
    // node->local_jacobian = local_jacobian;
    return node;
}

// --------------------------------------------------------------------
// Implementation of relative entropy when one argument is a vector
// --------------------------------------------------------------------
