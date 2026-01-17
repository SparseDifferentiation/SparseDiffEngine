#include "bivariate.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// --------------------------------------------------------------------
// Implementation of relative entropy when first argument is a vector
// and second argument is a scalar.
// --------------------------------------------------------------------
static void forward_vector_scalar(expr *node, const double *u)
{
    expr *x = node->left;
    expr *y = node->right;

    /* children's forward passes */
    x->forward(x, u);
    y->forward(y, u);

    /* local forward pass */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = x->value[i] * log(x->value[i] / y->value[0]);
    }
}

static void jacobian_init_vector_scalar(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;
    assert(y->d1 == 1 && y->d2 == 1);
    assert(x->var_id != NOT_A_VARIABLE && y->var_id != NOT_A_VARIABLE);
    assert(x->var_id != y->var_id);

    node->jacobian = new_csr_matrix(node->size, node->n_vars, 2 * node->size);

    if (x->var_id < y->var_id)
    {
        for (int j = 0; j < node->size; j++)
        {
            node->jacobian->i[2 * j] = x->var_id + j;
            node->jacobian->i[2 * j + 1] = y->var_id;
            node->jacobian->p[j] = 2 * j;
        }
    }
    else
    {
        for (int j = 0; j < node->size; j++)
        {
            node->jacobian->i[2 * j] = y->var_id;
            node->jacobian->i[2 * j + 1] = x->var_id + j;
            node->jacobian->p[j] = 2 * j;
        }
    }

    node->jacobian->p[node->size] = 2 * node->size;
}

static void eval_jacobian_vector_scalar(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    if (x->var_id < y->var_id)
    {
        for (int i = 0; i < node->size; i++)
        {
            node->jacobian->x[2 * i] = log(x->value[i] / y->value[0]) + 1;
            node->jacobian->x[2 * i + 1] = -x->value[i] / y->value[0];
        }
    }
    else
    {
        for (int i = 0; i < node->size; i++)
        {
            node->jacobian->x[2 * i] = -x->value[i] / y->value[0];
            node->jacobian->x[2 * i + 1] = log(x->value[i] / y->value[0]) + 1;
        }
    }
}

static void wsum_hess_init_vector_scalar(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;
    int var_id_x = x->var_id;
    int var_id_y = y->var_id;

    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, 3 * node->size + 1);
    CSR_Matrix *H = node->wsum_hess;

    if (var_id_x < var_id_y)
    {
        for (int i = 0; i < node->size; i++)
        {
            H->p[var_id_x + i] = 2 * i;
            H->i[2 * i] = var_id_x + i;
            H->i[2 * i + 1] = var_id_y;
        }

        int offset = 2 * node->size;
        for (int i = var_id_x + node->size; i <= var_id_y; i++)
        {
            H->p[i] = offset;
        }

        H->p[var_id_y + 1] = offset + node->size + 1;
        for (int i = 0; i < node->size; i++)
        {
            H->i[offset + i] = var_id_x + i;
        }
        H->i[offset + node->size] = var_id_y;

        for (int i = var_id_y + 1; i <= node->n_vars; i++)
        {
            H->p[i] = 3 * node->size + 1;
        }
    }
    else
    {
        H->p[var_id_y + 1] = node->size + 1;
        H->i[0] = var_id_y;
        for (int i = 0; i < node->size; i++)
        {
            H->i[i + 1] = var_id_x + i;
        }

        int offset = node->size + 1;
        for (int i = var_id_y + 1; i <= var_id_x; i++)
        {
            H->p[i] = offset;
        }

        for (int i = 0; i < node->size; i++)
        {
            H->p[var_id_x + i] = offset + 2 * i;
            H->i[offset + 2 * i] = var_id_y;
            H->i[offset + 2 * i + 1] = var_id_x + i;
        }

        for (int i = var_id_x + node->size; i <= node->n_vars; i++)
        {
            H->p[i] = 3 * node->size + 1;
        }
    }
}

static void eval_wsum_hess_vector_scalar(expr *node, const double *w)
{
    double *x = node->left->value;
    double y = node->right->value[0];
    double *H = node->wsum_hess->x;
    int var_id_x = node->left->var_id;
    int var_id_y = node->right->var_id;

    double diag_y = 0.0;
    for (int i = 0; i < node->size; i++)
    {
        diag_y += w[i] * x[i];
    }
    diag_y /= (y * y);

    if (var_id_x < var_id_y)
    {
        for (int i = 0; i < node->size; i++)
        {
            H[2 * i] = w[i] / x[i];
            H[2 * i + 1] = -w[i] / y;
        }

        int offset = 2 * node->size;
        for (int i = 0; i < node->size; i++)
        {
            H[offset + i] = -w[i] / y;
        }
        H[offset + node->size] = diag_y;
    }
    else
    {
        H[0] = diag_y;
        for (int i = 0; i < node->size; i++)
        {
            H[i + 1] = -w[i] / y;
        }

        int offset = node->size + 1;
        for (int i = 0; i < node->size; i++)
        {
            H[offset + 2 * i] = -w[i] / y;
            H[offset + 2 * i + 1] = w[i] / x[i];
        }
    }
}

static bool is_affine(const expr *node)
{
    (void) node;
    return false;
}

/* TODO: modify relative entropy atoms to use init_expr initialization? Also make
 * init_expr take in the hessian functions?*/
expr *new_rel_entr_second_arg_scalar(expr *left, expr *right)
{
    assert(right->d1 == 1 && right->d2 == 1);
    expr *node = new_expr(left->d1, left->d2, left->n_vars);
    node->left = left;
    node->right = right;
    expr_retain(left);
    expr_retain(right);
    node->forward = forward_vector_scalar;
    node->jacobian_init = jacobian_init_vector_scalar;
    node->eval_jacobian = eval_jacobian_vector_scalar;
    node->wsum_hess_init = wsum_hess_init_vector_scalar;
    node->eval_wsum_hess = eval_wsum_hess_vector_scalar;
    node->is_affine = is_affine;
    return node;
}
