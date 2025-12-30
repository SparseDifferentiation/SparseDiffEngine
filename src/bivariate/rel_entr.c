#include "bivariate.h"
#include <math.h>
#include <stdlib.h>

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

static void jacobian_init_vectors_args(expr *node)
{
    node->jacobian = new_csr_matrix(node->d1, node->n_vars, 2 * node->d1);

    expr *x = node->left;
    expr *y = node->right;

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
    // node->is_affine = is_affine_elementwise;
    // node->eval_local_jacobian = eval_local_jacobian;
    return node;
}

// --------------------------------------------------------------------
// Implementation of relative entropy when one argument is a vector
// --------------------------------------------------------------------
