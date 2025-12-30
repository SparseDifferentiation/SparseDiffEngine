#include "bivariate.h"
#include <math.h>
#include <stdlib.h>

// ------------------------------------------------------------------------------
// Implementation of elementwise multiplication when both arguments are vectors.
// If one argument is a scalar variable, the broadcasting should be represented
// as a linear operator child node? How to treat if one variable is a constant?
// ------------------------------------------------------------------------------
static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    expr *y = node->right;

    /* children's forward passes */
    x->forward(x, u);
    y->forward(y, u);

    /* local forward pass */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = x->value[i] * y->value[i];
    }
}

static void jacobian_init(expr *node)
{
    /* if a child is a variable we initialize its jacobian for a
       short chain rule implementation */
    if (node->left->var_id != -1)
    {
        node->left->jacobian_init(node->left);
    }

    if (node->right->var_id != -1)
    {
        node->right->jacobian_init(node->right);
    }

    node->dwork = (double *) malloc(2 * node->size * sizeof(double));
    int nnz_max = node->left->jacobian->nnz + node->right->jacobian->nnz;
    node->jacobian = new_csr_matrix(node->size, node->n_vars, nnz_max);
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* chain rule */
    sum_scaled_csr_matrices(x->jacobian, y->jacobian, node->jacobian, y->value,
                            x->value);
}

expr *new_elementwise_mult(expr *left, expr *right)
{
    expr *node = new_expr(left->d1, 1, left->n_vars);
    node->left = left;
    node->right = right;
    expr_retain(left);
    expr_retain(right);
    node->forward = forward;
    node->jacobian_init = jacobian_init;
    node->eval_jacobian = eval_jacobian;
    return node;
}
