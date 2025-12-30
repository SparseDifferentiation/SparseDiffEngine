#include "elementwise_univariate.h"
#include <math.h>

static void forward(expr *node, const double *u)
{
    /* child's forward pass */
    node->left->forward(node->left, u);

    /* local forward pass */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = log(node->left->value[i]);
    }
}

static void eval_local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->size; j++)
    {
        vals[j] = 1.0 / child->value[j];
    }
}

expr *new_log(expr *child)
{
    expr *node = new_expr(child->d1, child->d2, child->n_vars);
    node->left = child;
    expr_retain(child);
    node->forward = forward;
    node->jacobian_init = jacobian_init_elementwise;
    node->eval_jacobian = eval_jacobian_elementwise;
    node->is_affine = is_affine_elementwise;
    node->eval_local_jacobian = eval_local_jacobian;
    return node;
}
