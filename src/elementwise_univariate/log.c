#include "elementwise_univariate.h"
#include <math.h>

static void forward(expr *node, const double *u)
{
    /* child's forward pass */
    node->left->forward(node->left, u);

    /* local forward pass */
    for (int i = 0; i < node->m; i++)
    {
        node->value[i] = log(node->left->value[i]);
    }
}

static void eval_local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->m; j++)
    {
        vals[j] = 1.0 / child->value[j];
    }
}

expr *new_log(expr *child)
{
    expr *node = new_expr(child->m, child->n_vars);
    node->left = child;
    node->forward = forward;
    node->jacobian_init = jacobian_init_elementwise;
    node->eval_jacobian = eval_jacobian_elementwise;
    node->is_affine = is_affine_elementwise;
    node->eval_local_jacobian = eval_local_jacobian;
    return node;
}
