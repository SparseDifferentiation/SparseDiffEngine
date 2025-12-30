#include "elementwise_univariate.h"
#include <math.h>

static void forward(expr *node, const double *u)
{
    expr *child = node->left;

    /* child's forward pass */
    child->forward(child, u);

    /* local forward pass */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = -child->value[i] * log(child->value[i]);
    }
}

static void eval_local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->d1; j++)
    {
        vals[j] = -log(child->value[j]) - 1.0;
    }
}

expr *new_entr(expr *child)
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
