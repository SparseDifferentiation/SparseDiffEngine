#include "elementwise_univariate.h"
#include <math.h>

static void forward(expr *node, const double *u)
{

    /* child's forward pass */
    node->left->forward(node->left, u);

    double *x = node->left->value;

    /* local forward pass */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = pow(x[i], node->p);
    }
}

static void eval_local_jacobian(expr *node, double *vals)
{
    double *x = node->left->value;
    for (int j = 0; j < node->d1; j++)
    {
        vals[j] = node->p * pow(x[j], node->p - 1);
    }
}

expr *new_power(expr *child, int p)
{
    expr *node = new_expr(child->d1, child->d2, child->n_vars);
    node->p = p;
    node->left = child;
    expr_retain(child);
    node->forward = forward;
    node->jacobian_init = jacobian_init_elementwise;
    node->eval_jacobian = eval_jacobian_elementwise;
    node->is_affine = is_affine_elementwise;
    node->eval_local_jacobian = eval_local_jacobian;
    return node;
}
