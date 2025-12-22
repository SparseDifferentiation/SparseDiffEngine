#include "elementwise_univariate.h"
#include <math.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    /* child's forward pass */
    node->left->forward(node->left, u);

    /* local forward pass */
    for (int i = 0; i < node->m; i++)
    {
        node->value[i] = exp(node->left->value[i]);
    }
}

static void eval_local_jacobian(expr *node, double *vals)
{
    memcpy(vals, node->value, node->m * sizeof(double));
}

expr *new_exp(expr *child)
{
    if (!child) return NULL;

    expr *node = new_expr(child->m, child->n_vars);
    if (!node) return NULL;

    node->left = child;
    node->forward = forward;
    node->is_affine = is_affine_elementwise;
    node->jacobian_init = jacobian_init_elementwise;
    node->eval_jacobian = eval_jacobian_elementwise;
    node->eval_local_jacobian = eval_local_jacobian;

    return node;
}
