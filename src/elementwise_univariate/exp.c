#include "elementwise_univariate.h"
#include <math.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    /* child's forward pass */
    node->left->forward(node->left, u);

    /* local forward pass */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = exp(node->left->value[i]);
    }
}

static void eval_local_jacobian(expr *node, double *vals)
{
    memcpy(vals, node->value, node->d1 * sizeof(double));
}

expr *new_exp(expr *child)
{
    if (!child) return NULL;

    expr *node = new_expr(child->d1, child->d2, child->n_vars);
    if (!node) return NULL;

    node->left = child;
    expr_retain(child);
    node->forward = forward;
    node->is_affine = is_affine_elementwise;
    node->jacobian_init = jacobian_init_elementwise;
    node->eval_jacobian = eval_jacobian_elementwise;
    node->eval_local_jacobian = eval_local_jacobian;

    return node;
}
