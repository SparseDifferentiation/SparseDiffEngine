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

static void local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->size; j++)
    {
        vals[j] = 1.0 / child->value[j];
    }
}

static void local_wsum_hess(expr *node, double *out, double *w)
{
    double *x = node->left->value;

    for (int j = 0; j < node->size; j++)
    {
        out[j] = -w[j] / (x[j] * x[j]);
    }
}

expr *new_log(expr *child)
{
    expr *node = new_elementwise(child);
    node->forward = forward;
    node->local_jacobian = local_jacobian;
    node->local_wsum_hess = local_wsum_hess;
    return node;
}
