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

static void local_jacobian(expr *node, double *vals)
{
    memcpy(vals, node->value, node->size * sizeof(double));
}

static void local_wsum_hess(expr *node, double *out, const double *w)
{
    for (int j = 0; j < node->size; j++)
    {
        out[j] = w[j] * node->value[j];
    }
}

expr *new_exp(expr *child)
{
    expr *node = new_elementwise(child);
    node->forward = forward;
    node->local_jacobian = local_jacobian;
    node->local_wsum_hess = local_wsum_hess;
    return node;
}
