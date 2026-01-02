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

static void local_jacobian(expr *node, double *vals)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        vals[j] = node->p * pow(x[j], node->p - 1);
    }
}

static void local_wsum_hess(expr *node, double *out, double *w)
{
    double *x = node->left->value;
    double p = (double) node->p;

    for (int j = 0; j < node->size; j++)
    {
        out[j] = w[j] * p * (p - 1) * pow(x[j], p - 2);
    }
}

expr *new_power(expr *child, int p)
{
    expr *node = new_elementwise(child);
    node->p = p;
    node->forward = forward;
    node->local_jacobian = local_jacobian;
    node->local_wsum_hess = local_wsum_hess;
    return node;
}
