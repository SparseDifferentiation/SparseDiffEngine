#include "elementwise_univariate.h"
#include "subexpr.h"
#include <math.h>
#include <stdlib.h>

static void forward(expr *node, const double *u)
{

    /* child's forward pass */
    node->left->forward(node->left, u);

    /* local forward pass */
    double *x = node->left->value;
    double p = ((power_expr *) node)->p;
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = pow(x[i], p);
    }
}

static void local_jacobian(expr *node, double *vals)
{
    double *x = node->left->value;
    double p = ((power_expr *) node)->p;

    for (int j = 0; j < node->size; j++)
    {
        vals[j] = p * pow(x[j], p - 1);
    }
}

static void local_wsum_hess(expr *node, double *out, const double *w)
{
    double *x = node->left->value;
    double p = ((power_expr *) node)->p;

    for (int j = 0; j < node->size; j++)
    {
        out[j] = w[j] * p * (p - 1) * pow(x[j], p - 2);
    }
}

expr *new_power(expr *child, int p)
{
    /* Allocate the type-specific struct */
    power_expr *pnode = (power_expr *) calloc(1, sizeof(power_expr));
    expr *node = &pnode->base;
    init_elementwise(node, child);
    node->forward = forward;
    node->local_jacobian = local_jacobian;
    node->local_wsum_hess = local_wsum_hess;

    /* Set type-specific field */
    pnode->p = p;

    return node;
}
