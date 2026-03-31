#include "atoms/elementwise_full_dom.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

static const double INV_SQRT_2PI = 0.3989422804014326779399461;

static void forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);

    double *x = node->left->value;
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = 0.5 * (1.0 + erf(x[i] / M_SQRT2));
    }
}

static void local_jacobian(expr *node, double *vals)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        vals[j] = INV_SQRT_2PI * exp(-0.5 * x[j] * x[j]);
    }
}

static void local_wsum_hess(expr *node, double *out, const double *w)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        double phi = INV_SQRT_2PI * exp(-0.5 * x[j] * x[j]);
        out[j] = w[j] * (-x[j] * phi);
    }
}

expr *new_normal_cdf(expr *child)
{
    expr *node = new_elementwise(child);
    node->forward = forward;
    node->local_jacobian = local_jacobian;
    node->local_wsum_hess = local_wsum_hess;

    return node;
}
