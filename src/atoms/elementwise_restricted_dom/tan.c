#include "atoms/elementwise_restricted_dom.h"
#include <math.h>

static void tan_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = tan(node->left->value[i]);
    }
}

static void tan_eval_jacobian(expr *node)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        double c = cos(x[j]);
        node->jacobian->x[j] = 1.0 / (c * c);
    }
}

static void tan_eval_wsum_hess(expr *node, const double *w)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        double c = cos(x[j]);
        node->wsum_hess->x[j] = 2.0 * w[j] * node->value[j] / (c * c);
    }
}

expr *new_tan(expr *child)
{
    expr *node = new_restricted(child);
    node->forward = tan_forward;
    node->eval_jacobian = tan_eval_jacobian;
    node->eval_wsum_hess = tan_eval_wsum_hess;
    return node;
}
