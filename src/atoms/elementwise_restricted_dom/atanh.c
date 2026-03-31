#include "atoms/elementwise_restricted_dom.h"
#include <math.h>

static void atanh_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = atanh(node->left->value[i]);
    }
}

static void atanh_eval_jacobian(expr *node)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        node->jacobian->x[j] = 1.0 / (1.0 - x[j] * x[j]);
    }
}

static void atanh_eval_wsum_hess(expr *node, const double *w)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        double c = 1.0 - x[j] * x[j];
        node->wsum_hess->x[j] = w[j] * (2.0 * x[j]) / (c * c);
    }
}

expr *new_atanh(expr *child)
{
    expr *node = new_restricted(child);
    node->forward = atanh_forward;
    node->eval_jacobian = atanh_eval_jacobian;
    node->eval_wsum_hess = atanh_eval_wsum_hess;
    return node;
}
