#include "elementwise_restricted_dom.h"
#include <math.h>

static void log_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);

    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = log(node->left->value[i]);
    }
}

static void log_eval_jacobian(expr *node)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        node->jacobian->x[j] = 1.0 / x[j];
    }
}

static void log_eval_wsum_hess(expr *node, const double *w)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        node->wsum_hess->x[j] = -w[j] / (x[j] * x[j]);
    }
}

expr *new_log(expr *child)
{
    expr *node = new_restricted(child);
    node->forward = log_forward;
    node->eval_jacobian = log_eval_jacobian;
    node->eval_wsum_hess = log_eval_wsum_hess;
    return node;
}
