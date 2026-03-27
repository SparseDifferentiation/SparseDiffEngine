#include "elementwise_restricted_dom.h"
#include <math.h>

static void entr_forward(expr *node, const double *u)
{
    expr *child = node->left;
    child->forward(child, u);

    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = -child->value[i] * log(child->value[i]);
    }
}

static void entr_eval_jacobian(expr *node)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        node->jacobian->x[j] = -log(x[j]) - 1.0;
    }
}

static void entr_eval_wsum_hess(expr *node, const double *w)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        node->wsum_hess->x[j] = -w[j] / x[j];
    }
}

expr *new_entr(expr *child)
{
    expr *node = new_restricted(child);
    node->forward = entr_forward;
    node->eval_jacobian = entr_eval_jacobian;
    node->eval_wsum_hess = entr_eval_wsum_hess;
    return node;
}
