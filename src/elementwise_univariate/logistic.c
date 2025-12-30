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
        if (x[i] >= 0)
        {
            node->value[i] = x[i] + log(1.0 + exp(-x[i]));
        }
        else
        {
            node->value[i] = log(1.0 + exp(x[i]));
        }
    }
}

static void eval_local_jacobian(expr *node, double *vals)
{
    double *x = node->left->value;
    for (int j = 0; j < node->d1; j++)
    {
        if (x[j] >= 0)
        {
            vals[j] = 1.0 / (1.0 + exp(-x[j]));
        }
        else
        {
            double exp_x = exp(x[j]);
            vals[j] = exp_x / (1.0 + exp_x);
        }
    }
}

expr *new_logistic(expr *child)
{
    expr *node = new_expr(child->d1, child->d2, child->n_vars);
    node->left = child;
    expr_retain(child);
    node->forward = forward;
    node->jacobian_init = jacobian_init_elementwise;
    node->eval_jacobian = eval_jacobian_elementwise;
    node->is_affine = is_affine_elementwise;
    node->eval_local_jacobian = eval_local_jacobian;
    return node;
}
