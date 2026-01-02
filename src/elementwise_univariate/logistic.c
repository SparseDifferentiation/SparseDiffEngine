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

static void local_jacobian(expr *node, double *vals)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
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

static void local_wsum_hess(expr *node, double *out, double *w)
{
    double *x = node->left->value;
    double *sigmas;

    if (node->left->var_id != -1)
    {
        sigmas = node->jacobian->x;
    }
    else
    {
        sigmas = node->dwork;
    }

    // double sigma;

    for (int j = 0; j < node->size; j++)
    {
        /*
        if (x[j] >= 0)
        {
            sigma = 1.0 / (1.0 + exp(-x[j]));
        }
        else
        {
            double exp_x = exp(x[j]);
            sigma = exp_x / (1.0 + exp_x);
        }

        out[j] = w[j] * sigma * (1.0 - sigma);
        */
        out[j] = sigmas[j] * (1.0 - sigmas[j]) * w[j];
    }
}

expr *new_logistic(expr *child)
{
    expr *node = new_elementwise(child);
    node->forward = forward;
    node->local_jacobian = local_jacobian;
    node->local_wsum_hess = local_wsum_hess;
    return node;
}
