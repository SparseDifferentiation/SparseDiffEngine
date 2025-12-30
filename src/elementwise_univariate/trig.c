#include "elementwise_univariate.h"
#include <math.h>

/* ----------------------- sin ----------------------- */
static void sin_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = sin(node->left->value[i]);
    }
}

static void sin_eval_local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->d1; j++)
    {
        vals[j] = cos(child->value[j]);
    }
}

expr *new_sin(expr *child)
{
    expr *node = new_expr(child->d1, child->d2, child->n_vars);
    node->left = child;
    expr_retain(child);
    node->forward = sin_forward;
    node->jacobian_init = jacobian_init_elementwise;
    node->eval_jacobian = eval_jacobian_elementwise;
    node->eval_local_jacobian = sin_eval_local_jacobian;
    node->is_affine = is_affine_elementwise;
    return node;
}

/* ----------------------- cos ----------------------- */
static void cos_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = cos(node->left->value[i]);
    }
}

static void cos_eval_local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->d1; j++)
    {
        vals[j] = -sin(child->value[j]);
    }
}

expr *new_cos(expr *child)
{
    expr *node = new_expr(child->d1, child->d2, child->n_vars);
    node->left = child;
    expr_retain(child);
    node->forward = cos_forward;
    node->jacobian_init = jacobian_init_elementwise;
    node->eval_jacobian = eval_jacobian_elementwise;
    node->eval_local_jacobian = cos_eval_local_jacobian;
    node->is_affine = is_affine_elementwise;
    return node;
}

/* ----------------------- tan ----------------------- */
static void tan_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = tan(node->left->value[i]);
    }
}

static void tan_eval_local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->d1; j++)
    {
        double c = cos(child->value[j]);
        vals[j] = 1.0 / (c * c);
    }
}

expr *new_tan(expr *child)
{
    expr *node = new_expr(child->d1, child->d2, child->n_vars);
    node->left = child;
    expr_retain(child);
    node->forward = tan_forward;
    node->jacobian_init = jacobian_init_elementwise;
    node->eval_jacobian = eval_jacobian_elementwise;
    node->eval_local_jacobian = tan_eval_local_jacobian;
    node->is_affine = is_affine_elementwise;
    return node;
}
