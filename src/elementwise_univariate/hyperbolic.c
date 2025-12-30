#include "elementwise_univariate.h"
#include <math.h>

/* ----------------------- sinh ----------------------- */
static void sinh_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = sinh(node->left->value[i]);
    }
}

static void sinh_eval_local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->d1; j++)
    {
        vals[j] = cosh(child->value[j]);
    }
}

expr *new_sinh(expr *child)
{
    expr *node = new_expr(child->d1, child->d2, child->n_vars);
    node->left = child;
    expr_retain(child);
    node->forward = sinh_forward;
    node->jacobian_init = jacobian_init_elementwise;
    node->eval_jacobian = eval_jacobian_elementwise;
    node->eval_local_jacobian = sinh_eval_local_jacobian;
    node->is_affine = is_affine_elementwise;
    return node;
}

/* ----------------------- tanh ----------------------- */
static void tanh_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = tanh(node->left->value[i]);
    }
}

static void tanh_eval_local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->d1; j++)
    {
        double c = cosh(child->value[j]);
        vals[j] = 1.0 / (c * c);
    }
}

expr *new_tanh(expr *child)
{
    expr *node = new_expr(child->d1, child->d2, child->n_vars);
    node->left = child;
    expr_retain(child);
    node->forward = tanh_forward;
    node->jacobian_init = jacobian_init_elementwise;
    node->eval_jacobian = eval_jacobian_elementwise;
    node->eval_local_jacobian = tanh_eval_local_jacobian;
    node->is_affine = is_affine_elementwise;
    return node;
}

/* ----------------------- asinh ----------------------- */
static void asinh_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = asinh(node->left->value[i]);
    }
}

static void asinh_eval_local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->d1; j++)
    {
        vals[j] = 1.0 / sqrt(1.0 + child->value[j] * child->value[j]);
    }
}

expr *new_asinh(expr *child)
{
    expr *node = new_expr(child->d1, child->d2, child->n_vars);
    node->left = child;
    expr_retain(child);
    node->forward = asinh_forward;
    node->jacobian_init = jacobian_init_elementwise;
    node->eval_jacobian = eval_jacobian_elementwise;
    node->eval_local_jacobian = asinh_eval_local_jacobian;
    node->is_affine = is_affine_elementwise;
    return node;
}

/* ----------------------- atanh ----------------------- */
static void atanh_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = atanh(node->left->value[i]);
    }
}

static void atanh_eval_local_jacobian(expr *node, double *vals)
{
    expr *child = node->left;
    for (int j = 0; j < node->d1; j++)
    {
        vals[j] = 1.0 / (1.0 - child->value[j] * child->value[j]);
    }
}

expr *new_atanh(expr *child)
{
    expr *node = new_expr(child->d1, child->d2, child->n_vars);
    node->left = child;
    expr_retain(child);
    node->forward = atanh_forward;
    node->jacobian_init = jacobian_init_elementwise;
    node->eval_jacobian = eval_jacobian_elementwise;
    node->eval_local_jacobian = atanh_eval_local_jacobian;
    node->is_affine = is_affine_elementwise;
    return node;
}
