#include "affine.h"

static void add_forward(expr *node, const double *u)
{
    /* children's forward passes */
    node->left->forward(node->left, u);
    node->right->forward(node->right, u);

    /* add left and right values */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = node->left->value[i] + node->right->value[i];
    }
}

static void jacobian_init(expr *node)
{
    /* initialize children's jacobians */
    node->left->jacobian_init(node->left);
    node->right->jacobian_init(node->right);

    /* we never have to store more than the sum of children's nnz */
    int nnz_max = node->left->jacobian->nnz + node->right->jacobian->nnz;
    node->jacobian = new_csr_matrix(node->size, node->n_vars, nnz_max);
}

static void eval_jacobian(expr *node)
{
    /* evaluate children's jacobians */
    node->left->eval_jacobian(node->left);
    node->right->eval_jacobian(node->right);

    /* sum children's jacobians */
    sum_csr_matrices(node->left->jacobian, node->right->jacobian, node->jacobian);
}

static bool is_affine(expr *node)
{
    return node->left->is_affine(node->left) && node->right->is_affine(node->right);
}

expr *new_add(expr *left, expr *right)
{
    if (!left || !right) return NULL;
    if (left->d1 != right->d1) return NULL;
    if (left->d2 != right->d2) return NULL;

    expr *node = new_expr(left->d1, left->d2, left->n_vars);
    if (!node) return NULL;

    node->left = left;
    node->right = right;
    expr_retain(left);
    expr_retain(right);
    node->forward = add_forward;
    node->is_affine = is_affine;
    node->jacobian_init = jacobian_init;
    node->eval_jacobian = eval_jacobian;

    return node;
}
