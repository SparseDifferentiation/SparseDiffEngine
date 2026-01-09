#include "affine.h"

static void forward(expr *node, const double *u)
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

    /* fill sparsity pattern  */
    sum_csr_matrices_fill_sparsity(node->left->jacobian, node->right->jacobian,
                                   node->jacobian);
}

static void eval_jacobian(expr *node)
{
    /* evaluate children's jacobians */
    node->left->eval_jacobian(node->left);
    node->right->eval_jacobian(node->right);

    /* sum children's jacobians */
    sum_csr_matrices_fill_values(node->left->jacobian, node->right->jacobian,
                                 node->jacobian);
}

static void wsum_hess_init(expr *node)
{
    /* initialize children's wsum_hess */
    node->left->wsum_hess_init(node->left);
    node->right->wsum_hess_init(node->right);

    /* we never have to store more than the sum of children's nnz */
    int nnz_max = node->left->wsum_hess->nnz + node->right->wsum_hess->nnz;
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, nnz_max);

    /* TODO: we should fill sparsity pattern here for consistency */
    sum_csr_matrices_fill_sparsity(node->left->wsum_hess, node->right->wsum_hess,
                                   node->wsum_hess);
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* evaluate children's wsum_hess */
    node->left->eval_wsum_hess(node->left, w);
    node->right->eval_wsum_hess(node->right, w);

    /* sum children's wsum_hess */
    sum_csr_matrices_fill_values(node->left->wsum_hess, node->right->wsum_hess,
                                 node->wsum_hess);
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left) && node->right->is_affine(node->right);
}

expr *new_add(expr *left, expr *right)
{
    expr *node = new_expr(left->d1, left->d2, left->n_vars);
    node->left = left;
    node->right = right;
    expr_retain(left);
    expr_retain(right);
    node->forward = forward;
    node->is_affine = is_affine;
    node->jacobian_init = jacobian_init;
    node->eval_jacobian = eval_jacobian;
    node->wsum_hess_init = wsum_hess_init;
    node->eval_wsum_hess = eval_wsum_hess;

    // just for debugging, should be removed
    strcpy(node->name, "add");

    return node;
}
