#include "affine.h"

static void forward(expr *node, const double *u)
{
    /* child's forward pass */
    node->left->forward(node->left, u);

    /* negate values */
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = -node->left->value[i];
    }
}

static void jacobian_init(expr *node)
{
    /* initialize child's jacobian */
    node->left->jacobian_init(node->left);

    /* same sparsity pattern as child */
    CSR_Matrix *child_jac = node->left->jacobian;
    node->jacobian = new_csr_matrix(child_jac->m, child_jac->n, child_jac->nnz);

    /* copy row pointers and column indices (sparsity pattern is constant) */
    for (int i = 0; i <= child_jac->m; i++)
    {
        node->jacobian->p[i] = child_jac->p[i];
    }
    for (int k = 0; k < child_jac->nnz; k++)
    {
        node->jacobian->i[k] = child_jac->i[k];
    }
    node->jacobian->nnz = child_jac->nnz;
}

static void eval_jacobian(expr *node)
{
    /* evaluate child's jacobian */
    node->left->eval_jacobian(node->left);

    /* negate values only (sparsity pattern set in jacobian_init) */
    CSR_Matrix *child_jac = node->left->jacobian;
    for (int k = 0; k < child_jac->nnz; k++)
    {
        node->jacobian->x[k] = -child_jac->x[k];
    }
}

static void wsum_hess_init(expr *node)
{
    /* initialize child's wsum_hess */
    node->left->wsum_hess_init(node->left);

    /* same sparsity pattern as child */
    CSR_Matrix *child_hess = node->left->wsum_hess;
    node->wsum_hess = new_csr_matrix(child_hess->m, child_hess->n, child_hess->nnz);

    /* copy row pointers and column indices (sparsity pattern is constant) */
    for (int i = 0; i <= child_hess->m; i++)
    {
        node->wsum_hess->p[i] = child_hess->p[i];
    }
    for (int k = 0; k < child_hess->nnz; k++)
    {
        node->wsum_hess->i[k] = child_hess->i[k];
    }
    node->wsum_hess->nnz = child_hess->nnz;
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* For f(x) = -g(x): d²f/dx² = -d²g/dx² */
    node->left->eval_wsum_hess(node->left, w);

    /* negate values (sparsity pattern set in wsum_hess_init) */
    CSR_Matrix *child_hess = node->left->wsum_hess;
    for (int k = 0; k < child_hess->nnz; k++)
    {
        node->wsum_hess->x[k] = -child_hess->x[k];
    }
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_neg(expr *child)
{
    expr *node = new_expr(child->d1, child->d2, child->n_vars);
    node->left = child;
    expr_retain(child);
    node->forward = forward;
    node->is_affine = is_affine;
    node->jacobian_init = jacobian_init;
    node->eval_jacobian = eval_jacobian;
    node->wsum_hess_init = wsum_hess_init;
    node->eval_wsum_hess = eval_wsum_hess;

    return node;
}
