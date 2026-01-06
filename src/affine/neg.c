#include "affine.h"
#include <stdlib.h>

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
}

static void eval_jacobian(expr *node)
{
    /* evaluate child's jacobian */
    node->left->eval_jacobian(node->left);

    /* negate child's jacobian: copy structure, negate values */
    CSR_Matrix *child_jac = node->left->jacobian;
    CSR_Matrix *jac = node->jacobian;

    /* copy row pointers */
    for (int i = 0; i <= child_jac->m; i++)
    {
        jac->p[i] = child_jac->p[i];
    }

    /* copy column indices and negate values */
    int nnz = child_jac->p[child_jac->m];
    for (int k = 0; k < nnz; k++)
    {
        jac->i[k] = child_jac->i[k];
        jac->x[k] = -child_jac->x[k];
    }
    jac->nnz = nnz;
}

static void wsum_hess_init(expr *node)
{
    /* initialize child's wsum_hess */
    node->left->wsum_hess_init(node->left);

    /* same sparsity pattern as child */
    CSR_Matrix *child_hess = node->left->wsum_hess;
    node->wsum_hess = new_csr_matrix(child_hess->m, child_hess->n, child_hess->nnz);
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* For neg(x), d^2(-x)/dx^2 = 0, but we need to pass -w to child
     * Actually: d/dx(-x) = -I, so Hessian contribution is (-I)^T H (-I) = H
     * But the weight vector needs to be passed through: child sees same w */

    /* Negate weights for child (chain rule for linear transformation) */
    double *neg_w = (double *) malloc(node->size * sizeof(double));
    for (int i = 0; i < node->size; i++)
    {
        neg_w[i] = -w[i];
    }

    /* evaluate child's wsum_hess with negated weights */
    node->left->eval_wsum_hess(node->left, neg_w);
    free(neg_w);

    /* copy child's wsum_hess (the negation is already accounted for in weights) */
    CSR_Matrix *child_hess = node->left->wsum_hess;
    CSR_Matrix *hess = node->wsum_hess;

    for (int i = 0; i <= child_hess->m; i++)
    {
        hess->p[i] = child_hess->p[i];
    }

    int nnz = child_hess->p[child_hess->m];
    for (int k = 0; k < nnz; k++)
    {
        hess->i[k] = child_hess->i[k];
        hess->x[k] = child_hess->x[k];
    }
    hess->nnz = nnz;
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
