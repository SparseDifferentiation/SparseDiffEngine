#include "affine.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/* Reshape changes the shape of an expression without permuting data.
 * Only Fortran (column-major) order is supported, where reshape is a no-op
 * for the underlying data layout. */

static void forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    memcpy(node->value, node->left->value, node->size * sizeof(double));
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    x->jacobian_init(x);
    node->jacobian = new_csr_matrix(node->size, node->n_vars, x->jacobian->nnz);
    CSR_Matrix *jac = node->jacobian;
    memcpy(jac->p, x->jacobian->p, (x->size + 1) * sizeof(int));
    memcpy(jac->i, x->jacobian->i, x->jacobian->nnz * sizeof(int));
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    x->eval_jacobian(x);
    memcpy(node->jacobian->x, x->jacobian->x, x->jacobian->nnz * sizeof(double));
}

static void wsum_hess_init(expr *node)
{
    node->left->wsum_hess_init(node->left);
    CSR_Matrix *child_hess = node->left->wsum_hess;
    node->wsum_hess = new_csr_matrix(child_hess->m, child_hess->n, child_hess->nnz);
    memcpy(node->wsum_hess->p, child_hess->p, (child_hess->m + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, child_hess->i, child_hess->nnz * sizeof(int));
    node->wsum_hess->nnz = child_hess->nnz;
}

static void eval_wsum_hess(expr *node, const double *w)
{
    node->left->eval_wsum_hess(node->left, w);
    CSR_Matrix *child_hess = node->left->wsum_hess;
    CSR_Matrix *hess = node->wsum_hess;
    memcpy(hess->x, child_hess->x, child_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_reshape(expr *child, int d1, int d2)
{
    assert(d1 * d2 == child->size);
    expr *node = new_expr(d1, d2, child->n_vars);
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
