#include "affine.h"
#include <stdlib.h>

static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    /* child's forward pass */
    node->left->forward(node->left, u);

    /* y = A * x */
    csr_matvec(node->jacobian, x->value, node->value, x->var_id);
}

static bool is_affine(expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_linear(expr *u, const CSR_Matrix *A)
{
    expr *node = new_expr(A->m, 1, u->n_vars);
    if (!node) return NULL;

    node->left = u;
    expr_retain(u);
    node->forward = forward;
    node->is_affine = is_affine;

    /* allocate jacobian and copy A into it */
    // TODO: this should eventually be removed
    node->jacobian = new_csr_matrix(A->m, A->n, A->nnz);
    copy_csr_matrix(A, node->jacobian);

    node->A_csr = new_csr_matrix(A->m, A->n, A->nnz);
    copy_csr_matrix(A, node->A_csr);
    node->A_csc = csr_to_csc(A);

    return node;
}
