#include "affine/linear_op.h"
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
    expr *node = new_expr(A->m, u->n_vars);
    if (!node) return NULL;

    node->left = u;
    node->forward = forward;
    node->is_affine = is_affine;

    /* allocate jacobian and copy A into it */
    node->jacobian = new_csr_matrix(A->m, A->n, A->nnz);
    copy_csr_matrix(A, node->jacobian);

    return node;
}
