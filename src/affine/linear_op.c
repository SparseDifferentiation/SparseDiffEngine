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

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    linear_op_expr *lin_node = (linear_op_expr *) node;
    /* memory pointing to by A_csr will already be freed when the
       jacobian is freed, so free_csr_matrix(lin_node->A_csr) should
       be commented out */
    // free_csr_matrix(lin_node->A_csr);
    free_csc_matrix(lin_node->A_csc);
    lin_node->A_csr = NULL;
    lin_node->A_csc = NULL;
}

expr *new_linear(expr *u, const CSR_Matrix *A)
{
    /* Allocate the type-specific struct */
    linear_op_expr *lin_node = (linear_op_expr *) calloc(1, sizeof(linear_op_expr));
    expr *node = &lin_node->base;
    init_expr(node, A->m, 1, u->n_vars, forward, NULL, NULL, is_affine,
              free_type_data);
    node->left = u;
    expr_retain(u);

    /* Initialize type-specific fields */
    lin_node->A_csr = new_csr_matrix(A->m, A->n, A->nnz);
    copy_csr_matrix(A, lin_node->A_csr);
    lin_node->A_csc = csr_to_csc(A);

    /* what if we have A @ phi(x). Then I don't think this is correct. */
    node->jacobian = lin_node->A_csr;

    return node;
}
