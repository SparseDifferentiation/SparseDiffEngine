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

static void free_type_data(expr *node)
{
    linear_op_expr *lin_node = (linear_op_expr *) node;
    free_csr_matrix(lin_node->A_csr);
    free_csc_matrix(lin_node->A_csc);
}

/* Helper function to initialize a linear operator expr */
void init_linear_op(expr *node, expr *child, int d1, int d2)
{
    node->d1 = d1;
    node->d2 = d2;
    node->size = d1 * d2;
    node->n_vars = child->n_vars;
    node->var_id = -1;
    node->refcount = 1;
    node->left = child;
    node->right = NULL;
    node->dwork = NULL;
    node->iwork = NULL;
    node->value = (double *) calloc(node->size, sizeof(double));
    node->jacobian = NULL;
    node->wsum_hess = NULL;
    node->jacobian_init = NULL;
    node->wsum_hess_init = NULL;
    node->eval_jacobian = NULL;
    node->eval_wsum_hess = NULL;
    node->local_jacobian = NULL;
    node->local_wsum_hess = NULL;
    node->forward = forward;
    node->is_affine = is_affine;
    node->free_type_data = free_type_data;

    expr_retain(child);
}

expr *new_linear(expr *u, const CSR_Matrix *A)
{
    /* Allocate the type-specific struct */
    linear_op_expr *lin_node = (linear_op_expr *) malloc(sizeof(linear_op_expr));
    if (!lin_node) return NULL;

    expr *node = &lin_node->base;

    /* Initialize base linear operator fields */
    init_linear_op(node, u, A->m, 1);

    /* Check if allocation succeeded */
    if (!node->value)
    {
        free(lin_node);
        return NULL;
    }

    /* allocate jacobian and copy A into it */
    // TODO: this should eventually be removed
    node->jacobian = new_csr_matrix(A->m, A->n, A->nnz);
    copy_csr_matrix(A, node->jacobian);

    /* Initialize type-specific fields */
    lin_node->A_csr = new_csr_matrix(A->m, A->n, A->nnz);
    copy_csr_matrix(A, lin_node->A_csr);
    lin_node->A_csc = csr_to_csc(A);

    return node;
}
