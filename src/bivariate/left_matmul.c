#include "bivariate.h"
#include "subexpr.h"
#include <stdlib.h>

static void forward(expr *node, const double *u)
{
    expr *x = node->left;

    /* child's forward pass */
    node->left->forward(node->left, u);

    /* y = A * x */
    csr_matvec_wo_offset(((left_matmul_expr *) node)->A, x->value, node->value);
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    left_matmul_expr *lin_node = (left_matmul_expr *) node;
    free_csr_matrix(lin_node->A);
    if (lin_node->CSC_work)
    {
        free_csc_matrix(lin_node->CSC_work);
    }
    lin_node->A = NULL;
    lin_node->CSC_work = NULL;
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    left_matmul_expr *lin_node = (left_matmul_expr *) node;

    /* initialize child's jacobian and precompute sparsity of its transpose */
    x->jacobian_init(x);
    node->iwork = (int *) malloc(node->n_vars * sizeof(int));
    lin_node->CSC_work = csr_to_csc_fill_sparsity(x->jacobian, node->iwork);

    /* precompute sparsity of this node's jacobian */
    node->jacobian = csr_csc_matmul_alloc(lin_node->A, lin_node->CSC_work);
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    left_matmul_expr *lin_node = (left_matmul_expr *) node;

    /* evaluate child's jacobian and convert to CSC*/
    x->eval_jacobian(x);
    csr_to_csc_fill_values(x->jacobian, lin_node->CSC_work, node->iwork);

    /* compute this node's jacobian */
    csr_csc_matmul_fill_values(lin_node->A, lin_node->CSC_work, node->jacobian);
}

static void wsum_hess_init(expr *node)
{
    /* initialize child's hessian */
    expr *x = node->left;
    x->wsum_hess_init(x);

    /* allocate this node's hessian with the same sparsity as child's */
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, x->wsum_hess->nnz);
    memcpy(node->wsum_hess->p, x->wsum_hess->p, (node->n_vars + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, x->wsum_hess->i, x->wsum_hess->nnz * sizeof(int));

    /* work for computing A^T w*/
    int A_n = ((left_matmul_expr *) node)->A->n;
    node->dwork = (double *) malloc(A_n * sizeof(double));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* compute A^T w*/
    left_matmul_expr *lin_node = (left_matmul_expr *) node;
    csr_matvec_wo_offset(lin_node->AT, w, node->dwork);

    node->left->eval_wsum_hess(node->left, node->dwork);
    memcpy(node->wsum_hess->x, node->left->wsum_hess->x,
           node->wsum_hess->nnz * sizeof(double));
}

expr *new_left_matmul(expr *u, const CSR_Matrix *A)
{
    /* Allocate the type-specific struct */
    left_matmul_expr *lin_node =
        (left_matmul_expr *) calloc(1, sizeof(left_matmul_expr));
    expr *node = &lin_node->base;
    init_expr(node, A->m, 1, u->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, free_type_data);
    node->left = u;
    expr_retain(u);
    node->wsum_hess_init = wsum_hess_init;
    node->eval_wsum_hess = eval_wsum_hess;

    /* Initialize type-specific fields */
    lin_node->A = new_csr_matrix(A->m, A->n, A->nnz);
    node->iwork = (int *) malloc(A->n * sizeof(int));
    lin_node->AT = transpose(A, node->iwork);
    copy_csr_matrix(A, lin_node->A);
    lin_node->CSC_work = NULL;

    return node;
}
