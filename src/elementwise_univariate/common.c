#include "elementwise_univariate.h"
#include <stdlib.h>

void jacobian_init_elementwise(expr *node)
{
    expr *child = node->left;

    /* if the variable is a child */
    if (child->var_id != -1)
    {
        node->jacobian = new_csr_matrix(node->size, node->n_vars, node->size);
        for (int j = 0; j < node->size; j++)
        {
            node->jacobian->p[j] = j;
            node->jacobian->i[j] = j + child->var_id;
        }
        node->jacobian->p[node->size] = node->size;
    }
    /* otherwise it should be a linear operator */
    else
    {
        node->jacobian = new_csr_matrix(child->jacobian->m, child->jacobian->n,
                                        child->jacobian->nnz);
        node->dwork = (double *) malloc(node->size * sizeof(double));
    }
}

void eval_jacobian_elementwise(expr *node)
{
    expr *child = node->left;

    if (child->var_id != -1)
    {
        node->local_jacobian(node, node->jacobian->x);
    }
    else
    {
        /* Child must be a linear operator */
        linear_op_expr *lin_child = (linear_op_expr *) child;
        node->local_jacobian(node, node->dwork);
        diag_csr_mult(node->dwork, lin_child->A_csr, node->jacobian);
    }
}

void wsum_hess_init_elementwise(expr *node)
{
    expr *child = node->left;
    int id = child->var_id;
    int i;

    /* if the variable is a child*/
    if (id != -1)
    {
        node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, node->size);

        for (i = 0; i < node->size; i++)
        {
            node->wsum_hess->p[id + i] = i;
            node->wsum_hess->i[i] = id + i;
        }

        for (i = id + node->size; i <= node->n_vars; i++)
        {
            node->wsum_hess->p[i] = node->size;
        }
    }
    /* otherwise it will be a linear operator */
    else
    {
        linear_op_expr *lin_child = (linear_op_expr *) child;
        node->wsum_hess = ATA_alloc(lin_child->A_csc);
    }
}

void eval_wsum_hess_elementwise(expr *node, double *w)
{
    expr *child = node->left;

    if (child->var_id != -1)
    {
        node->local_wsum_hess(node, node->wsum_hess->x, w);
    }
    else
    {
        /* Child must be a linear operator */
        linear_op_expr *lin_child = (linear_op_expr *) child;
        node->local_wsum_hess(node, node->dwork, w);
        ATDA_values(lin_child->A_csc, node->dwork, node->wsum_hess);
    }
}

bool is_affine_elementwise(expr *node)
{
    (void) node;
    return false;
}

/* Helper function to initialize an already-allocated expr for elementwise operations
 * This is called when a power_expr or other type-specific struct is allocated
 * and we need to initialize the base expr fields
 */
void init_elementwise(expr *node, expr *child)
{
    /* Initialize dimensions from child */
    node->d1 = child->d1;
    node->d2 = child->d2;
    node->size = child->d1 * child->d2;
    node->n_vars = child->n_vars;
    node->var_id = -1;
    node->refcount = 1;

    /* Allocate the value array */
    node->value = (double *) calloc(node->size, sizeof(double));

    node->left = child;
    node->right = NULL;
    node->dwork = NULL;
    node->iwork = NULL;
    node->jacobian = NULL;
    node->wsum_hess = NULL;
    node->jacobian_init = jacobian_init_elementwise;
    node->wsum_hess_init = wsum_hess_init_elementwise;
    node->eval_jacobian = eval_jacobian_elementwise;
    node->eval_wsum_hess = eval_wsum_hess_elementwise;
    node->local_jacobian = NULL;
    node->local_wsum_hess = NULL;
    node->is_affine = is_affine_elementwise;
    node->forward = NULL;
    node->free_type_data = NULL;

    expr_retain(child);
}

expr *new_elementwise(expr *child)
{
    expr *node = (expr *) malloc(sizeof(expr));
    if (!node) return NULL;

    init_elementwise(node, child);
    return node;
}
