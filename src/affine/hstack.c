#include "affine.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    hstack_expr *hnode = (hstack_expr *) node;

    /* children's forward passes */
    for (int i = 0; i < hnode->n_args; i++)
    {
        hnode->args[i]->forward(hnode->args[i], u);
    }

    /* concatenate values horizontally */
    int offset = 0;
    for (int i = 0; i < hnode->n_args; i++)
    {
        expr *child = hnode->args[i];
        memcpy(node->value + offset, child->value, child->size * sizeof(double));
        offset += child->size;
    }
}

static void jacobian_init(expr *node)
{
    hstack_expr *hnode = (hstack_expr *) node;

    /* initialize children's jacobians */
    int nnz = 0;
    for (int i = 0; i < hnode->n_args; i++)
    {
        hnode->args[i]->jacobian_init(hnode->args[i]);
        nnz += hnode->args[i]->jacobian->nnz;
    }

    node->jacobian = new_csr_matrix(node->size, node->n_vars, nnz);
}

static void eval_jacobian(expr *node)
{
    hstack_expr *hnode = (hstack_expr *) node;

    /* evaluate children's jacobians */
    int row_offset = 0;
    CSR_Matrix *A = node->jacobian;
    A->nnz = 0;

    for (int i = 0; i < hnode->n_args; i++)
    {
        expr *child = hnode->args[i];
        child->eval_jacobian(child);
        CSR_Matrix *B = child->jacobian;

        /* copy columns and values */
        memcpy(A->x + A->nnz, B->x, B->nnz * sizeof(double));
        memcpy(A->i + A->nnz, B->i, B->nnz * sizeof(int));

        /* set row pointers */
        for (int r = 0; r < child->size; r++)
        {
            A->p[row_offset + r] = A->nnz + B->p[r];
        }

        A->nnz += B->nnz;
        row_offset += child->size;
    }
    A->p[node->size] = A->nnz;
}

static bool is_affine(expr *node)
{
    hstack_expr *hnode = (hstack_expr *) node;

    for (int i = 0; i < hnode->n_args; i++)
    {
        if (!hnode->args[i]->is_affine(hnode->args[i]))
        {
            return false;
        }
    }
    return true;
}

static void free_type_data(expr *node)
{
    hstack_expr *hnode = (hstack_expr *) node;
    for (int i = 0; i < hnode->n_args; i++)
    {
        free_expr(hnode->args[i]);
    }
}

/* Helper function to initialize an hstack expr */
void init_hstack(expr *node, int d1, int d2, int n_vars)
{
    node->d1 = d1;
    node->d2 = d2;
    node->size = d1 * d2;
    node->n_vars = n_vars;
    node->var_id = -1;
    node->refcount = 1;
    node->left = NULL;
    node->right = NULL;
    node->dwork = NULL;
    node->iwork = NULL;
    node->value = (double *) calloc(node->size, sizeof(double));
    node->jacobian = NULL;
    node->wsum_hess = NULL;
    node->jacobian_init = jacobian_init;
    node->wsum_hess_init = NULL;
    node->eval_jacobian = eval_jacobian;
    node->eval_wsum_hess = NULL;
    node->local_jacobian = NULL;
    node->local_wsum_hess = NULL;
    node->forward = forward;
    node->is_affine = is_affine;
    node->free_type_data = free_type_data;
}

expr *new_hstack(expr **args, int n_args, int n_vars)
{
    /* compute second dimension */
    int d2 = 0;
    for (int i = 0; i < n_args; i++)
    {
        d2 += args[i]->d2;
    }

    /* Allocate the type-specific struct */
    hstack_expr *hnode = (hstack_expr *) malloc(sizeof(hstack_expr));
    if (!hnode) return NULL;

    expr *node = &hnode->base;

    /* Initialize base hstack fields */
    init_hstack(node, args[0]->d1, d2, n_vars);

    /* Check if allocation succeeded */
    if (!node->value)
    {
        free(hnode);
        return NULL;
    }

    /* Set type-specific fields */
    hnode->args = args;
    hnode->n_args = n_args;

    for (int i = 0; i < n_args; i++)
    {
        expr_retain(args[i]);
    }

    return node;
}
