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

    /* precompute sparsity pattern of this jacobian's node */
    int row_offset = 0;
    CSR_Matrix *A = node->jacobian;
    A->nnz = 0;

    for (int i = 0; i < hnode->n_args; i++)
    {
        expr *child = hnode->args[i];
        CSR_Matrix *B = child->jacobian;

        /* copy columns */
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

static void eval_jacobian(expr *node)
{
    hstack_expr *hnode = (hstack_expr *) node;
    CSR_Matrix *A = node->jacobian;
    A->nnz = 0;

    for (int i = 0; i < hnode->n_args; i++)
    {
        expr *child = hnode->args[i];
        child->eval_jacobian(child);

        /* copy values */
        memcpy(A->x + A->nnz, child->jacobian->x,
               child->jacobian->nnz * sizeof(double));
        A->nnz += child->jacobian->nnz;
    }
}

static bool is_affine(const expr *node)
{
    const hstack_expr *hnode = (const hstack_expr *) node;

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

expr *new_hstack(expr **args, int n_args, int n_vars)
{
    /* compute second dimension */
    int d2 = 0;
    for (int i = 0; i < n_args; i++)
    {
        d2 += args[i]->d2;
    }

    /* Allocate the type-specific struct */
    hstack_expr *hnode = (hstack_expr *) calloc(1, sizeof(hstack_expr));
    expr *node = &hnode->base;
    init_expr(node, args[0]->d1, d2, n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, free_type_data);

    /* Set type-specific fields */
    hnode->args = args;
    hnode->n_args = n_args;

    for (int i = 0; i < n_args; i++)
    {
        expr_retain(args[i]);
    }

    return node;
}
