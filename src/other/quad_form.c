#include "other.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

static void forward(expr *node, const double *u)
{
    expr *x = node->left;

    /* children's forward passes */
    x->forward(x, u);

    /* local forward pass  */
    csr_matvec(node->Q, x->value, node->dwork, 0);

    node->value[0] = 0.0;

    for (int i = 0; i < x->d1; i++)
    {
        node->value[0] += x->value[i] * node->dwork[i];
    }
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    node->dwork = (double *) malloc(x->d1 * sizeof(double));

    /* if x is a variable */
    if (x->var_id != -1)
    {
        node->jacobian = new_csr_matrix(1, node->n_vars, x->d1);
        node->jacobian->p[0] = 0;
        node->jacobian->p[1] = x->d1;

        for (int j = 0; j < x->d1; j++)
        {
            node->jacobian->i[j] = x->var_id + j;
        }
    }
    else /* x is not a variable */
    {
        /* compute required allocation and allocate jacobian */
        bool *col_nz = (bool *) calloc(
            node->n_vars, sizeof(bool)); /* TODO: could use iwork here instead*/
        int nonzero_cols = count_nonzero_cols(x->jacobian, col_nz);
        node->jacobian = new_csr_matrix(1, node->n_vars, nonzero_cols + 1);

        /* precompute column indices */
        node->jacobian->nnz = 0;
        for (int j = 0; j < node->n_vars; j++)
        {
            if (col_nz[j])
            {
                node->jacobian->i[node->jacobian->nnz] = j;
                node->jacobian->nnz++;
            }
        }
        assert(nonzero_cols == node->jacobian->nnz);
        free(col_nz);

        node->jacobian->p[0] = 0;
        node->jacobian->p[1] = node->jacobian->nnz;

        /* store A^T of child's A to simplify chain rule computation */
        node->iwork = (int *) malloc(x->jacobian->n * sizeof(int));
        node->CSR_work = transpose(x->jacobian, node->iwork);
    }
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    CSR_Matrix *Q = node->Q;

    /* if x is a variable */
    if (x->var_id != -1)
    {
        csr_matvec(Q, x->value, node->jacobian->x, 0);

        for (int j = 0; j < x->d1; j++)
        {
            node->jacobian->x[j] *= 2.0;
        }
    }
    else /* x is not a variable */
    {
        /* local jacobian */
        csr_matvec(Q, x->value, node->dwork, 0);

        for (int j = 0; j < x->d1; j++)
        {
            node->dwork[j] *= 2.0;
        }

        /* chain rule */
        csr_matvec_fill_values(node->CSR_work, node->dwork, node->jacobian);
    }
}

expr *new_quad_form(expr *left, CSR_Matrix *Q)
{
    expr *node = new_expr(left->d1, 1, left->n_vars);
    node->left = left;
    expr_retain(left);
    node->Q = Q;
    node->forward = forward;
    node->jacobian_init = jacobian_init;
    node->eval_jacobian = eval_jacobian;
    return node;
}
