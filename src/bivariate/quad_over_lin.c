#include "bivariate.h"
#include "subexpr.h"
#include "utils/CSC_Matrix.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

// ------------------------------------------------------------------------------
// Implementation of quad-over-lin. The second argument will always be a variable
// that only appears in this node and as the left-hand side of another equality
// constraint.
// ------------------------------------------------------------------------------
static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    expr *y = node->right;

    /* children's forward passes */
    x->forward(x, u);
    y->forward(y, u);

    /* local forward pass */
    node->value[0] = 0.0;

    for (int i = 0; i < x->d1; i++)
    {
        node->value[0] += x->value[i] * x->value[i];
    }

    node->value[0] /= y->value[0];
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* if left node is a variable */
    if (x->var_id != -1)
    {
        node->jacobian = new_csr_matrix(1, node->n_vars, x->d1 + 1);
        node->jacobian->p[0] = 0;
        node->jacobian->p[1] = x->d1 + 1;

        /* if x has lower idx than y*/
        if (x->var_id < y->var_id)
        {
            for (int j = 0; j < x->d1; j++)
            {
                node->jacobian->i[j] = x->var_id + j;
            }
            node->jacobian->i[x->d1] = y->var_id;
        }
        else /* y has lower idx than x */
        {
            node->jacobian->i[0] = y->var_id;
            for (int j = 0; j < x->d1; j++)
            {
                node->jacobian->i[j + 1] = x->var_id + j;
            }
        }
    }
    else /* left node is not a variable (guaranteed to be a linear operator) */
    {
        linear_op_expr *lin_x = (linear_op_expr *) x;
        node->dwork = (double *) malloc(x->d1 * sizeof(double));

        /* compute required allocation and allocate jacobian */
        bool *col_nz = (bool *) calloc(
            node->n_vars, sizeof(bool)); /* TODO: could use iwork here instead*/
        int nonzero_cols = count_nonzero_cols(lin_x->base.jacobian, col_nz);
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

        /* insert y variable index at correct position */
        insert_idx(y->var_id, node->jacobian->i, node->jacobian->nnz);
        node->jacobian->nnz += 1;
        node->jacobian->p[0] = 0;
        node->jacobian->p[1] = node->jacobian->nnz;

        /* find position where y should be inserted */
        node->iwork = (int *) malloc(sizeof(int));
        for (int j = 0; j < node->jacobian->nnz; j++)
        {
            if (node->jacobian->i[j] == y->var_id)
            {
                node->iwork[0] = j;
                break;
            }
        }
    }
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;

    /* if x is a variable */
    if (x->var_id != -1)
    {
        /* if x has lower idx than y*/
        if (x->var_id < y->var_id)
        {
            for (int j = 0; j < x->d1; j++)
            {
                node->jacobian->x[j] = (2.0 * x->value[j]) / y->value[0];
            }
            node->jacobian->x[x->d1] = -node->value[0] / y->value[0];
        }
        else /* y has lower idx than x */
        {
            node->jacobian->x[0] = -node->value[0] / y->value[0];
            for (int j = 0; j < x->d1; j++)
            {
                node->jacobian->x[j + 1] = (2.0 * x->value[j]) / y->value[0];
            }
        }
    }
    else /* x is not a variable */
    {
        CSC_Matrix *A_csc = ((linear_op_expr *) x)->A_csc;

        /* local jacobian */
        for (int j = 0; j < x->d1; j++)
        {
            node->dwork[j] = (2.0 * x->value[j]) / y->value[0];
        }

        /* chain rule (no derivative wrt y) using CSC format */
        csc_matvec_fill_values(A_csc, node->dwork, node->jacobian);

        /* insert derivative wrt y at right place (for correctness this assumes
           that y does not appear in the denominator, but this will always be
           the case since y is a new variable for the numerator) */
        node->jacobian->x[node->iwork[0]] = -node->value[0] / y->value[0];
    }
}

expr *new_quad_over_lin(expr *left, expr *right)
{
    expr *node = new_expr(left->d1, 1, left->n_vars);
    node->left = left;
    node->right = right;
    expr_retain(left);
    expr_retain(right);
    node->forward = forward;
    node->jacobian_init = jacobian_init;
    node->eval_jacobian = eval_jacobian;
    return node;
}
