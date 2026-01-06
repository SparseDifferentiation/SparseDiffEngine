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
    if (x->var_id != NOT_A_VARIABLE)
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
    if (x->var_id != NOT_A_VARIABLE)
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
           that y does not appear in the numerator, but this will always be
           the case since y is a new variable for the denominator */
        node->jacobian->x[node->iwork[0]] = -node->value[0] / y->value[0];
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;
    expr *y = node->right;
    int var_id_x = x->var_id;
    int var_id_y = y->var_id;

    /* if left node is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, 3 * x->d1 + 1);
        CSR_Matrix *H = node->wsum_hess;

        /* if x has lower idx than y*/
        if (var_id_x < var_id_y)
        {
            /* x rows: each row has 2 entries (diagonal element + element for y) */
            for (int i = 0; i < x->d1; i++)
            {
                H->p[var_id_x + i] = 2 * i;
                H->i[2 * i] = var_id_x + i;
                H->i[2 * i + 1] = var_id_y;
            }

            /* rows between x and y are empty, all point to same offset */
            int offset = 2 * x->d1;
            for (int i = var_id_x + x->d1; i <= var_id_y; i++)
            {
                H->p[i] = offset;
            }

            /* y row has d1 + 1 entries */
            H->p[var_id_y + 1] = offset + x->d1 + 1;
            for (int i = 0; i < x->d1; i++)
            {
                H->i[offset + i] = var_id_x + i;
            }
            H->i[offset + x->d1] = var_id_y;

            /* remaining rows are empty */
            for (int i = var_id_y + 1; i <= node->n_vars; i++)
            {
                H->p[i] = 3 * x->d1 + 1;
            }
        }
        else /* y has lower idx than x */
        {
            /* y row has d1 + 1 entries */
            H->p[var_id_y + 1] = x->d1 + 1;
            H->i[0] = var_id_y;
            for (int i = 0; i < x->d1; i++)
            {
                H->i[i + 1] = var_id_x + i;
            }

            /* rows between y and x are empty, all point to same offset */
            int offset = x->d1 + 1;
            for (int i = var_id_y + 1; i <= var_id_x; i++)
            {
                H->p[i] = offset;
            }

            /* x rows: each row has 2 entries */
            for (int i = 0; i < x->d1; i++)
            {
                H->p[var_id_x + i] = offset + 2 * i;
                H->i[offset + 2 * i] = var_id_y;
                H->i[offset + 2 * i + 1] = var_id_x + i;
            }

            /* remaining rows are empty */
            for (int i = var_id_x + x->d1; i <= node->n_vars; i++)
            {
                H->p[i] = 3 * x->d1 + 1;
            }
        }
    }
    else
    {
        /* TODO: implement */
        assert(false && "not implemented");
    }
}

static void eval_wsum_hess(expr *node, const double *w)
{
    double *x = node->left->value;
    double y = node->right->value[0];
    double *H = node->wsum_hess->x;
    int var_id_x = node->left->var_id;
    int var_id_y = node->right->var_id;
    int x_d1 = node->left->d1;
    double a = (2.0 * w[0]) / y;
    double b = -(2.0 * w[0]) / (y * y);

    /* if left node is a variable */
    if (var_id_x != NOT_A_VARIABLE)
    {
        /* if x has lower idx than y*/
        if (var_id_x < var_id_y)
        {
            /* x rows*/
            for (int i = 0; i < x_d1; i++)
            {
                H[2 * i] = a;
                H[2 * i + 1] = b * x[i];
            }

            /* y row */
            int offset = 2 * x_d1;
            for (int i = 0; i < x_d1; i++)
            {
                H[offset + i] = b * x[i];
            }
            H[offset + x_d1] = -b * node->value[0];
        }
        else /* y has lower idx than x */
        {
            /* y row */
            H[0] = -b * node->value[0];
            for (int i = 0; i < x_d1; i++)
            {
                H[i + 1] = b * x[i];
            }

            /* x rows*/
            int offset = x_d1 + 1;
            for (int i = 0; i < x_d1; i++)
            {
                H[offset + 2 * i] = b * x[i];
                H[offset + 2 * i + 1] = a;
            }
        }
    }
    else
    {
        /* TODO: implement */
        assert(false && "not implemented");
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
    node->wsum_hess_init = wsum_hess_init;
    node->eval_wsum_hess = eval_wsum_hess;
    return node;
}
