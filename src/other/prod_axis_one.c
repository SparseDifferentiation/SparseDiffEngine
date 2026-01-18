#include "other.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IS_ZERO(x) (fabs((x)) < 1e-8)

static void forward(expr *node, const double *u)
{
    /* forward pass of child */
    expr *x = node->left;
    x->forward(x, u);

    prod_axis_one_expr *pnode = (prod_axis_one_expr *) node;
    int d1 = x->d1;
    int d2 = x->d2;

    /* initialize per-row statistics */
    for (int i = 0; i < d1; i++)
    {
        pnode->num_of_zeros[i] = 0;
        pnode->zero_index[i] = -1;
        pnode->prod_nonzero[i] = 1.0;
    }

    /* iterate over columns */
    for (int col = 0; col < d2; col++)
    {
        int start = col * d1;
        int end = start + d1;

        for (int idx = start; idx < end; idx++)
        {
            int row = idx - start;
            if (IS_ZERO(x->value[idx]))
            {
                pnode->num_of_zeros[row]++;
                pnode->zero_index[row] = col;
            }
            else
            {
                pnode->prod_nonzero[row] *= x->value[idx];
            }
        }
    }

    /* compute output values */
    for (int i = 0; i < d1; i++)
    {
        node->value[i] = (pnode->num_of_zeros[i] > 0) ? 0.0 : pnode->prod_nonzero[i];
    }
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;

    /* initialize child's jacobian */
    x->jacobian_init(x);

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        node->jacobian = new_csr_matrix(node->size, node->n_vars, x->size);

        /* set row pointers (each row has d2 nnzs) */
        for (int row = 0; row < x->d1; row++)
        {
            node->jacobian->p[row] = row * x->d2;
        }
        node->jacobian->p[x->d1] = x->size;

        /* set column indices */
        for (int row = 0; row < x->d1; row++)
        {
            int start = row * x->d2;
            for (int col = 0; col < x->d2; col++)
            {
                node->jacobian->i[start + col] = x->var_id + col * x->d1 + row;
            }
        }
    }
    else
    {
        assert(false && "child must be a variable");
    }
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    prod_axis_one_expr *pnode = (prod_axis_one_expr *) node;

    double *J_vals = node->jacobian->x;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        /* process each row */
        for (int row = 0; row < x->d1; row++)
        {
            int num_zeros = pnode->num_of_zeros[row];
            int start = row * x->d2;

            if (num_zeros == 0)
            {
                for (int col = 0; col < x->d2; col++)
                {
                    int idx = col * x->d1 + row;
                    J_vals[start + col] = node->value[row] / x->value[idx];
                }
            }
            else if (num_zeros == 1)
            {
                memset(J_vals + start, 0, sizeof(double) * x->d2);
                J_vals[start + pnode->zero_index[row]] = pnode->prod_nonzero[row];
            }
            else
            {
                memset(J_vals + start, 0, sizeof(double) * x->d2);
            }
        }
    }
    else
    {
        assert(false && "child must be a variable");
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        /* Hessian structure: for each row i of the input matrix,
         * the Hessian of the product (w[i] * prod(row i)) has entries
         * at the column indices corresponding to the columns in that row.
         * Each row i has d2 non-zero columns, and we have d2 x d2 interactions.
         * Total nnz = d1 * d2 * d2
         */
        int nnz = x->d1 * x->d2 * x->d2;
        node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, nnz);
        CSR_Matrix *H = node->wsum_hess;

        /* Fill row pointers and column indices */
        int nnz_per_row = x->d2;
        for (int i = 0; i < x->size; i++)
        {
            int matrix_row = i % x->d1;
            H->p[x->var_id + i] = i * nnz_per_row;

            /* For this variable (at matrix position [matrix_row, ...]),
             * the Hessian has entries with all columns in the same matrix_row */
            for (int j = 0; j < x->d2; j++)
            {
                /* Global column index for column j of this row */
                int global_col = x->var_id + matrix_row + j * x->d1;
                H->i[i * nnz_per_row + j] = global_col;
            }
        }

        /* fill row pointers for rows after the variable */
        for (int i = x->var_id + x->size; i <= node->n_vars; i++)
        {
            H->p[i] = nnz;
        }
    }
    else
    {
        assert(false && "child must be a variable");
    }

    /* print row pointers */
    // for (int i = 0; i <= node->n_vars; i++)
    //{
    //     printf("H->p[%d] = %d\n", i, node->wsum_hess->p[i]);
    // }
    //
    ///* print column indices */
    // for (int i = 0; i < node->wsum_hess->nnz; i++)
    //{
    //     printf("H->i[%d] = %d\n", i, node->wsum_hess->i[i]);
    // }

    printf("done wsum_hess_init prod_axis_one\n");
}

static inline void wsum_hess_row_no_zeros(expr *node, const double *w, int row,
                                          int d2)
{
    expr *x = node->left;
    prod_axis_one_expr *pnode = (prod_axis_one_expr *) node;
    CSR_Matrix *H = node->wsum_hess;
    double *H_vals = H->x;

    double scale = w[row] * node->value[row];
    int idx_i = row; /* Start with the row offset */

    /* For each variable in this row, fill in Hessian entries */
    for (int i = 0; i < d2; i++)
    {
        int var_i = x->var_id + row + i * x->d1;
        int row_start = H->p[var_i];
        idx_i = i * x->d1 + row; /* Element at matrix [row, i] */

        /* Row var_i has nnz entries with columns from the same matrix row */
        for (int j = 0; j < d2; j++)
        {
            if (i == j)
            {
                /* Diagonal entries are 0 */
                H_vals[row_start + j] = 0.0;
            }
            else
            {
                /* Off-diagonal: scale / (x[i] * x[j]) */
                int idx_j = j * x->d1 + row; /* Element at matrix [row, j] */
                H_vals[row_start + j] = scale / (x->value[idx_i] * x->value[idx_j]);
            }
        }
    }
    (void) pnode; /* suppress unused warning */
}

static inline void wsum_hess_row_one_zero(expr *node, const double *w, int row,
                                          int d2)
{
    expr *x = node->left;
    prod_axis_one_expr *pnode = (prod_axis_one_expr *) node;
    CSR_Matrix *H = node->wsum_hess;
    double *H_vals = H->x;

    int p = pnode->zero_index[row]; /* zero column index */
    double w_prod = w[row] * pnode->prod_nonzero[row];

    /* For each variable in this row */
    for (int i = 0; i < d2; i++)
    {
        int var_i = x->var_id + row + i * x->d1;
        int row_start = H->p[var_i];

        /* Row var_i has nnz entries with columns from the same matrix row */
        for (int j = 0; j < d2; j++)
        {
            if (i == p && j != p)
            {
                /* Row p (zero row), column j (nonzero) */
                int idx_j = j * x->d1 + row;
                H_vals[row_start + j] = w_prod / x->value[idx_j];
            }
            else if (j == p && i != p)
            {
                /* Row i (nonzero), column p (zero) */
                int idx_i = i * x->d1 + row;
                H_vals[row_start + j] = w_prod / x->value[idx_i];
            }
            else
            {
                /* All other entries are zero */
                H_vals[row_start + j] = 0.0;
            }
        }
    }
}

static inline void wsum_hess_row_two_zeros(expr *node, const double *w, int row,
                                           int d2)
{
    expr *x = node->left;
    prod_axis_one_expr *pnode = (prod_axis_one_expr *) node;
    CSR_Matrix *H = node->wsum_hess;
    double *H_vals = H->x;

    /* find indices p and q where row has zeros */
    int p = -1, q = -1;
    for (int c = 0; c < d2; c++)
    {
        int idx = c * node->left->d1 + row;
        if (IS_ZERO(x->value[idx]))
        {
            if (p == -1)
            {
                p = c;
            }
            else
            {
                q = c;
                break;
            }
        }
    }
    assert(p != -1 && q != -1);

    double hess_val = w[row] * pnode->prod_nonzero[row];

    /* For each variable in this row */
    for (int i = 0; i < d2; i++)
    {
        int var_i = x->var_id + row + i * x->d1;
        int row_start = H->p[var_i];

        /* Row var_i has nnz entries with columns from the same matrix row */
        for (int j = 0; j < d2; j++)
        {
            /* Only (p,q) and (q,p) are nonzero */
            if ((i == p && j == q) || (i == q && j == p))
            {
                H_vals[row_start + j] = hess_val;
            }
            else
            {
                H_vals[row_start + j] = 0.0;
            }
        }
    }
}

static inline void wsum_hess_row_many_zeros(expr *node, int row, int d2)
{
    CSR_Matrix *H = node->wsum_hess;
    double *H_vals = H->x;
    expr *x = node->left;

    /* For each variable in this row, zero out all entries */
    for (int i = 0; i < d2; i++)
    {
        int var_i = x->var_id + row + i * x->d1;
        int row_start = H->p[var_i];

        /* Each variable has d2 entries */
        for (int j = 0; j < d2; j++)
        {
            H_vals[row_start + j] = 0.0;
        }
    }
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    prod_axis_one_expr *pnode = (prod_axis_one_expr *) node;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        for (int row = 0; row < x->d1; row++)
        {
            int num_zeros = pnode->num_of_zeros[row];

            if (num_zeros == 0)
            {
                wsum_hess_row_no_zeros(node, w, row, x->d2);
            }
            else if (num_zeros == 1)
            {
                wsum_hess_row_one_zero(node, w, row, x->d2);
            }
            else if (num_zeros == 2)
            {
                wsum_hess_row_two_zeros(node, w, row, x->d2);
            }
            else
            {
                wsum_hess_row_many_zeros(node, row, x->d2);
            }
        }
    }
    else
    {
        assert(false && "child must be a variable");
    }
}

static bool is_affine(const expr *node)
{
    (void) node;
    return false;
}

static void free_type_data(expr *node)
{
    prod_axis_one_expr *pnode = (prod_axis_one_expr *) node;
    free(pnode->num_of_zeros);
    free(pnode->zero_index);
    free(pnode->prod_nonzero);
}

expr *new_prod_axis_one(expr *child)
{
    prod_axis_one_expr *pnode =
        (prod_axis_one_expr *) calloc(1, sizeof(prod_axis_one_expr));
    expr *node = &pnode->base;

    /* output is always a row vector 1 x d1 (one product per row) */
    init_expr(node, 1, child->d1, child->n_vars, forward, jacobian_init,
              eval_jacobian, is_affine, wsum_hess_init, eval_wsum_hess,
              free_type_data);

    /* allocate arrays to store per-row statistics */
    pnode->num_of_zeros = (int *) calloc(child->d1, sizeof(int));
    pnode->zero_index = (int *) calloc(child->d1, sizeof(int));
    pnode->prod_nonzero = (double *) calloc(child->d1, sizeof(double));

    node->left = child;
    expr_retain(child);

    return node;
}
