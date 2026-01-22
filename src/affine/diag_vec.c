// SPDX-License-Identifier: Apache-2.0

#include "affine.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* diag_vec: converts a vector of size n into an n×n diagonal matrix.
 * In Fortran (column-major) order, element i of the input maps to
 * position i*(n+1) in the flattened output (the diagonal positions). */

static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    int n = x->size;

    /* child's forward pass */
    x->forward(x, u);

    /* zero-initialize output */
    memset(node->value, 0, node->size * sizeof(double));

    /* place input elements on the diagonal */
    for (int i = 0; i < n; i++)
    {
        node->value[i * (n + 1)] = x->value[i];
    }
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    int n = x->size;
    x->jacobian_init(x);

    CSR_Matrix *Jx = x->jacobian;

    /* Output Jacobian has n² rows, but only n rows (diagonal positions) are non-zero.
     * We allocate space for the same nnz as child, but with n² rows. */
    CSR_Matrix *J = new_csr_matrix(node->size, node->n_vars, Jx->nnz);

    /* Build row pointers: rows at diagonal positions copy from child,
     * all other rows are empty */
    J->p[0] = 0;
    int child_row = 0;
    for (int out_row = 0; out_row < node->size; out_row++)
    {
        if (out_row == child_row * (n + 1) && child_row < n)
        {
            /* This is a diagonal position - copy sparsity from child row */
            int len = Jx->p[child_row + 1] - Jx->p[child_row];
            memcpy(J->i + J->p[out_row], Jx->i + Jx->p[child_row], len * sizeof(int));
            J->p[out_row + 1] = J->p[out_row] + len;
            child_row++;
        }
        else
        {
            /* Non-diagonal row - empty */
            J->p[out_row + 1] = J->p[out_row];
        }
    }

    node->jacobian = J;
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    int n = x->size;
    x->eval_jacobian(x);

    CSR_Matrix *J = node->jacobian;
    CSR_Matrix *Jx = x->jacobian;

    /* Copy values from child Jacobian to diagonal positions */
    int child_row = 0;
    for (int out_row = 0; out_row < node->size && child_row < n; out_row++)
    {
        if (out_row == child_row * (n + 1))
        {
            int len = J->p[out_row + 1] - J->p[out_row];
            memcpy(J->x + J->p[out_row], Jx->x + Jx->p[child_row], len * sizeof(double));
            child_row++;
        }
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;

    /* initialize child's wsum_hess */
    x->wsum_hess_init(x);

    /* workspace for extracting diagonal weights */
    node->dwork = (double *) calloc(x->size, sizeof(double));

    /* Copy child's Hessian structure (diag_vec is linear, so its own Hessian is zero) */
    CSR_Matrix *Hx = x->wsum_hess;
    node->wsum_hess = new_csr_matrix(Hx->m, Hx->n, Hx->nnz);
    memcpy(node->wsum_hess->p, Hx->p, (Hx->m + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, Hx->i, Hx->nnz * sizeof(int));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    int n = x->size;

    /* Extract weights from diagonal positions of w (which has n² elements) */
    for (int i = 0; i < n; i++)
    {
        node->dwork[i] = w[i * (n + 1)];
    }

    /* Evaluate child's Hessian with extracted weights */
    x->eval_wsum_hess(x, node->dwork);
    memcpy(node->wsum_hess->x, x->wsum_hess->x, x->wsum_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_diag_vec(expr *child)
{
    /* child must be a vector: either column (n, 1) or row (1, n) */
    assert(child->d1 == 1 || child->d2 == 1);

    /* n is the number of elements (works for both row and column vectors) */
    int n = child->size;
    expr *node = (expr *) calloc(1, sizeof(expr));
    init_expr(node, n, n, child->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, NULL);
    node->left = child;
    expr_retain(child);

    return node;
}
