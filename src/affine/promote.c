#include "affine.h"
#include "subexpr.h"
#include <stdlib.h>

/* Promote broadcasts a child expression to a larger shape.
 * Typically used to broadcast a scalar to a vector. */

static void forward(expr *node, const double *u)
{
    promote_expr *prom = (promote_expr *) node;

    /* child's forward pass */
    node->left->forward(node->left, u);

    /* broadcast child value to output shape */
    int child_size = node->left->size;
    for (int i = 0; i < node->size; i++)
    {
        /* replicate pattern: output[i] = child[i % child_size] */
        node->value[i] = node->left->value[i % child_size];
    }
    (void) prom; /* unused for now, shape info stored in d1/d2 */
}

static void jacobian_init(expr *node)
{
    /* initialize child's jacobian */
    node->left->jacobian_init(node->left);

    /* Each output row copies a row from child's jacobian (with wrapping).
     * For scalar->vector: all rows are copies of the single child row.
     * nnz = (output_size / child_size) * child_jac_nnz */
    CSR_Matrix *child_jac = node->left->jacobian;
    int child_size = node->left->size;
    int repeat = (node->size + child_size - 1) / child_size;
    int nnz_max = repeat * child_jac->nnz;
    if (nnz_max == 0) nnz_max = 1;

    node->jacobian = new_csr_matrix(node->size, node->n_vars, nnz_max);
}

static void eval_jacobian(expr *node)
{
    /* evaluate child's jacobian */
    node->left->eval_jacobian(node->left);

    CSR_Matrix *child_jac = node->left->jacobian;
    CSR_Matrix *jac = node->jacobian;
    int child_size = node->left->size;

    /* Build jacobian by replicating child's rows */
    jac->nnz = 0;
    for (int row = 0; row < node->size; row++)
    {
        jac->p[row] = jac->nnz;

        /* which child row does this output row correspond to? */
        int child_row = row % child_size;

        /* copy entries from child_jac row */
        for (int k = child_jac->p[child_row]; k < child_jac->p[child_row + 1]; k++)
        {
            jac->i[jac->nnz] = child_jac->i[k];
            jac->x[jac->nnz] = child_jac->x[k];
            jac->nnz++;
        }
    }
    jac->p[node->size] = jac->nnz;
}

static void wsum_hess_init(expr *node)
{
    /* initialize child's wsum_hess */
    node->left->wsum_hess_init(node->left);

    /* same sparsity as child since we're summing weights */
    CSR_Matrix *child_hess = node->left->wsum_hess;
    node->wsum_hess = new_csr_matrix(child_hess->m, child_hess->n, child_hess->nnz);
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* Sum weights that correspond to the same child element */
    int child_size = node->left->size;
    double *summed_w = (double *) calloc(child_size, sizeof(double));

    for (int i = 0; i < node->size; i++)
    {
        summed_w[i % child_size] += w[i];
    }

    /* evaluate child's wsum_hess with summed weights */
    node->left->eval_wsum_hess(node->left, summed_w);
    free(summed_w);

    /* copy child's wsum_hess */
    CSR_Matrix *child_hess = node->left->wsum_hess;
    CSR_Matrix *hess = node->wsum_hess;

    for (int i = 0; i <= child_hess->m; i++)
    {
        hess->p[i] = child_hess->p[i];
    }

    int nnz = child_hess->p[child_hess->m];
    for (int k = 0; k < nnz; k++)
    {
        hess->i[k] = child_hess->i[k];
        hess->x[k] = child_hess->x[k];
    }
    hess->nnz = nnz;
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_promote(expr *child, int d1, int d2)
{
    /* Allocate the type-specific struct */
    promote_expr *prom = (promote_expr *) calloc(1, sizeof(promote_expr));
    expr *node = &prom->base;

    init_expr(node, d1, d2, child->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, NULL);

    node->left = child;
    expr_retain(child);
    node->wsum_hess_init = wsum_hess_init;
    node->eval_wsum_hess = eval_wsum_hess;

    return node;
}
