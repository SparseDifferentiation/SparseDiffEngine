#include "affine.h"
#include <stdlib.h>
#include <string.h>

/* Promote broadcasts a child expression to a larger shape.
 * Typically used to broadcast a scalar to a vector. */

static void forward(expr *node, const double *u)
{
    /* child's forward pass */
    node->left->forward(node->left, u);

    /* broadcast child value to output shape */
    int child_size = node->left->size;
    for (int i = 0; i < node->size; i++)
    {
        /* replicate pattern: output[i] = child[i % child_size] */
        node->value[i] = node->left->value[i % child_size];
    }
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
    CSR_Matrix *jac = node->jacobian;

    /* Build sparsity pattern by replicating child's rows */
    jac->nnz = 0;
    for (int row = 0; row < node->size; row++)
    {
        jac->p[row] = jac->nnz;
        int child_row = row % child_size;
        for (int k = child_jac->p[child_row]; k < child_jac->p[child_row + 1]; k++)
        {
            jac->i[jac->nnz] = child_jac->i[k];
            jac->nnz++;
        }
    }
    jac->p[node->size] = jac->nnz;
}

static void eval_jacobian(expr *node)
{
    /* evaluate child's jacobian */
    node->left->eval_jacobian(node->left);

    CSR_Matrix *child_jac = node->left->jacobian;
    CSR_Matrix *jac = node->jacobian;
    int child_size = node->left->size;

    /* Copy values only (sparsity pattern set in jacobian_init) */
    int idx = 0;
    for (int row = 0; row < node->size; row++)
    {
        int child_row = row % child_size;
        for (int k = child_jac->p[child_row]; k < child_jac->p[child_row + 1]; k++)
        {
            jac->x[idx++] = child_jac->x[k];
        }
    }
}

static void wsum_hess_init(expr *node)
{
    /* initialize child's wsum_hess */
    node->left->wsum_hess_init(node->left);

    /* same sparsity as child since we're summing weights */
    CSR_Matrix *child_hess = node->left->wsum_hess;
    node->wsum_hess = new_csr_matrix(child_hess->m, child_hess->n, child_hess->nnz);

    /* copy row pointers and column indices (sparsity pattern is constant) */
    memcpy(node->wsum_hess->p, child_hess->p, (child_hess->m + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, child_hess->i, child_hess->nnz * sizeof(int));
    node->wsum_hess->nnz = child_hess->nnz;

    /* allocate workspace for summing weights */
    node->dwork = (double *) malloc(node->left->size * sizeof(double));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* Sum weights that correspond to the same child element */
    int child_size = node->left->size;
    double *summed_w = node->dwork;
    memset(summed_w, 0, child_size * sizeof(double));
    for (int i = 0; i < node->size; i++)
    {
        summed_w[i % child_size] += w[i];
    }

    /* evaluate child's wsum_hess with summed weights */
    node->left->eval_wsum_hess(node->left, summed_w);

    /* copy values only (sparsity pattern set in wsum_hess_init) */
    CSR_Matrix *child_hess = node->left->wsum_hess;
    for (int k = 0; k < child_hess->nnz; k++)
    {
        node->wsum_hess->x[k] = child_hess->x[k];
    }
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_promote(expr *child, int d1, int d2)
{
    expr *node = new_expr(d1, d2, child->n_vars);
    node->forward = forward;
    node->jacobian_init = jacobian_init;
    node->eval_jacobian = eval_jacobian;
    node->is_affine = is_affine;

    node->left = child;
    expr_retain(child);
    node->wsum_hess_init = wsum_hess_init;
    node->eval_wsum_hess = eval_wsum_hess;

    return node;
}
