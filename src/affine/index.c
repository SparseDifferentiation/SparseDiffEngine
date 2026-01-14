#include "affine.h"
#include "subexpr.h"
#include <stdlib.h>
#include <string.h>

/* Index/slicing: y = child[indices] where indices is a list of flat positions */

/* Check if indices array contains duplicates using a bitmap.
 * Returns true if duplicates exist, false otherwise. */
static bool check_for_duplicates(const int *indices, int n_idxs, int max_idx)
{
    bool *seen = (bool *) calloc(max_idx, sizeof(bool));
    bool has_dup = false;
    for (int i = 0; i < n_idxs && !has_dup; i++)
    {
        if (seen[indices[i]])
        {
            has_dup = true;
        }
        seen[indices[i]] = true;
    }
    free(seen);
    return has_dup;
}

static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    index_expr *idx = (index_expr *) node;

    /* child's forward pass */
    x->forward(x, u);

    /* gather selected elements */
    for (int i = 0; i < idx->n_idxs; i++)
    {
        node->value[i] = x->value[idx->indices[i]];
    }
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    index_expr *idx = (index_expr *) node;
    x->jacobian_init(x);

    CSR_Matrix *Jx = x->jacobian;
    CSR_Matrix *J = new_csr_matrix(node->size, node->n_vars, Jx->nnz);

    /* set sparsity pattern */
    J->p[0] = 0;
    for (int i = 0; i < idx->n_idxs; i++)
    {
        int row = idx->indices[i];
        int len = Jx->p[row + 1] - Jx->p[row];
        memcpy(J->i + J->p[i], Jx->i + Jx->p[row], len * sizeof(int));
        J->p[i + 1] = J->p[i] + len;
    }

    node->jacobian = J;
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    index_expr *idx = (index_expr *) node;
    x->eval_jacobian(x);

    CSR_Matrix *J = node->jacobian;
    CSR_Matrix *Jx = x->jacobian;

    for (int i = 0; i < idx->n_idxs; i++)
    {
        int row = idx->indices[i];
        int len = J->p[i + 1] - J->p[i];
        memcpy(J->x + J->p[i], Jx->x + Jx->p[row], len * sizeof(double));
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;

    /* initialize child's wsum_hess */
    x->wsum_hess_init(x);

    /* for setting weight vector to evaluate hessian of child */
    node->dwork = (double *) calloc(x->size, sizeof(double));

    /* in the implementation of eval_wsum_hess we evaluate the
       child's hessian with a weight vector that has w[i] = 0
       if i is not included in idx->indices. This can lead to
       many numerical zeros in child->wsum_hess that are actually
       structural zeros, but we do not try to exploit that sparsity
       right now. */
    CSR_Matrix *H_child = x->wsum_hess;
    node->wsum_hess = new_csr_matrix(H_child->m, H_child->n, H_child->nnz);
    memcpy(node->wsum_hess->p, H_child->p, (H_child->m + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, H_child->i, H_child->nnz * sizeof(int));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *child = node->left;
    index_expr *idx = (index_expr *) node;

    if (idx->has_duplicates)
    {
        /* zero and accumulate for repeated indices */
        memset(node->dwork, 0, child->size * sizeof(double));
        for (int i = 0; i < idx->n_idxs; i++)
        {
            node->dwork[idx->indices[i]] += w[i];
        }
    }
    else
    {
        /* direct write (no memset needed, no accumulation) */
        for (int i = 0; i < idx->n_idxs; i++)
        {
            node->dwork[idx->indices[i]] = w[i];
        }
    }

    /* delegate to child */
    child->eval_wsum_hess(child, node->dwork);

    /* copy values from child */
    memcpy(node->wsum_hess->x, child->wsum_hess->x,
           child->wsum_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    index_expr *idx = (index_expr *) node;
    if (idx->indices)
    {
        free(idx->indices);
        idx->indices = NULL;
    }
}

expr *new_index(expr *child, const int *indices, int n_idxs)
{
    /* allocate type-specific struct */
    index_expr *idx = (index_expr *) calloc(1, sizeof(index_expr));
    expr *node = &idx->base;

    /* output shape is (n_idxs, 1) - flattened */
    init_expr(node, n_idxs, 1, child->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, free_type_data);

    node->left = child;
    expr_retain(child);

    node->wsum_hess_init = wsum_hess_init;
    node->eval_wsum_hess = eval_wsum_hess;

    /* copy indices */
    idx->indices = (int *) malloc(n_idxs * sizeof(int));
    memcpy(idx->indices, indices, n_idxs * sizeof(int));
    idx->n_idxs = n_idxs;

    /* detect duplicates for Hessian optimization */
    idx->has_duplicates = check_for_duplicates(indices, n_idxs, child->size);
    return node;
}
