#include "affine.h"
#include "utils/int_double_pair.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    expr *x = node->left;

    /* child's forward pass */
    x->forward(x, u);

    /* local forward pass */
    double sum = 0.0;
    int row_spacing = x->d1 + 1;
    for (int idx = 0; idx < x->size; idx += row_spacing)
    {
        sum += x->value[idx];
    }

    node->value[0] = sum;
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;

    /* initialize child's jacobian */
    x->jacobian_init(x);

    /* count total nnz in the rows of the jacobian that should be summed */
    const CSR_Matrix *A = x->jacobian;
    int total_nnz = 0;
    int row_spacing = x->d1 + 1;
    for (int row = 0; row < A->m; row += row_spacing)
    {
        total_nnz += (A->p[row + 1] - A->p[row]);
    }

    node->jacobian = new_csr_matrix(1, node->n_vars, total_nnz);
    ((trace_expr *) node)->int_double_pairs = new_int_double_pair_array(total_nnz);
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    trace_expr *tnode = (trace_expr *) node;

    /* evaluate child's jacobian */
    x->eval_jacobian(x);

    /* local jacobian */
    sum_spaced_rows_into_row_csr(x->jacobian, node->jacobian,
                                 tnode->int_double_pairs, 0, x->d1 + 1);
}

/* Placeholders for Hessian-related functions */
static void wsum_hess_init(expr *node)
{
    expr *x = node->left;

    /* initialize child's hessian */
    x->wsum_hess_init(x);

    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, x->wsum_hess->nnz);
    node->dwork = (double *) calloc(x->size, sizeof(double));

    /* TODO: here we could copy over sparsity pattern once we have checked
    that all atoms fill their sparsity pattern in the init functions. Perhaps
    we should only take sparsity pattern of rows that are summed? Not the rows
    which will get zero weight in the hessian. That would be very cool.
    But must eval_wsum_hess then also ignore contributions with zero weight? that
    would be bad. */

    /* lets conclude that the hessian can be made more sophisticated */

    /* but perhaps let's keep it as simple as possible! */
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;

    int row_spacing = x->d1 + 1;
    for (int i = 0; i < x->size; i += row_spacing)
    {
        node->dwork[i] = w[0];
    }

    x->eval_wsum_hess(x, node->dwork);

    /* TODO: here we only need to copy over values once we have filled the sparsity
     * pattern in wsum_hess_init*/
    node->wsum_hess->nnz = x->wsum_hess->nnz;
    memcpy(node->wsum_hess->p, x->wsum_hess->p, sizeof(int) * (node->n_vars + 1));
    memcpy(node->wsum_hess->i, x->wsum_hess->i, sizeof(int) * x->wsum_hess->nnz);
    memcpy(node->wsum_hess->x, x->wsum_hess->x, sizeof(double) * x->wsum_hess->nnz);

    /* This might contain many many zeros actually! Hmm...*/
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    trace_expr *tnode = (trace_expr *) node;
    free_int_double_pair_array(tnode->int_double_pairs);
}

expr *new_trace(expr *child)
{
    /* Output is a single scalar */
    int d1 = 1;

    trace_expr *tnode = (trace_expr *) calloc(1, sizeof(trace_expr));
    expr *node = &tnode->base;
    init_expr(node, d1, 1, child->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, free_type_data);
    node->left = child;
    expr_retain(child);

    /* Hessian placeholders */
    node->wsum_hess_init = wsum_hess_init;
    node->eval_wsum_hess = eval_wsum_hess;

    /* Initialize type-specific fields */
    tnode->int_double_pairs = NULL;

    // just for debugging, should be removed
    strcpy(node->name, "trace");

    return node;
}
