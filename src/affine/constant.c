#include "affine.h"
#include <string.h>

static void forward(expr *node, const double *u)
{
    /* Constants don't depend on u; values are already set */
    (void) node;
    (void) u;
}

static void jacobian_init(expr *node)
{
    /* Constant jacobian is all zeros: size x n_vars with 0 nonzeros.
     * new_csr_matrix uses calloc for row pointers, so they're already 0. */
    node->jacobian = new_csr_matrix(node->size, node->n_vars, 0);
}

static void eval_jacobian(expr *node)
{
    /* Constant jacobian never changes - nothing to evaluate */
    (void) node;
}

static bool is_affine(const expr *node)
{
    (void) node;
    return true;
}

expr *new_constant(int d1, int d2, int n_vars, const double *values)
{
    expr *node = new_expr(d1, d2, n_vars);
    memcpy(node->value, values, d1 * d2 * sizeof(double));
    node->forward = forward;
    node->is_affine = is_affine;
    node->jacobian_init = jacobian_init;
    node->eval_jacobian = eval_jacobian;

    return node;
}
