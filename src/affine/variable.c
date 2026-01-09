#include "affine.h"
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    memcpy(node->value, u + node->var_id, node->d1 * node->d2 * sizeof(double));
}

static void jacobian_init(expr *node)
{
    node->jacobian = new_csr_matrix(node->size, node->n_vars, node->size);
    for (int j = 0; j < node->size; j++)
    {
        node->jacobian->p[j] = j;
        node->jacobian->i[j] = j + node->var_id;
        node->jacobian->x[j] = 1.0;
    }
    node->jacobian->p[node->size] = node->size;
}

static void eval_jacobian(expr *node)
{
    /* Jacobian is already initialized with correct values */
    (void) node;
}

static void eval_jacobian(expr *node)
{
    /* Variable jacobian never changes - nothing to evaluate */
    (void) node;
}

static bool is_affine(const expr *node)
{
    (void) node;
    return true;
}

expr *new_variable(int d1, int d2, int var_id, int n_vars)
{
    expr *node = new_expr(d1, d2, n_vars);
    node->forward = forward;
    node->var_id = var_id;
    node->is_affine = is_affine;
    node->jacobian_init = jacobian_init;
    node->eval_jacobian = eval_jacobian;

    return node;
}
