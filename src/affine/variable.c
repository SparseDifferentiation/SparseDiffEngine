#include "affine.h"
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    memcpy(node->value, u + node->var_id, node->d1 * node->d2 * sizeof(double));
}

static void jacobian_init(expr *node)
{
    int size = node->d1 * node->d2;
    node->jacobian = new_csr_matrix(size, node->n_vars, size);
    for (int j = 0; j < size; j++)
    {
        node->jacobian->p[j] = j;
        node->jacobian->i[j] = j + node->var_id;
        node->jacobian->x[j] = 1.0;
    }
    node->jacobian->p[size] = size;
}

static bool is_affine(expr *node)
{
    (void) node;
    return true;
}

expr *new_variable(int d1, int d2, int var_id, int n_vars)
{
    expr *node = new_expr(d1, d2, n_vars);
    if (!node) return NULL;

    node->forward = forward;
    node->var_id = var_id;
    node->is_affine = is_affine;
    node->jacobian_init = jacobian_init;
    // node->jacobian = NULL;

    return node;
}
