#include "affine/variable.h"
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    memcpy(node->value, u + node->var_id, node->m * sizeof(double));
}

static bool is_affine(expr *node)
{
    (void) node;
    return true;
}

expr *new_variable(int m, int var_id, int n_vars)
{
    expr *node = new_expr(m, n_vars);
    if (!node) return NULL;

    node->forward = forward;
    node->var_id = var_id;
    node->is_affine = is_affine;

    return node;
}
