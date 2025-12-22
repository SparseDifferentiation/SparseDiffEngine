#include "affine/constant.h"
#include <string.h>

static void forward(expr *node, const double *u)
{
    /* Constants don't depend on u; values are already set */
    (void) node;
    (void) u;
}

static bool is_affine(expr *node)
{
    (void) node;
    return true; /* constant is affine */
}

expr *new_constant(int m, const double *values)
{
    expr *node = new_expr(m, 0);
    if (!node) return NULL;

    /* Copy constant values */
    memcpy(node->value, values, m * sizeof(double));

    node->forward = forward;
    node->is_affine = is_affine;

    return node;
}
