#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_composite(void)
{
    double u[2] = {1.0, 2.0};
    double c[2] = {1.0, 1.0};

    /* Build tree: sin(exp(x) + c) */
    expr *var = new_variable(2, 1, 0, 2);
    expr *exp_node = new_exp(var);
    expr *const_node = new_parameter(2, 1, PARAM_FIXED, 0, c);
    expr *sum = new_add(exp_node, const_node);
    expr *sin_node = new_sin(sum);

    sin_node->forward(sin_node, u);

    double correct[2] = {sin(exp(1.0) + 1.0), sin(exp(2.0) + 1.0)};
    mu_assert("failed", cmp_double_array(sin_node->value, correct, 2));

    free_expr(sin_node);
    return 0;
}
