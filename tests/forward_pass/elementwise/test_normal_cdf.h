#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

const char *test_normal_cdf(void)
{
    double u[3] = {1.0, 2.0, 3.0};
    expr *var = new_variable(3, 1, 0, 3);
    expr *node = new_normal_cdf(var);
    node->forward(node, u);
    /* computed in python */
    double correct[3] = {0.8413447460685429,
                         0.9772498680518208,
                         0.9986501019683699};
    mu_assert("fail", cmp_double_array(node->value, correct, 3));
    free_expr(node);
    return 0;
}
