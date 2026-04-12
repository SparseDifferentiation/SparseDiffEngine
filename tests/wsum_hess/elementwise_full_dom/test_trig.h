#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_wsum_hess_sin(void)
{
    double u_vals[3] = {1.0, 2.0, 3.0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *sin_node = new_sin(x);
    sin_node->forward(sin_node, u_vals);
    jacobian_init(sin_node);
    wsum_hess_init(sin_node);
    sin_node->eval_wsum_hess(sin_node, w);

    /* Expected values on the diagonal: -w_i * sin(x_i) */
    double expected_x[3] = {-1.0 * sin(1.0), -2.0 * sin(2.0), -3.0 * sin(3.0)};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vals incorrect",
              cmp_double_array(sin_node->wsum_hess->x, expected_x, 3));
    mu_assert("rows incorrect",
              cmp_int_array(sin_node->wsum_hess->p, expected_p, 4));
    mu_assert("cols incorrect",
              cmp_int_array(sin_node->wsum_hess->i, expected_i, 3));

    free_expr(sin_node);

    return 0;
}

const char *test_wsum_hess_cos(void)
{
    double u_vals[3] = {1.0, 2.0, 3.0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *cos_node = new_cos(x);
    cos_node->forward(cos_node, u_vals);
    jacobian_init(cos_node);
    wsum_hess_init(cos_node);
    cos_node->eval_wsum_hess(cos_node, w);

    /* Expected values on the diagonal: -w_i * cos(x_i) */
    double expected_x[3] = {-1.0 * cos(1.0), -2.0 * cos(2.0), -3.0 * cos(3.0)};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vals incorrect",
              cmp_double_array(cos_node->wsum_hess->x, expected_x, 3));
    mu_assert("rows incorrect",
              cmp_int_array(cos_node->wsum_hess->p, expected_p, 4));
    mu_assert("cols incorrect",
              cmp_int_array(cos_node->wsum_hess->i, expected_i, 3));

    free_expr(cos_node);

    return 0;
}

const char *test_wsum_hess_tan(void)
{
    double u_vals[3] = {1.0, 2.0, 3.0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *tan_node = new_tan(x);
    tan_node->forward(tan_node, u_vals);
    jacobian_init(tan_node);
    wsum_hess_init(tan_node);
    tan_node->eval_wsum_hess(tan_node, w);

    /* Expected values on the diagonal: w_i * 2 * sin(x_i) / cos^3(x_i) */
    double expected_x[3] = {1.0 * 2.0 * sin(1.0) / pow(cos(1.0), 3),
                            2.0 * 2.0 * sin(2.0) / pow(cos(2.0), 3),
                            3.0 * 2.0 * sin(3.0) / pow(cos(3.0), 3)};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vals incorrect",
              cmp_double_array(tan_node->wsum_hess->x, expected_x, 3));
    mu_assert("rows incorrect",
              cmp_int_array(tan_node->wsum_hess->p, expected_p, 4));
    mu_assert("cols incorrect",
              cmp_int_array(tan_node->wsum_hess->i, expected_i, 3));

    free_expr(tan_node);

    return 0;
}
