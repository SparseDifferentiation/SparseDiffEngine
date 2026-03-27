#include "affine.h"
#include "bivariate.h"
#include "elementwise_full_dom.h"
#include "minunit.h"
#include "numerical_diff.h"

const char *test_jacobian_exp_sum(void)
{
    double u_vals[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *sum_x = new_sum(x, -1);
    expr *exp_sum_x = new_exp(sum_x);

    mu_assert("check_jacobian failed",
              check_jacobian(exp_sum_x, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(exp_sum_x);
    return 0;
}

const char *test_jacobian_exp_sum_mult(void)
{
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};

    expr *x = new_variable(2, 1, 0, 4);
    expr *y = new_variable(2, 1, 2, 4);
    expr *xy = new_elementwise_mult(x, y);
    expr *sum_xy = new_sum(xy, -1);
    expr *exp_sum_xy = new_exp(sum_xy);

    mu_assert("check_jacobian failed",
              check_jacobian(exp_sum_xy, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(exp_sum_xy);
    return 0;
}

const char *test_jacobian_sin_cos(void)
{
    double u_vals[5] = {0.5, 1.0, 1.5, 2.0, 2.5};

    expr *x = new_variable(5, 1, 0, 5);
    expr *cos_x = new_cos(x);
    expr *sin_cos_x = new_sin(cos_x);

    mu_assert("check_jacobian failed",
              check_jacobian(sin_cos_x, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(sin_cos_x);
    return 0;
}
