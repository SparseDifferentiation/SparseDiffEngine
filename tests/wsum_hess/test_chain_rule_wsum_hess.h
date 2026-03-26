#include "affine.h"
#include "bivariate.h"
#include "elementwise_full_dom.h"
#include "minunit.h"
#include "numerical_diff.h"

const char *test_wsum_hess_exp_sum(void)
{
    double u_vals[3] = {1.0, 2.0, 3.0};
    double w = 1.0;

    expr *x = new_variable(3, 1, 0, 3);
    expr *sum_x = new_sum(x, -1);
    expr *exp_sum_x = new_exp(sum_x);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(exp_sum_x, u_vals, &w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(exp_sum_x);
    return 0;
}

const char *test_wsum_hess_exp_sum_mult(void)
{
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};
    double w = 1.0;

    expr *x = new_variable(2, 1, 0, 4);
    expr *y = new_variable(2, 1, 2, 4);
    expr *xy = new_elementwise_mult(x, y);
    expr *sum_xy = new_sum(xy, -1);
    expr *exp_sum_xy = new_exp(sum_xy);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(exp_sum_xy, u_vals, &w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(exp_sum_xy);
    return 0;
}
