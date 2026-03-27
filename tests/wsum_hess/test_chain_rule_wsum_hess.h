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

const char *test_wsum_hess_exp_sum_matmul(void)
{
    /* exp(sum(X @ Y)) where X is 2x3, Y is 3x2
     * n_vars = 6 + 6 = 12 */
    double u_vals[12] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.5, 2.5, 0.1, 0.2, 0.3};
    double w = 1.0;

    expr *X = new_variable(2, 3, 0, 12);
    expr *Y = new_variable(3, 2, 6, 12);
    expr *XY = new_matmul(X, Y);
    expr *sum_XY = new_sum(XY, -1);
    expr *exp_sum_XY = new_exp(sum_XY);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(exp_sum_XY, u_vals, &w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(exp_sum_XY);
    return 0;
}

const char *test_wsum_hess_sin_sum_axis0_matmul(void)
{
    /* sin(sum(X @ Y, axis=0)) where X is 2x3, Y is 3x2
     * X@Y is 2x2, sum(axis=0) gives 1x2, sin gives 1x2
     * n_vars = 6 + 6 = 12 */
    double u_vals[12] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.5, 2.5, 0.1, 0.2, 0.3};
    double w[2] = {1.0, 1.0};

    expr *X = new_variable(2, 3, 0, 12);
    expr *Y = new_variable(3, 2, 6, 12);
    expr *XY = new_matmul(X, Y);
    expr *sum_XY = new_sum(XY, 0);
    expr *sin_sum_XY = new_sin(sum_XY);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(sin_sum_XY, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(sin_sum_XY);
    return 0;
}

const char *test_wsum_hess_logistic_sum_axis0_matmul(void)
{
    /* logistic(sum(X @ Y, axis=0)) where X is 2x3, Y is 3x2
     * n_vars = 6 + 6 = 12 */
    double u_vals[12] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.5, 2.5, 0.1, 0.2, 0.3};
    double w[2] = {1.0, 1.0};

    expr *X = new_variable(2, 3, 0, 12);
    expr *Y = new_variable(3, 2, 6, 12);
    expr *XY = new_matmul(X, Y);
    expr *sum_XY = new_sum(XY, 0);
    expr *logistic_sum_XY = new_logistic(sum_XY);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(logistic_sum_XY, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(logistic_sum_XY);
    return 0;
}
