#include "affine.h"
#include "bivariate_full_dom.h"
#include "elementwise_full_dom.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "test_helpers.h"

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

const char *test_wsum_hess_sin_cos(void)
{
    double u_vals[5] = {0.5, 1.0, 1.5, 2.0, 2.5};
    double w[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

    expr *x = new_variable(5, 1, 0, 5);
    expr *cos_x = new_cos(x);
    expr *sin_cos_x = new_sin(cos_x);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(sin_cos_x, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(sin_cos_x);
    return 0;
}

const char *test_wsum_hess_Ax_Bx_multiply(void)
{
    /* the first and last values are not used, but good to include them in test */
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};
    double w[2] = {1.33, 2.1};

    CSR_Matrix *A = new_csr_random(2, 2, 1.0);
    CSR_Matrix *B = new_csr_random(2, 2, 1.0);
    expr *x = new_variable(2, 1, 1, 4);
    expr *Ax = new_left_matmul(x, A);
    expr *Bx = new_left_matmul(x, B);
    expr *multiply = new_elementwise_mult(Ax, Bx);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(multiply, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(multiply);
    free_csr_matrix(A);
    free_csr_matrix(B);
    return 0;
}

const char *test_wsum_hess_AX_BX_multiply(void)
{
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};
    double w[4] = {1.1, 2.2, 3.3, 4.4};

    CSR_Matrix *A = new_csr_random(2, 2, 1.0);
    CSR_Matrix *B = new_csr_random(2, 2, 1.0);
    expr *X = new_variable(2, 2, 0, 4);
    expr *AX = new_left_matmul(X, A);
    expr *BX = new_left_matmul(X, B);
    expr *multiply = new_elementwise_mult(new_sin(AX), new_cos(BX));

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(multiply, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(multiply);
    free_csr_matrix(A);
    free_csr_matrix(B);
    return 0;
}
