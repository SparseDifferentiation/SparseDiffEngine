#include "affine.h"
#include "bivariate_full_dom.h"
#include "elementwise_full_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "utils/CSR_Matrix.h"
#include "test_helpers.h"

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

const char *test_jacobian_cos_sin_multiply(void)
{
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};

    expr *x = new_variable(2, 1, 0, 4);
    expr *y = new_variable(2, 1, 2, 4);
    expr *cos_x = new_cos(x);
    expr *cos_y = new_cos(y);
    expr *sum = new_add(cos_x, cos_y);
    expr *sin_y = new_sin(y);
    expr *multiply = new_elementwise_mult(sum, sin_y);

    mu_assert("check_jacobian failed",
              check_jacobian(multiply, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(multiply);
    return 0;
}

const char *test_jacobian_Ax_Bx_multiply(void)
{
    /* the first and last values are not used, but good to include them in test */
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};

    CSR_Matrix *A = new_csr_random(2, 2, 1.0);
    CSR_Matrix *B = new_csr_random(2, 2, 1.0);
    expr *x = new_variable(2, 1, 1, 4);
    expr *Ax = new_left_matmul(x, A);
    expr *Bx = new_left_matmul(x, B);
    expr *multiply = new_elementwise_mult(Ax, Bx);

    mu_assert("check_jacobian failed",
              check_jacobian(multiply, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(multiply);
    return 0;
}

const char *test_jacobian_AX_BX_multiply(void)
{
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};

    CSR_Matrix *A = new_csr_random(2, 2, 1.0);
    CSR_Matrix *B = new_csr_random(2, 2, 1.0);
    expr *X = new_variable(2, 2, 0, 4);
    expr *AX = new_left_matmul(X, A);
    expr *BX = new_left_matmul(X, B);
    expr *multiply = new_elementwise_mult(new_sin(AX), new_cos(BX));

    mu_assert("check_jacobian failed",
              check_jacobian(multiply, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(multiply);
    return 0;
}


