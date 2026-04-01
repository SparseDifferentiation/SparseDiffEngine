#include "atoms/affine.h"
#include "atoms/bivariate_full_dom.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/non_elementwise_full_dom.h"
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

const char *test_wsum_hess_x_x_multiply(void)
{
    /* the first and last values are not used, but good to include them in test */
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};
    double w[2] = {1.33, 2.1};
    expr *x = new_variable(2, 1, 1, 4);
    expr *multiply = new_elementwise_mult(x, x);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(multiply, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(multiply);
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

const char *test_wsum_hess_multiply_deep_composite(void)
{
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};
    double w[4] = {1.1, 2.2, 3.3, 4.4};

    CSR_Matrix *A = new_csr_random(2, 2, 1.0);
    CSR_Matrix *B = new_csr_random(2, 2, 1.0);
    expr *X = new_variable(2, 2, 0, 8);
    expr *Y = new_variable(2, 2, 0, 8);
    expr *AX = new_left_matmul(X, A);
    expr *BY = new_left_matmul(Y, B);
    expr *sin_AX = new_sin(AX);
    expr *cos_BY = new_cos(BY);
    expr *sin_AX_mult_sin_AX = new_elementwise_mult(sin_AX, sin_AX);
    expr *multiply = new_elementwise_mult(sin_AX_mult_sin_AX, cos_BY);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(multiply, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(multiply);
    free_csr_matrix(A);
    free_csr_matrix(B);
    return 0;
}

const char *test_wsum_hess_quad_form_Ax(void)
{
    double u_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double w = 1.0;

    CSR_Matrix *A = new_csr_random(3, 4, 1.0);

    /* Q = [1 2 0; 2 3 0; 0 0 4] (symmetric) */
    CSR_Matrix *Q = new_csr_matrix(3, 3, 5, NULL);
    double Qx[5] = {1.0, 2.0, 2.0, 3.0, 4.0};
    int Qi[5] = {0, 1, 0, 1, 2};
    int Qp[4] = {0, 2, 4, 5};
    memcpy(Q->x, Qx, 5 * sizeof(double));
    memcpy(Q->i, Qi, 5 * sizeof(int));
    memcpy(Q->p, Qp, 4 * sizeof(int));

    expr *x = new_variable(4, 1, 1, 6);
    expr *Ax = new_left_matmul(x, A);
    expr *node = new_quad_form(Ax, Q);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(node, u_vals, &w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    free_csr_matrix(A);
    free_csr_matrix(Q);
    return 0;
}

const char *test_wsum_hess_quad_form_sin_Ax(void)
{
    double u_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double w = 2.0;

    CSR_Matrix *A = new_csr_random(3, 4, 1.0);

    /* Q = [1 2 0; 2 3 0; 0 0 4] (symmetric) */
    CSR_Matrix *Q = new_csr_matrix(3, 3, 5, NULL);
    double Qx[5] = {1.0, 2.0, 2.0, 3.0, 4.0};
    int Qi[5] = {0, 1, 0, 1, 2};
    int Qp[4] = {0, 2, 4, 5};
    memcpy(Q->x, Qx, 5 * sizeof(double));
    memcpy(Q->i, Qi, 5 * sizeof(int));
    memcpy(Q->p, Qp, 4 * sizeof(int));

    expr *x = new_variable(4, 1, 1, 6);
    expr *Ax = new_left_matmul(x, A);
    expr *sin_Ax = new_sin(Ax);
    expr *node = new_quad_form(sin_Ax, Q);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(node, u_vals, &w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    free_csr_matrix(A);
    free_csr_matrix(Q);
    return 0;
}

const char *test_wsum_hess_matmul_exp_exp(void)
{
    /* Z = exp(X) @ exp(Y), X is 2x3, Y is 3x2, Z is 2x2 */
    double u_vals[12] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
    double w[4] = {1.0, 2.0, 3.0, 4.0};

    expr *X = new_variable(2, 3, 0, 12);
    expr *Y = new_variable(3, 2, 6, 12);
    expr *exp_X = new_exp(X);
    expr *exp_Y = new_exp(Y);
    expr *Z = new_matmul(exp_X, exp_Y);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(Z, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    return 0;
}

const char *test_wsum_hess_matmul_sin_cos(void)
{
    /* Z = sin(X) @ cos(Y), X is 2x2, Y is 2x3, Z is 2x3 */
    double u_vals[10] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
    double w[6] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};

    expr *X = new_variable(2, 2, 0, 10);
    expr *Y = new_variable(2, 3, 4, 10);
    expr *sin_X = new_sin(X);
    expr *cos_Y = new_cos(Y);
    expr *Z = new_matmul(sin_X, cos_Y);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(Z, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    return 0;
}

const char *test_wsum_hess_matmul_Ax_By(void)
{
    /* Z = (A @ X) @ (B @ Y), affine children */
    double u_vals[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double w[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    CSR_Matrix *A = new_csr_random(3, 2, 1.0);
    CSR_Matrix *B = new_csr_random(2, 3, 1.0);

    expr *X = new_variable(2, 2, 0, 10);
    expr *Y = new_variable(3, 2, 4, 10);
    expr *AX = new_left_matmul(X, A); /* 3x2 */
    expr *BY = new_left_matmul(Y, B); /* 2x2 */
    expr *Z = new_matmul(AX, BY);     /* 3x2 */

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(Z, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    free_csr_matrix(A);
    free_csr_matrix(B);
    return 0;
}

const char *test_wsum_hess_matmul_sin_Ax_cos_Bx(void)
{
    /* Z = sin(A @ X) @ cos(B @ X), shared variable */
    double u_vals[6] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    double w[4] = {1.0, 2.0, 3.0, 4.0};

    CSR_Matrix *A = new_csr_random(2, 3, 1.0);
    CSR_Matrix *B = new_csr_random(2, 3, 1.0);

    expr *X = new_variable(3, 2, 0, 6);
    expr *AX = new_left_matmul(X, A); /* 2x2 */
    expr *BX = new_left_matmul(X, B); /* 2x2 */
    expr *sin_AX = new_sin(AX);
    expr *cos_BX = new_cos(BX);
    expr *Z = new_matmul(sin_AX, cos_BX); /* 2x2 */

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(Z, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    free_csr_matrix(A);
    free_csr_matrix(B);
    return 0;
}

const char *test_wsum_hess_matmul_X_X(void)
{
    /* Z = X @ X, same leaf variable */
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};
    double w[4] = {1.0, 2.0, 3.0, 4.0};

    expr *X = new_variable(2, 2, 0, 4);
    expr *Z = new_matmul(X, X);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(Z, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    return 0;
}

const char *test_wsum_hess_quad_form_exp(void)
{
    double u_vals[3] = {0.5, 1.0, 1.5};
    double w = 3.0;

    /* Q = [1 2 0; 2 3 0; 0 0 4] (symmetric) */
    CSR_Matrix *Q = new_csr_matrix(3, 3, 5, NULL);
    double Qx[5] = {1.0, 2.0, 2.0, 3.0, 4.0};
    int Qi[5] = {0, 1, 0, 1, 2};
    int Qp[4] = {0, 2, 4, 5};
    memcpy(Q->x, Qx, 5 * sizeof(double));
    memcpy(Q->i, Qi, 5 * sizeof(int));
    memcpy(Q->p, Qp, 4 * sizeof(int));

    expr *x = new_variable(3, 1, 0, 3);
    expr *exp_x = new_exp(x);
    expr *node = new_quad_form(exp_x, Q);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(node, u_vals, &w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    free_csr_matrix(Q);
    return 0;
}
