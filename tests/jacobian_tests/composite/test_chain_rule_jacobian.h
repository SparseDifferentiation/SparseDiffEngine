#include "affine.h"
#include "bivariate_full_dom.h"
#include "elementwise_full_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "other.h"
#include "test_helpers.h"
#include "utils/CSR_Matrix.h"

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
    expr *Ax = new_left_matmul(NULL, x, A);
    expr *Bx = new_left_matmul(NULL, x, B);
    expr *multiply = new_elementwise_mult(Ax, Bx);

    mu_assert("check_jacobian failed",
              check_jacobian(multiply, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(multiply);
    free_csr_matrix(A);
    free_csr_matrix(B);
    return 0;
}

const char *test_jacobian_AX_BX_multiply(void)
{
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};

    CSR_Matrix *A = new_csr_random(2, 2, 1.0);
    CSR_Matrix *B = new_csr_random(2, 2, 1.0);
    expr *X = new_variable(2, 2, 0, 4);
    expr *AX = new_left_matmul(NULL, X, A);
    expr *BX = new_left_matmul(NULL, X, B);
    expr *multiply = new_elementwise_mult(new_sin(AX), new_cos(BX));

    mu_assert("check_jacobian failed",
              check_jacobian(multiply, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(multiply);
    free_csr_matrix(A);
    free_csr_matrix(B);
    return 0;
}

const char *test_jacobian_quad_form_Ax(void)
{
    /* (Ax)^T Q (Ax) where Q is symmetric */
    double u_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    CSR_Matrix *A = new_csr_random(3, 4, 1.0);

    /* Q = [1 2 0; 2 3 0; 0 0 4] */
    CSR_Matrix *Q = new_csr_matrix(3, 3, 5);
    double Qx[5] = {1.0, 2.0, 2.0, 3.0, 4.0};
    int Qi[5] = {0, 1, 0, 1, 2};
    int Qp[4] = {0, 2, 4, 5};
    memcpy(Q->x, Qx, 5 * sizeof(double));
    memcpy(Q->i, Qi, 5 * sizeof(int));
    memcpy(Q->p, Qp, 4 * sizeof(int));

    expr *x = new_variable(4, 1, 1, 6);
    expr *Ax = new_left_matmul(NULL, x, A);
    expr *sin_Ax = new_sin(Ax);
    expr *node = new_quad_form(sin_Ax, Q);

    mu_assert("check_jacobian failed",
              check_jacobian(node, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    free_csr_matrix(A);
    free_csr_matrix(Q);
    return 0;
}

const char *test_jacobian_quad_form_exp(void)
{
    /* exp(x)^T Q exp(x) where Q is symmetric */
    double u_vals[3] = {0.5, 1.0, 1.5};

    /* Q = [1 2 0; 2 3 0; 0 0 4] */
    CSR_Matrix *Q = new_csr_matrix(3, 3, 5);
    double Qx[5] = {1.0, 2.0, 2.0, 3.0, 4.0};
    int Qi[5] = {0, 1, 0, 1, 2};
    int Qp[4] = {0, 2, 4, 5};
    memcpy(Q->x, Qx, 5 * sizeof(double));
    memcpy(Q->i, Qi, 5 * sizeof(int));
    memcpy(Q->p, Qp, 4 * sizeof(int));

    expr *x = new_variable(3, 1, 0, 3);
    expr *exp_x = new_exp(x);
    expr *node = new_quad_form(exp_x, Q);

    mu_assert("check_jacobian failed",
              check_jacobian(node, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    free_csr_matrix(Q);
    return 0;
}

const char *test_jacobian_matmul_exp_exp(void)
{
    /* Z = exp(X) @ exp(Y), X is 2x3, Y is 3x2 */
    double u_vals[12] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};

    expr *X = new_variable(2, 3, 0, 12);
    expr *Y = new_variable(3, 2, 6, 12);
    expr *exp_X = new_exp(X);
    expr *exp_Y = new_exp(Y);
    expr *Z = new_matmul(exp_X, exp_Y);

    mu_assert("check_jacobian failed",
              check_jacobian(Z, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    return 0;
}

const char *test_jacobian_matmul_sin_cos(void)
{
    /* Z = sin(X) @ cos(Y), X is 2x2, Y is 2x3 */
    double u_vals[10] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};

    expr *X = new_variable(2, 2, 0, 10);
    expr *Y = new_variable(2, 3, 4, 10);
    expr *sin_X = new_sin(X);
    expr *cos_Y = new_cos(Y);
    expr *Z = new_matmul(sin_X, cos_Y);

    mu_assert("check_jacobian failed",
              check_jacobian(Z, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    return 0;
}

const char *test_jacobian_matmul_Ax_By(void)
{
    /* Z = (A @ X) @ (B @ Y) with constant matrices A, B */
    double u_vals[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    CSR_Matrix *A = new_csr_random(3, 2, 1.0);
    CSR_Matrix *B = new_csr_random(2, 3, 1.0);

    expr *X = new_variable(2, 2, 0, 10);    /* 2x2, vars 0-3 */
    expr *Y = new_variable(3, 2, 4, 10);    /* 3x2, vars 4-9 */
    expr *AX = new_left_matmul(NULL, X, A); /* 3x2 */
    expr *BY = new_left_matmul(NULL, Y, B); /* 2x2 */
    expr *Z = new_matmul(AX, BY);           /* 3x2 */

    mu_assert("check_jacobian failed",
              check_jacobian(Z, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    free_csr_matrix(A);
    free_csr_matrix(B);
    return 0;
}

const char *test_jacobian_matmul_sin_Ax_cos_Bx(void)
{
    /* Z = sin(A @ X) @ cos(B @ X), shared variable X */
    double u_vals[6] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};

    CSR_Matrix *A = new_csr_random(2, 3, 1.0);
    CSR_Matrix *B = new_csr_random(2, 3, 1.0);

    expr *X = new_variable(3, 2, 0, 6);     /* 3x2, vars 0-5 */
    expr *AX = new_left_matmul(NULL, X, A); /* 2x2 */
    expr *BX = new_left_matmul(NULL, X, B); /* 2x2 */
    expr *sin_AX = new_sin(AX);             /* 2x2 */
    expr *cos_BX = new_cos(BX);             /* 2x2 */
    expr *Z = new_matmul(sin_AX, cos_BX);   /* 2x2 */

    mu_assert("check_jacobian failed",
              check_jacobian(Z, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    free_csr_matrix(A);
    free_csr_matrix(B);
    return 0;
}

const char *test_jacobian_matmul_X_X(void)
{
    /* Z = X @ X, same leaf variable as both children */
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};

    expr *X = new_variable(2, 2, 0, 4);
    expr *Z = new_matmul(X, X); /* 2x2 */

    mu_assert("check_jacobian failed",
              check_jacobian(Z, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    return 0;
}
