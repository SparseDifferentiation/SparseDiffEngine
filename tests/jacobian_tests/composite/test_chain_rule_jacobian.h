#include "atoms/affine.h"
#include "atoms/bivariate_full_dom.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/non_elementwise_full_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "subexpr.h"
#include "test_helpers.h"
#include "utils/CSR_matrix.h"

const char *test_jacobian_exp_sum(void)
{
    double u_vals[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *sum_x = new_sum(x, -1);
    expr *exp_sum_x = new_exp(sum_x);

    mu_assert("check_jacobian failed",
              check_jacobian_num(exp_sum_x, u_vals, NUMERICAL_DIFF_DEFAULT_H));

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
              check_jacobian_num(exp_sum_xy, u_vals, NUMERICAL_DIFF_DEFAULT_H));

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
              check_jacobian_num(sin_cos_x, u_vals, NUMERICAL_DIFF_DEFAULT_H));

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
              check_jacobian_num(multiply, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(multiply);
    return 0;
}

const char *test_jacobian_Ax_Bx_multiply(void)
{
    /* the first and last values are not used, but good to include them in test */
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};

    CSR_matrix *A = new_csr_random(2, 2, 1.0);
    CSR_matrix *B = new_csr_random(2, 2, 1.0);
    expr *x = new_variable(2, 1, 1, 4);
    expr *Ax = new_left_matmul(NULL, x, A);
    expr *Bx = new_left_matmul(NULL, x, B);
    expr *multiply = new_elementwise_mult(Ax, Bx);

    mu_assert("check_jacobian failed",
              check_jacobian_num(multiply, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(multiply);
    free_CSR_matrix(A);
    free_CSR_matrix(B);
    return 0;
}

const char *test_jacobian_AX_BX_multiply(void)
{
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};

    CSR_matrix *A = new_csr_random(2, 2, 1.0);
    CSR_matrix *B = new_csr_random(2, 2, 1.0);
    expr *X = new_variable(2, 2, 0, 4);
    expr *AX = new_left_matmul(NULL, X, A);
    expr *BX = new_left_matmul(NULL, X, B);
    expr *multiply = new_elementwise_mult(new_sin(AX), new_cos(BX));

    mu_assert("check_jacobian failed",
              check_jacobian_num(multiply, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(multiply);
    free_CSR_matrix(A);
    free_CSR_matrix(B);
    return 0;
}

const char *test_jacobian_quad_form_Ax(void)
{
    /* (Ax)^T Q (Ax) where Q is symmetric */
    double u_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    CSR_matrix *A = new_csr_random(3, 4, 1.0);

    /* Q = [1 2 0; 2 3 0; 0 0 4] */
    CSR_matrix *Q = new_CSR_matrix(3, 3, 5);
    double Qx[5] = {1.0, 2.0, 2.0, 3.0, 4.0};
    int Qi[5] = {0, 1, 0, 1, 2};
    int Qp[4] = {0, 2, 4, 5};
    memcpy(Q->x, Qx, 5 * sizeof(double));
    memcpy(Q->i, Qi, 5 * sizeof(int));
    memcpy(Q->p, Qp, 4 * sizeof(int));

    expr *x = new_variable(4, 1, 1, 6);
    expr *Ax = new_left_matmul(NULL, x, A);
    expr *sin_Ax = new_sin(Ax);
    expr *node = new_quad_form_sparse(sin_Ax, Q);

    mu_assert("check_jacobian failed",
              check_jacobian_num(node, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    free_CSR_matrix(A);
    free_CSR_matrix(Q);
    return 0;
}

const char *test_jacobian_quad_form_exp(void)
{
    /* exp(x)^T Q exp(x) where Q is symmetric */
    double u_vals[3] = {0.5, 1.0, 1.5};

    /* Q = [1 2 0; 2 3 0; 0 0 4] */
    CSR_matrix *Q = new_CSR_matrix(3, 3, 5);
    double Qx[5] = {1.0, 2.0, 2.0, 3.0, 4.0};
    int Qi[5] = {0, 1, 0, 1, 2};
    int Qp[4] = {0, 2, 4, 5};
    memcpy(Q->x, Qx, 5 * sizeof(double));
    memcpy(Q->i, Qi, 5 * sizeof(int));
    memcpy(Q->p, Qp, 4 * sizeof(int));

    expr *x = new_variable(3, 1, 0, 3);
    expr *exp_x = new_exp(x);
    expr *node = new_quad_form_sparse(exp_x, Q);

    mu_assert("check_jacobian failed",
              check_jacobian_num(node, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    free_CSR_matrix(Q);
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
              check_jacobian_num(Z, u_vals, NUMERICAL_DIFF_DEFAULT_H));

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
              check_jacobian_num(Z, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    return 0;
}

const char *test_jacobian_matmul_Ax_By(void)
{
    /* Z = (A @ X) @ (B @ Y) with constant matrices A, B */
    double u_vals[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    CSR_matrix *A = new_csr_random(3, 2, 1.0);
    CSR_matrix *B = new_csr_random(2, 3, 1.0);

    expr *X = new_variable(2, 2, 0, 10);    /* 2x2, vars 0-3 */
    expr *Y = new_variable(3, 2, 4, 10);    /* 3x2, vars 4-9 */
    expr *AX = new_left_matmul(NULL, X, A); /* 3x2 */
    expr *BY = new_left_matmul(NULL, Y, B); /* 2x2 */
    expr *Z = new_matmul(AX, BY);           /* 3x2 */

    mu_assert("check_jacobian failed",
              check_jacobian_num(Z, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    free_CSR_matrix(A);
    free_CSR_matrix(B);
    return 0;
}

const char *test_jacobian_matmul_sin_Ax_cos_Bx(void)
{
    /* Z = sin(A @ X) @ cos(B @ X), shared variable X */
    double u_vals[6] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};

    CSR_matrix *A = new_csr_random(2, 3, 1.0);
    CSR_matrix *B = new_csr_random(2, 3, 1.0);

    expr *X = new_variable(3, 2, 0, 6);     /* 3x2, vars 0-5 */
    expr *AX = new_left_matmul(NULL, X, A); /* 2x2 */
    expr *BX = new_left_matmul(NULL, X, B); /* 2x2 */
    expr *sin_AX = new_sin(AX);             /* 2x2 */
    expr *cos_BX = new_cos(BX);             /* 2x2 */
    expr *Z = new_matmul(sin_AX, cos_BX);   /* 2x2 */

    mu_assert("check_jacobian failed",
              check_jacobian_num(Z, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    free_CSR_matrix(A);
    free_CSR_matrix(B);
    return 0;
}

const char *test_jacobian_matmul_X_X(void)
{
    /* Z = X @ X, same leaf variable as both children */
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};

    expr *X = new_variable(2, 2, 0, 4);
    expr *Z = new_matmul(X, X); /* 2x2 */

    mu_assert("check_jacobian failed",
              check_jacobian_num(Z, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    return 0;
}

/* Regression: atom that loops flatly over child->jacobian->x must work
   when the child Jacobian is a stacked_pd (from left_matmul_dense with
   p > 1). Pre-refactor, these would segfault because spd's base.x was
   NULL; the shared-buffer absorb makes them work uniformly. */
const char *test_jacobian_neg_left_matmul_dense(void)
{
    double u_vals[9] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    double A[9] = {1.0, 0.5, -0.3, 0.2, 1.0, 0.7, -0.1, 0.4, 1.0};

    expr *X = new_variable(3, 3, 0, 9);
    expr *AX = new_left_matmul_dense(NULL, X, 3, 3, A);
    expr *node = new_neg(AX);

    mu_assert("check_jacobian failed",
              check_jacobian_num(node, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    return 0;
}

const char *test_jacobian_scalar_mult_left_matmul_dense(void)
{
    double u_vals[9] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    double A[9] = {1.0, 0.5, -0.3, 0.2, 1.0, 0.7, -0.1, 0.4, 1.0};
    double a_val = 2.5;

    expr *X = new_variable(3, 3, 0, 9);
    expr *AX = new_left_matmul_dense(NULL, X, 3, 3, A);
    expr *a_param = new_parameter(1, 1, PARAM_FIXED, 9, &a_val);
    expr *node = new_scalar_mult(a_param, AX);

    mu_assert("check_jacobian failed",
              check_jacobian_num(node, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    return 0;
}

const char *test_jacobian_reshape_left_matmul_dense(void)
{
    double u_vals[9] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    double A[9] = {1.0, 0.5, -0.3, 0.2, 1.0, 0.7, -0.1, 0.4, 1.0};

    expr *X = new_variable(3, 3, 0, 9);
    expr *AX = new_left_matmul_dense(NULL, X, 3, 3, A);
    expr *node = new_reshape(AX, 9, 1);

    mu_assert("check_jacobian failed",
              check_jacobian_num(node, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    return 0;
}

/* Regression for BA_dense_kron_pd: left_matmul_dense whose child has a PD
   Jacobian and shape (n, p) with p > 1, triggering the kron pd path.
   Construct child as broadcast of sin(left_matmul_dense(A, x)) from
   (n_inner, 1) to (n_inner, p). */
const char *test_jacobian_left_matmul_dense_of_broadcast_sin_left_matmul_dense(void)
{
    int n_inner = 4;
    int n_outer_rows = 3;
    int p = 5;
    double u_vals[3] = {0.1, 0.2, 0.3}; /* n_vars = 3 */
    double A[12] = {1.0,  0.5, -0.3, 0.2, 1.0,  0.7,
                    -0.1, 0.4, 1.0,  0.3, -0.2, 0.6}; /* (n_inner, 3) */
    double C[12] = {1.0,  0.5, -0.3, 0.2, 1.0,  0.7,
                    -0.1, 0.4, 1.0,  0.3, -0.2, 0.6}; /* (n_outer_rows, n_inner) */

    expr *x = new_variable(3, 1, 0, 3);                       /* (3, 1) */
    expr *Ax = new_left_matmul_dense(NULL, x, n_inner, 3, A); /* (n_inner, 1) */
    expr *sin_Ax = new_sin(Ax);                      /* (n_inner, 1), PD jac */
    expr *bcast = new_broadcast(sin_Ax, n_inner, p); /* (n_inner, p), PD jac */
    expr *node = new_left_matmul_dense(NULL, bcast, n_outer_rows, n_inner,
                                       C); /* (n_outer_rows, p) */

    mu_assert("check_jacobian failed",
              check_jacobian_num(node, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    return 0;
}

/* Python regression: sum( multiply( reshape(sin(A@x), (m,1)),
                                     reshape(cos(B@y), (1,m)) ) )
   with A, B dense m x n and x, y vectors of length n. Reported segfault. */
const char *test_jacobian_sum_outer_product_sin_cos_left_matmul_dense(void)
{
    int m = 6;
    int n = 5;
    double u_vals[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    /* Deterministic A, B (m * n entries each, row-major). */
    double A[30] = {0.5488, 0.7152, 0.6028, 0.5449, 0.4237, 0.6459, 0.4376, 0.8918,
                    0.9637, 0.3834, 0.7917, 0.5289, 0.5680, 0.9256, 0.0710, 0.0871,
                    0.0202, 0.8326, 0.7782, 0.8700, 0.9786, 0.7992, 0.4615, 0.7805,
                    0.1183, 0.6399, 0.1434, 0.9447, 0.5218, 0.4147};
    double B[30] = {0.2645, 0.7742, 0.4561, 0.5684, 0.0188, 0.6176, 0.6121, 0.6169,
                    0.9437, 0.6818, 0.3595, 0.4370, 0.6976, 0.0602, 0.6668, 0.6706,
                    0.2104, 0.1289, 0.3154, 0.3637, 0.5702, 0.4386, 0.9884, 0.1020,
                    0.2089, 0.1613, 0.6531, 0.2533, 0.4663, 0.2444};

    expr *X = new_variable(n, 1, 0, 2 * n);
    expr *Y = new_variable(n, 1, n, 2 * n);
    expr *AX = new_left_matmul_dense(NULL, X, m, n, A); /* (m, 1) */
    expr *BY = new_left_matmul_dense(NULL, Y, m, n, B); /* (m, 1) */
    expr *sin_AX = new_sin(AX);                         /* (m, 1) */
    expr *cos_BY = new_cos(BY);                         /* (m, 1) */
    expr *cos_BY_row = new_reshape(cos_BY, 1, m);       /* (1, m) */
    expr *left = new_broadcast(sin_AX, m, m);           /* (m, m) */
    expr *right = new_broadcast(cos_BY_row, m, m);      /* (m, m) */
    expr *prod = new_elementwise_mult(left, right);     /* (m, m) */
    expr *node = new_sum(prod, -1);                     /* scalar */

    mu_assert("check_jacobian failed",
              check_jacobian_num(node, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    return 0;
}
