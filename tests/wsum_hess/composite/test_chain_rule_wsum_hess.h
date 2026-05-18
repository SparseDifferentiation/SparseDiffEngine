#include "atoms/affine.h"
#include "atoms/bivariate_full_dom.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/non_elementwise_full_dom.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "test_helpers.h"
#include <stdio.h>

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

/* Regression: sum(exp(A @ X.T)) — user-reported wrong-Hessian case.
   Triggers left_matmul_dense over the transpose of a multi-column
   variable, producing an spd Jacobian with non-trivial col_perm
   (because the child Jacobian is the transpose permutation PD).
   Uses the exact A and X values from the user report. */
const char *test_wsum_hess_sum_exp_left_matmul_dense_transpose(void)
{
    /* X = [[0.42, 0.65], [0.44, 0.89]] in row-major -> column-major
       vec is [X[0,0], X[1,0], X[0,1], X[1,1]] = [0.42, 0.44, 0.65, 0.89]. */
    double u_vals[4] = {0.42, 0.44, 0.65, 0.89};
    /* A row-major: A[0,0]=0.55, A[0,1]=0.72, A[1,0]=0.60, A[1,1]=0.54. */
    double A[4] = {0.55, 0.72, 0.60, 0.54};
    double w = 1.0;

    expr *X = new_variable(2, 2, 0, 4);
    expr *XT = new_transpose(X);
    expr *AX = new_left_matmul_dense(NULL, XT, 2, 2, A);
    expr *exp_AX = new_exp(AX);
    expr *node = new_sum(exp_AX, -1);

    /* Compute and print the analytical Hessian to compare with the
       user-reported c_hess_dense. */
    jacobian_init(node);
    wsum_hess_init(node);
    node->forward(node, u_vals);
    node->eval_jacobian(node);
    node->eval_wsum_hess(node, &w);

    CSR_matrix *H = node->wsum_hess->to_csr(node->wsum_hess);
    double dense[16] = {0};
    for (int r = 0; r < H->m; r++)
    {
        for (int kk = H->p[r]; kk < H->p[r + 1]; kk++)
        {
            dense[r * 4 + H->i[kk]] = H->x[kk];
        }
    }
    printf("\nC analytical Hessian (row-major 4x4):\n");
    for (int r = 0; r < 4; r++)
    {
        printf("  [%.6f %.6f %.6f %.6f]\n", dense[r * 4 + 0], dense[r * 4 + 1],
               dense[r * 4 + 2], dense[r * 4 + 3]);
    }

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(node, u_vals, &w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    return 0;
}

/* Regression: neg(sin(A @ X)) exercises neg's eval_jacobian flat ->x
   loop with an spd child (sin's jacobian inherits spd from
   left_matmul_dense via copy_sparsity) and neg's wsum_hess path. */
const char *test_wsum_hess_neg_sin_left_matmul_dense(void)
{
    double u_vals[9] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    double A[9] = {1.0, 0.5, -0.3, 0.2, 1.0, 0.7, -0.1, 0.4, 1.0};
    double w[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    expr *X = new_variable(3, 3, 0, 9);
    expr *AX = new_left_matmul_dense(NULL, X, 3, 3, A);
    expr *sin_AX = new_sin(AX);
    expr *node = new_neg(sin_AX);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(node, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    return 0;
}

/* sum(sin(A @ X)) where A is 3x3 dense, X is 3x3 variable. Mirrors a
   user-reported Python case that previously segfaulted because:
     (1) sum's eval_jacobian reads child->jacobian->x directly (NULL on spd)
     (2) sum's eval_wsum_hess memcpy's child->wsum_hess->x (NULL on spd)
   Both paths now route through to_csr / block-wise copy. */
const char *test_wsum_hess_sum_sin_left_matmul_dense(void)
{
    double u_vals[9] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    double A[9] = {1.0, 0.5, -0.3, 0.2, 1.0, 0.7, -0.1, 0.4, 1.0};
    double w = 1.0;

    expr *X = new_variable(3, 3, 0, 9);
    expr *AX = new_left_matmul_dense(NULL, X, 3, 3, A);
    expr *sin_AX = new_sin(AX);
    expr *node = new_sum(sin_AX, -1);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(node, u_vals, &w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
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

    CSR_matrix *A = new_csr_random(2, 2, 1.0);
    CSR_matrix *B = new_csr_random(2, 2, 1.0);
    expr *x = new_variable(2, 1, 1, 4);
    expr *Ax = new_left_matmul(NULL, x, A);
    expr *Bx = new_left_matmul(NULL, x, B);
    expr *multiply = new_elementwise_mult(Ax, Bx);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(multiply, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(multiply);
    free_CSR_matrix(A);
    free_CSR_matrix(B);
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

    CSR_matrix *A = new_csr_random(2, 2, 1.0);
    CSR_matrix *B = new_csr_random(2, 2, 1.0);
    expr *X = new_variable(2, 2, 0, 4);
    expr *AX = new_left_matmul(NULL, X, A);
    expr *BX = new_left_matmul(NULL, X, B);
    expr *multiply = new_elementwise_mult(new_sin(AX), new_cos(BX));

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(multiply, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(multiply);
    free_CSR_matrix(A);
    free_CSR_matrix(B);
    return 0;
}

const char *test_wsum_hess_multiply_deep_composite(void)
{
    double u_vals[4] = {1.0, 2.0, 3.0, 4.0};
    double w[4] = {1.1, 2.2, 3.3, 4.4};

    CSR_matrix *A = new_csr_random(2, 2, 1.0);
    CSR_matrix *B = new_csr_random(2, 2, 1.0);
    expr *X = new_variable(2, 2, 0, 8);
    expr *Y = new_variable(2, 2, 0, 8);
    expr *AX = new_left_matmul(NULL, X, A);
    expr *BY = new_left_matmul(NULL, Y, B);
    expr *sin_AX = new_sin(AX);
    expr *cos_BY = new_cos(BY);
    expr *sin_AX_mult_sin_AX = new_elementwise_mult(sin_AX, sin_AX);
    expr *multiply = new_elementwise_mult(sin_AX_mult_sin_AX, cos_BY);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(multiply, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(multiply);
    free_CSR_matrix(A);
    free_CSR_matrix(B);
    return 0;
}

const char *test_wsum_hess_quad_form_Ax(void)
{
    double u_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double w = 1.0;

    CSR_matrix *A = new_csr_random(3, 4, 1.0);

    /* Q = [1 2 0; 2 3 0; 0 0 4] (symmetric) */
    CSR_matrix *Q = new_CSR_matrix(3, 3, 5);
    double Qx[5] = {1.0, 2.0, 2.0, 3.0, 4.0};
    int Qi[5] = {0, 1, 0, 1, 2};
    int Qp[4] = {0, 2, 4, 5};
    memcpy(Q->x, Qx, 5 * sizeof(double));
    memcpy(Q->i, Qi, 5 * sizeof(int));
    memcpy(Q->p, Qp, 4 * sizeof(int));

    expr *x = new_variable(4, 1, 1, 6);
    expr *Ax = new_left_matmul(NULL, x, A);
    expr *node = new_quad_form(Ax, Q);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(node, u_vals, &w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    free_CSR_matrix(A);
    free_CSR_matrix(Q);
    return 0;
}

const char *test_wsum_hess_quad_form_sin_Ax(void)
{
    double u_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double w = 2.0;

    CSR_matrix *A = new_csr_random(3, 4, 1.0);

    /* Q = [1 2 0; 2 3 0; 0 0 4] (symmetric) */
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
    expr *node = new_quad_form(sin_Ax, Q);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(node, u_vals, &w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    free_CSR_matrix(A);
    free_CSR_matrix(Q);
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

    CSR_matrix *A = new_csr_random(3, 2, 1.0);
    CSR_matrix *B = new_csr_random(2, 3, 1.0);

    expr *X = new_variable(2, 2, 0, 10);
    expr *Y = new_variable(3, 2, 4, 10);
    expr *AX = new_left_matmul(NULL, X, A); /* 3x2 */
    expr *BY = new_left_matmul(NULL, Y, B); /* 2x2 */
    expr *Z = new_matmul(AX, BY);           /* 3x2 */

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(Z, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    free_CSR_matrix(A);
    free_CSR_matrix(B);
    return 0;
}

const char *test_wsum_hess_matmul_sin_Ax_cos_Bx(void)
{
    /* Z = sin(A @ X) @ cos(B @ X), shared variable */
    double u_vals[6] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    double w[4] = {1.0, 2.0, 3.0, 4.0};

    CSR_matrix *A = new_csr_random(2, 3, 1.0);
    CSR_matrix *B = new_csr_random(2, 3, 1.0);

    expr *X = new_variable(3, 2, 0, 6);
    expr *AX = new_left_matmul(NULL, X, A); /* 2x2 */
    expr *BX = new_left_matmul(NULL, X, B); /* 2x2 */
    expr *sin_AX = new_sin(AX);
    expr *cos_BX = new_cos(BX);
    expr *Z = new_matmul(sin_AX, cos_BX); /* 2x2 */

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(Z, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    free_CSR_matrix(A);
    free_CSR_matrix(B);
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
    CSR_matrix *Q = new_CSR_matrix(3, 3, 5);
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
    free_CSR_matrix(Q);
    return 0;
}
