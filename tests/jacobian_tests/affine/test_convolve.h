#include <math.h>
#include <stdio.h>
#include <string.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "subexpr.h"
#include "test_helpers.h"

const char *test_jacobian_convolve(void)
{
    /* Test: Jacobian of y = conv([1, 2, 3], x) for x = Variable(3).
     * J should equal T(kernel):
     *   [[1, 0, 0],
     *    [2, 1, 0],
     *    [3, 2, 1],
     *    [0, 3, 2],
     *    [0, 0, 3]]
     * stored in CSR_matrix with nnz = 9, shape 5 x 3. */
    double kernel[3] = {1.0, 2.0, 3.0};
    expr *kernel_param = new_parameter(3, 1, PARAM_FIXED, 3, kernel);
    expr *x = new_variable(3, 1, 0, 3);
    expr *y = new_convolve(kernel_param, x);

    double u[3] = {1.0, 2.0, 3.0};
    y->forward(y, u);
    jacobian_init(y);
    eval_jacobian(y);

    mu_assert("Jacobian should have 5 rows", y->jacobian->m == 5);
    mu_assert("Jacobian should have 3 columns", y->jacobian->n == 3);
    mu_assert("Jacobian should have 9 nonzeros", y->jacobian->nnz == 9);

    int expected_p[6] = {0, 1, 3, 6, 8, 9};
    int expected_i[9] = {0, 0, 1, 0, 1, 2, 1, 2, 2};
    double expected_x[9] = {1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0, 3.0};

    mu_assert("sparsity fail",
              cmp_sparsity(y->jacobian, expected_p, expected_i, 5, 9));
    mu_assert("vals fail", cmp_values(y->jacobian, expected_x, 9));

    free_expr(y);
    return 0;
}

const char *test_jacobian_convolve_composite(void)
{
    /* y = conv([1, 2, 3], log(x)) — verify via numerical differentiation that
     * the composite path (T(a) @ J_log) matches. */
    double kernel[3] = {1.0, 2.0, 3.0};
    double x_vals[3] = {1.0, 2.0, 3.0};

    expr *kernel_param = new_parameter(3, 1, PARAM_FIXED, 3, kernel);
    expr *x = new_variable(3, 1, 0, 3);
    expr *log_x = new_log(x);
    expr *y = new_convolve(kernel_param, log_x);

    mu_assert("check_jacobian failed",
              check_jacobian_num(y, x_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(y);
    return 0;
}

const char *test_jacobian_convolve_csc_consumer(void)
{
    /* y = B @ conv([1, 2, 3], exp(x)) with B a sparse CSR: the left matmul
     * (n_blocks = 1) reads conv's Jacobian through its CSC mirror. Checking at
     * two points catches a mirror left stale by the second evaluation. */
    double kernel[3] = {1.0, 2.0, 3.0};
    double u1[3] = {0.1, 0.2, 0.3};
    double u2[3] = {-0.4, 0.5, 1.0};

    CSR_matrix *B = new_CSR_matrix(2, 5, 5);
    int Bp[3] = {0, 3, 5};
    int Bi[5] = {0, 2, 4, 1, 3};
    double Bx[5] = {1.0, -2.0, 0.5, 3.0, 1.5};
    memcpy(B->p, Bp, 3 * sizeof(int));
    memcpy(B->i, Bi, 5 * sizeof(int));
    memcpy(B->x, Bx, 5 * sizeof(double));

    expr *kernel_param = new_parameter(3, 1, PARAM_FIXED, 3, kernel);
    expr *x = new_variable(3, 1, 0, 3);
    expr *conv = new_convolve(kernel_param, new_exp(x));
    expr *y = new_left_matmul(NULL, conv, B);

    mu_assert("check_jacobian failed at u1",
              check_jacobian_num(y, u1, NUMERICAL_DIFF_DEFAULT_H));
    mu_assert("check_jacobian failed at u2",
              check_jacobian_num(y, u2, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(y);
    free_CSR_matrix(B);
    return 0;
}

const char *test_jacobian_convolve_pd_child(void)
{
    /* y = conv([1, 2, 3], exp(D @ x)) with D dense: the child Jacobian is
     * permuted_dense, and so is conv's. */
    double kernel[3] = {1.0, 2.0, 3.0};
    double D[12] = {1.0, -0.5, 2.0, 0.5, 1.5, -1.0, 0.3, 0.2, -0.7, -1.2, 0.4, 0.9};
    double u[3] = {0.1, -0.2, 0.3};

    expr *kernel_param = new_parameter(3, 1, PARAM_FIXED, 3, kernel);
    expr *x = new_variable(3, 1, 0, 3);
    expr *child = new_exp(new_left_matmul_dense(NULL, x, 4, 3, D));
    expr *y = new_convolve(kernel_param, child);

    jacobian_init(y);
    mu_assert("child jacobian must be permuted_dense",
              child->jacobian->kind == MATRIX_PERMUTED_DENSE);
    mu_assert("jacobian must be permuted_dense",
              y->jacobian->kind == MATRIX_PERMUTED_DENSE);
    mu_assert("check_jacobian failed",
              check_jacobian_num(y, u, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(y);
    return 0;
}

const char *test_jacobian_convolve_spd_child(void)
{
    /* y = conv([1, 2, 3], exp(vec(A @ X))) with X a 3x2 matrix variable: the
     * dense left matmul has n_blocks = 2, so the child Jacobian is stacked_pd,
     * and so is conv's. */
    double kernel[3] = {1.0, 2.0, 3.0};
    double A[6] = {1.0, -0.5, 2.0, 0.5, 1.5, -1.0};
    double u1[6] = {0.1, 0.2, 0.3, -0.1, -0.2, -0.3};
    double u2[6] = {0.4, -0.3, 0.2, 0.1, 0.3, -0.4};

    expr *kernel_param = new_parameter(3, 1, PARAM_FIXED, 6, kernel);
    expr *X = new_variable(3, 2, 0, 6);
    expr *child =
        new_exp(new_reshape(new_left_matmul_dense(NULL, X, 2, 3, A), 4, 1));
    expr *y = new_convolve(kernel_param, child);

    jacobian_init(y);
    mu_assert("child jacobian must be stacked_pd",
              child->jacobian->kind == MATRIX_STACKED_PD);
    mu_assert("jacobian must be stacked_pd", y->jacobian->kind == MATRIX_STACKED_PD);
    mu_assert("check_jacobian failed at u1",
              check_jacobian_num(y, u1, NUMERICAL_DIFF_DEFAULT_H));
    mu_assert("check_jacobian failed at u2",
              check_jacobian_num(y, u2, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(y);
    return 0;
}
