#include <math.h>
#include <stdio.h>
#include <string.h>

#include "affine.h"
#include "bivariate.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_wsum_hess_dense_left_matmul(void)
{
    /* Test weighted sum of Hessian of A @ log(x) with dense A.
     * Same math as the sparse test:
     * x = [1, 2, 3], A = [1, 0, 2; 3, 0, 4; 5, 0, 6; 7, 0, 0], w = [1, 2, 3, 4]
     * A^T @ w = [50, 0, 28]
     * wsum_hess = diag(-50/1, -0/4, -28/9)
     */
    double x_vals[3] = {1.0, 2.0, 3.0};
    double w[4] = {1.0, 2.0, 3.0, 4.0};

    expr *x = new_variable(3, 1, 0, 3);
    double A_dense[12] = {1.0, 0.0, 2.0, 3.0, 0.0, 4.0,
                          5.0, 0.0, 6.0, 7.0, 0.0, 0.0};

    expr *log_x = new_log(x);
    expr *A_log_x = new_dense_left_matmul(log_x, A_dense, 4, 3);

    A_log_x->forward(A_log_x, x_vals);
    A_log_x->jacobian_init(A_log_x);
    A_log_x->wsum_hess_init(A_log_x);
    A_log_x->eval_wsum_hess(A_log_x, w);

    double expected_x[3] = {-50.0, -0.0, -28.0 / 9.0};
    int expected_i[3] = {0, 1, 2};
    int expected_p[4] = {0, 1, 2, 3};

    mu_assert("dense wsum_hess vals incorrect",
              cmp_double_array(A_log_x->wsum_hess->x, expected_x, 3));
    mu_assert("dense wsum_hess cols incorrect",
              cmp_int_array(A_log_x->wsum_hess->i, expected_i, 3));
    mu_assert("dense wsum_hess rows incorrect",
              cmp_int_array(A_log_x->wsum_hess->p, expected_p, 4));

    free_expr(A_log_x);
    return 0;
}

const char *test_wsum_hess_dense_left_matmul_matrix(void)
{
    /* Test with matrix-valued x (3x2), same math as sparse test.
     * x = [1,2,3,4,5,6], A same as above, w = [1,2,3,4,5,6,7,8]
     */
    double x_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double w[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    expr *x = new_variable(3, 2, 0, 6);
    double A_dense[12] = {1.0, 0.0, 2.0, 3.0, 0.0, 4.0,
                          5.0, 0.0, 6.0, 7.0, 0.0, 0.0};

    expr *log_x = new_log(x);
    expr *A_log_x = new_dense_left_matmul(log_x, A_dense, 4, 3);

    A_log_x->forward(A_log_x, x_vals);
    A_log_x->jacobian_init(A_log_x);
    A_log_x->wsum_hess_init(A_log_x);
    A_log_x->eval_wsum_hess(A_log_x, w);

    double expected_x[6] = {-50.0,       -0.0, -28.0 / 9.0,
                            -57.0 / 8.0, -0.0, -19.0 / 9.0};
    int expected_i[6] = {0, 1, 2, 3, 4, 5};
    int expected_p[7] = {0, 1, 2, 3, 4, 5, 6};

    mu_assert("dense wsum_hess matrix vals incorrect",
              cmp_double_array(A_log_x->wsum_hess->x, expected_x, 6));
    mu_assert("dense wsum_hess matrix cols incorrect",
              cmp_int_array(A_log_x->wsum_hess->i, expected_i, 6));
    mu_assert("dense wsum_hess matrix rows incorrect",
              cmp_int_array(A_log_x->wsum_hess->p, expected_p, 7));

    free_expr(A_log_x);
    return 0;
}

const char *test_wsum_hess_dense_left_matmul_composite(void)
{
    /* Test with A @ log(B @ x), same math as sparse test.
     * x = [1,2,3], B = 3x3 all ones, A = [1,0,2;3,0,4;5,0,6;7,0,0], w = [1,2,3,4]
     * Expected: 3x3 dense matrix, all entries = -13/6
     */
    double x_vals[3] = {1.0, 2.0, 3.0};
    double w[4] = {1.0, 2.0, 3.0, 4.0};

    expr *x = new_variable(3, 1, 0, 3);

    CSR_Matrix *B = new_csr_matrix(3, 3, 9);
    int B_p[4] = {0, 3, 6, 9};
    int B_i[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    double B_x[9] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    memcpy(B->p, B_p, 4 * sizeof(int));
    memcpy(B->i, B_i, 9 * sizeof(int));
    memcpy(B->x, B_x, 9 * sizeof(double));

    double A_dense[12] = {1.0, 0.0, 2.0, 3.0, 0.0, 4.0,
                          5.0, 0.0, 6.0, 7.0, 0.0, 0.0};

    expr *Bx = new_linear(x, B, NULL);
    expr *log_Bx = new_log(Bx);
    expr *A_log_Bx = new_dense_left_matmul(log_Bx, A_dense, 4, 3);

    A_log_Bx->forward(A_log_Bx, x_vals);
    A_log_Bx->jacobian_init(A_log_Bx);
    A_log_Bx->wsum_hess_init(A_log_Bx);
    A_log_Bx->eval_wsum_hess(A_log_Bx, w);

    double expected_x[9] = {-13.0 / 6.0, -13.0 / 6.0, -13.0 / 6.0,
                            -13.0 / 6.0, -13.0 / 6.0, -13.0 / 6.0,
                            -13.0 / 6.0, -13.0 / 6.0, -13.0 / 6.0};
    int expected_i[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int expected_p[4] = {0, 3, 6, 9};

    mu_assert("dense composite wsum_hess vals incorrect",
              cmp_double_array(A_log_Bx->wsum_hess->x, expected_x, 9));
    mu_assert("dense composite wsum_hess cols incorrect",
              cmp_int_array(A_log_Bx->wsum_hess->i, expected_i, 9));
    mu_assert("dense composite wsum_hess rows incorrect",
              cmp_int_array(A_log_Bx->wsum_hess->p, expected_p, 4));

    free_csr_matrix(B);
    free_expr(A_log_Bx);
    return 0;
}
