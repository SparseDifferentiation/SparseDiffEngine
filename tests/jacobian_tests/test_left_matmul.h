#include <math.h>
#include <stdio.h>

#include "bivariate.h"
#include "elementwise_full_dom.h"
#include "elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "test_helpers.h"

const char *test_jacobian_left_matmul_log(void)
{
    /* Test Jacobian of A @ log(x) where:
     * x is 3x1 variable at x = [1, 2, 3]
     * A is 4x3 sparse matrix [1, 0, 2; 3, 0, 4; 5, 0, 6; 7, 0, 0]
     * Output: A @ log(x) is 4x1
     *
     * Jacobian is d(A @ log(x))/dx = A @ diag(1/x)
     * At x = [1, 2, 3], this is:
     * [1,   0, 2/3  ]
     * [3,   0, 4/3  ]
     * [5,   0, 2    ]
     * [7,   0, 0    ]
     *
     * Stored in CSR format (4x3 sparse):
     * nnz = 7
     * p = [0, 2, 4, 6, 7]
     * i = [0, 2, 0, 2, 0, 2, 0]
     * x = [1.0, 2.0/3.0, 3.0, 4.0/3.0, 5.0, 2.0, 7.0]
     */
    double x_vals[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(3, 1, 0, 3);

    /* Create sparse matrix A in CSR format */
    CSR_Matrix *A = new_csr_matrix(4, 3, 7);
    int A_p[5] = {0, 2, 4, 6, 7};
    int A_i[7] = {0, 2, 0, 2, 0, 2, 0};
    double A_x[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    memcpy(A->p, A_p, 5 * sizeof(int));
    memcpy(A->i, A_i, 7 * sizeof(int));
    memcpy(A->x, A_x, 7 * sizeof(double));

    expr *log_x = new_log(x);
    expr *A_log_x = new_left_matmul(log_x, A);

    A_log_x->forward(A_log_x, x_vals);
    A_log_x->jacobian_init(A_log_x);
    A_log_x->eval_jacobian(A_log_x);

    /* Expected jacobian values: A @ diag(1/x) */
    double expected_Ax[7] = {
        1.0,       /* row 0, col 0: 1 * (1/1) */
        2.0 / 3.0, /* row 0, col 2: 2 * (1/3) */
        3.0,       /* row 1, col 0: 3 * (1/1) */
        4.0 / 3.0, /* row 1, col 2: 4 * (1/3) */
        5.0,       /* row 2, col 0: 5 * (1/1) */
        2.0,       /* row 2, col 2: 6 * (1/3) */
        7.0        /* row 3, col 0: 7 * (1/1) */
    };
    int expected_Ai[7] = {0, 2, 0, 2, 0, 2, 0};
    int expected_Ap[5] = {0, 2, 4, 6, 7};

    mu_assert("vals fail", cmp_double_array(A_log_x->jacobian->x, expected_Ax, 7));
    mu_assert("cols fail", cmp_int_array(A_log_x->jacobian->i, expected_Ai, 7));
    mu_assert("rows fail", cmp_int_array(A_log_x->jacobian->p, expected_Ap, 5));

    free_csr_matrix(A);
    free_expr(A_log_x);
    return 0;
}

const char *test_jacobian_left_matmul_log_matrix(void)
{
    /* x is 3x2, vectorized column-wise: [1,2,3 | 4,5,6] */
    double x_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    expr *x = new_variable(3, 2, 0, 6);

    /* Create sparse matrix A in CSR format (4x3) */
    CSR_Matrix *A = new_csr_matrix(4, 3, 7);
    int A_p[5] = {0, 2, 4, 6, 7};
    int A_i[7] = {0, 2, 0, 2, 0, 2, 0};
    double A_x[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    memcpy(A->p, A_p, 5 * sizeof(int));
    memcpy(A->i, A_i, 7 * sizeof(int));
    memcpy(A->x, A_x, 7 * sizeof(double));

    expr *log_x = new_log(x);
    expr *A_log_x = new_left_matmul(log_x, A);

    A_log_x->forward(A_log_x, x_vals);
    A_log_x->jacobian_init(A_log_x);
    A_log_x->eval_jacobian(A_log_x);

    /* Expected Jacobian: block-diagonal repeat of A scaled by diag(1./x) */
    double expected_Ax[14] = {/* first column block (x = [1, 2, 3]) */
                              1.0, 2.0 / 3.0, 3.0, 4.0 / 3.0, 5.0, 2.0, 7.0,
                              /* second column block (x = [4, 5, 6]) */
                              0.25, 1.0 / 3.0, 0.75, 2.0 / 3.0, 1.25, 1.0, 1.75};
    int expected_Ai[14] = {0, 2, 0, 2, 0, 2, 0, 3, 5, 3, 5, 3, 5, 3};
    int expected_Ap[9] = {0, 2, 4, 6, 7, 9, 11, 13, 14};

    mu_assert("vals fail", cmp_double_array(A_log_x->jacobian->x, expected_Ax, 14));
    mu_assert("cols fail", cmp_int_array(A_log_x->jacobian->i, expected_Ai, 14));
    mu_assert("rows fail", cmp_int_array(A_log_x->jacobian->p, expected_Ap, 9));

    free_csr_matrix(A);
    free_expr(A_log_x);
    return 0;
}

const char *test_jacobian_left_matmul_exp_composite(void)
{
    /* Test Jacobian of A @ exp(B @ x) */
    double x_vals[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(3, 1, 0, 3);

    /* Create B matrix (3x3 all ones) */
    CSR_Matrix *B = new_csr_matrix(3, 3, 9);
    int B_p[4] = {0, 3, 6, 9};
    int B_i[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    double B_x[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    memcpy(B->p, B_p, 4 * sizeof(int));
    memcpy(B->i, B_i, 9 * sizeof(int));
    memcpy(B->x, B_x, 9 * sizeof(double));

    /* Create A matrix */
    CSR_Matrix *A = new_csr_matrix(4, 3, 7);
    int A_p[5] = {0, 2, 4, 6, 7};
    int A_i[7] = {0, 2, 0, 2, 0, 2, 0};
    double A_x[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(A->p, A_p, 5 * sizeof(int));
    memcpy(A->i, A_i, 7 * sizeof(int));
    memcpy(A->x, A_x, 7 * sizeof(double));

    expr *Bx = new_linear(x, B, NULL);
    expr *exp_Bx = new_exp(Bx);
    expr *A_exp_Bx = new_left_matmul(exp_Bx, A);

    mu_assert("check_jacobian failed",
              check_jacobian(A_exp_Bx, x_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_csr_matrix(A);
    free_csr_matrix(B);
    free_expr(A_exp_Bx);
    return 0;
}
