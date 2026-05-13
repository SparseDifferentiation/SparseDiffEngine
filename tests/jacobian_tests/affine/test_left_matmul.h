#include <math.h>
#include <stdio.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "test_helpers.h"
#include "utils/permuted_dense.h"

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
     * Stored in CSR_matrix format (4x3 sparse):
     * nnz = 7
     * p = [0, 2, 4, 6, 7]
     * i = [0, 2, 0, 2, 0, 2, 0]
     * x = [1.0, 2.0/3.0, 3.0, 4.0/3.0, 5.0, 2.0, 7.0]
     */
    double x_vals[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(3, 1, 0, 3);

    /* Create sparse matrix A in CSR_matrix format */
    CSR_matrix *A = new_CSR_matrix(4, 3, 7);
    int A_p[5] = {0, 2, 4, 6, 7};
    int A_i[7] = {0, 2, 0, 2, 0, 2, 0};
    double A_x[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    memcpy(A->p, A_p, 5 * sizeof(int));
    memcpy(A->i, A_i, 7 * sizeof(int));
    memcpy(A->x, A_x, 7 * sizeof(double));

    expr *log_x = new_log(x);
    expr *A_log_x = new_left_matmul(NULL, log_x, A);

    A_log_x->forward(A_log_x, x_vals);
    jacobian_init(A_log_x);
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

    mu_assert("vals fail", cmp_values(A_log_x->jacobian, expected_Ax, 7));
    mu_assert("sparsity fail",
              cmp_sparsity(A_log_x->jacobian, expected_Ap, expected_Ai, 4, 7));

    free_CSR_matrix(A);
    free_expr(A_log_x);
    return 0;
}

const char *test_jacobian_left_matmul_log_matrix(void)
{
    /* x is 3x2, vectorized column-wise: [1,2,3 | 4,5,6] */
    double x_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    expr *x = new_variable(3, 2, 0, 6);

    /* Create sparse matrix A in CSR_matrix format (4x3) */
    CSR_matrix *A = new_CSR_matrix(4, 3, 7);
    int A_p[5] = {0, 2, 4, 6, 7};
    int A_i[7] = {0, 2, 0, 2, 0, 2, 0};
    double A_x[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    memcpy(A->p, A_p, 5 * sizeof(int));
    memcpy(A->i, A_i, 7 * sizeof(int));
    memcpy(A->x, A_x, 7 * sizeof(double));

    expr *log_x = new_log(x);
    expr *A_log_x = new_left_matmul(NULL, log_x, A);

    A_log_x->forward(A_log_x, x_vals);
    jacobian_init(A_log_x);
    A_log_x->eval_jacobian(A_log_x);

    /* Expected Jacobian: block-diagonal repeat of A scaled by diag(1./x) */
    double expected_Ax[14] = {/* first column block (x = [1, 2, 3]) */
                              1.0, 2.0 / 3.0, 3.0, 4.0 / 3.0, 5.0, 2.0, 7.0,
                              /* second column block (x = [4, 5, 6]) */
                              0.25, 1.0 / 3.0, 0.75, 2.0 / 3.0, 1.25, 1.0, 1.75};
    int expected_Ai[14] = {0, 2, 0, 2, 0, 2, 0, 3, 5, 3, 5, 3, 5, 3};
    int expected_Ap[9] = {0, 2, 4, 6, 7, 9, 11, 13, 14};

    mu_assert("vals fail", cmp_values(A_log_x->jacobian, expected_Ax, 14));
    mu_assert("sparsity fail",
              cmp_sparsity(A_log_x->jacobian, expected_Ap, expected_Ai, 8, 14));

    free_CSR_matrix(A);
    free_expr(A_log_x);
    return 0;
}

const char *test_jacobian_left_matmul_exp_composite(void)
{
    /* Test Jacobian of A @ exp(B @ x) */
    double x_vals[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(3, 1, 0, 3);

    /* Create B matrix (3x3 all ones) */
    CSR_matrix *B = new_CSR_matrix(3, 3, 9);
    int B_p[4] = {0, 3, 6, 9};
    int B_i[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    double B_x[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    memcpy(B->p, B_p, 4 * sizeof(int));
    memcpy(B->i, B_i, 9 * sizeof(int));
    memcpy(B->x, B_x, 9 * sizeof(double));

    /* Create A matrix */
    CSR_matrix *A = new_CSR_matrix(4, 3, 7);
    int A_p[5] = {0, 2, 4, 6, 7};
    int A_i[7] = {0, 2, 0, 2, 0, 2, 0};
    double A_x[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(A->p, A_p, 5 * sizeof(int));
    memcpy(A->i, A_i, 7 * sizeof(int));
    memcpy(A->x, A_x, 7 * sizeof(double));

    expr *Bx = new_linear(x, B, NULL);
    expr *exp_Bx = new_exp(Bx);
    expr *A_exp_Bx = new_left_matmul(NULL, exp_Bx, A);

    mu_assert("check_jacobian failed",
              check_jacobian_num(A_exp_Bx, x_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_CSR_matrix(A);
    free_CSR_matrix(B);
    free_expr(A_exp_Bx);
    return 0;
}

/* outer = A2 @ (A1 @ x). Inner left_matmul produces a PD Jacobian via the
   leaf-var fast path. Outer left_matmul sees a PD child Jacobian and must
   fire the produce_pd_jacobian_from_child branch via BA_pd_matrices_*.

   x is a length-2 leaf variable at var_id=0, n_vars=2.
   A1 is 3x2: [[1,2],[3,4],[5,6]] (row-major).
   A2 is 4x3: [[1,0,1],[0,1,0],[1,0,1],[0,1,0]] (row-major).
   Expected outer->jacobian: PD of shape (4, 2), row_perm=[0..3],
   col_perm=[0,1], X = A2 @ A1 = [[6,8],[3,4],[6,8],[3,4]]. */
const char *test_jacobian_left_matmul_pd_from_composite_child(void)
{
    double A1_data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double A2_data[12] = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                          1.0, 0.0, 1.0, 0.0, 1.0, 0.0};

    expr *x = new_variable(2, 1, 0, 2);
    expr *A1_x = new_left_matmul_dense(NULL, x, 3, 2, A1_data);
    expr *A2_A1_x = new_left_matmul_dense(NULL, A1_x, 4, 3, A2_data);

    double x_vals[2] = {0.5, -1.5};
    A2_A1_x->forward(A2_A1_x, x_vals);
    jacobian_init(A2_A1_x);
    A2_A1_x->eval_jacobian(A2_A1_x);

    /* Structural: outer's Jacobian must be PD (produced by the new
       produce_pd_jacobian_from_child branch). */
    mu_assert("outer Jacobian should be PD",
              A2_A1_x->jacobian->is_permuted_dense);
    permuted_dense *pd = (permuted_dense *) A2_A1_x->jacobian;
    mu_assert("global m", A2_A1_x->jacobian->m == 4);
    mu_assert("global n", A2_A1_x->jacobian->n == 2);
    mu_assert("m0", pd->m0 == 4);
    mu_assert("n0", pd->n0 == 2);
    int expected_row_perm[4] = {0, 1, 2, 3};
    int expected_col_perm[2] = {0, 1};
    mu_assert("row_perm", cmp_int_array(pd->row_perm, expected_row_perm, 4));
    mu_assert("col_perm", cmp_int_array(pd->col_perm, expected_col_perm, 2));

    /* Numerical: X = A2 @ A1 (row-major 4x2). */
    double expected_X[8] = {6.0, 8.0, 3.0, 4.0, 6.0, 8.0, 3.0, 4.0};
    mu_assert("X values", cmp_double_array(pd->X, expected_X, 8));

    /* Cross-check against numerical differentiation for paranoia. */
    mu_assert("check_jacobian failed",
              check_jacobian_num(A2_A1_x, x_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(A2_A1_x);
    return 0;
}
