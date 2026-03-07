#include <math.h>
#include <stdio.h>
#include <string.h>

#include "bivariate.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_jacobian_dense_left_matmul_log(void)
{
    /* Test Jacobian of A @ log(x) where A is dense, identical to the sparse test.
     * x is 3x1 variable at x = [1, 2, 3]
     * A is 4x3 dense matrix [1, 0, 2; 3, 0, 4; 5, 0, 6; 7, 0, 0]
     * Expected Jacobian = A @ diag(1/x):
     * [1, 0, 2/3; 3, 0, 4/3; 5, 0, 2; 7, 0, 0]
     */
    double x_vals[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(3, 1, 0, 3);

    /* A is 4x3 dense row-major */
    double A_dense[12] = {1.0, 0.0, 2.0, 3.0, 0.0, 4.0,
                          5.0, 0.0, 6.0, 7.0, 0.0, 0.0};

    expr *log_x = new_log(x);
    expr *A_log_x = new_dense_left_matmul(log_x, A_dense, 4, 3);

    A_log_x->forward(A_log_x, x_vals);
    A_log_x->jacobian_init(A_log_x);
    A_log_x->eval_jacobian(A_log_x);

    /* Since A is dense, the jacobian may have entries where A has zeros.
     * For dense_left_matmul, ALL m rows get entries if any child jacobian
     * entry exists in the block. Since child is a variable (identity jacobian),
     * all 3 columns have entries, so all 4 rows get all 3 columns.
     * The values at zero positions of A will be 0.0. */
    double expected_Ax[12] = {
        1.0, 0.0, 2.0 / 3.0, /* row 0: A[0,:] @ diag(1/x) */
        3.0, 0.0, 4.0 / 3.0, /* row 1 */
        5.0, 0.0, 2.0,       /* row 2 */
        7.0, 0.0, 0.0        /* row 3 */
    };
    int expected_Ai[12] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
    int expected_Ap[5] = {0, 3, 6, 9, 12};

    mu_assert("dense_left_matmul vals fail",
              cmp_double_array(A_log_x->jacobian->x, expected_Ax, 12));
    mu_assert("dense_left_matmul cols fail",
              cmp_int_array(A_log_x->jacobian->i, expected_Ai, 12));
    mu_assert("dense_left_matmul rows fail",
              cmp_int_array(A_log_x->jacobian->p, expected_Ap, 5));

    free_expr(A_log_x);
    return 0;
}

const char *test_jacobian_dense_left_matmul_log_matrix(void)
{
    /* x is 3x2, vectorized column-wise: [1,2,3 | 4,5,6]
     * A is 4x3 dense: [1, 0, 2; 3, 0, 4; 5, 0, 6; 7, 0, 0]
     * Jacobian = block-diag(A @ diag(1/x_col0), A @ diag(1/x_col1))
     * = 8x6 block-diagonal with two 4x3 blocks */
    double x_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    expr *x = new_variable(3, 2, 0, 6);

    double A_dense[12] = {1.0, 0.0, 2.0, 3.0, 0.0, 4.0,
                          5.0, 0.0, 6.0, 7.0, 0.0, 0.0};

    expr *log_x = new_log(x);
    expr *A_log_x = new_dense_left_matmul(log_x, A_dense, 4, 3);

    A_log_x->forward(A_log_x, x_vals);
    A_log_x->jacobian_init(A_log_x);
    A_log_x->eval_jacobian(A_log_x);

    /* 8x6 Jacobian, block-diagonal with 4x3 dense blocks */
    double expected_Ax[24] = {
        /* block 0: A @ diag(1/[1,2,3]) */
        1.0, 0.0, 2.0 / 3.0, 3.0, 0.0, 4.0 / 3.0, 5.0, 0.0, 2.0, 7.0, 0.0, 0.0,
        /* block 1: A @ diag(1/[4,5,6]) */
        0.25, 0.0, 1.0 / 3.0, 0.75, 0.0, 2.0 / 3.0, 1.25, 0.0, 1.0, 1.75, 0.0, 0.0};
    int expected_Ai[24] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
                           3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5};
    int expected_Ap[9] = {0, 3, 6, 9, 12, 15, 18, 21, 24};

    mu_assert("dense_left_matmul_matrix vals fail",
              cmp_double_array(A_log_x->jacobian->x, expected_Ax, 24));
    mu_assert("dense_left_matmul_matrix cols fail",
              cmp_int_array(A_log_x->jacobian->i, expected_Ai, 24));
    mu_assert("dense_left_matmul_matrix rows fail",
              cmp_int_array(A_log_x->jacobian->p, expected_Ap, 9));

    free_expr(A_log_x);
    return 0;
}

const char *test_jacobian_dense_left_matmul_log_composite(void)
{
    /* Test Jacobian of A @ log(B @ x) where both A and B are dense.
     * x is 3x1 at x = [1, 2, 3]
     * B is 3x3 all ones, A is 4x3 dense: [1, 0, 2; 3, 0, 4; 5, 0, 6; 7, 0, 0]
     *
     * B @ x = [6, 6, 6]^T
     * Jacobian = A @ diag(1/(B@x)) @ B
     * = A @ diag([1/6, 1/6, 1/6]) @ all_ones(3x3)
     * Row 0: (1+0+2)/6 * [1,1,1] = [1/2, 1/2, 1/2]
     * Row 1: (3+0+4)/6 * [1,1,1] = [7/6, 7/6, 7/6]
     * Row 2: (5+0+6)/6 * [1,1,1] = [11/6, 11/6, 11/6]
     * Row 3: (7+0+0)/6 * [1,1,1] = [7/6, 7/6, 7/6]
     */
    double x_vals[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(3, 1, 0, 3);

    /* B as CSR (for new_linear) */
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
    A_log_Bx->eval_jacobian(A_log_Bx);

    double expected_Ax[12] = {
        0.5,        0.5,        0.5,        /* row 0 */
        7.0 / 6.0,  7.0 / 6.0,  7.0 / 6.0,  /* row 1 */
        11.0 / 6.0, 11.0 / 6.0, 11.0 / 6.0, /* row 2 */
        7.0 / 6.0,  7.0 / 6.0,  7.0 / 6.0   /* row 3 */
    };
    int expected_Ai[12] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
    int expected_Ap[5] = {0, 3, 6, 9, 12};

    mu_assert("dense composite vals fail",
              cmp_double_array(A_log_Bx->jacobian->x, expected_Ax, 12));
    mu_assert("dense composite cols fail",
              cmp_int_array(A_log_Bx->jacobian->i, expected_Ai, 12));
    mu_assert("dense composite rows fail",
              cmp_int_array(A_log_Bx->jacobian->p, expected_Ap, 5));

    free_csr_matrix(B);
    free_expr(A_log_Bx);
    return 0;
}
