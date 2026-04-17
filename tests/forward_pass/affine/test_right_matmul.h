#include <string.h>

#include "atoms/affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_right_matmul(void)
{
    /* Test: Y = u @ A where
     * u is 2x3 variable: [[1, 2, 3], [4, 5, 6]]
     * A is 3x2 sparse:   [[1, 0], [0, 2], [3, 4]]
     * Y = u @ A = [[10, 16], [22, 34]]
     */

    /* Create u variable (2 x 3) */
    expr *u = new_variable(2, 3, 0, 6);

    /* Constant sparse matrix A (3 x 2) in CSR */
    CSR_Matrix *A = new_csr_matrix(3, 2, 4);
    int A_p[4] = {0, 1, 2, 4};
    int A_i[4] = {0, 1, 0, 1};
    double A_x[4] = {1.0, 2.0, 3.0, 4.0};
    memcpy(A->p, A_p, 4 * sizeof(int));
    memcpy(A->i, A_i, 4 * sizeof(int));
    memcpy(A->x, A_x, 4 * sizeof(double));

    /* Build expression Y = u @ A */
    expr *Y = new_right_matmul(NULL, u, A);

    /* Variable values in column-major order: cols [1,4], [2,5], [3,6] */
    double u_vals[6] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};

    /* Evaluate forward pass */
    Y->forward(Y, u_vals);

    /* Expected result (2 x 2) in column-major order: cols [10,22], [16,34] */
    double expected[4] = {10.0, 22.0, 16.0, 34.0};

    /* Verify dimensions */
    mu_assert("right_matmul result should have d1=2", Y->d1 == 2);
    mu_assert("right_matmul result should have d2=2", Y->d2 == 2);
    mu_assert("right_matmul result should have size=4", Y->size == 4);

    /* Verify values */
    mu_assert("right_matmul forward pass value mismatch",
              cmp_double_array(Y->value, expected, 4));

    free_csr_matrix(A);
    free_expr(Y);
    return 0;
}

const char *test_right_matmul_vector(void)
{
    /* Test: numpy 1D broadcast with u shape (3,) stored as (1, 3).
     * u = [1, 2, 3] @ A (same 3x2 sparse as above) = [10, 16]
     */

    /* Create u variable (1 x 3) */
    expr *u = new_variable(1, 3, 0, 3);

    /* Constant sparse matrix A (3 x 2) in CSR */
    CSR_Matrix *A = new_csr_matrix(3, 2, 4);
    int A_p[4] = {0, 1, 2, 4};
    int A_i[4] = {0, 1, 0, 1};
    double A_x[4] = {1.0, 2.0, 3.0, 4.0};
    memcpy(A->p, A_p, 4 * sizeof(int));
    memcpy(A->i, A_i, 4 * sizeof(int));
    memcpy(A->x, A_x, 4 * sizeof(double));

    /* Build expression Y = u @ A */
    expr *Y = new_right_matmul(NULL, u, A);

    /* Variable values */
    double u_vals[3] = {1.0, 2.0, 3.0};

    /* Evaluate forward pass */
    Y->forward(Y, u_vals);

    /* Expected result (1 x 2) */
    double expected[2] = {10.0, 16.0};

    /* Verify dimensions */
    mu_assert("right_matmul (vector) result should have d1=1", Y->d1 == 1);
    mu_assert("right_matmul (vector) result should have d2=2", Y->d2 == 2);

    /* Verify values */
    mu_assert("right_matmul (vector) forward pass value mismatch",
              cmp_double_array(Y->value, expected, 2));

    free_csr_matrix(A);
    free_expr(Y);
    return 0;
}
