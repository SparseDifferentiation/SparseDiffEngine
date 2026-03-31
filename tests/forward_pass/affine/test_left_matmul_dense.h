#include <stdio.h>
#include <stdlib.h>

#include "atoms/affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_left_matmul_dense(void)
{
    /* Test: Z = A @ X where
     * A is 3x3 (row-major): [1 2 3; 4 5 6; 7 8 9]
     * X is 3x3 variable (col-major): [1 4 7; 2 5 8; 3 6 9]
     *
     * Z = A @ X = [14   32   50]
     *             [32   77  122]
     *             [50  122  194]
     */

    /* Create X variable (3 x 3) */
    expr *X = new_variable(3, 3, 0, 9);

    /* Constant matrix A in row-major order */
    double A_data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    /* Build expression Z = A @ X */
    expr *Z = new_left_matmul_dense(X, 3, 3, A_data);

    /* Variable values in column-major order */
    double u[9] = {1.0, 2.0, 3.0,  /* first column */
                   4.0, 5.0, 6.0,  /* second column */
                   7.0, 8.0, 9.0}; /* third column */

    /* Evaluate forward pass */
    Z->forward(Z, u);

    /* Expected result (3 x 3) in column-major order */
    double expected[9] = {14.0, 32.0,  50.0,   /* first column */
                          32.0, 77.0,  122.0,  /* second column */
                          50.0, 122.0, 194.0}; /* third column */

    /* Verify dimensions */
    mu_assert("left_matmul_dense result should have d1=3", Z->d1 == 3);
    mu_assert("left_matmul_dense result should have d2=3", Z->d2 == 3);
    mu_assert("left_matmul_dense result should have size=9", Z->size == 9);

    /* Verify values */
    mu_assert("Left matmul dense forward pass test failed",
              cmp_double_array(Z->value, expected, 9));

    free_expr(Z);
    return 0;
}
