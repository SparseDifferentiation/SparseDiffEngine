#include <stdio.h>
#include <stdlib.h>

#include "atoms/affine.h"
#include "expr.h"
#include "minunit.h"
#include "subexpr.h"
#include "test_helpers.h"

const char *test_kron_forward_const_left(void)
{
    /* Z = kron(A, B), A = [[1,2],[3,4]] constant, B = [[5,6],[7,8]] variable.
     * np.kron gives a 4x4; check its column-major flatten. */
    double A[4] = {1.0, 3.0, 2.0, 4.0}; /* col-major [[1,2],[3,4]] */
    expr *A_param = new_parameter(2, 2, PARAM_FIXED, 4, A);
    expr *B = new_variable(2, 2, 0, 4);
    expr *Z = new_kron(A_param, B, 1, 2, 2, 2, 2);

    double u[4] = {5.0, 7.0, 6.0, 8.0}; /* col-major [[5,6],[7,8]] */
    Z->forward(Z, u);

    double expected[16] = {5,  7,  15, 21, 6,  8,  18, 24,
                           10, 14, 20, 28, 12, 16, 24, 32};

    mu_assert("kron const-left d1=4", Z->d1 == 4);
    mu_assert("kron const-left d2=4", Z->d2 == 4);
    mu_assert("kron const-left forward failed",
              cmp_double_array(Z->value, expected, 16));

    free_expr(Z);
    return 0;
}

const char *test_kron_forward_const_right(void)
{
    /* Z = kron(A, B), A = [[5,6],[7,8]] variable, B = [[1,2],[3,4]] constant. */
    double B[4] = {1.0, 3.0, 2.0, 4.0}; /* col-major [[1,2],[3,4]] */
    expr *B_param = new_parameter(2, 2, PARAM_FIXED, 4, B);
    expr *A = new_variable(2, 2, 0, 4);
    expr *Z = new_kron(B_param, A, 0, 2, 2, 2, 2);

    double u[4] = {5.0, 7.0, 6.0, 8.0}; /* col-major [[5,6],[7,8]] */
    Z->forward(Z, u);

    double expected[16] = {5, 15, 7, 21, 10, 20, 14, 28,
                           6, 18, 8, 24, 12, 24, 16, 32};

    mu_assert("kron const-right d1=4", Z->d1 == 4);
    mu_assert("kron const-right d2=4", Z->d2 == 4);
    mu_assert("kron const-right forward failed",
              cmp_double_array(Z->value, expected, 16));

    free_expr(Z);
    return 0;
}

const char *test_kron_forward_scalar(void)
{
    /* Z = kron(y, B), y a scalar variable, B = [[1,2],[3,4]] constant.
     * Z = y * B; for y = 3 expect [[3,6],[9,12]] (col-major 3*B). */
    double B[4] = {1.0, 3.0, 2.0, 4.0};
    expr *B_param = new_parameter(2, 2, PARAM_FIXED, 1, B);
    expr *y = new_variable(1, 1, 0, 1);
    expr *Z = new_kron(B_param, y, 0, 1, 1, 2, 2);

    double u[1] = {3.0};
    Z->forward(Z, u);

    double expected[4] = {3.0, 9.0, 6.0, 12.0};

    mu_assert("kron scalar d1=2", Z->d1 == 2);
    mu_assert("kron scalar d2=2", Z->d2 == 2);
    mu_assert("kron scalar forward failed", cmp_double_array(Z->value, expected, 4));

    free_expr(Z);
    return 0;
}
