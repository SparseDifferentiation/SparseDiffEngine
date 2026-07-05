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
     * All 4 blocks of A are active. */
    double A[4] = {1.0, 3.0, 2.0, 4.0}; /* col-major [[1,2],[3,4]] */
    int active[4] = {0, 1, 2, 3};
    expr *A_param = new_parameter(2, 2, PARAM_FIXED, 4, A);
    expr *B = new_variable(2, 2, 0, 4);
    expr *Z = new_left_kron(A_param, B, 2, 2, 2, 2, active, 4);

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
    int active[4] = {0, 1, 2, 3};
    expr *B_param = new_parameter(2, 2, PARAM_FIXED, 4, B);
    expr *A = new_variable(2, 2, 0, 4);
    expr *Z = new_right_kron(B_param, A, 2, 2, 2, 2, active, 4);

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
    /* Z = kron(y, B), y a scalar variable, B = [[1,2],[3,4]] constant (right).
     * Z = y * B; for y = 3 expect [[3,6],[9,12]] (col-major 3*B). */
    double B[4] = {1.0, 3.0, 2.0, 4.0};
    int active[4] = {0, 1, 2, 3};
    expr *B_param = new_parameter(2, 2, PARAM_FIXED, 1, B);
    expr *y = new_variable(1, 1, 0, 1);
    expr *Z = new_right_kron(B_param, y, 1, 1, 2, 2, active, 4);

    double u[1] = {3.0};
    Z->forward(Z, u);

    double expected[4] = {3.0, 9.0, 6.0, 12.0};

    mu_assert("kron scalar d1=2", Z->d1 == 2);
    mu_assert("kron scalar d2=2", Z->d2 == 2);
    mu_assert("kron scalar forward failed", cmp_double_array(Z->value, expected, 4));

    free_expr(Z);
    return 0;
}

const char *test_kron_forward_sparse(void)
{
    /* Z = kron(I_3, X), X = [[1,2],[3,4]] variable. Only the 3 diagonal blocks of
     * the identity are active; off-diagonal blocks are structurally zero. */
    double I3[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}; /* col-major identity */
    int active[3] = {0, 4, 8};                  /* col-major diagonal positions */
    expr *I_param = new_parameter(3, 3, PARAM_FIXED, 4, I3);
    expr *X = new_variable(2, 2, 0, 4);
    expr *Z = new_left_kron(I_param, X, 3, 3, 2, 2, active, 3);

    double u[4] = {1.0, 3.0, 2.0, 4.0}; /* col-major [[1,2],[3,4]] */
    Z->forward(Z, u);

    double expected[36] = {0};
    int pos[12] = {0, 1, 6, 7, 14, 15, 20, 21, 28, 29, 34, 35};
    double vals[12] = {1, 3, 2, 4, 1, 3, 2, 4, 1, 3, 2, 4};
    for (int t = 0; t < 12; t++)
        expected[pos[t]] = vals[t];

    mu_assert("kron sparse d1=6", Z->d1 == 6);
    mu_assert("kron sparse d2=6", Z->d2 == 6);
    mu_assert("kron sparse forward failed",
              cmp_double_array(Z->value, expected, 36));

    free_expr(Z);
    return 0;
}
