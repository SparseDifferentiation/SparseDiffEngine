#include <stdio.h>

#include "atoms/affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_upper_tri_forward_4x4(void)
{
    /* 4x4 matrix variable (column-major): [1..16]
     * Matrix:  1  5   9  13
     *          2  6  10  14
     *          3  7  11  15
     *          4  8  12  16
     * Upper tri in row-major order (matching CVXPY):
     *   Row 0: (0,1)=5,  (0,2)=9,  (0,3)=13
     *   Row 1: (1,2)=10, (1,3)=14
     *   Row 2: (2,3)=15
     * Flat indices: 4, 8, 12, 9, 13, 14
     *
     * NOTE: column-major order would give [4, 8, 9, 12, 13, 14]
     * instead. This test verifies the row-major ordering. */
    double u[16];
    for (int k = 0; k < 16; k++)
    {
        u[k] = (double) (k + 1);
    }
    expr *var = new_variable(4, 4, 0, 16);
    expr *ut = new_upper_tri(var);

    mu_assert("upper_tri 4x4 d1", ut->d1 == 6);
    mu_assert("upper_tri 4x4 d2", ut->d2 == 1);

    ut->forward(ut, u);

    double expected[6] = {5.0, 9.0, 13.0, 10.0, 14.0, 15.0};
    mu_assert("upper_tri forward 4x4",
              cmp_double_array(ut->value, expected, 6));

    free_expr(ut);
    return 0;
}
