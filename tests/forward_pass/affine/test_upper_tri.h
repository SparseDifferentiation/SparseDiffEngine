// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>

#include "atoms/affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_upper_tri_forward(void)
{
    /* 3x3 matrix variable (column-major): [1,2,3,4,5,6,7,8,9]
     * Matrix:  1 4 7
     *          2 5 8
     *          3 6 9
     * Upper tri (i < j): (0,1)=4, (0,2)=7, (1,2)=8
     * Flat indices: 3, 6, 7 */
    double u[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    expr *var = new_variable(3, 3, 0, 9);
    expr *ut = new_upper_tri(var);

    mu_assert("upper_tri d1", ut->d1 == 3);
    mu_assert("upper_tri d2", ut->d2 == 1);

    ut->forward(ut, u);

    double expected[3] = {4.0, 7.0, 8.0};
    mu_assert("upper_tri forward", cmp_double_array(ut->value, expected, 3));

    free_expr(ut);
    return 0;
}
