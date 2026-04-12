#include <stdio.h>

#include "atoms/affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_diag_mat_forward(void)
{
    /* 3x3 matrix variable (column-major): [1,2,3,4,5,6,7,8,9]
     * Matrix:  1 4 7
     *          2 5 8
     *          3 6 9
     * Diagonal: (0,0)=1, (1,1)=5, (2,2)=9 */
    double u[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    expr *var = new_variable(3, 3, 0, 9);
    expr *dm = new_diag_mat(var);

    mu_assert("diag_mat d1", dm->d1 == 3);
    mu_assert("diag_mat d2", dm->d2 == 1);

    dm->forward(dm, u);

    double expected[3] = {1.0, 5.0, 9.0};
    mu_assert("diag_mat forward", cmp_double_array(dm->value, expected, 3));

    free_expr(dm);
    return 0;
}
