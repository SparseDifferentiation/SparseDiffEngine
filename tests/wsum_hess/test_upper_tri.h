// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_wsum_hess_upper_tri_log(void)
{
    /* upper_tri(log(X)) where X is 3x3, w = [1, 1, 1]
     * X (column-major): [1, 2, 3, 4, 5, 6, 7, 8, 9]
     * Upper tri flat indices: [3, 6, 7]
     * Hessian of log is diag(-1/x^2)
     * Weights scatter: parent_w[3]=1, parent_w[6]=1, parent_w[7]=1
     * All other parent_w entries = 0
     *
     * H[k,k] = parent_w[k] * (-1/x[k]^2):
     * H[0,0] = 0, H[1,1] = 0, H[2,2] = 0
     * H[3,3] = -1/16 = -0.0625
     * H[4,4] = 0, H[5,5] = 0
     * H[6,6] = -1/49
     * H[7,7] = -1/64 = -0.015625
     * H[8,8] = 0 */
    double u[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double w[3] = {1.0, 1.0, 1.0};

    expr *var = new_variable(3, 3, 0, 9);
    expr *log_node = new_log(var);
    expr *ut = new_upper_tri(log_node);

    ut->forward(ut, u);
    ut->jacobian_init(ut);
    ut->wsum_hess_init(ut);
    ut->eval_wsum_hess(ut, w);

    double expected_x[9] = {0.0, 0.0,         0.0,       -0.0625, 0.0,
                            0.0, -1.0 / 49.0, -0.015625, 0.0};
    int expected_p[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int expected_i[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    mu_assert("upper_tri log hess vals",
              cmp_double_array(ut->wsum_hess->x, expected_x, 9));
    mu_assert("upper_tri log hess p",
              cmp_int_array(ut->wsum_hess->p, expected_p, 10));
    mu_assert("upper_tri log hess i",
              cmp_int_array(ut->wsum_hess->i, expected_i, 9));

    free_expr(ut);
    return 0;
}
