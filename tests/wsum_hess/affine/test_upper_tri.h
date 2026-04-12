#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "atoms/affine.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_wsum_hess_upper_tri_log(void)
{
    /* upper_tri(log(X)) where X is 4x4, w = [1, 1, 1, 1, 1, 1]
     * X (column-major): [1..16]
     * Row-major upper tri indices: [4, 8, 12, 9, 13, 14]
     * Weights scatter to parent_w: 1.0 at positions
     * {4, 8, 9, 12, 13, 14}, zero elsewhere.
     *
     * Hessian of log is diag(-1/x^2).
     * H[k,k] = parent_w[k] * (-1/x[k]^2):
     *   H[4,4]   = -1/25  = -0.04
     *   H[8,8]   = -1/81
     *   H[9,9]   = -1/100 = -0.01
     *   H[12,12] = -1/169
     *   H[13,13] = -1/196
     *   H[14,14] = -1/225
     *   All others = 0 */
    double u[16];
    for (int k = 0; k < 16; k++)
    {
        u[k] = (double) (k + 1);
    }
    double w[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    expr *var = new_variable(4, 4, 0, 16);
    expr *log_node = new_log(var);
    expr *ut = new_upper_tri(log_node);

    ut->forward(ut, u);
    jacobian_init(ut);
    wsum_hess_init(ut);
    ut->eval_wsum_hess(ut, w);

    double expected_x[16] = {0.0,          0.0,          0.0,          0.0,
                             -1.0 / 25.0,  0.0,          0.0,          0.0,
                             -1.0 / 81.0,  -1.0 / 100.0, 0.0,          0.0,
                             -1.0 / 169.0, -1.0 / 196.0, -1.0 / 225.0, 0.0};
    int expected_p[17] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int expected_i[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    mu_assert("upper_tri log hess vals",
              cmp_double_array(ut->wsum_hess->x, expected_x, 16));
    mu_assert("upper_tri log hess p",
              cmp_int_array(ut->wsum_hess->p, expected_p, 17));
    mu_assert("upper_tri log hess i",
              cmp_int_array(ut->wsum_hess->i, expected_i, 16));

    free_expr(ut);
    return 0;
}
