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

const char *test_wsum_hess_diag_mat_log(void)
{
    /* diag_mat(log(X)) where X is 2x2, w = [1, 1]
     * X = [[1, 3], [2, 4]] (column-major: [1, 2, 3, 4])
     * Diagonal positions: flat indices 0 and 3
     * Hessian of log is diag(-1/x^2)
     * Weights scatter: parent_w = [1, 0, 0, 1]
     * H[0,0] = -1 * 1/1 = -1.0
     * H[1,1] = 0 (no weight)
     * H[2,2] = 0 (no weight)
     * H[3,3] = -1 * 1/16 = -0.0625 */
    double u[4] = {1.0, 2.0, 3.0, 4.0};
    double w[2] = {1.0, 1.0};

    expr *var = new_variable(2, 2, 0, 4);
    expr *log_node = new_log(var);
    expr *dm = new_diag_mat(log_node);

    dm->forward(dm, u);
    dm->jacobian_init(dm);
    dm->wsum_hess_init(dm);
    dm->eval_wsum_hess(dm, w);

    double expected_x[4] = {-1.0, 0.0, 0.0, -0.0625};
    int expected_p[5] = {0, 1, 2, 3, 4};
    int expected_i[4] = {0, 1, 2, 3};

    mu_assert("diag_mat log hess vals",
              cmp_double_array(dm->wsum_hess->x, expected_x, 4));
    mu_assert("diag_mat log hess p", cmp_int_array(dm->wsum_hess->p, expected_p, 5));
    mu_assert("diag_mat log hess i", cmp_int_array(dm->wsum_hess->i, expected_i, 4));

    free_expr(dm);
    return 0;
}
