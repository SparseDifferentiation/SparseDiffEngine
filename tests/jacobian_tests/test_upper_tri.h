// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_upper_tri_jacobian_variable(void)
{
    /* upper_tri of a 3x3 variable (9 vars total)
     * Upper tri flat indices: [3, 6, 7]
     * Jacobian is 3x9 CSR: row k has col indices[k] */
    double u[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    expr *var = new_variable(3, 3, 0, 9);
    expr *ut = new_upper_tri(var);

    ut->forward(ut, u);
    ut->jacobian_init(ut);
    ut->eval_jacobian(ut);

    double expected_x[3] = {1.0, 1.0, 1.0};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {3, 6, 7};

    mu_assert("upper_tri jac vals",
              cmp_double_array(ut->jacobian->x, expected_x, 3));
    mu_assert("upper_tri jac p", cmp_int_array(ut->jacobian->p, expected_p, 4));
    mu_assert("upper_tri jac i", cmp_int_array(ut->jacobian->i, expected_i, 3));

    free_expr(ut);
    return 0;
}

const char *test_upper_tri_jacobian_of_log(void)
{
    /* upper_tri(log(X)) where X is 3x3 variable
     * Upper tri flat indices: [3, 6, 7]
     * X values at those positions: x[3]=4, x[6]=7, x[7]=8
     * d/dx log at those positions:
     * Row 0: 1/4 = 0.25 at col 3
     * Row 1: 1/7 at col 6
     * Row 2: 1/8 = 0.125 at col 7 */
    double u[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    expr *var = new_variable(3, 3, 0, 9);
    expr *log_node = new_log(var);
    expr *ut = new_upper_tri(log_node);

    ut->forward(ut, u);
    ut->jacobian_init(ut);
    ut->eval_jacobian(ut);

    double expected_x[3] = {0.25, 1.0 / 7.0, 0.125};
    int expected_i[3] = {3, 6, 7};

    mu_assert("upper_tri log jac vals",
              cmp_double_array(ut->jacobian->x, expected_x, 3));
    mu_assert("upper_tri log jac cols",
              cmp_int_array(ut->jacobian->i, expected_i, 3));

    free_expr(ut);
    return 0;
}
