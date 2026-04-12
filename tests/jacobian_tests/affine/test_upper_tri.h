#include <stdio.h>

#include "atoms/affine.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_upper_tri_jacobian_variable(void)
{
    /* upper_tri of a 4x4 variable (16 vars total)
     * Row-major upper tri indices: [4, 8, 12, 9, 13, 14]
     * Jacobian is 6x16 CSR: row k has a single 1.0 at col indices[k] */
    double u[16];
    for (int k = 0; k < 16; k++)
    {
        u[k] = (double) (k + 1);
    }
    expr *var = new_variable(4, 4, 0, 16);
    expr *ut = new_upper_tri(var);

    ut->forward(ut, u);
    jacobian_init(ut);
    ut->eval_jacobian(ut);

    double expected_x[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    int expected_p[7] = {0, 1, 2, 3, 4, 5, 6};
    int expected_i[6] = {4, 8, 12, 9, 13, 14};

    mu_assert("upper_tri jac vals",
              cmp_double_array(ut->jacobian->x, expected_x, 6));
    mu_assert("upper_tri jac p",
              cmp_int_array(ut->jacobian->p, expected_p, 7));
    mu_assert("upper_tri jac i",
              cmp_int_array(ut->jacobian->i, expected_i, 6));

    free_expr(ut);
    return 0;
}

const char *test_upper_tri_jacobian_of_log(void)
{
    /* upper_tri(log(X)) where X is 4x4 variable
     * Row-major upper tri indices: [4, 8, 12, 9, 13, 14]
     * Values at those positions: u[4]=5, u[8]=9, u[12]=13,
     *                            u[9]=10, u[13]=14, u[14]=15
     * d/dx log at those positions: 1/5, 1/9, 1/13, 1/10, 1/14, 1/15 */
    double u[16];
    for (int k = 0; k < 16; k++)
    {
        u[k] = (double) (k + 1);
    }
    expr *var = new_variable(4, 4, 0, 16);
    expr *log_node = new_log(var);
    expr *ut = new_upper_tri(log_node);

    ut->forward(ut, u);
    jacobian_init(ut);
    ut->eval_jacobian(ut);

    double expected_x[6] = {0.2, 1.0 / 9.0, 1.0 / 13.0,
                            0.1, 1.0 / 14.0, 1.0 / 15.0};
    int expected_i[6] = {4, 8, 12, 9, 13, 14};

    mu_assert("upper_tri log jac vals",
              cmp_double_array(ut->jacobian->x, expected_x, 6));
    mu_assert("upper_tri log jac cols",
              cmp_int_array(ut->jacobian->i, expected_i, 6));

    free_expr(ut);
    return 0;
}
