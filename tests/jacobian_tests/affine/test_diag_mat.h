#include <stdio.h>

#include "atoms/affine.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_diag_mat_jacobian_variable(void)
{
    /* diag_mat of a 2x2 variable (4 vars total)
     * Diagonal indices in column-major: [0, 3]
     * Jacobian is 2x4 CSR: row 0 has col 0, row 1 has col 3 */
    double u[4] = {1.0, 2.0, 3.0, 4.0};
    expr *var = new_variable(2, 2, 0, 4);
    expr *dm = new_diag_mat(var);

    dm->forward(dm, u);
    jacobian_init(dm);
    dm->eval_jacobian(dm);

    double expected_x[2] = {1.0, 1.0};
    int expected_p[3] = {0, 1, 2};
    int expected_i[2] = {0, 3};

    mu_assert("diag_mat jac vals", cmp_double_array(dm->jacobian->x, expected_x, 2));
    mu_assert("diag_mat jac p", cmp_int_array(dm->jacobian->p, expected_p, 3));
    mu_assert("diag_mat jac i", cmp_int_array(dm->jacobian->i, expected_i, 2));

    free_expr(dm);
    return 0;
}

const char *test_diag_mat_jacobian_of_log(void)
{
    /* diag_mat(log(X)) where X is 2x2 variable
     * X = [[1, 3], [2, 4]] (column-major: [1, 2, 3, 4])
     * Diagonal: x[0]=1, x[3]=4
     * d/dx log at diagonal positions:
     * Row 0: 1/1 = 1.0 at col 0
     * Row 1: 1/4 = 0.25 at col 3 */
    double u[4] = {1.0, 2.0, 3.0, 4.0};
    expr *var = new_variable(2, 2, 0, 4);
    expr *log_node = new_log(var);
    expr *dm = new_diag_mat(log_node);

    dm->forward(dm, u);
    jacobian_init(dm);
    dm->eval_jacobian(dm);

    double expected_x[2] = {1.0, 0.25};
    int expected_i[2] = {0, 3};

    mu_assert("diag_mat log jac vals",
              cmp_double_array(dm->jacobian->x, expected_x, 2));
    mu_assert("diag_mat log jac cols",
              cmp_int_array(dm->jacobian->i, expected_i, 2));

    free_expr(dm);
    return 0;
}
