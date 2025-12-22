#include <stdio.h>

#include "affine/variable.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_jacobian_log()
{
    double u_vals[5] = {0.0, 0.0, 1.0, 2.0, 3.0};
    double expected_Ax[3] = {1.0, 0.5, 0.333333333};
    int expected_Ap[4] = {0, 1, 2, 3};
    int expected_Ai[3] = {2, 3, 4};
    expr *u = new_variable(3, 2, 5);
    expr *log_node = new_log(u);
    log_node->forward(log_node, u_vals);
    log_node->jacobian_init(log_node);
    log_node->eval_jacobian(log_node);
    mu_assert("vals fail", cmp_double_array(log_node->jacobian->x, expected_Ax, 3));
    mu_assert("rows fail", cmp_int_array(log_node->jacobian->p, expected_Ap, 4));
    mu_assert("cols fail", cmp_int_array(log_node->jacobian->i, expected_Ai, 3));
    free_expr(log_node);
    return 0;
}
