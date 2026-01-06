#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_promote_scalar_to_vector(void)
{
    /* Promote scalar to 4-element vector */
    double u[1] = {5.0};
    expr *var = new_variable(1, 1, 0, 1);
    expr *promote_node = new_promote(var, 4, 1);
    promote_node->forward(promote_node, u);

    double expected[4] = {5.0, 5.0, 5.0, 5.0};
    mu_assert("promote scalar->vector forward failed",
              cmp_double_array(promote_node->value, expected, 4));

    free_expr(promote_node);
    free_expr(var);
    return 0;
}

const char *test_promote_scalar_jacobian(void)
{
    /* Promote scalar to 3-element vector, check jacobian */
    double u[1] = {2.0};
    expr *var = new_variable(1, 1, 0, 1);
    expr *promote_node = new_promote(var, 3, 1);
    promote_node->forward(promote_node, u);
    promote_node->jacobian_init(promote_node);
    promote_node->eval_jacobian(promote_node);

    /* Jacobian is 3x1 with all 1s (each output depends on same input) */
    double expected_x[3] = {1.0, 1.0, 1.0};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 0, 0};

    mu_assert("promote jacobian vals fail",
              cmp_double_array(promote_node->jacobian->x, expected_x, 3));
    mu_assert("promote jacobian rows fail",
              cmp_int_array(promote_node->jacobian->p, expected_p, 4));
    mu_assert("promote jacobian cols fail",
              cmp_int_array(promote_node->jacobian->i, expected_i, 3));

    free_expr(promote_node);
    free_expr(var);
    return 0;
}

const char *test_promote_vector_jacobian(void)
{
    /* Promote 2-vector to 4-vector, check jacobian */
    double u[2] = {1.0, 2.0};
    expr *var = new_variable(2, 1, 0, 2);
    expr *promote_node = new_promote(var, 4, 1);
    promote_node->forward(promote_node, u);

    /* Pattern repeats: [1, 2, 1, 2] */
    double expected_val[4] = {1.0, 2.0, 1.0, 2.0};
    mu_assert("promote vector forward failed",
              cmp_double_array(promote_node->value, expected_val, 4));

    promote_node->jacobian_init(promote_node);
    promote_node->eval_jacobian(promote_node);

    /* Jacobian is 4x2:
     * Row 0: [1, 0] (output 0 depends on input 0)
     * Row 1: [0, 1] (output 1 depends on input 1)
     * Row 2: [1, 0] (output 2 depends on input 0)
     * Row 3: [0, 1] (output 3 depends on input 1)
     */
    double expected_x[4] = {1.0, 1.0, 1.0, 1.0};
    int expected_p[5] = {0, 1, 2, 3, 4};
    int expected_i[4] = {0, 1, 0, 1};

    mu_assert("promote vector jacobian vals fail",
              cmp_double_array(promote_node->jacobian->x, expected_x, 4));
    mu_assert("promote vector jacobian rows fail",
              cmp_int_array(promote_node->jacobian->p, expected_p, 5));
    mu_assert("promote vector jacobian cols fail",
              cmp_int_array(promote_node->jacobian->i, expected_i, 4));

    free_expr(promote_node);
    free_expr(var);
    return 0;
}
