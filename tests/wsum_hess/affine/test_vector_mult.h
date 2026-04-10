#include <math.h>
#include <stdio.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "subexpr.h"
#include "test_helpers.h"

/* Test: y = a ∘ log(x) where a is a parameter vector */

const char *test_wsum_hess_vector_mult_log_vector(void)
{
    /* Create variable x: [1.0, 2.0, 4.0] */
    double u_vals[3] = {1.0, 2.0, 4.0};
    expr *x = new_variable(3, 1, 0, 3);

    /* Create log node: log(x) */
    expr *log_node = new_log(x);

    /* Create vector mult node: y = [2.0, 3.0, 4.0] ∘ log(x) */
    double a[3] = {2.0, 3.0, 4.0};
    expr *a_param = new_parameter(3, 1, PARAM_FIXED, 3, a);
    expr *y = new_vector_mult(a_param, log_node);

    /* Forward pass */
    y->forward(y, u_vals);

    /* Initialize and evaluate weighted Hessian with w = [1.0, 0.5, 0.25] */
    jacobian_init(y);
    wsum_hess_init(y);
    double w[3] = {1.0, 0.5, 0.25};
    y->eval_wsum_hess(y, w);

    double expected_x[3] = {-2.0, -0.375, -0.0625};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vector mult log hess: x values fail",
              cmp_double_array(y->wsum_hess->x, expected_x, 3));
    mu_assert("vector mult log hess: row pointers fail",
              cmp_int_array(y->wsum_hess->p, expected_p, 4));
    mu_assert("vector mult log hess: column indices fail",
              cmp_int_array(y->wsum_hess->i, expected_i, 3));

    free_expr(y);
    return 0;
}

const char *test_wsum_hess_vector_mult_log_matrix(void)
{
    /* Create variable x as 2x2 matrix: [[1.0, 2.0], [4.0, 8.0]] */
    double u_vals[4] = {1.0, 2.0, 4.0, 8.0};
    expr *x = new_variable(2, 2, 0, 4);

    /* Create log node: log(x) */
    expr *log_node = new_log(x);

    /* Create vector mult node: y = [1.5, 2.5, 3.5, 4.5] ∘ log(x) */
    double a[4] = {1.5, 2.5, 3.5, 4.5};
    expr *a_param = new_parameter(4, 1, PARAM_FIXED, 4, a);
    expr *y = new_vector_mult(a_param, log_node);

    /* Forward pass */
    y->forward(y, u_vals);

    /* Initialize and evaluate weighted Hessian with w = [1.0, 1.0, 1.0, 1.0] */
    jacobian_init(y);
    wsum_hess_init(y);
    double w[4] = {1.0, 1.0, 1.0, 1.0};
    y->eval_wsum_hess(y, w);

    double expected_x[4] = {-1.5, -0.625, -0.21875, -0.0703125};
    int expected_p[5] = {0, 1, 2, 3, 4};
    int expected_i[4] = {0, 1, 2, 3};

    mu_assert("vector mult log hess matrix: x values fail",
              cmp_double_array(y->wsum_hess->x, expected_x, 4));
    mu_assert("vector mult log hess matrix: row pointers fail",
              cmp_int_array(y->wsum_hess->p, expected_p, 5));
    mu_assert("vector mult log hess matrix: column indices fail",
              cmp_int_array(y->wsum_hess->i, expected_i, 4));

    free_expr(y);
    return 0;
}
