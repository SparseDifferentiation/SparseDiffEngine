#include "affine.h"
#include "bivariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>

const char *test_jacobian_rel_entr_vector_args_1()
{
    // var = (z, x, w, y) where z is 2 x 1, x is 3 x 1, w is 2 x 1, y is 3 x 1
    // we compute jacobian of x log(x) - x log(y)
    double u_vals[10] = {0, 0, 1.0, 2.0, 3.0, 0, 0, 4.0, 5.0, 6.0};
    expr *x = new_variable(3, 1, 2, 10);
    expr *y = new_variable(3, 1, 7, 10);
    expr *node = new_rel_entr_vector_args(x, y);

    node->forward(node, u_vals);
    node->jacobian_init(node);
    node->eval_jacobian(node);

    double a = log(1.0 / 4.0) + 1.0;
    double b = log(2.0 / 5.0) + 1.0;
    double c = log(3.0 / 6.0) + 1.0;
    double d = -1.0 / 4.0;
    double e = -2.0 / 5.0;
    double f = -3.0 / 6.0;

    double expected_Ax[6] = {a, d, b, e, c, f};
    int expected_Ap[4] = {0, 2, 4, 6};
    int expected_Ai[6] = {2, 7, 3, 8, 4, 9};

    mu_assert("vals fail", cmp_double_array(node->jacobian->x, expected_Ax, 6));
    mu_assert("rows fail", cmp_int_array(node->jacobian->p, expected_Ap, 4));
    mu_assert("cols fail", cmp_int_array(node->jacobian->i, expected_Ai, 6));
    free_expr(node);
    free_expr(x);
    free_expr(y);
    return 0;
}

const char *test_jacobian_rel_entr_vector_args_2()
{
    // var = (z, y, w, x) where z is 2 x 1, x is 3 x 1, w is 2 x 1, y is 3 x 1
    // we compute jacobian of x log(x) - x log(y)
    double u_vals[10] = {0, 0, 4, 5, 6, 0, 0, 1.0, 2.0, 3.0};
    expr *x = new_variable(3, 1, 7, 10);
    expr *y = new_variable(3, 1, 2, 10);
    expr *node = new_rel_entr_vector_args(x, y);

    node->forward(node, u_vals);
    node->jacobian_init(node);
    node->eval_jacobian(node);

    double a = log(1.0 / 4.0) + 1.0;
    double b = log(2.0 / 5.0) + 1.0;
    double c = log(3.0 / 6.0) + 1.0;
    double d = -1.0 / 4.0;
    double e = -2.0 / 5.0;
    double f = -3.0 / 6.0;

    double expected_Ax[6] = {d, a, e, b, f, c};
    int expected_Ap[4] = {0, 2, 4, 6};
    int expected_Ai[6] = {2, 7, 3, 8, 4, 9};

    mu_assert("vals fail", cmp_double_array(node->jacobian->x, expected_Ax, 6));
    mu_assert("rows fail", cmp_int_array(node->jacobian->p, expected_Ap, 4));
    mu_assert("cols fail", cmp_int_array(node->jacobian->i, expected_Ai, 6));
    free_expr(node);
    free_expr(x);
    free_expr(y);
    return 0;
}
