#include "affine.h"
#include "bivariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>

const char *test_wsum_hess_rel_entr_1()
{
    // Total 10 variables: [?, x0, x1, x2, ?, ?, y0, y1, y2, ?]
    // x has var_id = 1, y has var_id = 6
    // x = [1, 2, 3], y = [4, 5, 6]
    // w = [1, 2, 3]

    double u_vals[10] = {0, 1.0, 2.0, 3.0, 0, 0, 4.0, 5.0, 6.0, 0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 1, 10);
    expr *y = new_variable(3, 1, 6, 10);
    expr *node = new_rel_entr_vector_args(x, y);

    node->forward(node, u_vals);
    node->wsum_hess_init(node);
    node->eval_wsum_hess(node, w);

    int expected_p[11] = {0, 0, 2, 4, 6, 6, 6, 8, 10, 12, 12};
    int expected_i[12] = {1, 6, 2, 7, 3, 8, 1, 6, 2, 7, 3, 8};
    double expected_x[12] = {
        1.0,   -0.25,  // row 1
        1.0,   -0.4,   // row 2
        1.0,   -0.5,   // row 3
        -0.25, 0.0625, // row 6
        -0.4,  0.16,   // row 7
        -0.5,  0.25    // row 8
    };

    mu_assert("p array fails", cmp_int_array(node->wsum_hess->p, expected_p, 11));
    mu_assert("i array fails", cmp_int_array(node->wsum_hess->i, expected_i, 12));
    mu_assert("x array fails", cmp_double_array(node->wsum_hess->x, expected_x, 12));

    free_expr(node);
    return 0;
}

const char *test_wsum_hess_rel_entr_2()
{
    // Total 10 variables: [?, y0, y1, y2, ?, ?, x0, x1, x2, ?]
    // y has var_id = 1, x has var_id = 6
    // x = [1, 2, 3], y = [4, 5, 6]
    // w = [1, 2, 3]

    double u_vals[10] = {0, 4.0, 5.0, 6.0, 0, 0, 1.0, 2.0, 3.0, 0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 6, 10);
    expr *y = new_variable(3, 1, 1, 10);
    expr *node = new_rel_entr_vector_args(x, y);

    node->forward(node, u_vals);
    node->wsum_hess_init(node);
    node->eval_wsum_hess(node, w);

    int expected_p[11] = {0, 0, 2, 4, 6, 6, 6, 8, 10, 12, 12};
    int expected_i[12] = {1, 6, 2, 7, 3, 8, 1, 6, 2, 7, 3, 8};
    double expected_x[12] = {
        0.0625, -0.25, // row 1
        0.16,   -0.4,  // row 2
        0.25,   -0.5,  // row 3
        -0.25,  1.0,   // row 6
        -0.4,   1.0,   // row 7
        -0.5,   1.0    // row 8
    };

    mu_assert("p array fails", cmp_int_array(node->wsum_hess->p, expected_p, 11));
    mu_assert("i array fails", cmp_int_array(node->wsum_hess->i, expected_i, 12));
    mu_assert("x array fails", cmp_double_array(node->wsum_hess->x, expected_x, 12));

    free_expr(node);
    return 0;
}
