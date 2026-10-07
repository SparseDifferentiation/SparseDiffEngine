#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_sum_axis_neg1(void)
{
    /* Create a 3x2 constant matrix stored column-wise:
       [1, 4]
       [2, 5]
       [3, 6]
       Stored as: [1, 2, 3, 4, 5, 6]
    */
    double values[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    expr *const_node = new_parameter(3, 2, PARAM_FIXED, 0, values);
    expr *sum_node = new_sum(const_node, -1);
    sum_node->forward(sum_node, NULL);

    /* Expected: 1 + 2 + 3 + 4 + 5 + 6 */
    double expected = 21.0;

    mu_assert("Sum with axis=-1 test failed",
              fabs(sum_node->value[0] - expected) < 1e-10);

    free_expr(sum_node);
    return 0;
}

const char *test_sum_axis_0(void)
{
    /* Create a 3x2 constant matrix stored column-wise:
       [1, 4]
       [2, 5]
       [3, 6]
       Stored as: [1, 2, 3, 4, 5, 6]
    */
    double values[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    expr *const_node = new_parameter(3, 2, PARAM_FIXED, 0, values);
    expr *sum_node = new_sum(const_node, 0);
    sum_node->forward(sum_node, NULL);

    /* Expected: sum along rows (axis=0), result is 1x2
       [1 + 2 + 3, 4 + 5 + 6]
    */
    double expected[2] = {6.0, 15.0};

    mu_assert("Sum with axis=0 test failed",
              cmp_double_array(sum_node->value, expected, 2));

    free_expr(sum_node);
    return 0;
}

const char *test_sum_axis_1(void)
{
    /* Create a 3x2 constant matrix stored column-wise:
       [1, 4]
       [2, 5]
       [3, 6]
       Stored as: [1, 2, 3, 4, 5, 6]
    */
    double values[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    expr *const_node = new_parameter(3, 2, PARAM_FIXED, 0, values);
    expr *sum_node = new_sum(const_node, 1);
    sum_node->forward(sum_node, NULL);

    /* Expected: sum along columns (axis=1), result is 3x1
       [1 + 4]
       [2 + 5]
       [3 + 6]
    */
    double expected[3] = {5.0, 7.0, 9.0};

    mu_assert("Sum with axis=1 test failed",
              cmp_double_array(sum_node->value, expected, 3));

    free_expr(sum_node);
    return 0;
}
