#include <stdio.h>
#include <stdlib.h>

#include "atoms/affine.h"
#include "expr.h"
#include "minunit.h"
#include "subexpr.h"
#include "test_helpers.h"

const char *test_convolve_forward(void)
{
    /* Test: y = conv([1, 2, 3], x) where x = [1, 2, 3].
     * Expected y = [1, 4, 10, 12, 9]. */
    double kernel[3] = {1.0, 2.0, 3.0};
    expr *kernel_param = new_parameter(3, 1, PARAM_FIXED, 3, kernel);
    expr *x = new_variable(3, 1, 0, 3);
    expr *y = new_convolve(kernel_param, x);

    double u[3] = {1.0, 2.0, 3.0};
    y->forward(y, u);

    double expected[5] = {1.0, 4.0, 10.0, 12.0, 9.0};

    mu_assert("convolve result should have d1=5", y->d1 == 5);
    mu_assert("convolve result should have d2=1", y->d2 == 1);
    mu_assert("convolve result should have size=5", y->size == 5);
    mu_assert("Convolve forward pass test failed",
              cmp_double_array(y->value, expected, 5));

    free_expr(y);
    return 0;
}

const char *test_convolve_forward_param(void)
{
    /* Same as test_convolve_forward, but the kernel is an updatable parameter
     * (param_id = 0). Exercises the needs_parameter_refresh path in forward(). */
    double kernel[3] = {1.0, 2.0, 3.0};
    expr *kernel_param = new_parameter(3, 1, 0, 3, kernel);
    expr *x = new_variable(3, 1, 0, 3);
    expr *y = new_convolve(kernel_param, x);

    double u[3] = {1.0, 2.0, 3.0};
    y->forward(y, u);

    double expected[5] = {1.0, 4.0, 10.0, 12.0, 9.0};

    mu_assert("Convolve forward pass with param kernel failed",
              cmp_double_array(y->value, expected, 5));

    free_expr(y);
    return 0;
}
