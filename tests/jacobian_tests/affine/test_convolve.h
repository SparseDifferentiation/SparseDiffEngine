#include <math.h>
#include <stdio.h>

#include "atoms/affine.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "subexpr.h"
#include "test_helpers.h"

const char *test_jacobian_convolve(void)
{
    /* Test: Jacobian of y = conv([1, 2, 3], x) for x = Variable(3).
     * J should equal T(kernel):
     *   [[1, 0, 0],
     *    [2, 1, 0],
     *    [3, 2, 1],
     *    [0, 3, 2],
     *    [0, 0, 3]]
     * stored in CSR with nnz = 9, shape 5 x 3. */
    double kernel[3] = {1.0, 2.0, 3.0};
    expr *kernel_param = new_parameter(3, 1, PARAM_FIXED, 3, kernel);
    expr *x = new_variable(3, 1, 0, 3);
    expr *y = new_convolve(kernel_param, x);

    double u[3] = {1.0, 2.0, 3.0};
    y->forward(y, u);
    jacobian_init(y);
    y->eval_jacobian(y);

    mu_assert("Jacobian should have 5 rows", y->jacobian->m == 5);
    mu_assert("Jacobian should have 3 columns", y->jacobian->n == 3);
    mu_assert("Jacobian should have 9 nonzeros", y->jacobian->nnz == 9);

    int expected_p[6] = {0, 1, 3, 6, 8, 9};
    int expected_i[9] = {0, 0, 1, 0, 1, 2, 1, 2, 2};
    double expected_x[9] = {1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0, 3.0};

    mu_assert("Convolve Jacobian row pointers incorrect",
              cmp_int_array(y->jacobian->p, expected_p, 6));
    mu_assert("Convolve Jacobian column indices incorrect",
              cmp_int_array(y->jacobian->i, expected_i, 9));
    mu_assert("Convolve Jacobian values incorrect",
              cmp_double_array(y->jacobian->x, expected_x, 9));

    free_expr(y);
    return 0;
}

const char *test_jacobian_convolve_composite(void)
{
    /* y = conv([1, 2, 3], log(x)) — verify via numerical differentiation that
     * the composite path (T(a) @ J_log) matches. */
    double kernel[3] = {1.0, 2.0, 3.0};
    double x_vals[3] = {1.0, 2.0, 3.0};

    expr *kernel_param = new_parameter(3, 1, PARAM_FIXED, 3, kernel);
    expr *x = new_variable(3, 1, 0, 3);
    expr *log_x = new_log(x);
    expr *y = new_convolve(kernel_param, log_x);

    mu_assert("check_jacobian failed",
              check_jacobian_num(y, x_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(y);
    return 0;
}
