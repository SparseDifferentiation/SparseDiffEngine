#include <stdio.h>

#include "atoms/affine.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "subexpr.h"
#include "test_helpers.h"

const char *test_wsum_hess_convolve(void)
{
    /* y = conv([1, 2, 3], x) is linear in x when the kernel is constant.
     * Child x is a leaf variable (zero Hessian), so y's weighted Hessian
     * should be zero for any weight vector w. */
    double kernel[3] = {1.0, 2.0, 3.0};
    expr *kernel_param = new_parameter(3, 1, PARAM_FIXED, 3, kernel);
    expr *x = new_variable(3, 1, 0, 3);
    expr *y = new_convolve(kernel_param, x);

    double u[3] = {1.0, 2.0, 3.0};
    double w[5] = {1.0, -1.0, 2.0, 3.0, -2.0};

    y->forward(y, u);
    jacobian_init(y);
    wsum_hess_init(y);
    y->eval_wsum_hess(y, w);

    mu_assert("Convolve wsum_hess should be 3x3", y->wsum_hess->m == 3);
    mu_assert("Convolve wsum_hess should be square", y->wsum_hess->n == 3);
    mu_assert("Convolve wsum_hess should have zero nonzeros",
              y->wsum_hess->nnz == 0);

    free_expr(y);
    return 0;
}

const char *test_wsum_hess_convolve_composite(void)
{
    /* y = conv([1, 2, 3], log(x)) — nonlinear in x, so backprop of weights
     * through the Toeplitz should produce the same Hessian as a numerical
     * second-derivative check. */
    double kernel[3] = {1.0, 2.0, 3.0};
    double x_vals[3] = {1.0, 2.0, 3.0};
    double w[5] = {1.0, -1.0, 2.0, 3.0, -2.0};

    expr *kernel_param = new_parameter(3, 1, PARAM_FIXED, 3, kernel);
    expr *x = new_variable(3, 1, 0, 3);
    expr *log_x = new_log(x);
    expr *y = new_convolve(kernel_param, log_x);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(y, x_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(y);
    return 0;
}
