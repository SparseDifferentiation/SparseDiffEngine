#include <stdio.h>

#include "atoms/affine.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "subexpr.h"
#include "test_helpers.h"

const char *test_wsum_hess_kron(void)
{
    /* kron(A, B) is linear in a leaf variable B, so its weighted Hessian is
     * zero for any weights. */
    double A[4] = {1.0, 3.0, 2.0, 4.0};
    int active[4] = {0, 1, 2, 3};
    expr *A_param = new_parameter(2, 2, PARAM_FIXED, 4, A);
    expr *B = new_variable(2, 2, 0, 4);
    expr *Z = new_left_kron(A_param, B, 2, 2, 2, 2, active, 4);

    double u[4] = {5.0, 7.0, 6.0, 8.0};
    double w[16] = {1, -1, 2, 3, -2, 1, 0, 2, -1, 1, 3, -3, 2, 1, -2, 1};

    Z->forward(Z, u);
    jacobian_init(Z);
    wsum_hess_init(Z);
    Z->eval_wsum_hess(Z, w);

    mu_assert("kron wsum_hess square", Z->wsum_hess->m == 4 && Z->wsum_hess->n == 4);
    mu_assert("kron wsum_hess zero for linear arg", Z->wsum_hess->nnz == 0);

    free_expr(Z);
    return 0;
}

const char *test_wsum_hess_kron_composite(void)
{
    /* Z = kron([[1,2],[3,4]], log(X)) — nonlinear in X; backprop of weights
     * through the linear gather must match a numerical second-derivative. */
    double A[4] = {1.0, 3.0, 2.0, 4.0};
    int active[4] = {0, 1, 2, 3};
    double x_vals[4] = {1.0, 2.0, 3.0, 4.0};
    double w[16] = {1, -1, 2, 3, -2, 1, 0, 2, -1, 1, 3, -3, 2, 1, -2, 1};

    expr *A_param = new_parameter(2, 2, PARAM_FIXED, 4, A);
    expr *X = new_variable(2, 2, 0, 4);
    expr *log_X = new_log(X);
    expr *Z = new_left_kron(A_param, log_X, 2, 2, 2, 2, active, 4);

    mu_assert("kron composite wsum_hess check failed",
              check_wsum_hess(Z, x_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    return 0;
}
