#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "test_helpers.h"

const char *test_wsum_hess_kron_left(void)
{
    /* wsum_hess of kron(C, log(x)) where
     *   x is 2x1 variable at x = [1, 2]
     *   C is 2x2 sparse: [[1, 2], [0, 3]]
     * Weights w = [1, 2, 3, 4, 5, 6, 7, 8].
     *
     * w_child[s] = sum over active (i,j) of C[i,j] * w[(j*q+l)*mp + i*p + k]
     * (p=2, q=1, mp=4):
     *   w_child[0] = 1*w[0] + 2*w[4] + 3*w[6] = 1 + 10 + 21 = 32
     *   w_child[1] = 1*w[1] + 2*w[5] + 3*w[7] = 2 + 12 + 24 = 38
     * Hessian of log: H[k,k] = -1/x[k]^2 -> [-32, -9.5]. */
    double x_vals[2] = {1.0, 2.0};
    double w[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    expr *x = new_variable(2, 1, 0, 2);

    CSR_Matrix *C = new_csr_matrix(2, 2, 3);
    int C_p[3] = {0, 2, 3};
    int C_i[3] = {0, 1, 1};
    double C_x[3] = {1.0, 2.0, 3.0};
    memcpy(C->p, C_p, 3 * sizeof(int));
    memcpy(C->i, C_i, 3 * sizeof(int));
    memcpy(C->x, C_x, 3 * sizeof(double));

    expr *log_x = new_log(x);
    expr *Z = new_kron_left(NULL, log_x, C, 2, 1);

    Z->forward(Z, x_vals);
    jacobian_init(Z);
    wsum_hess_init(Z);
    Z->eval_wsum_hess(Z, w);

    double expected_x[2] = {-32.0, -9.5};
    int expected_i[2] = {0, 1};
    int expected_p[3] = {0, 1, 2};

    mu_assert("kron_left hess vals fail",
              cmp_double_array(Z->wsum_hess->x, expected_x, 2));
    mu_assert("kron_left hess cols fail",
              cmp_int_array(Z->wsum_hess->i, expected_i, 2));
    mu_assert("kron_left hess rows fail",
              cmp_int_array(Z->wsum_hess->p, expected_p, 3));

    free_csr_matrix(C);
    free_expr(Z);
    return 0;
}

const char *test_wsum_hess_kron_left_composite(void)
{
    /* Verify weight propagation through kron_left when the child has a
     * non-trivial Hessian, by numerical differentiation against
     * kron(C, exp(x)). exp is a full-domain elementwise atom so it
     * composes on top of a variable child correctly here. */
    double x_vals[2] = {0.3, -0.7};
    double w[8] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5};

    expr *x = new_variable(2, 1, 0, 2);

    CSR_Matrix *C = new_csr_matrix(2, 2, 3);
    int C_p[3] = {0, 2, 3};
    int C_i[3] = {0, 1, 1};
    double C_x[3] = {1.0, 2.0, 3.0};
    memcpy(C->p, C_p, 3 * sizeof(int));
    memcpy(C->i, C_i, 3 * sizeof(int));
    memcpy(C->x, C_x, 3 * sizeof(double));

    expr *exp_x = new_exp(x);
    expr *Z = new_kron_left(NULL, exp_x, C, 2, 1);

    mu_assert("kron_left composite wsum_hess check failed",
              check_wsum_hess(Z, x_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_csr_matrix(C);
    free_expr(Z);
    return 0;
}
