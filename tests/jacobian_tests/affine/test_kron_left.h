#include <math.h>
#include <stdio.h>
#include <string.h>

#include "atoms/affine.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_jacobian_kron_left_log(void)
{
    /* Jacobian of kron(C, log(x)) where
     *   x is 2x1 variable at x = [1, 2]
     *   C is 2x2 sparse: [[1, 2], [0, 3]]
     * Output kron(C, log(x)) is 4x2, vectorized to 8x1.
     *
     * Each active output row r (C[i,j] != 0) is C[i,j] * d log(x_s)/dx_s
     * at col s = l*p + k. Since d log(x_k)/dx_v = delta_{k,v}/x_k:
     *   r=0: [1, 0]; r=1: [0, 1/2];
     *   r=2: zero;   r=3: zero;
     *   r=4: [2, 0]; r=5: [0, 1];
     *   r=6: [3, 0]; r=7: [0, 1.5].
     */
    double x_vals[2] = {1.0, 2.0};
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
    Z->eval_jacobian(Z);

    double expected_x[6] = {1.0, 0.5, 2.0, 1.0, 3.0, 1.5};
    int expected_i[6] = {0, 1, 0, 1, 0, 1};
    int expected_p[9] = {0, 1, 2, 2, 2, 3, 4, 5, 6};

    mu_assert("kron_left jac vals fail",
              cmp_double_array(Z->jacobian->x, expected_x, 6));
    mu_assert("kron_left jac cols fail",
              cmp_int_array(Z->jacobian->i, expected_i, 6));
    mu_assert("kron_left jac rows fail",
              cmp_int_array(Z->jacobian->p, expected_p, 9));

    free_csr_matrix(C);
    free_expr(Z);
    return 0;
}

const char *test_jacobian_kron_left_log_matrix(void)
{
    /* Jacobian of kron(C, log(x)) where
     *   x is 2x2 variable, col-major [1,2,3,4]
     *   C is 2x1 sparse: [[1], [2]]
     * Output is 4x2, vectorized to 8x1. Every output row is active.
     * J[r,var] = C[i,j] / x[s] at col s, zero elsewhere.
     */
    double x_vals[4] = {1.0, 2.0, 3.0, 4.0};
    expr *x = new_variable(2, 2, 0, 4);

    CSR_Matrix *C = new_csr_matrix(2, 1, 2);
    int C_p[3] = {0, 1, 2};
    int C_i[2] = {0, 0};
    double C_x[2] = {1.0, 2.0};
    memcpy(C->p, C_p, 3 * sizeof(int));
    memcpy(C->i, C_i, 2 * sizeof(int));
    memcpy(C->x, C_x, 2 * sizeof(double));

    expr *log_x = new_log(x);
    expr *Z = new_kron_left(NULL, log_x, C, 2, 2);

    Z->forward(Z, x_vals);
    jacobian_init(Z);
    Z->eval_jacobian(Z);

    double expected_x[8] = {1.0,       0.5,  2.0,       1.0,
                            1.0 / 3.0, 0.25, 2.0 / 3.0, 0.5};
    int expected_i[8] = {0, 1, 0, 1, 2, 3, 2, 3};
    int expected_p[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    mu_assert("kron_left matrix jac vals fail",
              cmp_double_array(Z->jacobian->x, expected_x, 8));
    mu_assert("kron_left matrix jac cols fail",
              cmp_int_array(Z->jacobian->i, expected_i, 8));
    mu_assert("kron_left matrix jac rows fail",
              cmp_int_array(Z->jacobian->p, expected_p, 9));

    free_csr_matrix(C);
    free_expr(Z);
    return 0;
}
