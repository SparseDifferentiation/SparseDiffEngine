#include <math.h>
#include <stdio.h>

#include "atoms/affine.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "subexpr.h"
#include "test_helpers.h"

const char *test_jacobian_kron_const_left(void)
{
    /* Z = kron([[1,2],[3,4]], B), B a 2x2 leaf variable. Each output row has a
     * single nonzero: column = the B entry it gathers, value = the A entry. */
    double A[4] = {1.0, 3.0, 2.0, 4.0};
    expr *A_param = new_parameter(2, 2, PARAM_FIXED, 4, A);
    expr *B = new_variable(2, 2, 0, 4);
    expr *Z = new_kron(A_param, B, 1, 2, 2, 2, 2);

    double u[4] = {5.0, 7.0, 6.0, 8.0};
    Z->forward(Z, u);
    jacobian_init(Z);
    Z->eval_jacobian(Z);

    mu_assert("kron J rows", Z->jacobian->m == 16);
    mu_assert("kron J cols", Z->jacobian->n == 4);
    mu_assert("kron J nnz", Z->jacobian->nnz == 16);

    int expected_p[17] = {0, 1, 2, 3, 4, 5, 6, 7, 8,
                          9, 10, 11, 12, 13, 14, 15, 16};
    int expected_i[16] = {0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3};
    double expected_x[16] = {1, 1, 3, 3, 1, 1, 3, 3, 2, 2, 4, 4, 2, 2, 4, 4};

    mu_assert("kron const-left J sparsity",
              cmp_sparsity(Z->jacobian, expected_p, expected_i, 16, 16));
    mu_assert("kron const-left J values",
              cmp_values(Z->jacobian, expected_x, 16));

    free_expr(Z);
    return 0;
}

const char *test_jacobian_kron_const_right(void)
{
    /* Z = kron(A, [[1,2],[3,4]]), A a 2x2 leaf variable (const_is_left = 0). */
    double B[4] = {1.0, 3.0, 2.0, 4.0};
    expr *B_param = new_parameter(2, 2, PARAM_FIXED, 4, B);
    expr *A = new_variable(2, 2, 0, 4);
    expr *Z = new_kron(B_param, A, 0, 2, 2, 2, 2);

    double u[4] = {5.0, 7.0, 6.0, 8.0};
    Z->forward(Z, u);
    jacobian_init(Z);
    Z->eval_jacobian(Z);

    mu_assert("kron J rows", Z->jacobian->m == 16);
    mu_assert("kron J nnz", Z->jacobian->nnz == 16);

    int expected_p[17] = {0, 1, 2, 3, 4, 5, 6, 7, 8,
                          9, 10, 11, 12, 13, 14, 15, 16};
    int expected_i[16] = {0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3};
    double expected_x[16] = {1, 3, 1, 3, 2, 4, 2, 4, 1, 3, 1, 3, 2, 4, 2, 4};

    mu_assert("kron const-right J sparsity",
              cmp_sparsity(Z->jacobian, expected_p, expected_i, 16, 16));
    mu_assert("kron const-right J values",
              cmp_values(Z->jacobian, expected_x, 16));

    free_expr(Z);
    return 0;
}

const char *test_jacobian_kron_composite(void)
{
    /* Z = kron([[1,2],[3,4]], log(X)) — composite variable operand; check the
     * gathered/scaled Jacobian against finite differences. */
    double A[4] = {1.0, 3.0, 2.0, 4.0};
    double x_vals[4] = {1.0, 2.0, 3.0, 4.0};

    expr *A_param = new_parameter(2, 2, PARAM_FIXED, 4, A);
    expr *X = new_variable(2, 2, 0, 4);
    expr *log_X = new_log(X);
    expr *Z = new_kron(A_param, log_X, 1, 2, 2, 2, 2);

    mu_assert("kron composite Jacobian check failed",
              check_jacobian_num(Z, x_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(Z);
    return 0;
}
