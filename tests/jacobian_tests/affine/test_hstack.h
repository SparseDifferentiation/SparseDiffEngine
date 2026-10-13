#include <math.h>
#include <stdio.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "test_helpers.h"

const char *test_jacobian_hstack_vectors(void)
{
    /* Test Jacobian of hstack(log(x), exp(x), sin(x)) where x is 3x1
     * x = [1, 2, 3] at global indices [0, 1, 2]
     * Output is 3x3 matrix (9 elements) stored column-wise:
     * [log(1), log(2), log(3), exp(1), exp(2), exp(3), sin(1), sin(2), sin(3)]
     *
     * Jacobian is 9x3 sparse matrix:
     * d(log(x))/dx: diagonal with [1/1, 1/2, 1/3]
     * d(exp(x))/dx: diagonal with [exp(1), exp(2), exp(3)]
     * d(sin(x))/dx: diagonal with [cos(1), cos(2), cos(3)]
     */
    double u[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(3, 1, 0, 3);

    expr *log_x = new_log(x);
    expr *exp_x = new_exp(x);
    expr *sin_x = new_sin(x);

    expr *args[3] = {log_x, exp_x, sin_x};
    expr *stack = new_hstack(args, 3, 3);

    stack->forward(stack, u);
    jacobian_init(stack);
    eval_jacobian(stack);

    /* Expected jacobian: 9x3 with 9 nonzeros (diagonal blocks) */
    double expected_Ax[9] = {1.0,      0.5,      1.0 / 3.0, /* d(log)/dx */
                             exp(1.0), exp(2.0), exp(3.0),  /* d(exp)/dx */
                             cos(1.0), cos(2.0), cos(3.0)}; /* d(sin)/dx */

    int expected_Ai[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int expected_Ap[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    mu_assert("vals fail", cmp_values(stack->jacobian, expected_Ax, 9));
    mu_assert("sparsity fail",
              cmp_sparsity(stack->jacobian, expected_Ap, expected_Ai, 9, 9));

    free_expr(stack);
    return 0;
}

const char *test_jacobian_hstack_matrix(void)
{
    /* Test Jacobian of hstack(log(x), exp(x), sin(x)) where x is 3x2
     * x stored column-wise: [1, 3, 5, 2, 4, 6]
     * Output is 3x6 matrix (18 elements):
     * [log(1), log(3), log(5), log(2), log(4), log(6),
     *  exp(1), exp(3), exp(5), exp(2), exp(4), exp(6),
     *  sin(1), sin(3), sin(5), sin(2), sin(4), sin(6)]
     *
     * Jacobian is 18x6 sparse matrix (diagonal):
     * Each child contributes a 6x6 diagonal block
     */
    double u[6] = {1.0, 3.0, 5.0, 2.0, 4.0, 6.0};
    expr *x = new_variable(3, 2, 0, 6);

    expr *log_x = new_log(x);
    expr *exp_x = new_exp(x);
    expr *sin_x = new_sin(x);

    expr *args[3] = {log_x, exp_x, sin_x};
    expr *stack = new_hstack(args, 3, 6);

    stack->forward(stack, u);
    jacobian_init(stack);
    eval_jacobian(stack);

    /* Expected jacobian: 18x6 with 18 nonzeros (diagonal) */
    double expected_Ax[18] = {
        1.0,      1.0 / 3.0, 1.0 / 5.0, 0.5,      0.25,     1.0 / 6.0, /* log */
        exp(1.0), exp(3.0),  exp(5.0),  exp(2.0), exp(4.0), exp(6.0),  /* exp */
        cos(1.0), cos(3.0),  cos(5.0),  cos(2.0), cos(4.0), cos(6.0)}; /* sin */

    int expected_Ai[18] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
    int expected_Ap[19] = {0,  1,  2,  3,  4,  5,  6,  7,  8, 9,
                           10, 11, 12, 13, 14, 15, 16, 17, 18};

    mu_assert("vals fail", cmp_values(stack->jacobian, expected_Ax, 18));
    mu_assert("sparsity fail",
              cmp_sparsity(stack->jacobian, expected_Ap, expected_Ai, 18, 18));

    free_expr(stack);
    return 0;
}

/* hstack(A @ exp(X), sin(X)) for a 2x2 variable X: the left_matmul child's
   Jacobian is kron(I_2, A) diag(exp(x)), a stacked_pd, so the stack reads it
   blockwise rather than through a CSR view. Checked at two points so the
   second evaluation goes through the refill. */
const char *test_jacobian_hstack_stacked_pd_child(void)
{
    double A[4] = {1.0, 2.0, -0.5, 3.0};
    double u1[4] = {0.1, 0.2, 0.3, 0.4};
    double u2[4] = {-0.7, 0.5, 1.1, -0.2};
    expr *X = new_variable(2, 2, 0, 4);
    expr *f = new_left_matmul_dense(NULL, new_exp(X), 2, 2, A);
    expr *g = new_sin(X);
    expr *args[2] = {f, g};
    expr *stack = new_hstack(args, 2, 4);

    mu_assert("check_jacobian at u1 failed",
              check_jacobian_num(stack, u1, NUMERICAL_DIFF_DEFAULT_H));
    mu_assert("child jacobian is stacked_pd",
              f->jacobian->kind == MATRIX_STACKED_PD);
    mu_assert("check_jacobian at u2 failed",
              check_jacobian_num(stack, u2, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(stack);
    return 0;
}
