#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>

const char *test_wsum_hess_vstack_vectors(void)
{
    /* vstack([log(x), exp(x)]) where x is (3,1)
     * Output (6,1): [log(x0), log(x1), log(x2),
     *                exp(x0), exp(x1), exp(x2)]
     * w = [1, 2, 3, 4, 5, 6]
     *
     * Hessian (3x3 diagonal):
     *   (0,0): w[0]*(-1/x0^2) + w[3]*exp(x0) = -1 + 4*e
     *   (1,1): w[1]*(-1/x1^2) + w[4]*exp(x1) = -2/4 + 5*e^2
     *   (2,2): w[2]*(-1/x2^2) + w[5]*exp(x2) = -3/9 + 6*e^3
     */
    double u[3] = {1.0, 2.0, 3.0};
    double w[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *log_x = new_log(x);
    expr *exp_x = new_exp(x);

    expr *args[2] = {log_x, exp_x};
    expr *stack = new_vstack(args, 2, 3);

    stack->forward(stack, u);
    stack->jacobian_init(stack);
    stack->wsum_hess_init(stack);
    stack->eval_wsum_hess(stack, w);

    double expected_x[3] = {-1.0 + 4.0 * exp(1.0), -0.5 + 5.0 * exp(2.0),
                            -1.0 / 3.0 + 6.0 * exp(3.0)};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vstack hess vectors: vals",
              cmp_double_array(stack->wsum_hess->x, expected_x, 3));
    mu_assert("vstack hess vectors: rows",
              cmp_int_array(stack->wsum_hess->p, expected_p, 4));
    mu_assert("vstack hess vectors: cols",
              cmp_int_array(stack->wsum_hess->i, expected_i, 3));

    free_expr(stack);
    return 0;
}

const char *test_wsum_hess_vstack_matrix(void)
{
    /* vstack([log(x), exp(y)]) where x is (2,3), y is (1,3)
     * x at var_id 0 (6 vars), y at var_id 6 (3 vars)
     *
     * Output (3,3) flat column-wise:
     *   [log(x0), log(x1), exp(y0), log(x2), log(x3), exp(y1),
     *    log(x4), log(x5), exp(y2)]
     * w = [1, 2, 3, 4, 5, 6, 7, 8, 9]
     *
     * Hessian (9x9 diagonal):
     *   x vars (0-5): w[k]*(-1/xk^2) where k maps output→var
     *     (0,0): w[0]*(-1/1^2) = -1
     *     (1,1): w[1]*(-1/2^2) = -0.5
     *     (2,2): w[3]*(-1/3^2) = -4/9
     *     (3,3): w[4]*(-1/4^2) = -5/16
     *     (4,4): w[6]*(-1/5^2) = -7/25
     *     (5,5): w[7]*(-1/6^2) = -8/36
     *   y vars (6-8): w[k]*exp(yk) where k maps output→var
     *     (6,6): w[2]*exp(7) = 3*exp(7)
     *     (7,7): w[5]*exp(8) = 6*exp(8)
     *     (8,8): w[8]*exp(9) = 9*exp(9)
     */
    double u[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double w[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    expr *x = new_variable(2, 3, 0, 9);
    expr *y = new_variable(1, 3, 6, 9);

    expr *log_x = new_log(x);
    expr *exp_y = new_exp(y);

    expr *args[2] = {log_x, exp_y};
    expr *stack = new_vstack(args, 2, 9);

    stack->forward(stack, u);
    stack->jacobian_init(stack);
    stack->wsum_hess_init(stack);
    stack->eval_wsum_hess(stack, w);

    double expected_x[9] = {-1.0,            /* x0: w[0]*(-1/1) */
                            -0.5,            /* x1: w[1]*(-1/4) */
                            -4.0 / 9.0,      /* x2: w[3]*(-1/9) */
                            -5.0 / 16.0,     /* x3: w[4]*(-1/16) */
                            -7.0 / 25.0,     /* x4: w[6]*(-1/25) */
                            -8.0 / 36.0,     /* x5: w[7]*(-1/36) */
                            3.0 * exp(7.0),  /* y0: w[2]*exp(7) */
                            6.0 * exp(8.0),  /* y1: w[5]*exp(8) */
                            9.0 * exp(9.0)}; /* y2: w[8]*exp(9) */
    int expected_p[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int expected_i[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    mu_assert("vstack hess matrix: vals",
              cmp_double_array(stack->wsum_hess->x, expected_x, 9));
    mu_assert("vstack hess matrix: rows",
              cmp_int_array(stack->wsum_hess->p, expected_p, 10));
    mu_assert("vstack hess matrix: cols",
              cmp_int_array(stack->wsum_hess->i, expected_i, 9));

    free_expr(stack);
    return 0;
}
