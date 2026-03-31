#include <math.h>
#include <stdio.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_jacobian_vstack_vectors(void)
{
    /* vstack([log(x), exp(x)]) where x is (3,1)
     * Output (6,1) flat: [log(1), log(2), log(3), exp(1), exp(2), exp(3)]
     *
     * Jacobian is 6x3:
     *   row 0: d(log(x0))/dx0 = 1/1 at col 0
     *   row 1: d(log(x1))/dx1 = 1/2 at col 1
     *   row 2: d(log(x2))/dx2 = 1/3 at col 2
     *   row 3: d(exp(x0))/dx0 = e^1 at col 0
     *   row 4: d(exp(x1))/dx1 = e^2 at col 1
     *   row 5: d(exp(x2))/dx2 = e^3 at col 2
     */
    double u[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(3, 1, 0, 3);

    expr *log_x = new_log(x);
    expr *exp_x = new_exp(x);

    expr *args[2] = {log_x, exp_x};
    expr *stack = new_vstack(args, 2, 3);

    stack->forward(stack, u);
    jacobian_init(stack);
    stack->eval_jacobian(stack);

    double expected_x[6] = {1.0, 0.5, 1.0 / 3.0, exp(1.0), exp(2.0), exp(3.0)};
    int expected_i[6] = {0, 1, 2, 0, 1, 2};
    int expected_p[7] = {0, 1, 2, 3, 4, 5, 6};

    mu_assert("vstack jac vectors: vals",
              cmp_double_array(stack->jacobian->x, expected_x, 6));
    mu_assert("vstack jac vectors: cols",
              cmp_int_array(stack->jacobian->i, expected_i, 6));
    mu_assert("vstack jac vectors: rows",
              cmp_int_array(stack->jacobian->p, expected_p, 7));

    free_expr(stack);
    return 0;
}

const char *test_jacobian_vstack_matrix(void)
{
    /* vstack([log(x), exp(y)]) where x is (2,3), y is (1,3)
     * x at var_id 0 (6 vars), y at var_id 6 (3 vars), total 9 vars
     *
     * Output (3,3) flat column-wise:
     *   [log(x0), log(x1), exp(y0), log(x2), log(x3), exp(y1),
     *    log(x4), log(x5), exp(y2)]
     *
     * Jacobian is 9x9 sparse:
     *   row 0 (log(x0)): 1/x0 at col 0
     *   row 1 (log(x1)): 1/x1 at col 1
     *   row 2 (exp(y0)): e^y0 at col 6
     *   row 3 (log(x2)): 1/x2 at col 2
     *   row 4 (log(x3)): 1/x3 at col 3
     *   row 5 (exp(y1)): e^y1 at col 7
     *   row 6 (log(x4)): 1/x4 at col 4
     *   row 7 (log(x5)): 1/x5 at col 5
     *   row 8 (exp(y2)): e^y2 at col 8
     */
    double u[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    expr *x = new_variable(2, 3, 0, 9);
    expr *y = new_variable(1, 3, 6, 9);

    expr *log_x = new_log(x);
    expr *exp_y = new_exp(y);

    expr *args[2] = {log_x, exp_y};
    expr *stack = new_vstack(args, 2, 9);

    stack->forward(stack, u);
    jacobian_init(stack);
    stack->eval_jacobian(stack);

    double expected_x[9] = {1.0,      0.5, exp(7.0),  1.0 / 3.0, 0.25,
                            exp(8.0), 0.2, 1.0 / 6.0, exp(9.0)};
    int expected_i[9] = {0, 1, 6, 2, 3, 7, 4, 5, 8};
    int expected_p[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    mu_assert("vstack jac matrix: vals",
              cmp_double_array(stack->jacobian->x, expected_x, 9));
    mu_assert("vstack jac matrix: cols",
              cmp_int_array(stack->jacobian->i, expected_i, 9));
    mu_assert("vstack jac matrix: rows",
              cmp_int_array(stack->jacobian->p, expected_p, 10));

    free_expr(stack);
    return 0;
}
