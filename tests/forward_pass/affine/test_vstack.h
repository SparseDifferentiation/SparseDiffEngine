#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_vstack_forward_vectors(void)
{
    /* vstack([log(x), exp(x)]) where x is (3,1)
     * Output is (6,1): [log(1), log(2), log(3), exp(1), exp(2), exp(3)]
     * For d2=1, vstack == hstack (no interleaving needed)
     */
    double u[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(3, 1, 0, 3);

    expr *log_x = new_log(x);
    expr *exp_x = new_exp(x);

    expr *args[2] = {log_x, exp_x};
    expr *stack = new_vstack(args, 2, 3);

    mu_assert("vstack vectors: wrong d1", stack->d1 == 6);
    mu_assert("vstack vectors: wrong d2", stack->d2 == 1);

    stack->forward(stack, u);

    double expected[6] = {log(1.0), log(2.0), log(3.0),
                          exp(1.0), exp(2.0), exp(3.0)};

    mu_assert("vstack forward vectors failed",
              cmp_double_array(stack->value, expected, 6));

    free_expr(stack);
    return 0;
}

const char *test_vstack_forward_matrix(void)
{
    /* vstack([log(x), exp(y)]) where x is (2,3), y is (1,3)
     * x stored column-wise: [1, 2, 3, 4, 5, 6]
     *   x = [1 3 5]
     *       [2 4 6]
     * y stored column-wise: [7, 8, 9]
     *   y = [7 8 9]
     *
     * Output is (3,3), stored column-wise:
     *   result = [log(1) log(3) log(5)]
     *            [log(2) log(4) log(6)]
     *            [exp(7) exp(8) exp(9)]
     *
     * Flat column-wise:
     *   [log(1), log(2), exp(7), log(3), log(4), exp(8),
     *    log(5), log(6), exp(9)]
     */
    double u[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    expr *x = new_variable(2, 3, 0, 9);
    expr *y = new_variable(1, 3, 6, 9);

    expr *log_x = new_log(x);
    expr *exp_y = new_exp(y);

    expr *args[2] = {log_x, exp_y};
    expr *stack = new_vstack(args, 2, 9);

    mu_assert("vstack matrix: wrong d1", stack->d1 == 3);
    mu_assert("vstack matrix: wrong d2", stack->d2 == 3);

    stack->forward(stack, u);

    double expected[9] = {log(1.0), log(2.0), exp(7.0), log(3.0), log(4.0),
                          exp(8.0), log(5.0), log(6.0), exp(9.0)};

    mu_assert("vstack forward matrix failed",
              cmp_double_array(stack->value, expected, 9));

    free_expr(stack);
    return 0;
}
