#include <stdio.h>

#include "atoms/affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_neg_jacobian(void)
{
    double u[3] = {1.0, 2.0, 3.0};
    expr *var = new_variable(3, 1, 0, 3);
    expr *neg_node = new_neg(var);
    neg_node->forward(neg_node, u);
    jacobian_init(neg_node);
    neg_node->eval_jacobian(neg_node);

    /* Jacobian of neg(x) is -I (diagonal with -1) */
    double expected_x[3] = {-1.0, -1.0, -1.0};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vals fail", cmp_values(neg_node->jacobian, expected_x, 3));
    mu_assert("sparsity fail",
              cmp_sparsity(neg_node->jacobian, expected_p, expected_i, 3, 3));

    free_expr(neg_node);
    return 0;
}

const char *test_neg_chain(void)
{
    /* Test neg(neg(x)) = x */
    double u[3] = {1.0, 2.0, 3.0};
    expr *var = new_variable(3, 1, 0, 3);
    expr *neg1 = new_neg(var);
    expr *neg2 = new_neg(neg1);
    neg2->forward(neg2, u);

    /* neg(neg(x)) should equal x */
    mu_assert("neg chain forward failed", cmp_double_array(neg2->value, u, 3));

    jacobian_init(neg2);
    neg2->eval_jacobian(neg2);

    /* Jacobian of neg(neg(x)) is (-1)*(-1)*I = I */
    double expected_x[3] = {1.0, 1.0, 1.0};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vals fail", cmp_values(neg2->jacobian, expected_x, 3));
    mu_assert("sparsity fail",
              cmp_sparsity(neg2->jacobian, expected_p, expected_i, 3, 3));

    free_expr(neg2);
    return 0;
}
