#include <math.h>
#include <stdio.h>

#include "atoms/affine.h"
#include "atoms/bivariate_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "test_helpers.h"

/* y at var_id 2, x at var_id 7 (left has the lower index) */
const char *test_jacobian_atan2_1(void)
{
    double u[10] = {0.0, 0.0, 1.0, -2.0, 0.5, 0.0, 0.0, 2.0, 1.0, -1.5};

    expr *y = new_variable(3, 1, 2, 10);
    expr *x = new_variable(3, 1, 7, 10);
    expr *node = new_atan2(y, x);
    mu_assert("new_atan2 failed", node != NULL);

    node->forward(node, u);
    double expected[3] = {atan2(1.0, 2.0), atan2(-2.0, 1.0), atan2(0.5, -1.5)};
    mu_assert("forward fail", cmp_double_array(node->value, expected, 3));

    mu_assert("check_jacobian failed",
              check_jacobian_num(node, u, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    return 0;
}

/* y at var_id 7, x at var_id 2 (left has the higher index) */
const char *test_jacobian_atan2_2(void)
{
    double u[10] = {0.0, 0.0, 2.0, 1.0, -1.5, 0.0, 0.0, 1.0, -2.0, 0.5};

    expr *y = new_variable(3, 1, 7, 10);
    expr *x = new_variable(3, 1, 2, 10);
    expr *node = new_atan2(y, x);
    mu_assert("new_atan2 failed", node != NULL);

    node->forward(node, u);
    double expected[3] = {atan2(1.0, 2.0), atan2(-2.0, 1.0), atan2(0.5, -1.5)};
    mu_assert("forward fail", cmp_double_array(node->value, expected, 3));

    mu_assert("check_jacobian failed",
              check_jacobian_num(node, u, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    return 0;
}

/* leaf-only contract: non-variable, same variable and shape mismatch all
   return NULL and leave the children owned by the caller */
const char *test_atan2_rejects_bad_args(void)
{
    CSR_matrix *A = new_csr_random(3, 4, 1.0);
    expr *x = new_variable(4, 1, 0, 8);
    expr *x3 = new_variable(3, 1, 4, 8);
    expr *Ax = new_left_matmul(NULL, x, A);

    mu_assert("affine child should be NULL", new_atan2(Ax, x3) == NULL);
    mu_assert("same variable should be NULL", new_atan2(x, x) == NULL);
    mu_assert("shape mismatch should be NULL", new_atan2(x, x3) == NULL);

    free_expr(Ax);
    free_expr(x3);
    free_CSR_matrix(A);
    return 0;
}
