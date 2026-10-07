#include <math.h>
#include <stdio.h>

#include "atoms/affine.h"
#include "atoms/bivariate_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "test_helpers.h"

/* y at var_id 1, x at var_id 6 (left has the lower index) */
const char *test_wsum_hess_atan2_1(void)
{
    double u[10] = {0.0, 1.0, -2.0, 0.5, 0.0, 0.0, 2.0, 1.0, -1.5, 0.0};
    double w[3] = {1.0, -2.0, 3.0};

    expr *y = new_variable(3, 1, 1, 10);
    expr *x = new_variable(3, 1, 6, 10);
    expr *node = new_atan2(y, x);
    mu_assert("new_atan2 failed", node != NULL);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(node, u, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    return 0;
}

/* y at var_id 6, x at var_id 1 (left has the higher index) */
const char *test_wsum_hess_atan2_2(void)
{
    double u[10] = {0.0, 2.0, 1.0, -1.5, 0.0, 0.0, 1.0, -2.0, 0.5, 0.0};
    double w[3] = {1.0, -2.0, 3.0};

    expr *y = new_variable(3, 1, 6, 10);
    expr *x = new_variable(3, 1, 1, 10);
    expr *node = new_atan2(y, x);
    mu_assert("new_atan2 failed", node != NULL);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(node, u, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(node);
    return 0;
}
