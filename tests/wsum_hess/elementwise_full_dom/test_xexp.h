#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_wsum_hess_xexp(void)
{
    double u_vals[3] = {1.0, 2.0, 3.0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *xexp_node = new_xexp(x);
    xexp_node->forward(xexp_node, u_vals);
    jacobian_init(xexp_node);
    wsum_hess_init(xexp_node);
    xexp_node->eval_wsum_hess(xexp_node, w);

    /* Expected values on the diagonal: w_i * (2+x_i) * exp(x_i) */
    double expected_x[3] = {1.0 * 3.0 * exp(1.0), 2.0 * 4.0 * exp(2.0),
                            3.0 * 5.0 * exp(3.0)};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vals fail", cmp_values(xexp_node->wsum_hess, expected_x, 3));
    mu_assert("sparsity fail",
              cmp_sparsity(xexp_node->wsum_hess, expected_p, expected_i, 3, 3));

    free_expr(xexp_node);

    return 0;
}
