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

const char *test_wsum_hess_entr(void)
{
    double u_vals[3] = {1.0, 2.0, 3.0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *entr_node = new_entr(x);
    entr_node->forward(entr_node, u_vals);
    jacobian_init(entr_node);
    wsum_hess_init(entr_node);
    entr_node->eval_wsum_hess(entr_node, w);

    /* Expected values on the diagonal: -w_i/x_i */
    double expected_x[3] = {-1.0, -1.0, -1.0};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vals incorrect",
              cmp_double_array(entr_node->wsum_hess->x, expected_x, 3));
    mu_assert("rows incorrect",
              cmp_int_array(entr_node->wsum_hess->p, expected_p, 4));
    mu_assert("cols incorrect",
              cmp_int_array(entr_node->wsum_hess->i, expected_i, 3));

    free_expr(entr_node);

    return 0;
}
