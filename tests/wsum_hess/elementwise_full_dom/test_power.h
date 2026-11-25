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

const char *test_wsum_hess_power(void)
{
    double u_vals[3] = {1.0, 2.0, 3.0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *power_node = new_power(x, 3.0);
    power_node->forward(power_node, u_vals);
    jacobian_init(power_node);
    wsum_hess_init(power_node);
    eval_wsum_hess(power_node, w);

    /* Expected values on the diagonal: w_i * 6 * x_i */
    double expected_x[3] = {6.0 * 1.0, 6.0 * 4.0, 6.0 * 9.0};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("vals fail", cmp_values(power_node->wsum_hess, expected_x, 3));
    mu_assert("sparsity fail",
              cmp_sparsity(power_node->wsum_hess, expected_p, expected_i, 3, 3));

    free_expr(power_node);

    return 0;
}

/* p = 2 and p = -1 use every special case in power.c, p = 0.5 uses pow */
const char *test_power_special_exponents(void)
{
    double u_vals[3] = {0.5, 1.5, 3.0};
    double w[3] = {1.0, -2.0, 3.0};
    double exponents[3] = {2.0, -1.0, 0.5};

    for (int k = 0; k < 3; k++)
    {
        double p = exponents[k];
        double f[3], df[3], d2f[3];
        for (int i = 0; i < 3; i++)
        {
            f[i] = pow(u_vals[i], p);
            df[i] = p * pow(u_vals[i], p - 1);
            d2f[i] = w[i] * p * (p - 1) * pow(u_vals[i], p - 2);
        }

        expr *node = new_power(new_variable(3, 1, 0, 3), p);
        node->forward(node, u_vals);
        jacobian_init(node);
        eval_jacobian(node);
        wsum_hess_init(node);
        eval_wsum_hess(node, w);

        mu_assert("forward fail", cmp_double_array(node->value, f, 3));
        mu_assert("jacobian fail", cmp_values(node->jacobian, df, 3));
        mu_assert("hessian fail", cmp_values(node->wsum_hess, d2f, 3));

        free_expr(node);
    }

    return 0;
}
