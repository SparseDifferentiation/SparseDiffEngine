#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"
#include "elementwise_full_dom.h"
#include "elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "test_helpers.h"

const char *test_wsum_hess_log(void)
{
    /* Test: wsum_hess of log(x) where x = [1, 2, 3] (3x1) at global variable index 2
     * Total 7 variables, weight w = [1, 2, 3]
     *
     * For log(x), the Hessian of log(x_i) w.r.t x_i is -1/x_i^2
     * Weighted sum of Hessian diagonal: w_i * (-1/x_i^2)
     * For w = [1, 2, 3], x = [1, 2, 3]:
     *   out[0] = 1 * (-1/1^2) = -1
     *   out[1] = 2 * (-1/2^2) = -0.5
     *   out[2] = 3 * (-1/3^2) = -1/3
     */

    double u_vals[7] = {0, 0, 1.0, 2.0, 3.0, 0, 0};
    double w[3] = {1.0, 2.0, 3.0};

    expr *x = new_variable(3, 1, 2, 7);
    expr *log_node = new_log(x);
    log_node->forward(log_node, u_vals);
    jacobian_init(log_node);
    wsum_hess_init(log_node);
    log_node->eval_wsum_hess(log_node, w);

    /* Expected values on the diagonal: -w_i/x_i^2 */
    double expected_x[3] = {-1.0, -0.5, -1.0 / 3.0};
    int expected_p[8] = {0, 0, 0, 1, 2, 3, 3, 3};
    int expected_i[3] = {2, 3, 4};

    mu_assert("vals incorrect",
              cmp_double_array(log_node->wsum_hess->x, expected_x, 3));
    mu_assert("rows incorrect",
              cmp_int_array(log_node->wsum_hess->p, expected_p, 8));
    mu_assert("cols incorrect",
              cmp_int_array(log_node->wsum_hess->i, expected_i, 3));

    free_expr(log_node);

    return 0;
}

const char *test_wsum_hess_exp_composite(void)
{
    double u_vals[5] = {1, 2, 3, 4, 5};
    double w[3] = {-1, -2, -3};
    double Ax[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int Ai[] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    int Ap[] = {0, 5, 10, 15};
    CSR_Matrix *A_csr = new_csr_matrix(3, 5, 15);
    memcpy(A_csr->x, Ax, 15 * sizeof(double));
    memcpy(A_csr->i, Ai, 15 * sizeof(int));
    memcpy(A_csr->p, Ap, 4 * sizeof(int));

    expr *x = new_variable(5, 1, 0, 5);
    expr *Ax_node = new_linear(x, A_csr, NULL);
    expr *exp_node = new_exp(Ax_node);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(exp_node, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_csr_matrix(A_csr);
    free_expr(exp_node);

    return 0;
}
