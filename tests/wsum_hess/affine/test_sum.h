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

const char *test_wsum_hess_sum_exp_linear(void)
{
    double Ax[6] = {1, 1, 2, 3, 1, -1};
    int Ai[6] = {0, 1, 0, 1, 0, 1};
    int Ap[4] = {0, 2, 4, 6};
    CSR_Matrix *A = new_csr_matrix(3, 2, 6);
    memcpy(A->x, Ax, 6 * sizeof(double));
    memcpy(A->i, Ai, 6 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));
    double x_vals[2] = {2.0, 1.0};
    double w = 1.5;

    expr *x = new_variable(2, 1, 0, 2);
    expr *Ax_node = new_linear(x, A, NULL);
    expr *exp_node = new_exp(Ax_node);
    expr *sum_node = new_sum(exp_node, -1);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(sum_node, x_vals, &w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(sum_node);
    free_csr_matrix(A);

    return 0;
}

const char *test_wsum_hess_sum_log_axis0(void)
{
    /* Test: wsum_hess of sum(log(x), axis=0) where x is 3x2
     * x = [[1, 4],
     *      [2, 5],
     *      [3, 6]]
     */

    double x[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double w[2] = {-1.0, -2.0};
    expr *x_node = new_variable(3, 2, 0, 6);
    expr *log_node = new_log(x_node);
    expr *sum_node = new_sum(log_node, 0);

    sum_node->forward(sum_node, x);
    sum_node->jacobian_init(sum_node);
    sum_node->wsum_hess_init(sum_node);
    sum_node->eval_wsum_hess(sum_node, w);

    /* Expected diagonal values */
    double expected_x[6] = {-w[0] / (x[0] * x[0]), -w[0] / (x[1] * x[1]),
                            -w[0] / (x[2] * x[2]), -w[1] / (x[3] * x[3]),
                            -w[1] / (x[4] * x[4]), -w[1] / (x[5] * x[5])};
    int expected_p[7] = {0, 1, 2, 3, 4, 5, 6};
    int expected_i[6] = {0, 1, 2, 3, 4, 5};

    mu_assert("vals incorrect",
              cmp_double_array(sum_node->wsum_hess->x, expected_x, 6));
    mu_assert("rows incorrect",
              cmp_int_array(sum_node->wsum_hess->p, expected_p, 7));
    mu_assert("cols incorrect",
              cmp_int_array(sum_node->wsum_hess->i, expected_i, 6));

    free_expr(sum_node);

    return 0;
}

const char *test_wsum_hess_sum_log_axis1(void)
{
    /* Test: wsum_hess of sum(log(x), axis=1) where x is 3x2
     * x = [[1, 4],
     *      [2, 5],
     *      [3, 6]]
     */

    double x[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double w[3] = {-1.0, -2.0, -3.0};
    expr *x_node = new_variable(3, 2, 0, 6);
    expr *log_node = new_log(x_node);
    expr *sum_node = new_sum(log_node, 1);

    sum_node->forward(sum_node, x);
    sum_node->jacobian_init(sum_node);
    sum_node->wsum_hess_init(sum_node);
    sum_node->eval_wsum_hess(sum_node, w);

    /* Expected diagonal values */
    double expected_x[6] = {-w[0] / (x[0] * x[0]), -w[1] / (x[1] * x[1]),
                            -w[2] / (x[2] * x[2]), -w[0] / (x[3] * x[3]),
                            -w[1] / (x[4] * x[4]), -w[2] / (x[5] * x[5])};
    int expected_p[7] = {0, 1, 2, 3, 4, 5, 6};
    int expected_i[6] = {0, 1, 2, 3, 4, 5};

    mu_assert("vals incorrect",
              cmp_double_array(sum_node->wsum_hess->x, expected_x, 6));
    mu_assert("rows incorrect",
              cmp_int_array(sum_node->wsum_hess->p, expected_p, 7));
    mu_assert("cols incorrect",
              cmp_int_array(sum_node->wsum_hess->i, expected_i, 6));

    free_expr(sum_node);

    return 0;
}
