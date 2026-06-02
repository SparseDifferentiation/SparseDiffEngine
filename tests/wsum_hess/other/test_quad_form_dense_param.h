#include "atoms/affine.h"
#include "atoms/non_elementwise_full_dom.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"
#include <math.h>
#include <string.h>

/* Parametric dense path of quad_form: y = x' P x where P is fed by a parameter
 * and refreshed each solve. Same setup as the constant test (x is 3x1 at global
 * index 2, total variables = 5, x = [1, 2, 3]), then the parameter is updated
 * and the derivatives are re-verified.
 *
 * P1 = [1 2 0; 2 3 0; 0 0 4]: value = 57, grad = [10,16,24], Hessian = 4 P1.
 * P2 = [2 1 0; 1 4 0; 0 0 5]: value = 67, grad = [8,18,30],  Hessian = 4 P2.
 */
const char *test_wsum_hess_quad_form_dense_param(void)
{
    double u_vals[5] = {0.0, 0.0, 1.0, 2.0, 3.0};
    double w = 2.0;

    /* row-major 3x3 dense P (symmetric, so column-major == row-major) */
    double P1[9] = {1.0, 2.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0};

    expr *param_P = new_parameter(3, 3, 0, 5, P1); /* param_id 0 = updatable */
    expr *x = new_variable(3, 1, 2, 5);
    expr *node =
        new_quad_form_dense(x, 3, NULL, param_P); /* parametric: data NULL */

    jacobian_init(node);
    wsum_hess_init(node);

    node->forward(node, u_vals);
    node->eval_jacobian(node);
    node->eval_wsum_hess(node, &w);

    int expected_jp[2] = {0, 3};
    int expected_ji[3] = {2, 3, 4};
    int expected_hp[6] = {0, 0, 0, 3, 6, 9};
    int expected_hi[9] = {2, 3, 4, 2, 3, 4, 2, 3, 4};

    /* --- parameter P1 --- */
    mu_assert("param quad_form value (P1) fail", fabs(node->value[0] - 57.0) < 1e-9);
    double grad1[3] = {10.0, 16.0, 24.0};
    mu_assert("param quad_form jacobian vals (P1) fail",
              cmp_values(node->jacobian, grad1, 3));
    mu_assert("param quad_form jacobian sparsity (P1) fail",
              cmp_sparsity(node->jacobian, expected_jp, expected_ji, 1, 3));
    mu_assert("param quad_form hessian not permuted_dense",
              node->wsum_hess->is_permuted_dense);
    double hess1[9] = {4.0, 8.0, 0.0, 8.0, 12.0, 0.0, 0.0, 0.0, 16.0};
    mu_assert("param quad_form hessian vals (P1) fail",
              cmp_values(node->wsum_hess, hess1, 9));
    mu_assert("param quad_form hessian sparsity (P1) fail",
              cmp_sparsity(node->wsum_hess, expected_hp, expected_hi, 5, 9));

    /* --- update the parameter and re-evaluate (sparsity is reused) --- */
    double P2[9] = {2.0, 1.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 5.0};
    memcpy(param_P->value, P2, 9 * sizeof(double));
    expr_set_needs_refresh(node);

    node->forward(node, u_vals);
    node->eval_jacobian(node);
    node->eval_wsum_hess(node, &w);

    /* --- parameter P2 --- */
    mu_assert("param quad_form value (P2) fail", fabs(node->value[0] - 67.0) < 1e-9);
    double grad2[3] = {8.0, 18.0, 30.0};
    mu_assert("param quad_form jacobian vals (P2) fail",
              cmp_values(node->jacobian, grad2, 3));
    mu_assert("param quad_form jacobian sparsity (P2) fail",
              cmp_sparsity(node->jacobian, expected_jp, expected_ji, 1, 3));
    double hess2[9] = {8.0, 4.0, 0.0, 4.0, 16.0, 0.0, 0.0, 0.0, 20.0};
    mu_assert("param quad_form hessian vals (P2) fail",
              cmp_values(node->wsum_hess, hess2, 9));
    mu_assert("param quad_form hessian sparsity (P2) fail",
              cmp_sparsity(node->wsum_hess, expected_hp, expected_hi, 5, 9));

    free_expr(node);
    return 0;
}
