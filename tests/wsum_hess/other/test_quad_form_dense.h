#include "atoms/affine.h"
#include "atoms/non_elementwise_full_dom.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"
#include <math.h>
#include <string.h>

/* Dense path of quad_form: y = x' P x with a dense (permuted_dense) P.
 * x is 3x1 with global index 2, total variables = 5.
 * P = [1 2 0; 2 3 0; 0 0 4] (symmetric), x = [1, 2, 3].
 *   value    = x' P x = 57
 *   gradient = 2 P x = [10, 16, 24] on columns {2, 3, 4}
 *   Hessian  = 2 w P  (full dense block over rows/cols {2, 3, 4})
 */
const char *test_wsum_hess_quad_form_dense(void)
{
    double u_vals[5] = {0.0, 0.0, 1.0, 2.0, 3.0};
    double w = 2.0;

    /* row-major 3x3 dense P */
    double P[9] = {1.0, 2.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0};

    expr *x = new_variable(3, 1, 2, 5);
    expr *node = new_quad_form_dense(x, 3, P, NULL);

    jacobian_init(node);
    node->forward(node, u_vals);
    node->eval_jacobian(node);
    wsum_hess_init(node);
    node->eval_wsum_hess(node, &w);

    /* forward value */
    mu_assert("dense quad_form value fail", fabs(node->value[0] - 57.0) < 1e-9);

    /* gradient = 2 P x on columns {2,3,4} */
    double expected_grad[3] = {10.0, 16.0, 24.0};
    int expected_jp[2] = {0, 3};
    int expected_ji[3] = {2, 3, 4};
    mu_assert("dense quad_form jacobian vals fail",
              cmp_values(node->jacobian, expected_grad, 3));
    mu_assert("dense quad_form jacobian sparsity fail",
              cmp_sparsity(node->jacobian, expected_jp, expected_ji, 1, 3));

    /* Hessian = 2 w P = 4 P as a dense block over rows/cols {2,3,4} */
    mu_assert("dense quad_form hessian is not permuted_dense",
              node->wsum_hess->is_permuted_dense);
    int expected_hp[6] = {0, 0, 0, 3, 6, 9};
    int expected_hi[9] = {2, 3, 4, 2, 3, 4, 2, 3, 4};
    double expected_hx[9] = {4.0, 8.0, 0.0, 8.0, 12.0, 0.0, 0.0, 0.0, 16.0};
    mu_assert("dense quad_form hessian sparsity fail",
              cmp_sparsity(node->wsum_hess, expected_hp, expected_hi, 5, 9));
    mu_assert("dense quad_form hessian vals fail",
              cmp_values(node->wsum_hess, expected_hx, 9));

    free_expr(node);
    return 0;
}
