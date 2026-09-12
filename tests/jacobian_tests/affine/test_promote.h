#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "atoms/affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"
#include "utils/permuted_dense.h"

const char *test_promote_scalar_jacobian(void)
{
    /* Promote scalar to 3-element vector, check jacobian */
    double u[1] = {2.0};
    expr *var = new_variable(1, 1, 0, 1);
    expr *promote_node = new_promote(var, 3, 1);
    promote_node->forward(promote_node, u);
    jacobian_init(promote_node);
    eval_jacobian(promote_node);

    /* Jacobian is 3x1 with all 1s (each output depends on same input) */
    double expected_x[3] = {1.0, 1.0, 1.0};
    int expected_p[4] = {0, 1, 2, 3};
    int expected_i[3] = {0, 0, 0};

    mu_assert("vals fail", cmp_values(promote_node->jacobian, expected_x, 3));
    mu_assert("sparsity fail",
              cmp_sparsity(promote_node->jacobian, expected_p, expected_i, 3, 3));

    free_expr(promote_node);
    return 0;
}

const char *test_promote_scalar_to_matrix_jacobian(void)
{
    /* Promote scalar to 2x3 matrix, check jacobian */
    double u[1] = {7.0};
    expr *var = new_variable(1, 1, 0, 1);
    expr *promote_node = new_promote(var, 2, 3);
    promote_node->forward(promote_node, u);

    /* Forward: all 6 elements should be 7.0 */
    double expected_val[6] = {7.0, 7.0, 7.0, 7.0, 7.0, 7.0};
    mu_assert("promote scalar->matrix forward failed",
              cmp_double_array(promote_node->value, expected_val, 6));

    jacobian_init(promote_node);
    eval_jacobian(promote_node);

    /* Jacobian is 6x1 with all 1s (each output depends on same scalar input) */
    double expected_x[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    int expected_p[7] = {0, 1, 2, 3, 4, 5, 6};
    int expected_i[6] = {0, 0, 0, 0, 0, 0};

    mu_assert("vals fail", cmp_values(promote_node->jacobian, expected_x, 6));
    mu_assert("sparsity fail",
              cmp_sparsity(promote_node->jacobian, expected_p, expected_i, 6, 6));

    free_expr(promote_node);
    return 0;
}

/* A pd child Jacobian stays pd through promote: AU = A @ u with A a 1x2 dense
   constant and u a 2x1 variable has a (1, 2) pd Jacobian with m0 = 1;
   promote(AU, 3, 2) gathers that single row six times. */
const char *test_promote_jacobian_pd_preserved(void)
{
    double A[2] = {1.5, -2.0};
    expr *u = new_variable(2, 1, 0, 2);
    expr *AU = new_left_matmul_dense(NULL, u, 1, 2, A);
    expr *P = new_promote(AU, 3, 2);

    double u_vals[2] = {0.5, -1.5};
    jacobian_init(P);
    P->forward(P, u_vals);
    eval_jacobian(P);

    mu_assert("promote Jacobian should be PD", P->jacobian->is_permuted_dense);
    permuted_dense *pd = (permuted_dense *) P->jacobian;
    mu_assert("shape", P->jacobian->m == 6 && P->jacobian->n == 2);
    mu_assert("m0", pd->m0 == 6);
    mu_assert("n0", pd->n0 == 2);
    int expected_row_perm[6] = {0, 1, 2, 3, 4, 5};
    mu_assert("row_perm", cmp_int_array(pd->row_perm, expected_row_perm, 6));
    double expected_X[12] = {1.5, -2.0, 1.5, -2.0, 1.5, -2.0,
                             1.5, -2.0, 1.5, -2.0, 1.5, -2.0};
    mu_assert("X values", cmp_double_array(pd->X, expected_X, 12));

    free_expr(P);
    return 0;
}
