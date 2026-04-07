#ifndef TEST_PARAM_PROB_H
#define TEST_PARAM_PROB_H

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "atoms/affine.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "problem.h"
#include "subexpr.h"
#include "test_helpers.h"

const char *test_param_scalar_mult_problem(void)
{
    int n = 2;

    /* minimize a * sum(log(x)), with a parameter */
    expr *x = new_variable(2, 1, 0, n);
    expr *log_x = new_log(x);
    expr *a_param = new_parameter(1, 1, 0, n, NULL);
    expr *scaled = new_scalar_mult(a_param, log_x);
    expr *objective = new_sum(scaled, -1);
    problem *prob = new_problem(objective, NULL, 0, false);

    /* register parameters and fill sparsity patterns */
    expr *param_nodes[1] = {a_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* point for evaluating */
    double x_vals[2] = {1.0, 2.0};

    /* test 1: a=3 */
    double theta[1] = {3.0};
    problem_update_params(prob, theta);
    double obj_val = problem_objective_forward(prob, x_vals);
    problem_gradient(prob);
    double expected_obj = 3.0 * log(2.0);
    mu_assert("vals fail", fabs(obj_val - expected_obj) < 1e-10);
    double grad[2] = {3.0, 1.5};
    mu_assert("vals fail", cmp_double_array(prob->gradient_values, grad, 2));

    /* test 2: a=5 */
    theta[0] = 5.0;
    problem_update_params(prob, theta);
    obj_val = problem_objective_forward(prob, x_vals);
    problem_gradient(prob);
    expected_obj = 5.0 * log(2.0);
    mu_assert("vals fail", fabs(obj_val - expected_obj) < 1e-10);
    grad[0] = 5.0;
    grad[1] = 2.5;
    mu_assert("vals fail", cmp_double_array(prob->gradient_values, grad, 2));

    free_problem(prob);

    return 0;
}

const char *test_param_vector_mult_problem(void)
{
    int n = 2;

    /* minimize sum(x) subject to p ∘ x, with p parameter */
    expr *x = new_variable(2, 1, 0, n);
    expr *objective = new_sum(x, -1);
    expr *p_param = new_parameter(2, 1, 0, n, NULL);
    expr *constraint = new_vector_mult(p_param, x);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    /* register parameters and fill sparsity patterns */
    expr *param_nodes[1] = {p_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* point for evaluating and utilities for test */
    double x_vals[2] = {1.0, 2.0};
    int Ap[3] = {0, 1, 2};
    int Ai[2] = {0, 1};
    double Ax[2] = {3.0, 4.0};

    /* test 1: p=[3,4] */
    double theta[2] = {3.0, 4.0};
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    double constrs[2] = {3.0, 8.0};
    problem_jacobian(prob);
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 2));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, Ax, 2));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 3));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 2));

    /* test 2: p=[5,6] */
    theta[0] = 5.0;
    theta[1] = 6.0;
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    constrs[0] = 5.0;
    constrs[1] = 12.0;
    Ax[0] = 5.0;
    Ax[1] = 6.0;
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 2));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, Ax, 2));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 3));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 2));
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 2));

    free_problem(prob);

    return 0;
}

const char *test_param_left_matmul_problem(void)
{
    int n = 2;

    /* minimize sum(x) subject to Ax = ?, with A parameter */
    expr *x = new_variable(2, 1, 0, n);
    expr *objective = new_sum(x, -1);
    expr *A_param = new_parameter(2, 2, 0, n, NULL);

    /* dense 2x2 matrix */
    double Ax[4] = {2.0, 0.0, 0.0, 1.0};
    expr *constraint = new_left_matmul_dense(A_param, x, 2, 2, Ax);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    /* register parameters and fill sparsity patterns*/
    expr *param_nodes[1] = {A_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* point for evaluating and utilities for test */
    double x_vals[2] = {1.0, 2.0};
    int Ap[3] = {0, 2, 4};
    int Ai[4] = {0, 1, 0, 1};

    /* test 1: initial jacobian */
    problem_constraint_forward(prob, x_vals);
    double constrs[2] = {2.0, 2.0};
    problem_jacobian(prob);
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 2));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, Ax, 4));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 3));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 4));

    /* test 2: A = [[1,2],[3,4]] (column-major [1,3,2,4]) */
    double theta[4] = {1.0, 3.0, 2.0, 4.0};
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    constrs[0] = 5;
    constrs[1] = 11;
    Ax[0] = 1.0;
    Ax[1] = 2.0;
    Ax[2] = 3.0;
    Ax[3] = 4.0;
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 2));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, Ax, 4));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 3));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 4));
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 2));

    free_problem(prob);

    return 0;
}

const char *test_param_right_matmul_problem(void)
{
    int n = 2;

    /* minimize sum(x) subject to xA = ?, with A parameter */
    expr *x = new_variable(1, 2, 0, n);
    expr *objective = new_sum(x, -1);
    expr *A_param = new_parameter(2, 2, 0, n, NULL);

    /* dense 2x2 matrix */
    double Ax[4] = {2.0, 0.0, 0.0, 1.0};
    expr *constraint = new_right_matmul_dense(A_param, x, 2, 2, Ax);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    /* register parameters and fill sparsity patterns */
    expr *param_nodes[1] = {A_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* point for evaluating and utilities for test */
    double x_vals[2] = {1.0, 2.0};
    int Ap[3] = {0, 2, 4};
    int Ai[4] = {0, 1, 0, 1};

    /* test 1: initial jacobian */
    problem_constraint_forward(prob, x_vals);
    double constrs[2] = {2.0, 2.0};
    problem_jacobian(prob);
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 2));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, Ax, 4));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 3));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 4));

    /* test 2: A = [[1,2],[3,4]] (column-major [1,3,2,4]) */
    double theta[4] = {1.0, 3.0, 2.0, 4.0};
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    constrs[0] = 7;
    constrs[1] = 10;
    Ax[0] = 1.0;
    Ax[1] = 3.0;
    Ax[2] = 2.0;
    Ax[3] = 4.0;
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 2));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, Ax, 4));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 3));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 4));
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 2));

    free_problem(prob);

    return 0;
}

const char *test_param_fixed_skip_in_update(void)
{
    int n = 2;

    /* minimize a * sum(log(x)) + b * sum(x), a fixed, b updatable */
    expr *x = new_variable(2, 1, 0, n);
    expr *log_x = new_log(x);
    double a_val = 2.0;
    expr *a_param = new_parameter(1, 1, PARAM_FIXED, n, &a_val);
    expr *a_log = new_scalar_mult(a_param, log_x);
    expr *sum_a_log = new_sum(a_log, -1);

    expr *b_param = new_parameter(1, 1, 0, n, NULL);
    expr *b_x = new_scalar_mult(b_param, x);
    expr *sum_b_x = new_sum(b_x, -1);

    expr *objective = new_add(sum_a_log, sum_b_x);
    problem *prob = new_problem(objective, NULL, 0, false);

    /* register parameters and fill sparsity patterns */
    expr *param_nodes[2] = {a_param, b_param};
    problem_register_params(prob, param_nodes, 2);
    problem_init_derivatives(prob);

    /* point for evaluating */
    double x_vals[2] = {1.0, 2.0};

    /* test 1: b=3, a stays 2 */
    double theta[1] = {3.0};
    problem_update_params(prob, theta);
    mu_assert("vals fail", fabs(a_param->value[0] - 2.0) < 1e-10);
    double obj_val = problem_objective_forward(prob, x_vals);
    problem_gradient(prob);
    double expected_obj = 2.0 * log(2.0) + 9.0;
    mu_assert("vals fail", fabs(obj_val - expected_obj) < 1e-10);
    double grad[2] = {5.0, 4.0};
    mu_assert("vals fail", cmp_double_array(prob->gradient_values, grad, 2));

    /* test 2: b=5, a stays 2 */
    theta[0] = 5.0;
    problem_update_params(prob, theta);
    mu_assert("vals fail", fabs(a_param->value[0] - 2.0) < 1e-10);
    obj_val = problem_objective_forward(prob, x_vals);
    problem_gradient(prob);
    expected_obj = 2.0 * log(2.0) + 15.0;
    mu_assert("vals fail", fabs(obj_val - expected_obj) < 1e-10);
    grad[0] = 7.0;
    grad[1] = 6.0;
    mu_assert("vals fail", cmp_double_array(prob->gradient_values, grad, 2));

    free_problem(prob);

    return 0;
}

const char *test_param_left_matmul_rectangular(void)
{
    int n = 2;

    /* minimize sum(x) subject to Ax = ?, with A parameter (3x2) */
    expr *x = new_variable(2, 1, 0, n);
    expr *objective = new_sum(x, -1);
    expr *A_param = new_parameter(3, 2, 0, n, NULL);

    /* dense 3x2 matrix */
    double Ax[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    expr *constraint = new_left_matmul_dense(A_param, x, 3, 2, Ax);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    /* register parameters and fill sparsity patterns */
    expr *param_nodes[1] = {A_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* point for evaluating and utilities for test */
    double x_vals[2] = {1.0, 2.0};
    int Ap[4] = {0, 2, 4, 6};
    int Ai[6] = {0, 1, 0, 1, 0, 1};

    /* test 1: initial jacobian */
    problem_constraint_forward(prob, x_vals);
    double constrs[3] = {5.0, 11.0, 17.0};
    problem_jacobian(prob);
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 3));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, Ax, 6));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 4));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 6));

    /* test 2: A = [[7,8],[9,10],[11,12]] (column-major [7,9,11,8,10,12]) */
    double theta[6] = {7.0, 9.0, 11.0, 8.0, 10.0, 12.0};
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    constrs[0] = 23.0;
    constrs[1] = 29.0;
    constrs[2] = 35.0;
    Ax[0] = 7.0;
    Ax[1] = 8.0;
    Ax[2] = 9.0;
    Ax[3] = 10.0;
    Ax[4] = 11.0;
    Ax[5] = 12.0;
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 3));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, Ax, 6));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 4));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 6));
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 3));

    free_problem(prob);

    return 0;
}

const char *test_param_right_matmul_rectangular(void)
{
    int n = 2;

    /* minimize sum(x) subject to xA = ?, with A parameter (2x3) */
    expr *x = new_variable(1, 2, 0, n);
    expr *objective = new_sum(x, -1);
    expr *A_param = new_parameter(2, 3, 0, n, NULL);

    /* dense 2x3 matrix */
    double Ax[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    expr *constraint = new_right_matmul_dense(A_param, x, 2, 3, Ax);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    /* register parameters and fill sparsity patterns */
    expr *param_nodes[1] = {A_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* point for evaluating and utilities for test */
    double x_vals[2] = {1.0, 2.0};
    int Ap[4] = {0, 2, 4, 6};
    int Ai[6] = {0, 1, 0, 1, 0, 1};

    /* test 1: initial jacobian */
    problem_constraint_forward(prob, x_vals);
    double constrs[3] = {9.0, 12.0, 15.0};
    problem_jacobian(prob);
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 3));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 4));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 6));
    double jac_x[6] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 6));

    /* test 2: A = [[7,8,9],[10,11,12]] (column-major [7,10,8,11,9,12]) */
    double theta[6] = {7.0, 10.0, 8.0, 11.0, 9.0, 12.0};
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    constrs[0] = 27.0;
    constrs[1] = 30.0;
    constrs[2] = 33.0;
    jac_x[0] = 7.0;
    jac_x[1] = 10.0;
    jac_x[2] = 8.0;
    jac_x[3] = 11.0;
    jac_x[4] = 9.0;
    jac_x[5] = 12.0;
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 3));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 6));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 4));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 6));
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 3));

    free_problem(prob);

    return 0;
}

const char *test_param_shared_left_matmul_problem(void)
{
    int n = 4;

    /* minimize sum(x) subject to A@x and A@y, shared A parameter */
    expr *x = new_variable(2, 1, 0, n);
    expr *y = new_variable(2, 1, 2, n);
    expr *objective = new_sum(x, -1);
    expr *A_param = new_parameter(2, 2, 0, n, NULL);

    /* dense 2x2 identity */
    double Ax[4] = {1.0, 0.0, 0.0, 1.0};
    expr *constraints[2];
    constraints[0] = new_left_matmul_dense(A_param, x, 2, 2, Ax);
    constraints[1] = new_left_matmul_dense(A_param, y, 2, 2, Ax);
    problem *prob = new_problem(objective, constraints, 2, false);

    /* register parameters and fill sparsity patterns */
    expr *param_nodes[1] = {A_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* point for evaluating and utilities for test */
    double x_vals[4] = {1.0, 2.0, 3.0, 4.0};
    int Ap[5] = {0, 2, 4, 6, 8};
    int Ai[8] = {0, 1, 0, 1, 2, 3, 2, 3};

    /* test 1: initial identity jacobian */
    problem_constraint_forward(prob, x_vals);
    double constrs[4] = {1.0, 2.0, 3.0, 4.0};
    problem_jacobian(prob);
    double jac_x[8] = {1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0};
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 4));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 8));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 5));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 8));

    /* test 2: A = [[1,2],[3,4]] (column-major [1,3,2,4]) */
    double theta[4] = {1.0, 3.0, 2.0, 4.0};
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    constrs[0] = 5.0;
    constrs[1] = 11.0;
    constrs[2] = 11.0;
    constrs[3] = 25.0;
    jac_x[0] = 1.0;
    jac_x[1] = 2.0;
    jac_x[2] = 3.0;
    jac_x[3] = 4.0;
    jac_x[4] = 1.0;
    jac_x[5] = 2.0;
    jac_x[6] = 3.0;
    jac_x[7] = 4.0;
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 4));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 8));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 5));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 8));
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 4));

    free_problem(prob);

    return 0;
}

#endif /* TEST_PARAM_PROB_H */
