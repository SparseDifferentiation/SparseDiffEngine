#ifndef TEST_PARAM_BROADCAST_H
#define TEST_PARAM_BROADCAST_H

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "atoms/affine.h"
#include "expr.h"
#include "minunit.h"
#include "problem.h"
#include "subexpr.h"
#include "test_helpers.h"

const char *test_param_broadcast_vector_mult(void)
{
    int n = 6;

    /* minimize sum(x) subject to broadcast(p) ∘ x, with p parameter */
    expr *x = new_variable(2, 3, 0, n);
    expr *objective = new_sum(x, -1);
    expr *p_param = new_parameter(1, 3, 0, n, NULL);
    expr *p_bcast = new_broadcast(p_param, 2, 3);
    expr *constraint = new_vector_mult(p_bcast, x);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    /* register parameters and fill sparsity patterns */
    expr *param_nodes[1] = {p_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* point for evaluating */
    double x_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    /* test 1: p=[1,2,3] */
    double theta[3] = {1.0, 2.0, 3.0};
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    double constrs[6] = {1.0, 2.0, 6.0, 8.0, 15.0, 18.0};
    problem_jacobian(prob);
    double jac_x[6] = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0};
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 6));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 6));

    /* test 2: p=[10,20,30] */
    theta[0] = 10.0;
    theta[1] = 20.0;
    theta[2] = 30.0;
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    constrs[0] = 10.0;
    constrs[1] = 20.0;
    constrs[2] = 60.0;
    constrs[3] = 80.0;
    constrs[4] = 150.0;
    constrs[5] = 180.0;
    jac_x[0] = 10.0;
    jac_x[1] = 10.0;
    jac_x[2] = 20.0;
    jac_x[3] = 20.0;
    jac_x[4] = 30.0;
    jac_x[5] = 30.0;
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 6));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 6));

    free_problem(prob);
    return 0;
}

const char *test_param_sum_scalar_mult(void)
{
    int n = 3;

    /* minimize sum(x) subject to sum(p) * x, with p parameter */
    expr *x = new_variable(3, 1, 0, n);
    expr *objective = new_sum(x, -1);
    expr *p_param = new_parameter(2, 1, 0, n, NULL);
    expr *p_sum = new_sum(p_param, -1);
    expr *constraint = new_scalar_mult(p_sum, x);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    /* register parameters and fill sparsity patterns */
    expr *param_nodes[1] = {p_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* point for evaluating */
    double x_vals[3] = {1.0, 2.0, 3.0};

    /* test 1: p=[1,2], sum(p)=3 */
    double theta[2] = {1.0, 2.0};
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    double constrs[3] = {3.0, 6.0, 9.0};
    problem_jacobian(prob);
    double jac_x[3] = {3.0, 3.0, 3.0};
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 3));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 3));

    /* test 2: p=[5,10], sum(p)=15 */
    theta[0] = 5.0;
    theta[1] = 10.0;
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    constrs[0] = 15.0;
    constrs[1] = 30.0;
    constrs[2] = 45.0;
    jac_x[0] = 15.0;
    jac_x[1] = 15.0;
    jac_x[2] = 15.0;
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 3));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 3));

    free_problem(prob);
    return 0;
}

const char *test_param_broadcast_left_matmul(void)
{
    int n = 2;

    /* minimize sum(x) subject to broadcast(p)@x, with p parameter */
    expr *x = new_variable(2, 1, 0, n);
    expr *objective = new_sum(x, -1);
    expr *p_param = new_parameter(1, 2, 0, n, NULL);
    expr *p_bcast = new_broadcast(p_param, 3, 2);
    double Ax[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    expr *constraint = new_left_matmul_dense(p_bcast, x, 3, 2, Ax);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    /* register parameters and fill sparsity patterns */
    expr *param_nodes[1] = {p_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* point for evaluating and utilities for test */
    double x_vals[2] = {3.0, 4.0};
    int Ap[4] = {0, 2, 4, 6};
    int Ai[6] = {0, 1, 0, 1, 0, 1};

    /* test 1: p=[1,2] */
    double theta[2] = {1.0, 2.0};
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    double constrs[3] = {11.0, 11.0, 11.0};
    problem_jacobian(prob);
    double jac_x[6] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 3));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 6));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 4));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 6));

    /* test 2: p=[5,10] */
    theta[0] = 5.0;
    theta[1] = 10.0;
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    constrs[0] = 55.0;
    constrs[1] = 55.0;
    constrs[2] = 55.0;
    jac_x[0] = 5.0;
    jac_x[1] = 10.0;
    jac_x[2] = 5.0;
    jac_x[3] = 10.0;
    jac_x[4] = 5.0;
    jac_x[5] = 10.0;
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 3));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 6));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 4));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 6));

    free_problem(prob);
    return 0;
}

#endif /* TEST_PARAM_BROADCAST_H */
