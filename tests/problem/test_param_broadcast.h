#ifndef TEST_PARAM_BROADCAST_H
#define TEST_PARAM_BROADCAST_H

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "atoms/affine.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "problem.h"
#include "subexpr.h"
#include "test_helpers.h"

const char *test_constant_broadcast_vector_mult(void)
{
    int n = 6;

    /* minimize sum(x) subject to broadcast(c) ∘ x, with c constant */
    expr *x = new_variable(2, 3, 0, n);
    expr *objective = new_sum(x, -1);
    double c_vals[3] = {1.0, 2.0, 3.0};
    expr *c = new_parameter(1, 3, PARAM_FIXED, n, c_vals);
    expr *c_bcast = new_broadcast(c, 2, 3);
    expr *constraint = new_vector_mult(c_bcast, x);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);
    problem_init_derivatives(prob);

    /* point for evaluating */
    double x_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    problem_constraint_forward(prob, x_vals);
    double constrs[6] = {1.0, 2.0, 6.0, 8.0, 15.0, 18.0};
    problem_jacobian(prob);
    double jac_x[6] = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0};
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 6));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 6));

    mu_assert("check_jacobian failed",
              check_jacobian_num(constraint, x_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_problem(prob);
    return 0;
}

const char *test_constant_promote_vector_mult(void)
{
    int n = 6;

    /* minimize sum(x) subject to promote(c) ∘ x, with c constant */
    expr *x = new_variable(2, 3, 0, n);
    expr *objective = new_sum(x, -1);
    double c_vals = 3.0;
    expr *c = new_parameter(1, 1, PARAM_FIXED, n, &c_vals);
    expr *c_bcast = new_promote(c, 2, 3);
    expr *constraint = new_vector_mult(c_bcast, x);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    problem_init_derivatives(prob);

    /* point for evaluating */
    double x_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    problem_constraint_forward(prob, x_vals);
    double constrs[6] = {3.0, 6.0, 9.0, 12.0, 15.0, 18.0};
    problem_jacobian(prob);
    double jac_x[6] = {3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 6));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 6));

    mu_assert("check_jacobian failed",
              check_jacobian_num(constraint, x_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_problem(prob);
    return 0;
}

const char *test_param_broadcast_vector_mult(void)
{
    int n = 6;

    /* minimize sum(x) subject to broadcast(p) ∘ x, with p parameter */
    expr *x = new_variable(2, 3, 0, n);
    expr *objective = new_sum(x, -1);
    double c_vals[3] = {1.0, 2.0, 3.0};
    expr *c = new_parameter(1, 3, 0, n, c_vals);
    expr *c_bcast = new_broadcast(c, 2, 3);
    expr *constraint = new_vector_mult(c_bcast, x);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    expr *param_nodes[1] = {c};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* point for evaluating */
    double x_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    problem_constraint_forward(prob, x_vals);
    double constrs[6] = {1.0, 2.0, 6.0, 8.0, 15.0, 18.0};
    problem_jacobian(prob);
    double jac_x[6] = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0};
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 6));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 6));

    mu_assert("check_jacobian failed",
              check_jacobian_num(constraint, x_vals, NUMERICAL_DIFF_DEFAULT_H));

    /* second iteration after updating parameter */
    double theta[3] = {5.0, 4.0, 3.0};
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    double updated_constrs[6] = {5.0, 10.0, 12.0, 16.0, 15.0, 18.0};
    double updated_jac_x[6] = {5.0, 5.0, 4.0, 4.0, 3.0, 3.0};
    mu_assert("vals fail",
              cmp_double_array(prob->constraint_values, updated_constrs, 6));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, updated_jac_x, 6));

    mu_assert("check_jacobian failed",
              check_jacobian_num(constraint, x_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_problem(prob);
    return 0;
}

const char *test_param_promote_vector_mult(void)
{
    int n = 6;

    /* minimize sum(x) subject to promote(p) ∘ x, with p parameter */
    expr *x = new_variable(2, 3, 0, n);
    expr *objective = new_sum(x, -1);
    double c_vals = 3.0;
    expr *c = new_parameter(1, 1, 0, n, &c_vals);
    expr *c_bcast = new_promote(c, 2, 3);
    expr *constraint = new_vector_mult(c_bcast, x);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    expr *param_nodes[1] = {c};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* point for evaluating */
    double x_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    problem_constraint_forward(prob, x_vals);
    double constrs[6] = {3.0, 6.0, 9.0, 12.0, 15.0, 18.0};
    problem_jacobian(prob);
    double jac_x[6] = {3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 6));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 6));

    mu_assert("check_jacobian failed",
              check_jacobian_num(constraint, x_vals, NUMERICAL_DIFF_DEFAULT_H));

    /* second iteration after updating parameter */
    double theta = 5.0;
    problem_update_params(prob, &theta);
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    double updated_constrs[6] = {5.0, 10.0, 15.0, 20.0, 25.0, 30.0};
    double updated_jac_x[6] = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
    mu_assert("vals fail",
              cmp_double_array(prob->constraint_values, updated_constrs, 6));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, updated_jac_x, 6));

    mu_assert("check_jacobian failed",
              check_jacobian_num(constraint, x_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_problem(prob);
    return 0;
}

const char *test_const_sum_scalar_mult(void)
{
    int n = 6;

    /* minimize sum(x) subject to sum(c) * x, with c constant */
    expr *x = new_variable(1, 1, 0, n);
    expr *objective = new_sum(x, -1);
    double c_vals[3] = {1.0, 2.0, 3.0};
    expr *c = new_parameter(1, 3, PARAM_FIXED, n, c_vals);
    expr *c_sum = new_sum(c, -1);
    expr *constraint = new_scalar_mult(c_sum, x);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    problem_init_derivatives(prob);

    /* point for evaluating */
    double x_vals[1] = {4.0};

    problem_constraint_forward(prob, x_vals);
    double constrs[1] = {6.0 * 4.0};
    problem_jacobian(prob);
    double jac_x[1] = {6.0};
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 1));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 1));

    mu_assert("check_jacobian failed",
              check_jacobian_num(constraint, x_vals, NUMERICAL_DIFF_DEFAULT_H));
    free_problem(prob);
    return 0;
}

const char *test_param_sum_scalar_mult(void)
{
    int n = 6;

    /* minimize sum(x) subject to sum(p) * x, with p parameter */
    expr *x = new_variable(1, 1, 0, n);
    expr *objective = new_sum(x, -1);
    double c_vals[3] = {1.0, 2.0, 3.0};
    expr *c = new_parameter(1, 3, 0, n, c_vals);
    expr *c_sum = new_sum(c, -1);
    expr *constraint = new_scalar_mult(c_sum, x);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    expr *param_nodes[1] = {c};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* point for evaluating */
    double x_vals[1] = {4.0};

    problem_constraint_forward(prob, x_vals);
    double constrs[1] = {6.0 * 4.0};
    problem_jacobian(prob);
    double jac_x[1] = {6.0};
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 1));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, jac_x, 1));

    mu_assert("check_jacobian failed",
              check_jacobian_num(constraint, x_vals, NUMERICAL_DIFF_DEFAULT_H));
    
    /* second iteration after updating parameter */
    double theta[3] = {5.0, 4.0, 3.0};
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    double updated_constrs[1] = {12.0 * 4.0};
    double updated_jac_x[1] = {12.0};
    mu_assert("vals fail",
              cmp_double_array(prob->constraint_values, updated_constrs, 1));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, updated_jac_x, 1));

    mu_assert("check_jacobian failed",
              check_jacobian_num(constraint, x_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_problem(prob);
    return 0;
}

#endif /* TEST_PARAM_BROADCAST_H */
