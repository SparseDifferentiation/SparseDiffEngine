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

const char *test_const_hstack_left_matmul(void)
{
    int n = 4;

    /* minimize hstack(p1, p2) @ x, where p1 and p2 are fixed */
    expr *x = new_variable(2 * n, 1, 0, 2 * n);
    double p1_vals[4] = {1.0, 2.0, 3.0, 0.0};
    expr *p1 = new_parameter(1, n, PARAM_FIXED, 2 * n, p1_vals);
    double p2_vals[4] = {4.0, 0.0, 5.0, 6.0};
    expr *p2 = new_parameter(1, n, PARAM_FIXED, 2 * n, p2_vals);
    expr *param_nodes[2] = {p1, p2};
    expr *p_hstack = new_hstack(param_nodes, 2, 2 * n);
    expr *objective = new_left_matmul_dense(p_hstack, x, 1, 2 * n, NULL);
    problem *prob = new_problem(objective, NULL, 0, false);

    problem_init_derivatives(prob);

    /* point for evaluating */
    double x_vals[8] = {2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0};

    problem_objective_forward(prob, x_vals);
    double obj_val = 1.0 * 2.0 + 2.0 * 2.0 + 3.0 * 2.0 + 0.0 * 2.0 + 4.0 * 1.0 +
                     0.0 * 1.0 + 5.0 * 1.0 + 6.0 * 1.0;
    mu_assert("vals fail", fabs(prob->objective->value[0] - obj_val) < 1e-6);

    problem_gradient(prob);
    double grad_x[8] = {1.0, 2.0, 3.0, 0.0, 4.0, 0.0, 5.0, 6.0};
    mu_assert("vals fail", cmp_double_array(prob->gradient_values, grad_x, 8));

    free_problem(prob);
    return 0;
}

const char *test_param_hstack_left_matmul(void)
{
    int n = 4;

    /* minimize hstack(p1, p2) @ x, where p1 and p2 are parameter */
    expr *x = new_variable(2 * n, 1, 0, 2 * n);
    double p1_vals[4] = {1.0, 2.0, 3.0, 0.0};
    expr *p1 = new_parameter(1, n, 0, 2 * n, p1_vals);
    double p2_vals[4] = {4.0, 0.0, 5.0, 6.0};
    expr *p2 = new_parameter(1, n, n, 2 * n, p2_vals);
    expr *param_nodes[2] = {p1, p2};
    expr *p_hstack = new_hstack(param_nodes, 2, 2 * n);
    expr *objective = new_left_matmul_dense(p_hstack, x, 1, 2 * n, NULL);
    problem *prob = new_problem(objective, NULL, 0, false);

    problem_register_params(prob, param_nodes, 2);
    problem_init_derivatives(prob);

    /* point for evaluating */
    double x_vals[8] = {2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0};

    problem_objective_forward(prob, x_vals);
    double obj_val = 1.0 * 2.0 + 2.0 * 2.0 + 3.0 * 2.0 + 0.0 * 2.0 + 4.0 * 1.0 +
                     0.0 * 1.0 + 5.0 * 1.0 + 6.0 * 1.0;
    mu_assert("vals fail", fabs(prob->objective->value[0] - obj_val) < 1e-6);

    problem_gradient(prob);
    double grad_x[8] = {1.0, 2.0, 3.0, 0.0, 4.0, 0.0, 5.0, 6.0};
    mu_assert("vals fail", cmp_double_array(prob->gradient_values, grad_x, 8));

    double theta[8] = {5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0};
    problem_update_params(prob, theta);

    problem_objective_forward(prob, x_vals);
    double updated_obj_val = 5.0 * 2.0 + 4.0 * 2.0 + 3.0 * 2.0 + 2.0 * 2.0 +
                             1.0 * 1.0 + 0.0 * 1.0 + 1.0 * 1.0 + 2.0 * 1.0;
    mu_assert("vals fail", fabs(prob->objective->value[0] - updated_obj_val) < 1e-6);

    problem_gradient(prob);
    double updated_grad_x[8] = {5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0};
    mu_assert("vals fail",
              cmp_double_array(prob->gradient_values, updated_grad_x, 8));

    free_problem(prob);
    return 0;
}

const char *test_param_scalar_mult_convolve(void)
{
    int n = 3;

    /* minimize sum(x) subject to conv(s * kernel, x), where kernel is a fixed
       length-4 vector and s is an updatable scalar parameter. Scaling s
       rescales the entire kernel, so Jacobian and constraint values scale
       linearly with s. */
    expr *x = new_variable(3, 1, 0, n);
    expr *objective = new_sum(x, -1);

    double kernel_vals[4] = {1.0, 2.0, 3.0, 4.0};
    expr *kernel = new_parameter(4, 1, PARAM_FIXED, n, kernel_vals);
    double s_vals[1] = {1.0};
    expr *s = new_parameter(1, 1, 0, n, s_vals);
    expr *scaled_kernel = new_scalar_mult(s, kernel);

    expr *constraint = new_convolve(scaled_kernel, x);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    expr *param_nodes[1] = {s};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* point for evaluating and sparsity arrays (T is 6x3 with 12 nonzeros) */
    double x_vals[3] = {1.0, 2.0, 3.0};
    int Ap[7] = {0, 1, 3, 6, 9, 11, 12};
    int Ai[12] = {0, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 2};

    /* test 1: s = 1, effective kernel = [1, 2, 3, 4] */
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    double constrs[6] = {1.0, 4.0, 10.0, 16.0, 17.0, 12.0};
    double Ax[12] = {1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 4.0, 3.0, 4.0};
    mu_assert("vals fail", cmp_double_array(prob->constraint_values, constrs, 6));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 7));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 12));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, Ax, 12));

    mu_assert("check_jacobian failed",
              check_jacobian_num(constraint, x_vals, NUMERICAL_DIFF_DEFAULT_H));

    /* test 2: s = 10 via problem_update_params, effective kernel = [10,20,30,40] */
    double theta[1] = {10.0};
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    double updated_constrs[6] = {10.0, 40.0, 100.0, 160.0, 170.0, 120.0};
    double updated_Ax[12] = {10.0, 20.0, 10.0, 30.0, 20.0, 10.0,
                             40.0, 30.0, 20.0, 40.0, 30.0, 40.0};
    mu_assert("vals fail",
              cmp_double_array(prob->constraint_values, updated_constrs, 6));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 7));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 12));
    mu_assert("vals fail", cmp_double_array(prob->jacobian->x, updated_Ax, 12));

    mu_assert("check_jacobian failed",
              check_jacobian_num(constraint, x_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_problem(prob);
    return 0;
}

#endif /* TEST_PARAM_BROADCAST_H */
