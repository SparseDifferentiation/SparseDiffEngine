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

    free_problem(prob);
    return 0;
}
#endif /* TEST_PARAM_BROADCAST_H */
