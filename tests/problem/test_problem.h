#ifndef TEST_PROBLEM_H
#define TEST_PROBLEM_H

#include <math.h>
#include <stdio.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "problem.h"
#include "test_helpers.h"

/*
 * Test problem: minimize sum(log(x))
 *               subject to x >= 1 (as x - 1 >= 0)
 *
 * With x of size 3, n_vars = 3
 */
const char *test_problem_new_free(void)
{
    /* Create expressions */
    expr *x = new_variable(3, 1, 0, 3);
    expr *log_x = new_log(x);
    expr *objective = new_sum(log_x, -1);

    /* Create constraint: x - 1 (represented as just x for simplicity) */
    expr *x_constraint = new_variable(3, 1, 0, 3);

    expr *constraints[1] = {x_constraint};

    /* Create problem */
    problem *prob = new_problem(objective, constraints, 1);

    mu_assert("new_problem failed", prob != NULL);
    mu_assert("n_vars wrong", prob->n_vars == 3);
    mu_assert("n_constraints wrong", prob->n_constraints == 1);
    mu_assert("total_constraint_size wrong", prob->total_constraint_size == 3);

    /* Free problem (owns and frees all expressions) */
    free_problem(prob);

    return 0;
}

/*
 * Test problem_forward: minimize sum(log(x))
 *                       subject to x (as constraint)
 */
const char *test_problem_forward(void)
{
    expr *x = new_variable(3, 1, 0, 3);
    expr *log_x = new_log(x);
    expr *objective = new_sum(log_x, -1);

    expr *x_constraint = new_variable(3, 1, 0, 3);
    expr *constraints[1] = {x_constraint};

    problem *prob = new_problem(objective, constraints, 1);

    double u[3] = {1.0, 2.0, 3.0};
    problem_allocate(prob, u);

    double obj_val = problem_forward(prob, u);

    /* Expected: sum(log([1, 2, 3])) = 0 + log(2) + log(3) */
    double expected_obj = log(1.0) + log(2.0) + log(3.0);
    mu_assert("objective value wrong", fabs(obj_val - expected_obj) < 1e-10);

    /* Constraint values should be [1, 2, 3] */
    mu_assert("constraint[0] wrong", fabs(prob->constraint_values[0] - 1.0) < 1e-10);
    mu_assert("constraint[1] wrong", fabs(prob->constraint_values[1] - 2.0) < 1e-10);
    mu_assert("constraint[2] wrong", fabs(prob->constraint_values[2] - 3.0) < 1e-10);

    free_problem(prob);

    return 0;
}

/*
 * Test problem_gradient: gradient of sum(log(x)) = [1/x1, 1/x2, 1/x3]
 */
const char *test_problem_gradient(void)
{
    expr *x = new_variable(3, 1, 0, 3);
    expr *log_x = new_log(x);
    expr *objective = new_sum(log_x, -1);

    problem *prob = new_problem(objective, NULL, 0);

    double u[3] = {1.0, 2.0, 4.0};
    problem_allocate(prob, u);

    double *grad = problem_gradient(prob, u);

    /* Expected gradient: [1/1, 1/2, 1/4] = [1.0, 0.5, 0.25] */
    mu_assert("grad[0] wrong", fabs(grad[0] - 1.0) < 1e-10);
    mu_assert("grad[1] wrong", fabs(grad[1] - 0.5) < 1e-10);
    mu_assert("grad[2] wrong", fabs(grad[2] - 0.25) < 1e-10);

    free_problem(prob);

    return 0;
}

/*
 * Test problem_jacobian: one constraint log(x)
 * Jacobian of log(x): diag([1/x1, 1/x2])
 */
const char *test_problem_jacobian(void)
{
    int n_vars = 2;

    /* Create separate expression trees */
    expr *x_obj = new_variable(2, 1, 0, n_vars);
    expr *log_obj = new_log(x_obj);
    expr *objective = new_sum(log_obj, -1);

    expr *x_c1 = new_variable(2, 1, 0, n_vars);
    expr *log_c1 = new_log(x_c1);

    expr *constraints[1] = {log_c1};

    problem *prob = new_problem(objective, constraints, 1);

    double u[2] = {2.0, 4.0};
    problem_allocate(prob, u);

    CSR_Matrix *jac = problem_jacobian(prob, u);

    /* Check dimensions */
    mu_assert("jac rows wrong", jac->m == 2);
    mu_assert("jac cols wrong", jac->n == 2);

    /* Check row pointers: each row has 1 element */
    mu_assert("jac->p[0] wrong", jac->p[0] == 0);
    mu_assert("jac->p[1] wrong", jac->p[1] == 1);
    mu_assert("jac->p[2] wrong", jac->p[2] == 2);

    /* Check column indices */
    mu_assert("jac->i[0] wrong", jac->i[0] == 0);
    mu_assert("jac->i[1] wrong", jac->i[1] == 1);

    /* Check values: [1/2, 1/4] */
    mu_assert("jac->x[0] wrong", fabs(jac->x[0] - 0.5) < 1e-10);
    mu_assert("jac->x[1] wrong", fabs(jac->x[1] - 0.25) < 1e-10);

    free_problem(prob);

    return 0;
}

#endif /* TEST_PROBLEM_H */
