#ifndef TEST_PARAM_PROB_H
#define TEST_PARAM_PROB_H

#include <math.h>
#include <stdio.h>

#include "affine.h"
#include "bivariate.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "problem.h"
#include "test_helpers.h"

/*
 * Test 1: param_scalar_mult in objective
 *
 * Problem: minimize a * sum(log(x)), no constraints, x size 2
 *   a is a scalar parameter (param_id=0)
 *
 * At x=[1,2], a=3:
 *   obj = 3*(log(1)+log(2)) = 3*log(2)
 *   gradient = [3/1, 3/2] = [3.0, 1.5]
 *
 * After update a=5:
 *   obj = 5*log(2)
 *   gradient = [5.0, 2.5]
 */
const char *test_param_scalar_mult_problem(void)
{
    int n_vars = 2;

    /* Build tree: sum(a * log(x)) */
    expr *x = new_variable(2, 1, 0, n_vars);
    expr *log_x = new_log(x);
    expr *a_param = new_parameter(1, 1, 0, n_vars, NULL);
    expr *scaled = new_scalar_mult(a_param, log_x);
    expr *objective = new_sum(scaled, -1);

    /* Create problem (no constraints) */
    problem *prob = new_problem(objective, NULL, 0, true);

    /* Register parameter */
    expr *param_nodes[1] = {a_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* Set a=3 and evaluate at x=[1,2] */
    double theta[1] = {3.0};
    problem_update_params(prob, theta);

    double u[2] = {1.0, 2.0};
    double obj_val = problem_objective_forward(prob, u);
    problem_gradient(prob);

    double expected_obj = 3.0 * log(2.0);
    mu_assert("obj wrong (a=3)", fabs(obj_val - expected_obj) < 1e-10);

    double expected_grad[2] = {3.0, 1.5};
    mu_assert("gradient wrong (a=3)",
              cmp_double_array(prob->gradient_values, expected_grad, 2));

    /* Update a=5 and re-evaluate */
    theta[0] = 5.0;
    problem_update_params(prob, theta);

    obj_val = problem_objective_forward(prob, u);
    problem_gradient(prob);

    expected_obj = 5.0 * log(2.0);
    mu_assert("obj wrong (a=5)", fabs(obj_val - expected_obj) < 1e-10);

    double expected_grad2[2] = {5.0, 2.5};
    mu_assert("gradient wrong (a=5)",
              cmp_double_array(prob->gradient_values, expected_grad2, 2));

    free_problem(prob);

    return 0;
}

/*
 * Test 2: param_vector_mult in constraint
 *
 * Problem: minimize sum(x), subject to p ∘ x, x size 2
 *   p is a vector parameter of size 2 (param_id=0)
 *
 * At x=[1,2], p=[3,4]:
 *   constraint_values = [3, 8]
 *   jacobian = diag([3, 4])
 *
 * After update p=[5,6]:
 *   constraint_values = [5, 12]
 *   jacobian = diag([5, 6])
 */
const char *test_param_vector_mult_problem(void)
{
    int n_vars = 2;

    /* Objective: sum(x) */
    expr *x_obj = new_variable(2, 1, 0, n_vars);
    expr *objective = new_sum(x_obj, -1);

    /* Constraint: p ∘ x */
    expr *x_con = new_variable(2, 1, 0, n_vars);
    expr *p_param = new_parameter(2, 1, 0, n_vars, NULL);
    expr *constraint = new_vector_mult(p_param, x_con);

    expr *constraints[1] = {constraint};

    /* Create problem */
    problem *prob = new_problem(objective, constraints, 1, true);

    expr *param_nodes[1] = {p_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* Set p=[3,4] and evaluate at x=[1,2] */
    double theta[2] = {3.0, 4.0};
    problem_update_params(prob, theta);

    double u[2] = {1.0, 2.0};
    problem_constraint_forward(prob, u);
    problem_jacobian(prob);

    double expected_cv[2] = {3.0, 8.0};
    mu_assert("constraint values wrong (p=[3,4])",
              cmp_double_array(prob->constraint_values, expected_cv, 2));

    CSR_Matrix *jac = prob->jacobian;
    mu_assert("jac rows wrong", jac->m == 2);
    mu_assert("jac cols wrong", jac->n == 2);

    int expected_p[3] = {0, 1, 2};
    mu_assert("jac->p wrong (p=[3,4])", cmp_int_array(jac->p, expected_p, 3));

    int expected_i[2] = {0, 1};
    mu_assert("jac->i wrong (p=[3,4])", cmp_int_array(jac->i, expected_i, 2));

    double expected_x[2] = {3.0, 4.0};
    mu_assert("jac->x wrong (p=[3,4])", cmp_double_array(jac->x, expected_x, 2));

    /* Update p=[5,6] and re-evaluate */
    double theta2[2] = {5.0, 6.0};
    problem_update_params(prob, theta2);

    problem_constraint_forward(prob, u);
    problem_jacobian(prob);

    double expected_cv2[2] = {5.0, 12.0};
    mu_assert("constraint values wrong (p=[5,6])",
              cmp_double_array(prob->constraint_values, expected_cv2, 2));

    double expected_x2[2] = {5.0, 6.0};
    mu_assert("jac->x wrong (p=[5,6])", cmp_double_array(jac->x, expected_x2, 2));

    free_problem(prob);

    return 0;
}

/*
 * Test 3: left_param_matmul in constraint
 *
 * Problem: minimize sum(x), subject to A @ x, x size 2, A is 2x2
 *   A is a 2x2 matrix parameter (param_id=0, size=4, column-major)
 *   A = [[1,2],[3,4]] → column-major theta = [1,3,2,4]
 *
 * At x=[1,2]:
 *   constraint_values = [1*1+2*2, 3*1+4*2] = [5, 11]
 *   jacobian = [[1,2],[3,4]]
 *
 * After update A = [[5,6],[7,8]] → theta = [5,7,6,8]:
 *   constraint_values = [5*1+6*2, 7*1+8*2] = [17, 23]
 *   jacobian = [[5,6],[7,8]]
 */
const char *test_param_left_matmul_problem(void)
{
    int n_vars = 2;

    /* Objective: sum(x) */
    expr *x_obj = new_variable(2, 1, 0, n_vars);
    expr *objective = new_sum(x_obj, -1);

    /* Constraint: A @ x */
    expr *x_con = new_variable(2, 1, 0, n_vars);
    expr *A_param = new_parameter(2, 2, 0, n_vars, NULL);
    expr *constraint = new_left_matmul(A_param, x_con);

    expr *constraints[1] = {constraint};

    /* Create problem */
    problem *prob = new_problem(objective, constraints, 1, true);

    expr *param_nodes[1] = {A_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    /* Set A = [[1,2],[3,4]], column-major: [1,3,2,4] */
    double theta[4] = {1.0, 3.0, 2.0, 4.0};
    problem_update_params(prob, theta);

    double u[2] = {1.0, 2.0};
    problem_constraint_forward(prob, u);
    problem_jacobian(prob);

    double expected_cv[2] = {5.0, 11.0};
    mu_assert("constraint values wrong (A1)",
              cmp_double_array(prob->constraint_values, expected_cv, 2));

    CSR_Matrix *jac = prob->jacobian;
    mu_assert("jac rows wrong", jac->m == 2);
    mu_assert("jac cols wrong", jac->n == 2);

    /* Dense jacobian = [[1,2],[3,4]], CSR: row 0 → cols 0,1 vals 1,2;
     *                                      row 1 → cols 0,1 vals 3,4 */
    int expected_p[3] = {0, 2, 4};
    mu_assert("jac->p wrong (A1)", cmp_int_array(jac->p, expected_p, 3));

    int expected_i[4] = {0, 1, 0, 1};
    mu_assert("jac->i wrong (A1)", cmp_int_array(jac->i, expected_i, 4));

    double expected_x[4] = {1.0, 2.0, 3.0, 4.0};
    mu_assert("jac->x wrong (A1)", cmp_double_array(jac->x, expected_x, 4));

    /* Update A = [[5,6],[7,8]], column-major: [5,7,6,8] */
    double theta2[4] = {5.0, 7.0, 6.0, 8.0};
    problem_update_params(prob, theta2);

    problem_constraint_forward(prob, u);
    problem_jacobian(prob);

    double expected_cv2[2] = {17.0, 23.0};
    mu_assert("constraint values wrong (A2)",
              cmp_double_array(prob->constraint_values, expected_cv2, 2));

    double expected_x2[4] = {5.0, 6.0, 7.0, 8.0};
    mu_assert("jac->x wrong (A2)", cmp_double_array(jac->x, expected_x2, 4));

    free_problem(prob);

    return 0;
}

#endif /* TEST_PARAM_PROB_H */
