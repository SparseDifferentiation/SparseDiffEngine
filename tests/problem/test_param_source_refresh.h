#ifndef TEST_PARAM_SOURCE_REFRESH_H
#define TEST_PARAM_SOURCE_REFRESH_H

/* Composite param_source refresh across problem_update_params.
 *
 * A param_source subtree is evaluated by the owning atom's gated recursion,
 * not the main forward walk, so gated nodes *inside* a composite source
 * (promote, nested mults, cached matmuls) are only re-evaluated if the owner
 * marks the subtree before forwarding it. These tests pin the shapes that
 * served stale values before that mark existed:
 *
 *   1. left_matmul whose coefficient is p (.) A     ("(p*A) @ x")
 *   2. quad_form whose matrix is g (.) Sigma        ("quad_form(x, g*Sig)")
 *   3. two gated levels: coefficient (c*p) (.) A    ("((2p)*A) @ x")
 *   4. kron whose left operand is p (.) A           ("kron(p*A, X)")
 *
 * Bare-parameter sources are covered by test_param_prob.h; these are the
 * composite counterparts.
 */

#include <math.h>
#include <stdio.h>

#include "atoms/affine.h"
#include "atoms/non_elementwise_full_dom.h"
#include "expr.h"
#include "minunit.h"
#include "problem.h"
#include "subexpr.h"
#include "test_helpers.h"

/* (p * A) @ x: the elementwise product of a promoted scalar parameter with a
   constant matrix feeds left_matmul as its param_source. */
const char *test_composite_source_left_matmul(void)
{
    int n = 2;

    expr *x = new_variable(2, 1, 0, n);
    expr *objective = new_sum(x, -1);

    double theta[1] = {2.0};
    expr *p = new_parameter(1, 1, 0, n, theta);
    /* column-major A = [[1,2],[3,4]] */
    double A_vals[4] = {1.0, 3.0, 2.0, 4.0};
    expr *A_const = new_parameter(2, 2, PARAM_FIXED, n, A_vals);
    expr *coeff = new_vector_mult(new_promote(p, 2, 2), A_const);

    expr *constraint = new_left_matmul_dense(coeff, x, 2, 2, NULL);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    expr *param_nodes[1] = {p};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    double x_vals[2] = {1.0, 2.0};
    int Ap[3] = {0, 2, 4};
    int Ai[4] = {0, 1, 0, 1};

    /* p = 2: effective A is 2*A, so A@x = 2*[5,11] */
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    double constrs[2] = {10.0, 22.0};
    double Ax[4] = {2.0, 4.0, 6.0, 8.0};
    mu_assert("initial vals fail",
              cmp_double_array(prob->constraint_values, constrs, 2));
    mu_assert("initial jac fail", cmp_double_array(prob->jacobian->x, Ax, 4));
    mu_assert("rows fail", cmp_int_array(prob->jacobian->p, Ap, 3));
    mu_assert("cols fail", cmp_int_array(prob->jacobian->i, Ai, 4));

    /* p = 10: the promote inside the source must re-evaluate */
    theta[0] = 10.0;
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    constrs[0] = 50.0;
    constrs[1] = 110.0;
    Ax[0] = 10.0;
    Ax[1] = 20.0;
    Ax[2] = 30.0;
    Ax[3] = 40.0;
    mu_assert("stale constraint values after update",
              cmp_double_array(prob->constraint_values, constrs, 2));
    mu_assert("stale jacobian values after update",
              cmp_double_array(prob->jacobian->x, Ax, 4));

    free_problem(prob);

    return 0;
}

/* quad_form(x, g * Sigma): the scaled constant matrix feeds quad_form's
   cached Q through its param_source. */
const char *test_composite_source_quad_form(void)
{
    int n = 2;

    expr *x = new_variable(2, 1, 0, n);

    double theta[1] = {1.0};
    expr *g = new_parameter(1, 1, 0, n, theta);
    /* Sigma = diag(2, 3); symmetric, so major order is irrelevant */
    double S_vals[4] = {2.0, 0.0, 0.0, 3.0};
    expr *S_const = new_parameter(2, 2, PARAM_FIXED, n, S_vals);
    expr *Q_src = new_vector_mult(new_promote(g, 2, 2), S_const);

    expr *objective = new_quad_form_dense(x, 2, NULL, Q_src);
    problem *prob = new_problem(objective, NULL, 0, false);

    expr *param_nodes[1] = {g};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    double x_vals[2] = {1.0, 2.0};

    /* g = 1: x'(g Sigma)x = 2*1 + 3*4 = 14; gradient 2(g Sigma)x = [4, 12] */
    double obj_val = problem_objective_forward(prob, x_vals);
    problem_gradient(prob);
    double grad[2] = {4.0, 12.0};
    mu_assert("initial obj fail", fabs(obj_val - 14.0) < 1e-10);
    mu_assert("initial grad fail", cmp_double_array(prob->gradient_values, grad, 2));

    /* g = 5: Q's cached values must be re-copied from the re-evaluated source */
    theta[0] = 5.0;
    problem_update_params(prob, theta);
    obj_val = problem_objective_forward(prob, x_vals);
    problem_gradient(prob);
    grad[0] = 20.0;
    grad[1] = 60.0;
    mu_assert("stale objective after update", fabs(obj_val - 70.0) < 1e-10);
    mu_assert("stale gradient after update",
              cmp_double_array(prob->gradient_values, grad, 2));

    free_problem(prob);

    return 0;
}

/* ((c * p) * A) @ x: two gated levels inside the source (scalar_mult under
   promote under vector_mult) — the mark must recurse through all of them. */
const char *test_composite_source_nested_gates(void)
{
    int n = 2;

    expr *x = new_variable(2, 1, 0, n);
    expr *objective = new_sum(x, -1);

    double theta[1] = {1.0};
    expr *p = new_parameter(1, 1, 0, n, theta);
    double c_val = 2.0;
    expr *c_const = new_parameter(1, 1, PARAM_FIXED, n, &c_val);
    expr *scaled = new_scalar_mult(p, c_const); /* = c * p, a gated node */
    /* column-major A = [[1,2],[3,4]] */
    double A_vals[4] = {1.0, 3.0, 2.0, 4.0};
    expr *A_const = new_parameter(2, 2, PARAM_FIXED, n, A_vals);
    expr *coeff = new_vector_mult(new_promote(scaled, 2, 2), A_const);

    expr *constraint = new_left_matmul_dense(coeff, x, 2, 2, NULL);
    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    expr *param_nodes[1] = {p};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    double x_vals[2] = {1.0, 2.0};

    /* p = 1: effective A is 2*A */
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    double constrs[2] = {10.0, 22.0};
    double Ax[4] = {2.0, 4.0, 6.0, 8.0};
    mu_assert("initial vals fail",
              cmp_double_array(prob->constraint_values, constrs, 2));
    mu_assert("initial jac fail", cmp_double_array(prob->jacobian->x, Ax, 4));

    /* p = 5: effective A is 10*A; both gated levels must re-evaluate */
    theta[0] = 5.0;
    problem_update_params(prob, theta);
    problem_constraint_forward(prob, x_vals);
    problem_jacobian(prob);
    constrs[0] = 50.0;
    constrs[1] = 110.0;
    Ax[0] = 10.0;
    Ax[1] = 20.0;
    Ax[2] = 30.0;
    Ax[3] = 40.0;
    mu_assert("stale constraint values after update",
              cmp_double_array(prob->constraint_values, constrs, 2));
    mu_assert("stale jacobian values after update",
              cmp_double_array(prob->jacobian->x, Ax, 4));

    free_problem(prob);

    return 0;
}

/* kron(p * A, X): the scaled constant operand feeds the kron atom's
   coefficient values through its param_source. */
const char *test_composite_source_kron(void)
{
    int n = 4;

    expr *X = new_variable(2, 2, 0, n);

    double theta[1] = {2.0};
    expr *p = new_parameter(1, 1, 0, n, theta);
    /* column-major A = [[1,2],[3,4]]; all four blocks active */
    double A_vals[4] = {1.0, 3.0, 2.0, 4.0};
    expr *A_const = new_parameter(2, 2, PARAM_FIXED, n, A_vals);
    expr *coeff = new_vector_mult(new_promote(p, 2, 2), A_const);
    int active[4] = {0, 1, 2, 3};

    expr *Z = new_left_kron(coeff, X, 2, 2, 2, 2, active, 4);
    expr *objective = new_sum(Z, -1);
    problem *prob = new_problem(objective, NULL, 0, false);

    expr *param_nodes[1] = {p};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    double x_vals[4] = {1.0, 1.0, 1.0, 1.0};

    /* p = 2: sum(kron(2A, ones)) = 2 * sum(A) * 4 = 80;
       gradient: each X entry sees sum(2A) = 20 */
    double obj_val = problem_objective_forward(prob, x_vals);
    problem_gradient(prob);
    double grad[4] = {20.0, 20.0, 20.0, 20.0};
    mu_assert("initial obj fail", fabs(obj_val - 80.0) < 1e-10);
    mu_assert("initial grad fail", cmp_double_array(prob->gradient_values, grad, 4));

    /* p = 10: the promote and mult inside the operand must re-evaluate */
    theta[0] = 10.0;
    problem_update_params(prob, theta);
    obj_val = problem_objective_forward(prob, x_vals);
    problem_gradient(prob);
    grad[0] = grad[1] = grad[2] = grad[3] = 100.0;
    mu_assert("stale objective after update", fabs(obj_val - 400.0) < 1e-10);
    mu_assert("stale gradient after update",
              cmp_double_array(prob->gradient_values, grad, 4));

    free_problem(prob);

    return 0;
}

#endif /* TEST_PARAM_SOURCE_REFRESH_H */
