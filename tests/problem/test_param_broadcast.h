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

/*
 * Tests for param_source being a non-trivial expression tree
 * (broadcast, sum, etc.) rather than a bare parameter node.
 *
 * The key invariant: after problem_update_params, consuming nodes must
 * call param_source->forward() to propagate updated parameter values
 * through any intermediate nodes before reading param_source->value.
 */

/* ------------------------------------------------------------------ */
/* vector_mult(broadcast(parameter), variable)                        */
/*                                                                    */
/* p is (1,3), broadcast ROW to (2,3), x is (2,3) variable.          */
/* broadcast(p) = [p0,p0, p1,p1, p2,p2] in column-major.             */
/* constraint_k = broadcast(p)_k * x_k (elementwise).                */
/* Jacobian = diag(broadcast(p)).                                     */
/* ------------------------------------------------------------------ */
const char *test_param_broadcast_vector_mult(void)
{
    int n = 6;

    expr *x = new_variable(2, 3, 0, n);
    expr *objective = new_sum(x, -1);

    expr *p_param = new_parameter(1, 3, 0, n, NULL);
    expr *p_bcast = new_broadcast(p_param, 2, 3);
    expr *constraint = new_vector_mult(p_bcast, x);

    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    expr *param_nodes[1] = {p_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    double x_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    /* --- theta = [1, 2, 3] ---------------------------------------- */
    /* broadcast = [1,1, 2,2, 3,3]                                    */
    /* constraint = [1*1, 1*2, 2*3, 2*4, 3*5, 3*6]                   */
    /*            = [1, 2, 6, 8, 15, 18]                              */
    double theta1[3] = {1.0, 2.0, 3.0};
    problem_update_params(prob, theta1);
    problem_constraint_forward(prob, x_vals);

    double fwd1[6] = {1.0, 2.0, 6.0, 8.0, 15.0, 18.0};
    mu_assert("bcast vmul fwd1", cmp_double_array(prob->constraint_values, fwd1, 6));

    problem_jacobian(prob);
    double jac1[6] = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0};
    mu_assert("bcast vmul jac1", cmp_double_array(prob->jacobian->x, jac1, 6));

    /* --- theta = [10, 20, 30] ------------------------------------- */
    /* broadcast = [10,10, 20,20, 30,30]                              */
    /* constraint = [10, 20, 60, 80, 150, 180]                        */
    double theta2[3] = {10.0, 20.0, 30.0};
    problem_update_params(prob, theta2);
    problem_constraint_forward(prob, x_vals);

    double fwd2[6] = {10.0, 20.0, 60.0, 80.0, 150.0, 180.0};
    mu_assert("bcast vmul fwd2", cmp_double_array(prob->constraint_values, fwd2, 6));

    problem_jacobian(prob);
    double jac2[6] = {10.0, 10.0, 20.0, 20.0, 30.0, 30.0};
    mu_assert("bcast vmul jac2", cmp_double_array(prob->jacobian->x, jac2, 6));

    free_problem(prob);
    return 0;
}

/* ------------------------------------------------------------------ */
/* scalar_mult(sum(parameter), variable)                              */
/*                                                                    */
/* p is (2,1), sum(p) is (1,1) scalar = p0+p1.                       */
/* x is (3,1) variable.                                               */
/* constraint_i = (p0+p1) * x_i.                                     */
/* Jacobian = diag(p0+p1, p0+p1, p0+p1).                             */
/* ------------------------------------------------------------------ */
const char *test_param_sum_scalar_mult(void)
{
    int n = 3;

    expr *x = new_variable(3, 1, 0, n);
    expr *objective = new_sum(x, -1);

    expr *p_param = new_parameter(2, 1, 0, n, NULL);
    expr *p_sum = new_sum(p_param, -1);
    expr *constraint = new_scalar_mult(p_sum, x);

    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    expr *param_nodes[1] = {p_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    double x_vals[3] = {1.0, 2.0, 3.0};

    /* --- theta = [1, 2], sum(p) = 3 ------------------------------- */
    /* constraint = [3, 6, 9]                                         */
    double theta1[2] = {1.0, 2.0};
    problem_update_params(prob, theta1);
    problem_constraint_forward(prob, x_vals);

    double fwd1[3] = {3.0, 6.0, 9.0};
    mu_assert("sum smul fwd1", cmp_double_array(prob->constraint_values, fwd1, 3));

    problem_jacobian(prob);
    double jac1[3] = {3.0, 3.0, 3.0};
    mu_assert("sum smul jac1", cmp_double_array(prob->jacobian->x, jac1, 3));

    /* --- theta = [5, 10], sum(p) = 15 ----------------------------- */
    /* constraint = [15, 30, 45]                                      */
    double theta2[2] = {5.0, 10.0};
    problem_update_params(prob, theta2);
    problem_constraint_forward(prob, x_vals);

    double fwd2[3] = {15.0, 30.0, 45.0};
    mu_assert("sum smul fwd2", cmp_double_array(prob->constraint_values, fwd2, 3));

    problem_jacobian(prob);
    double jac2[3] = {15.0, 15.0, 15.0};
    mu_assert("sum smul jac2", cmp_double_array(prob->jacobian->x, jac2, 3));

    free_problem(prob);
    return 0;
}

/* ------------------------------------------------------------------ */
/* left_matmul_dense(broadcast(parameter), variable)                  */
/*                                                                    */
/* p is (1,2), broadcast ROW to (3,2).                                */
/* broadcast stores column-major: [a,a,a, b,b,b].                    */
/* refresh_dense_left interprets this as A in column-major, giving    */
/* A = [[a,b],[a,b],[a,b]] (3x2).                                    */
/* x is (2,1) variable.                                               */
/* constraint = A @ x, a (3,1) vector.                                */
/* Jacobian = A (each row = [a, b]).                                  */
/* ------------------------------------------------------------------ */
const char *test_param_broadcast_left_matmul(void)
{
    int n = 2;

    expr *x = new_variable(2, 1, 0, n);
    expr *objective = new_sum(x, -1);

    expr *p_param = new_parameter(1, 2, 0, n, NULL);
    expr *p_bcast = new_broadcast(p_param, 3, 2);

    /* initial data (overwritten on first param refresh) */
    double A_init[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    expr *constraint = new_left_matmul_dense(p_bcast, x, 3, 2, A_init);

    expr *constraints[1] = {constraint};
    problem *prob = new_problem(objective, constraints, 1, false);

    expr *param_nodes[1] = {p_param};
    problem_register_params(prob, param_nodes, 1);
    problem_init_derivatives(prob);

    double x_vals[2] = {3.0, 4.0};
    int Ap[4] = {0, 2, 4, 6};
    int Ai[6] = {0, 1, 0, 1, 0, 1};

    /* --- theta = [1, 2] ------------------------------------------- */
    /* A = [[1,2],[1,2],[1,2]], A@x = [11, 11, 11]                    */
    double theta1[2] = {1.0, 2.0};
    problem_update_params(prob, theta1);
    problem_constraint_forward(prob, x_vals);

    double fwd1[3] = {11.0, 11.0, 11.0};
    mu_assert("bcast lmul fwd1", cmp_double_array(prob->constraint_values, fwd1, 3));

    problem_jacobian(prob);
    double jac1[6] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    mu_assert("bcast lmul jac1", cmp_double_array(prob->jacobian->x, jac1, 6));
    mu_assert("bcast lmul rows1", cmp_int_array(prob->jacobian->p, Ap, 4));
    mu_assert("bcast lmul cols1", cmp_int_array(prob->jacobian->i, Ai, 6));

    /* --- theta = [5, 10] ------------------------------------------ */
    /* A = [[5,10],[5,10],[5,10]], A@x = [55, 55, 55]                 */
    double theta2[2] = {5.0, 10.0};
    problem_update_params(prob, theta2);
    problem_constraint_forward(prob, x_vals);

    double fwd2[3] = {55.0, 55.0, 55.0};
    mu_assert("bcast lmul fwd2", cmp_double_array(prob->constraint_values, fwd2, 3));

    problem_jacobian(prob);
    double jac2[6] = {5.0, 10.0, 5.0, 10.0, 5.0, 10.0};
    mu_assert("bcast lmul jac2", cmp_double_array(prob->jacobian->x, jac2, 6));
    mu_assert("bcast lmul rows2", cmp_int_array(prob->jacobian->p, Ap, 4));
    mu_assert("bcast lmul cols2", cmp_int_array(prob->jacobian->i, Ai, 6));

    free_problem(prob);
    return 0;
}

#endif /* TEST_PARAM_BROADCAST_H */
