#include <math.h>
#include <stdio.h>

#include "atoms/affine.h"
#include "atoms/bivariate_full_dom.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/non_elementwise_full_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "problem.h"
#include "subexpr.h"
#include "test_helpers.h"

/* Tests for DAGs in which a node has several parents: every pass must
   evaluate a shared node once (not once per path from the root), and the
   Hessian must still be right. */

/* Objective x_N^2 with x_{k+1} = x_k + h (x_k - x_k^3) + h u_k. Each x_k has
   three parents, so the root is reached by 3^N paths: without per-node guards
   this test does not finish. Variables are (x_0, u_0, ..., u_{N-1}). */
#define SHARED_DAG_N 30
static expr *build_shared_chain(void)
{
    int n_vars = SHARED_DAG_N + 1;
    double h = 0.1;
    expr *h_param = new_parameter(1, 1, PARAM_FIXED, n_vars, &h);
    expr *u = new_variable(SHARED_DAG_N, 1, 1, n_vars);
    expr *x = new_variable(1, 1, 0, n_vars);

    for (int k = 0; k < SHARED_DAG_N; k++)
    {
        expr *drift = new_add(x, new_neg(new_power(x, 3.0)));
        expr *u_k = new_index(u, 1, 1, &k, 1);
        x = new_add(new_add(x, new_scalar_mult(h_param, drift)),
                    new_scalar_mult(h_param, u_k));
    }
    return new_power(x, 2.0);
}

const char *test_shared_dag_chain(void)
{
    double u[SHARED_DAG_N + 1];
    u[0] = 0.5;
    for (int k = 0; k < SHARED_DAG_N; k++)
    {
        u[k + 1] = 0.3 * sin(0.7 * k);
    }
    double w = 1.5;

    expr *obj = build_shared_chain();
    mu_assert("chain jacobian",
              check_jacobian_num(obj, u, NUMERICAL_DIFF_DEFAULT_H));
    mu_assert("chain hessian",
              check_wsum_hess(obj, u, &w, NUMERICAL_DIFF_DEFAULT_H));

    /* a second pass at another point and weight */
    u[0] = -0.4;
    w = -0.7;
    mu_assert("chain hessian, second pass",
              check_wsum_hess(obj, u, &w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(obj);
    return 0;
}

/* g = exp(x) under the atoms that scale their child's Hessian (neg,
   scalar_mult) and as both children of a multiply */
const char *test_shared_dag_scaling_atoms(void)
{
    double u1[3] = {0.1, -0.2, 0.3};
    double u2[3] = {-0.3, 0.4, 0.2};
    double w1[3] = {1.0, -2.0, 0.5};
    double w2[3] = {-0.5, 1.5, 2.0};
    double a = 2.5;

    expr *x = new_variable(3, 1, 0, 3);
    expr *g = new_exp(x);
    expr *a_param = new_parameter(1, 1, PARAM_FIXED, 3, &a);
    expr *e = new_add(new_add(new_neg(g), new_scalar_mult(a_param, g)),
                      new_elementwise_mult(g, g));

    mu_assert("jacobian", check_jacobian_num(e, u1, NUMERICAL_DIFF_DEFAULT_H));
    mu_assert("hessian", check_wsum_hess(e, u1, w1, NUMERICAL_DIFF_DEFAULT_H));
    mu_assert("hessian, second pass",
              check_wsum_hess(e, u2, w2, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(e);
    return 0;
}

/* g = exp(x) under a quad_form (which scales by 2w) and a second parent */
const char *test_shared_dag_quad_form(void)
{
    double u1[3] = {0.1, -0.2, 0.3};
    double u2[3] = {-0.3, 0.4, 0.2};
    double w1 = 1.5;
    double w2 = -0.7;
    double Q[9] = {2.0, 0.5, -1.0, 0.5, 1.0, 0.3, -1.0, 0.3, 3.0};

    expr *x = new_variable(3, 1, 0, 3);
    expr *g = new_exp(x);
    expr *e = new_add(new_quad_form_dense(g, 3, Q, NULL), new_sum(new_sin(g), -1));

    mu_assert("jacobian", check_jacobian_num(e, u1, NUMERICAL_DIFF_DEFAULT_H));
    mu_assert("hessian", check_wsum_hess(e, u1, &w1, NUMERICAL_DIFF_DEFAULT_H));
    mu_assert("hessian, second pass",
              check_wsum_hess(e, u2, &w2, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(e);
    return 0;
}

/* The constraint g = exp(x) is a root of the problem, appears as a constraint
   twice, and also sits inside the objective sum(sin(g)): all its in-edges
   are delivered in every problem_hessian call. Compared with the same problem
   built without any sharing. */
static problem *build_root_problem(bool shared)
{
    expr *g = new_exp(new_variable(3, 1, 0, 3));
    expr *g_obj = shared ? g : new_exp(new_variable(3, 1, 0, 3));
    expr *g_dup = shared ? g : new_exp(new_variable(3, 1, 0, 3));
    expr *constraints[2] = {g, g_dup};
    problem *prob = new_problem(new_sum(new_sin(g_obj), -1), constraints, 2, false);
    problem_init_derivatives(prob);
    return prob;
}

const char *test_shared_dag_root_inside_other_root(void)
{
    double u[2][3] = {{0.1, -0.2, 0.3}, {-0.3, 0.4, 0.2}};
    double obj_w[2] = {1.5, -0.7};
    double w[2][6] = {{1.0, -2.0, 0.5, 0.3, 0.7, -1.1},
                      {-0.5, 1.5, 2.0, 0.9, -0.2, 0.4}};

    problem *shared = build_root_problem(true);
    problem *plain = build_root_problem(false);
    problem *probs[2] = {shared, plain};
    double H[2][9];

    for (int pass = 0; pass < 2; pass++)
    {
        for (int k = 0; k < 2; k++)
        {
            problem_objective_forward(probs[k], u[pass]);
            problem_constraint_forward(probs[k], u[pass]);
            problem_gradient(probs[k]);
            problem_jacobian(probs[k]);
            problem_hessian(probs[k], obj_w[pass], w[pass]);
            memset(H[k], 0, sizeof(H[k]));
            CSR_matrix *L = probs[k]->lagrange_hessian;
            for (int i = 0; i < 3; i++)
            {
                for (int jj = L->p[i]; jj < L->p[i + 1]; jj++)
                {
                    H[k][3 * i + L->i[jj]] += L->x[jj];
                }
            }
        }
        mu_assert("lagrange hessian differs from the unshared problem",
                  cmp_double_array(H[0], H[1], 9));
        mu_assert("lagrange hessian is zero", fabs(H[1][0]) > 1e-3);
    }

    free_problem(shared);
    free_problem(plain);
    return 0;
}
