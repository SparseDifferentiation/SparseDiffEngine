#ifndef PROFILE_LASSO_H
#define PROFILE_LASSO_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "expr.h"
#include "minunit.h"
#include "problem.h"
#include "subexpr.h"
#include "utils/Timer.h"

/* Dense lasso over a lambda path.
 *
 *   variables:   x (n), t (n)            -> n_vars = 2n
 *   objective:   sum((A x - b)^2) + lam * sum(t)
 *   constraints: x - t <= 0,  -x - t <= 0
 *
 * lam is the ONLY registered parameter. A is a fixed dense buffer, so the
 * whole A@x subtree is parameter-free -- the case the has_params prune is
 * meant to exploit. Reports forward / gradient / jacobian time separately so
 * the dense-Jacobian cost is not hidden behind the dgemv.
 */
const char *profile_lasso(void)
{
    const int m = 2000;
    const int n = 785;
    const int nv = 2 * n;
    const int n_sweep = 50;

    double *A_data = (double *) malloc((size_t) m * n * sizeof(double));
    double *negb = (double *) malloc((size_t) m * sizeof(double));
    double *u = (double *) malloc((size_t) nv * sizeof(double));
    srand(42);
    for (int i = 0; i < m * n; i++) A_data[i] = (double) rand() / RAND_MAX - 0.5;
    for (int i = 0; i < m; i++) negb[i] = (double) rand() / RAND_MAX - 0.5;
    for (int i = 0; i < nv; i++) u[i] = (double) rand() / RAND_MAX - 0.5;

    /* ---- objective: sum((Ax - b)^2) + lam*sum(t) ---- */
    expr *x = new_variable(n, 1, 0, nv);
    expr *t = new_variable(n, 1, n, nv);

    expr *Ax = new_left_matmul_dense(NULL, x, m, n, A_data);
    expr *b_const = new_parameter(m, 1, PARAM_FIXED, nv, negb);
    expr *resid = new_add(Ax, b_const);
    expr *sq = new_power(resid, 2.0);
    expr *ssq = new_sum(sq, -1);

    double lam0 = 1.0;
    expr *lam = new_parameter(1, 1, 0, nv, &lam0);
    expr *sum_t = new_sum(t, -1);
    expr *pen = new_scalar_mult(lam, sum_t);

    expr *objective = new_add(ssq, pen);

    /* ---- constraints: x - t <= 0, -x - t <= 0 ---- */
    expr *c1 = new_add(x, new_neg(t));
    expr *c2 = new_add(new_neg(x), new_neg(t));
    expr *constraints[2] = {c1, c2};

    problem *prob = new_problem(objective, constraints, 2, false);
    expr *param_nodes[1] = {lam};
    problem_register_params(prob, param_nodes, 1);

    Timer t_init;
    clock_gettime(CLOCK_MONOTONIC, &t_init.start);
    problem_init_derivatives(prob);
    clock_gettime(CLOCK_MONOTONIC, &t_init.end);
    double sec_init = GET_ELAPSED_SECONDS(t_init);

    /* ---- lambda path ---- */
    Timer t_upd, t_fwd, t_grad, t_jac;
    double sec_upd = 0, sec_fwd = 0, sec_grad = 0, sec_jac = 0;
    double checksum = 0.0;

    for (int k = 0; k < n_sweep; k++)
    {
        double theta[1] = {0.01 + 0.02 * k};

        clock_gettime(CLOCK_MONOTONIC, &t_upd.start);
        problem_update_params(prob, theta);
        clock_gettime(CLOCK_MONOTONIC, &t_upd.end);
        sec_upd += GET_ELAPSED_SECONDS(t_upd);

        clock_gettime(CLOCK_MONOTONIC, &t_fwd.start);
        checksum += problem_objective_forward(prob, u);
        problem_constraint_forward(prob, u);
        clock_gettime(CLOCK_MONOTONIC, &t_fwd.end);
        sec_fwd += GET_ELAPSED_SECONDS(t_fwd);

        clock_gettime(CLOCK_MONOTONIC, &t_grad.start);
        problem_gradient(prob);
        clock_gettime(CLOCK_MONOTONIC, &t_grad.end);
        sec_grad += GET_ELAPSED_SECONDS(t_grad);

        clock_gettime(CLOCK_MONOTONIC, &t_jac.start);
        problem_jacobian(prob);
        clock_gettime(CLOCK_MONOTONIC, &t_jac.end);
        sec_jac += GET_ELAPSED_SECONDS(t_jac);

        checksum += prob->gradient_values[0] + prob->jacobian->x[0];
    }

    printf("\n  [lasso] m=%d n=%d n_vars=%d sweep=%d\n", m, n, nv, n_sweep);
    printf("  [lasso] jacobian nnz            = %d\n", prob->jacobian->nnz);
    printf("  [lasso] objective jac nnz       = %d\n", objective->jacobian->nnz);
    printf("  [lasso] Ax jac nnz              = %d\n", Ax->jacobian->nnz);
    printf("  [lasso] init_derivatives        = %.4f s\n", sec_init);
    printf("  [lasso] update_params  (total)  = %.4f s  (%.3f ms/iter)\n", sec_upd,
           1e3 * sec_upd / n_sweep);
    printf("  [lasso] forward        (total)  = %.4f s  (%.3f ms/iter)\n", sec_fwd,
           1e3 * sec_fwd / n_sweep);
    printf("  [lasso] gradient       (total)  = %.4f s  (%.3f ms/iter)\n", sec_grad,
           1e3 * sec_grad / n_sweep);
    printf("  [lasso] jacobian       (total)  = %.4f s  (%.3f ms/iter)\n", sec_jac,
           1e3 * sec_jac / n_sweep);
    printf("  [lasso] checksum                = %.6f\n", checksum);

    /* free_problem releases objective + constraints (it retained them). */
    free_problem(prob);
    free(A_data);
    free(negb);
    free(u);

    return 0;
}

#endif /* PROFILE_LASSO_H */
