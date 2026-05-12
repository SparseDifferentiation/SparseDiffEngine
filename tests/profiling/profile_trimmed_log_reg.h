#ifndef PROFILE_TRIMMED_LOG_REG_H
#define PROFILE_TRIMMED_LOG_REG_H

#include <stdio.h>
#include <stdlib.h>

#include "atoms/affine.h"
#include "atoms/bivariate_full_dom.h"
#include "atoms/elementwise_full_dom.h"
#include "expr.h"
#include "minunit.h"
#include "subexpr.h"
#include "utils/Timer.h"

/* Profile Jacobian + Hessian of:
     obj = sum( w ∘ logistic( -(y ∘ (A·theta)) ) )

   theta (n x 1), w (m x 1)  : variables (n_vars = n + m)
   A     (m x n)             : dense constant
   y     (m x 1)             : constant in {-1, +1}, wrapped as PARAM_FIXED

   Forward pass is excluded from timing. */
const char *profile_trimmed_log_reg(void)
{
    int m = 2000;
    int n = 785;
    int N_HESS_ITERS = 10;
    int n_vars = n + m;

    /* ---- Random inputs ---- */
    srand(42);
    double *A_data = (double *) malloc((size_t) m * n * sizeof(double));
    double *y_data = (double *) malloc((size_t) m * sizeof(double));
    double *u = (double *) malloc((size_t) n_vars * sizeof(double));
    for (int i = 0; i < m * n; i++)
    {
        A_data[i] = (double) rand() / RAND_MAX - 0.5;
    }
    for (int i = 0; i < m; i++)
    {
        y_data[i] = (rand() % 2 == 0) ? 1.0 : -1.0;
    }
    for (int i = 0; i < n_vars; i++)
    {
        u[i] = (double) rand() / RAND_MAX - 0.5;
    }

    /* ---- Build expression DAG ---- */
    expr *theta = new_variable(n, 1, 0, n_vars);
    expr *w = new_variable(m, 1, n, n_vars);

    expr *y_param = new_parameter(m, 1, PARAM_FIXED, n_vars, y_data);

    expr *A_theta = new_left_matmul_dense(NULL, theta, m, n, A_data);
    expr *y_A_theta = new_vector_mult(y_param, A_theta);
    expr *neg_node = new_neg(y_A_theta);
    expr *sig = new_logistic(neg_node);
    expr *w_sig = new_elementwise_mult(w, sig);
    expr *obj = new_sum(w_sig, -1);

    jacobian_init(obj);
    wsum_hess_init(obj);

    /* Forward (untimed). */
    obj->forward(obj, u);

    double w_one = 1.0;
    Timer t_jac, t_hess;
    clock_gettime(CLOCK_MONOTONIC, &t_jac.start);
    obj->eval_jacobian(obj);
    clock_gettime(CLOCK_MONOTONIC, &t_jac.end);

    obj->eval_wsum_hess(obj, &w_one); /* warm-up */
    clock_gettime(CLOCK_MONOTONIC, &t_hess.start);
    for (int it = 0; it < N_HESS_ITERS; it++)
    {
        obj->eval_wsum_hess(obj, &w_one);
    }
    clock_gettime(CLOCK_MONOTONIC, &t_hess.end);

    double sec_jac = GET_ELAPSED_SECONDS(t_jac);
    double sec_hess = GET_ELAPSED_SECONDS(t_hess) / N_HESS_ITERS;

    printf("\n");
    printf("trimmed_log_reg (m=%d, n=%d):\n", m, n);
    printf("  jacobian = %10.6fs   hessian = %10.6fs  (avg over %d)\n", sec_jac,
           sec_hess, N_HESS_ITERS);

    free_expr(obj);
    free(A_data);
    free(y_data);
    free(u);
    return 0;
}

#endif /* PROFILE_TRIMMED_LOG_REG_H */
