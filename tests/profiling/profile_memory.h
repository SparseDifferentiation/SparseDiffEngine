#ifndef PROFILE_MEMORY_H
#define PROFILE_MEMORY_H

#include <stdlib.h>

#include "atoms/affine.h"
#include "atoms/bivariate_full_dom.h"
#include "atoms/elementwise_full_dom.h"
#include "expr.h"
#include "problem.h"

/* Reproduces the Python test
       n = 100, rank = 10
       A in R^{n x n}, B, C in R^{n x rank}
       cost = sum(exp(A @ (B @ C.T)))
   and lets free_problem print the SparseDiff banner so we can compare
   the C "Peak memory" number against the Python report. Bounds on B and C
   from the Python snippet are solver-level, so they are omitted here
   (matching the "Number of constraints: 0" line in the Python output). */
const char *profile_memory(void)
{
    const int n = 100;
    const int rank = 10;
    const int n_vars = 2 * n * rank;
    srand(0);

    double *A_data = (double *) malloc((size_t) n * n * sizeof(double));
    double *u = (double *) malloc((size_t) n_vars * sizeof(double));
    for (int k = 0; k < n * n; k++)
    {
        A_data[k] = ((double) rand() / (double) RAND_MAX) - 0.5;
    }
    for (int k = 0; k < n_vars; k++)
    {
        u[k] = ((double) rand() / (double) RAND_MAX) - 0.5;
    }

    expr *B = new_variable(n, rank, 0, n_vars);
    expr *C = new_variable(n, rank, n * rank, n_vars);
    expr *CT = new_transpose(C);
    expr *BCT = new_matmul(B, CT);
    expr *AX = new_left_matmul_dense(NULL, BCT, n, n, A_data);
    expr *e = new_exp(AX);
    expr *cost = new_sum(e, -1);

    problem *prob = new_problem(cost, NULL, 0, /*verbose=*/true);

    problem_init_derivatives(prob);
    problem_objective_forward(prob, u);
    problem_gradient(prob);
    problem_hessian(prob, 1.0, NULL);

    free_problem(prob);
    free(A_data);
    free(u);
    return 0;
}

#endif /* PROFILE_MEMORY_H */
