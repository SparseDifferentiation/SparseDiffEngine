#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "atoms/affine.h"
#include "atoms/bivariate_full_dom.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"
#include "utils/Timer.h"

static void profile_expr(expr *node, double *x_vals, const char *label)
{
    Timer timer;

    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    node->forward(node, x_vals);
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    printf("%s forward time:       %8.3f seconds\n", label,
           GET_ELAPSED_SECONDS(timer));

    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    jacobian_init(node);
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    printf("%s jacobian init time: %8.3f seconds\n", label,
           GET_ELAPSED_SECONDS(timer));

    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    node->eval_jacobian(node);
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    printf("%s jacobian eval time: %8.3f seconds\n", label,
           GET_ELAPSED_SECONDS(timer));

    double *w = (double *) malloc(node->size * sizeof(double));
    for (int i = 0; i < node->size; i++)
    {
        w[i] = 1.0;
    }

    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    wsum_hess_init(node);
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    printf("%s hessian init time:  %8.3f seconds\n", label,
           GET_ELAPSED_SECONDS(timer));

    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    node->eval_wsum_hess(node, w);
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    printf("%s hessian eval time:  %8.3f seconds\n", label,
           GET_ELAPSED_SECONDS(timer));

    free(w);
}

const char *profile_left_matmul(void)
{
    int n = 100;
    expr *X = new_variable(n, n, 0, n * n);

    double *x_vals = (double *) malloc(n * n * sizeof(double));
    double *A_data = (double *) malloc(n * n * sizeof(double));
    for (int i = 0; i < n * n; i++)
    {
        x_vals[i] = 1.0;
        A_data[i] = 1.0;
    }

    /* --- sparse CSR path --- */
    CSR_Matrix *A = new_csr_matrix(n, n, n * n);
    memcpy(A->x, A_data, n * n * sizeof(double));
    for (int row = 0; row < n; row++)
    {
        A->p[row] = row * n;
        for (int col = 0; col < n; col++)
        {
            A->i[row * n + col] = col;
        }
    }
    A->p[n] = n * n;

    expr *AX_sparse = new_left_matmul(X, A);
    free_csr_matrix(A);
    profile_expr(AX_sparse, x_vals, "left_matmul        ");

    /* --- dense path --- */
    expr *AX_dense = new_left_matmul_dense(X, n, n, A_data);
    profile_expr(AX_dense, x_vals, "left_matmul_dense  ");

    free(A_data);
    free(x_vals);
    free_expr(AX_sparse);
    free_expr(AX_dense);
    return 0;
}

const char *profile_matmul_lstsq(void)
{
    /* sum(power(A - matmul(X, Y), 2))
       X is m x k, Y is k x n, A is m x n */
    int m = 200, k = 10, n = 100;
    int n_vars = m * k + k * n;

    expr *X = new_variable(m, k, 0, n_vars);
    expr *Y = new_variable(k, n, m * k, n_vars);
    expr *XY = new_matmul(X, Y);
    expr *sq = new_power(XY, 2.0);

    double *u = (double *) calloc(n_vars, sizeof(double));
    for (int i = 0; i < n_vars; i++)
    {
        u[i] = 0.1 * (i + 1);
    }

    profile_expr(sq, u, "matmul_lstsq       ");

    free(u);
    return 0;
}
