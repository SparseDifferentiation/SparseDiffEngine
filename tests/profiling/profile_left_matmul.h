#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"
#include "bivariate.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "minunit.h"
#include "subexpr.h"
#include "test_helpers.h"
#include "utils/Timer.h"

const char *profile_left_matmul()
{
    /* A @ X where A is 100 x 100 dense (all ones) and X is 100 x 100 variable */
    int n = 100;
    expr *X = new_variable(n, n, 0, n * n);

    /* Create n x n parameter of all ones (column-major, but all ones so order
     * doesn't matter) */
    double *A_vals = (double *) malloc(n * n * sizeof(double));
    for (int i = 0; i < n * n; i++)
    {
        A_vals[i] = 1.0;
    }
    expr *A_param = new_parameter(n, n, PARAM_FIXED, n, A_vals);
    free(A_vals);

    expr *AX = new_left_matmul(A_param, X);

    double *x_vals = (double *) malloc(n * n * sizeof(double));
    for (int i = 0; i < n * n; i++)
    {
        x_vals[i] = 1.0;
    }

    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    AX->forward(AX, x_vals);
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    printf("left_matmul forward time: %8.3f seconds\n", GET_ELAPSED_SECONDS(timer));
    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    AX->jacobian_init(AX);
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    printf("left_matmul jacobian init time: %8.3f seconds\n",
           GET_ELAPSED_SECONDS(timer));
    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    AX->eval_jacobian(AX);
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    printf("left_matmul jacobian eval time: %8.3f seconds\n",
           GET_ELAPSED_SECONDS(timer));

    free(x_vals);
    free_expr(AX);
    return 0;
}
