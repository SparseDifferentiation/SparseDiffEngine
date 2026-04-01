#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "expr.h"
#include "utils/CSR_Matrix.h"
#include "utils/Timer.h"

#define EPSILON 1e-7

#define ABS_TOL 1e-6
#define REL_TOL 1e-6

int is_equal_double(double a, double b)
{
    return fabs(a - b) <= fmax(ABS_TOL, REL_TOL * fmax(fabs(a), fabs(b)));
}

int cmp_double_array(const double *actual, const double *expected, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (!is_equal_double(actual[i], expected[i]))
        {
            printf("  FAILED: actual[%d] = %f, expected %f\n", i, actual[i],
                   expected[i]);
            return 0;
        }
    }
    return 1;
}

int cmp_int_array(const int *actual, const int *expected, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (actual[i] != expected[i])
        {
            printf("  FAILED: actual[%d] = %d, expected %d\n", i, actual[i],
                   expected[i]);
            return 0;
        }
    }
    return 1;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Standard normal via Box-Muller transform */
static double randn(void)
{
    double u1 = ((double) rand() + 1.0) / ((double) RAND_MAX + 1.0);
    double u2 = ((double) rand() + 1.0) / ((double) RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

CSR_Matrix *new_csr_random(int m, int n, double density)
{
    /* Single pass: over-allocate, fill, then copy to exact size */
    int cap = (int) ((double) m * (double) n * density * 1.5) + m;
    int *tmp_p = (int *) malloc(((size_t) m + 1) * sizeof(int));
    int *tmp_i = (int *) malloc((size_t) cap * sizeof(int));
    double *tmp_x = (double *) malloc((size_t) cap * sizeof(double));

    int nnz = 0;
    for (int r = 0; r < m; r++)
    {
        tmp_p[r] = nnz;
        for (int c = 0; c < n; c++)
        {
            double u = (double) rand() / (double) RAND_MAX;
            if (u < density)
            {
                if (nnz >= cap)
                {
                    cap *= 2;
                    tmp_i = (int *) realloc(tmp_i, (size_t) cap * sizeof(int));
                    tmp_x = (double *) realloc(tmp_x, (size_t) cap * sizeof(double));
                }
                tmp_i[nnz] = c;
                tmp_x[nnz] = randn();
                nnz++;
            }
        }
    }
    tmp_p[m] = nnz;

    CSR_Matrix *A = new_csr_matrix(m, n, nnz);
    memcpy(A->p, tmp_p, ((size_t) m + 1) * sizeof(int));
    memcpy(A->i, tmp_i, (size_t) nnz * sizeof(int));
    memcpy(A->x, tmp_x, (size_t) nnz * sizeof(double));

    free(tmp_p);
    free(tmp_i);
    free(tmp_x);
    return A;
}

void profile_expr(expr *node, double *x_vals, const char *label)
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

    free(w);
}
