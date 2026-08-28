#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "expr.h"
#include "utils/CSR_matrix.h"
#include "utils/matrix.h"

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

int cmp_sparsity(matrix *M, const int *exp_p, const int *exp_i, int m, int nnz)
{
    if (M->m != m)
    {
        printf("  FAILED: M->m = %d, expected %d\n", M->m, m);
        return 0;
    }
    if (M->nnz != nnz)
    {
        printf("  FAILED: M->nnz = %d, expected %d\n", M->nnz, nnz);
        return 0;
    }
    CSR_matrix *csr = M->to_csr(M);
    return cmp_int_array(csr->p, exp_p, m + 1) && cmp_int_array(csr->i, exp_i, nnz);
}

int cmp_values(const matrix *M, const double *exp_x, int nnz)
{
    if (M->nnz != nnz)
    {
        printf("  FAILED: M->nnz = %d, expected %d\n", M->nnz, nnz);
        return 0;
    }
    return cmp_double_array(M->x, exp_x, nnz);
}

int csr_is_valid(const CSR_matrix *A)
{
    if (A->p[0] != 0)
    {
        printf("  FAILED: p[0] = %d, expected 0\n", A->p[0]);
        return 0;
    }
    for (int i = 0; i < A->m; i++)
    {
        if (A->p[i] > A->p[i + 1])
        {
            printf("  FAILED: p[%d] = %d > p[%d] = %d\n", i, A->p[i], i + 1,
                   A->p[i + 1]);
            return 0;
        }
    }
    if (A->p[A->m] != A->nnz)
    {
        printf("  FAILED: p[m] = %d, but nnz = %d\n", A->p[A->m], A->nnz);
        return 0;
    }
    for (int jj = 0; jj < A->nnz; jj++)
    {
        if (A->i[jj] < 0 || A->i[jj] >= A->n)
        {
            printf("  FAILED: i[%d] = %d out of range [0, %d)\n", jj, A->i[jj],
                   A->n);
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

CSR_matrix *new_csr_random(int m, int n, double density)
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

    CSR_matrix *A = new_CSR_matrix(m, n, nnz);
    memcpy(A->p, tmp_p, ((size_t) m + 1) * sizeof(int));
    memcpy(A->i, tmp_i, (size_t) nnz * sizeof(int));
    memcpy(A->x, tmp_x, (size_t) nnz * sizeof(double));

    free(tmp_p);
    free(tmp_i);
    free(tmp_x);
    return A;
}
