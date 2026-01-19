#include "utils/mini_numpy.h"
#include <string.h>

void repeat(double *result, const double *a, int len, int repeats)
{
    int idx = 0;
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < repeats; j++)
        {
            result[idx++] = a[i];
        }
    }
}

void tile_double(double *result, const double *a, int len, int tiles)
{
    for (int i = 0; i < tiles; i++)
    {
        memcpy(result + i * len, a, len * sizeof(double));
    }
}

void tile_int(int *result, const int *a, int len, int tiles)
{
    for (int i = 0; i < tiles; i++)
    {
        memcpy(result + i * len, a, len * sizeof(int));
    }
}

void scaled_ones(double *result, int size, double value)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = value;
    }
}

void mat_mat_mult(const double *X, const double *Y, double *Z, int m, int k, int n)
{
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < m; ++i)
        {
            Z[i + j * m] = 0.0;
            for (int l = 0; l < k; ++l)
            {
                Z[i + j * m] += X[i + l * m] * Y[l + j * k];
            }
        }
    }
}
