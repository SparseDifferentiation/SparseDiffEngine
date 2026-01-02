#include "utils/mini_numpy.h"

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

void tile(double *result, const double *a, int len, int tiles)
{
    int idx = 0;
    for (int i = 0; i < tiles; i++)
    {
        for (int j = 0; j < len; j++)
        {
            result[idx++] = a[j];
        }
    }
}

void scaled_ones(double *result, int size, double value)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = value;
    }
}
