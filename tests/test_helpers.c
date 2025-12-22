#include <math.h>
#include <stdio.h>

#include "expr.h"

#define EPSILON 1e-9

int cmp_double_array(const double *actual, const double *expected, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (fabs(actual[i] - expected[i]) > EPSILON)
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
