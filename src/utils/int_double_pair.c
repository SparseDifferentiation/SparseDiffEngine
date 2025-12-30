#include "utils/int_double_pair.h"
#include <stdlib.h>

static int compare_int_double_pair(const void *a, const void *b)
{
    const int_double_pair *pair_a = (const int_double_pair *) a;
    const int_double_pair *pair_b = (const int_double_pair *) b;

    if (pair_a->col < pair_b->col) return -1;
    if (pair_a->col > pair_b->col) return 1;
    return 0;
}

int_double_pair *new_int_double_pair_array(int size)
{
    return (int_double_pair *) malloc(size * sizeof(int_double_pair));
}

void set_int_double_pair_array(int_double_pair *pair, int *ints, double *doubles,
                               int size)
{
    for (int k = 0; k < size; k++)
    {
        pair[k].col = ints[k];
        pair[k].val = doubles[k];
    }
}

void free_int_double_pair_array(int_double_pair *array)
{
    free(array);
}

void sort_int_double_pair_array(int_double_pair *array, int size)
{
    qsort(array, size, sizeof(int_double_pair), compare_int_double_pair);
}
