#include "utils/utils.h"
#include <stdlib.h>

/* Helper function to compare integers for qsort */
static int compare_int_asc(const void *a, const void *b)
{
    int ia = *((const int *) a);
    int ib = *((const int *) b);
    return (ia > ib) - (ia < ib);
}

void sort_int_array(int *array, int size)
{
    qsort(array, size, sizeof(int), compare_int_asc);
}
