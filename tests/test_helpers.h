#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include "expr.h"
#include "utils/CSR_Matrix.h"

/* Compare two double arrays directly
 * Returns 1 if all values match, 0 otherwise */
int cmp_double_array(const double *actual, const double *expected,
                     int size);

/* Compare two int arrays directly
 * Returns 1 if all values match, 0 otherwise */
int cmp_int_array(const int *actual, const int *expected, int size);

/* Create a random m x n CSR matrix with approximate nonzero density
 * in [0, 1]. Nonzero values are standard Gaussian (Box-Muller). */
CSR_Matrix *new_csr_random(int m, int n, double density);

#endif /* TEST_HELPERS_H */
