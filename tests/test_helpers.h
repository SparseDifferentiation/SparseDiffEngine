#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include "expr.h"
#include "utils/CSR_Matrix.h"

/* Compare two double arrays directly
 * Returns 1 if all values match, 0 otherwise */
int cmp_double_array(const double *actual, const double *expected, int size);

/* Compare two int arrays directly
 * Returns 1 if all values match, 0 otherwise */
int cmp_int_array(const int *actual, const int *expected, int size);

/* Create a random m x n CSR matrix with approximate nonzero density
 * in [0, 1]. Nonzero values are standard Gaussian (Box-Muller). */
CSR_Matrix *new_csr_random(int m, int n, double density);

/* Allocate and fill an array of `size` doubles with uniform random
 * values in [-0.5, 0.5].  Caller owns the returned pointer. */
double *new_random_dense_data(int size);

/* Run the full profiling pipeline (forward, jacobian init/eval,
 * hessian init/eval) on `e` with variable values `u`, printing
 * timing results prefixed by `label`. */
void profile_expr(expr *e, const double *u, const char *label);

#endif /* TEST_HELPERS_H */
