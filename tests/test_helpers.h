#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include "expr.h"
#include "utils/CSR_Matrix.h"
#include "utils/matrix.h"

/* Compare two double arrays directly
 * Returns 1 if all values match, 0 otherwise */
int cmp_double_array(const double *actual, const double *expected, int size);

/* Compare two int arrays directly
 * Returns 1 if all values match, 0 otherwise */
int cmp_int_array(const int *actual, const int *expected, int size);

/* Verify M has shape (m, *) with exactly nnz entries, and that M's CSR
 * row pointers and column indices match exp_p (length m+1) and exp_i
 * (length nnz). Returns 1 on full match, 0 otherwise. */
int cmp_sparsity(Matrix *M, const int *exp_p, const int *exp_i, int m, int nnz);

/* Verify M has nnz entries and that its value array matches exp_x of
 * length nnz. Returns 1 on full match, 0 otherwise. */
int cmp_values(const Matrix *M, const double *exp_x, int nnz);

/* Create a random m x n CSR matrix with approximate nonzero density
 * in [0, 1]. Nonzero values are standard Gaussian (Box-Muller). */
CSR_Matrix *new_csr_random(int m, int n, double density);

#endif /* TEST_HELPERS_H */
