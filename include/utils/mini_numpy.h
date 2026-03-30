#ifndef MINI_NUMPY_H
#define MINI_NUMPY_H

/* Repeat each element of array 'a' 'repeats' times
 * Example: a = [1, 2], len = 2, repeats = 3
 *          result = [1, 1, 1, 2, 2, 2]
 */
void repeat(double *result, const double *a, int len, int repeats);

/* Tile array 'a' 'tiles' times
 * Example: a = [1, 2], len = 2, tiles = 3
 *          result = [1, 2, 1, 2, 1, 2]
 */
void tile_double(double *result, const double *a, int len, int tiles);
void tile_int(int *result, const int *a, int len, int tiles);

/* Fill array with 'size' copies of 'value'
 * Example: size = 5, value = 3.0
 *          result = [3.0, 3.0, 3.0, 3.0, 3.0]
 */
void scaled_ones(double *result, int size, double value);

/* Naive implementation of Z = X @ Y, X is m x k, Y is k x n, Z is m x n */
void mat_mat_mult(const double *X, const double *Y, double *Z, int m, int k, int n);

/* Compute v = (Y kron I_m) @ w where Y is k x n (col-major), w has
   length m*n, and v has length m*k.  Equivalently, reshape w as the
   m x n matrix W (col-major) and compute v = vec(W @ Y^T). */
void Y_kron_I_vec(int m, int k, int n, const double *Y, const double *w, double *v);

/* Compute v = (I_n kron X^T) @ w where X is m x k (col-major), w has
   length m*n, and v has length k*n.  Equivalently, reshape w as the
   m x n matrix W (col-major) and compute v = vec(X^T @ W). */
void I_kron_XT_vec(int m, int k, int n, const double *X, const double *w, double *v);

#endif /* MINI_NUMPY_H */
