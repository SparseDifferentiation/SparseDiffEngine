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

#endif /* MINI_NUMPY_H */
