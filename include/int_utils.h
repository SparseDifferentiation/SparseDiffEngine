/*
 * Copyright 2026 Daniel Cederberg and William Zhang
 *
 * This file is part of the SparseDiffEngine project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef INT_UTILS_H
#define INT_UTILS_H

/* Small integer helpers the atoms use when sizing sparsity patterns. */

#include <limits.h>
#include <stdlib.h>

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

/* Dense cell count a*b for allocation upper bounds, clamped to INT_MAX on overflow.
   Used as MIN(loose_nnz_bound, sat_mul_int(rows, cols)): a real nnz bound is always
   < INT_MAX, so on overflow the MIN correctly falls back to the loose bound instead
   of selecting a wrapped (negative or small-positive) product. */
static inline int sat_mul_int(int a, int b)
{
    if (a <= 0 || b <= 0) return 0;
    if (a > INT_MAX / b) return INT_MAX;
    return a * b;
}

static inline int int_utils_compare_asc(const void *a, const void *b)
{
    int x = *(const int *) a;
    int y = *(const int *) b;
    return (x > y) - (x < y);
}

/* Sort an array of integers in ascending order. */
static inline void sort_int_array(int *array, int size)
{
    qsort(array, size, sizeof(int), int_utils_compare_asc);
}

#endif /* INT_UTILS_H */
