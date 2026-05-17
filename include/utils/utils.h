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
#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

/* Sort an array of integers in ascending order */
void sort_int_array(int *array, int size);

/* Return true if sorted index arrays a_idx and b_idx (lengths a_len, b_len)
   share any value, where b_idx entries are shifted by b_offset before
   comparison (a_idx[ai] == b_idx[bi] - b_offset). Use b_offset = 0 for a
   plain intersection check.  */
bool has_overlap(const int *a_idx, int a_len, const int *b_idx, int b_len,
                 int b_offset);

/* Find positions where two sorted, ascending int arrays match. For each
   match (a[ii] == b[jj]) writes ii into idx_a and jj into idx_b. Returns
   the count. Buffers idx_a and idx_b must have capacity >= min(a_len,
   b_len); no allocation is performed. */
int sorted_intersect_indices(const int *a, int a_len, const int *b, int b_len,
                             int *idx_a, int *idx_b);

#endif // UTILS_H
