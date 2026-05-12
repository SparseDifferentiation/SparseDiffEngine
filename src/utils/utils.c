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

bool has_overlap(const int *a_idx, int a_len, const int *b_idx, int b_len,
                 int b_offset)
{
    int ai = 0, bi = 0;
    while (ai < a_len && bi < b_len)
    {
        if (a_idx[ai] == b_idx[bi] - b_offset) return true;
        if (a_idx[ai] < b_idx[bi] - b_offset)
        {
            ai++;
        }
        else
        {
            bi++;
        }
    }
    return false;
}
