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

#include "utils/iVec.h"
#include "utils/tracked_alloc.h"
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

int sorted_intersect_indices(const int *a, int a_len, const int *b, int b_len,
                             int *idx_a, int *idx_b)
{
    int s = 0;
    int ii = 0, jj = 0;
    while (ii < a_len && jj < b_len)
    {
        int ra = a[ii];
        int rb = b[jj];
        if (ra == rb)
        {
            idx_a[s] = ii;
            idx_b[s] = jj;
            s++;
            ii++;
            jj++;
        }
        else if (ra < rb)
        {
            ii++;
        }
        else
        {
            jj++;
        }
    }
    return s;
}

void sorted_union_int_arrays(const int *const *arrs, const int *lens, int n_arrs,
                             iVec *out)
{
    iVec_clear_no_resize(out);
    int *cursor = (int *) SP_CALLOC(n_arrs, sizeof(int));
    while (1)
    {
        int min_val = 0;
        int min_arr = -1;
        for (int a = 0; a < n_arrs; a++)
        {
            if (cursor[a] >= lens[a])
            {
                continue;
            }
            int v = arrs[a][cursor[a]];
            if (min_arr == -1 || v < min_val)
            {
                min_val = v;
                min_arr = a;
            }
        }
        if (min_arr == -1)
        {
            break;
        }
        if (out->len == 0 || out->data[out->len - 1] != min_val)
        {
            iVec_append(out, min_val);
        }
        cursor[min_arr]++;
    }
    free(cursor);
}

void cumsum(int *p, int n)
{
    for (int i = 0; i < n; i++)
    {
        p[i + 1] += p[i];
    }
}
