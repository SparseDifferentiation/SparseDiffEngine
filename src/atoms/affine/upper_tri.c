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
#include "atoms/affine.h"
#include <assert.h>
#include <stdlib.h>

/* Extract strict upper triangular elements (excluding diagonal)
 * from a square matrix, in ROW-MAJOR order to match CVXPY.
 *
 * NOTE: This is an exception to the engine's column-major
 * convention. CVXPY's upper_tri iterates row-by-row across
 * columns (i outer, j inner), so we do the same here for
 * compatibility.
 *
 * For an (n, n) column-major matrix, element (i, j) with
 * i < j is at flat index j * n + i.
 * Output has n * (n - 1) / 2 elements. */

expr *new_upper_tri(expr *child)
{
    assert(child->d1 == child->d2);
    int n = child->d1;
    int n_elems = n * (n - 1) / 2;

    int *indices = NULL;
    if (n_elems > 0)
    {
        indices = (int *) malloc((size_t) n_elems * sizeof(int));
        int k = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                indices[k++] = j * n + i;
            }
        }
        assert(k == n_elems);
    }

    expr *node = new_index(child, n_elems, 1, indices, n_elems);
    free(indices);
    return node;
}
