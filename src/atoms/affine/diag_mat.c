/*
 * Copyright 2026 Daniel Cederberg and William Zhang
 *
 * This file is part of the DNLP-differentiation-engine project.
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
// SPDX-License-Identifier: Apache-2.0

#include "atoms/affine.h"
#include <assert.h>
#include <stdlib.h>

/* Extract diagonal from a square matrix into a column vector.
 * For an (n, n) matrix in column-major order, diagonal element i
 * is at flat index i * (n + 1). */

expr *new_diag_mat(expr *child)
{
    assert(child->d1 == child->d2);
    int n = child->d1;

    int *indices = (int *) malloc((size_t) n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        indices[i] = i * (n + 1);
    }

    expr *node = new_index(child, n, 1, indices, n);
    free(indices);
    return node;
}
