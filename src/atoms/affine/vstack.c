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
#include "atoms/affine.h"
#include <assert.h>
#include <stdlib.h>

/*
 * vstack(args) = transpose(hstack(transpose(args[0]),
 *                                 transpose(args[1]),
 *                                 ...))
 *
 * All args must share the same d2 (number of columns).
 */
expr *new_vstack(expr **args, int n_args, int n_vars)
{
    assert(n_args > 0);
    for (int i = 1; i < n_args; i++)
    {
        assert(args[i]->d2 == args[0]->d2);
    }

    expr **transposed = (expr **) malloc(n_args * sizeof(expr *));
    for (int i = 0; i < n_args; i++)
    {
        transposed[i] = new_transpose(args[i]);
    }

    expr *hstacked = new_hstack(transposed, n_args, n_vars);
    free(transposed);

    return new_transpose(hstacked);
}
