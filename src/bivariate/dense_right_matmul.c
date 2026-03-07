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
#include "affine.h"
#include "bivariate.h"
#include "subexpr.h"
#include <stdlib.h>

/* This file implements the atom 'dense_right_matmul' corresponding to the
   operation y = f(x) @ A, where A is a given dense matrix.
   We implement this by expressing right matmul in terms of dense left matmul
   and transpose: f(x) @ A = (A^T @ f(x)^T)^T. */
expr *new_dense_right_matmul(expr *u, const double *A_dense, int m, int n)
{
    /* We express: u @ A = (A^T @ u^T)^T
       A is m x n, so A^T is n x m. */
    double *AT = (double *) malloc(n * m * sizeof(double));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            AT[j * m + i] = A_dense[i * n + j];
        }
    }

    expr *u_transpose = new_transpose(u);
    expr *left_matmul_node = new_dense_left_matmul(u_transpose, AT, n, m);
    expr *node = new_transpose(left_matmul_node);

    free(AT);
    return node;
}
