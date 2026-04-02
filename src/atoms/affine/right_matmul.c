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

#include "utils/CSR_Matrix.h"
#include "utils/tracked_alloc.h"
#include <stdlib.h>

/* This file implements the atom 'right_matmul' corresponding to the operation y =
   f(x) @ A, where A is a given matrix and f(x) is an arbitrary expression.
    We implement this by expressing right matmul in terms of left matmul and
    transpose: f(x) @ A = (A^T @ f(x)^T)^T. */
expr *new_right_matmul(expr *u, const CSR_Matrix *A)
{
    /* We can express right matmul using left matmul and transpose:
       u @ A = (A^T @ u^T)^T. */
    int *work_transpose = (int *) SP_MALLOC(A->n * sizeof(int));
    CSR_Matrix *AT = transpose(A, work_transpose);

    expr *u_transpose = new_transpose(u);
    expr *left_matmul = new_left_matmul(u_transpose, AT);
    expr *node = new_transpose(left_matmul);

    free_csr_matrix(AT);
    free(work_transpose);
    return node;
}

expr *new_right_matmul_dense(expr *u, int m, int n, const double *data)
{
    /* We express: u @ A = (A^T @ u^T)^T
       A is m x n, so A^T is n x m. */
    double *AT = (double *) SP_MALLOC(n * m * sizeof(double));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            AT[j * m + i] = data[i * n + j];
        }
    }

    expr *u_transpose = new_transpose(u);
    expr *left_matmul_node = new_left_matmul_dense(u_transpose, n, m, AT);
    expr *node = new_transpose(left_matmul_node);

    free(AT);
    return node;
}
