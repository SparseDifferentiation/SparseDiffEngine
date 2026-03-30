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
#include "utils/dense_matrix.h"
#include "utils/cblas_wrapper.h"
#include "utils/linalg_dense_sparse_matmuls.h"
#include <stdlib.h>
#include <string.h>

static void dense_block_left_mult_vec(const Matrix *A, const double *x, double *y,
                                      int p)
{
    const Dense_Matrix *dm = (const Dense_Matrix *) A;
    int m = dm->base.m;
    int n = dm->base.n;

    /* y = kron(I_p, A) @ x via a single dgemm call:
       Treat x as n x p (column-major blocks) and y as m x p.
       But x and y are stored as p blocks of length n and m
       respectively (i.e. block-interleaved). This is the same as
       treating them as row-major matrices of shape p x n and
       p x m, so:
       y (p x m) = x (p x n) * A^T (n x m), all row-major.
       cblas with RowMajor: C = alpha * A * B + beta * C
       where A = x (p x n), B = A^T (n x m), C = y (p x m). */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, p, m, n, 1.0, x, n, dm->x,
                n, 0.0, y, m);
}

static void dense_free(Matrix *A)
{
    Dense_Matrix *dm = (Dense_Matrix *) A;
    free(dm->x);
    free(dm->work);
    free(dm);
}

Matrix *new_dense_matrix(int m, int n, const double *data)
{
    Dense_Matrix *dm = (Dense_Matrix *) calloc(1, sizeof(Dense_Matrix));
    dm->base.m = m;
    dm->base.n = n;
    dm->base.block_left_mult_vec = dense_block_left_mult_vec;
    dm->base.block_left_mult_sparsity = I_kron_A_alloc;
    dm->base.block_left_mult_values = I_kron_A_fill_values;
    dm->base.free_fn = dense_free;
    dm->x = (double *) malloc(m * n * sizeof(double));
    memcpy(dm->x, data, m * n * sizeof(double));
    dm->work = (double *) malloc(n * sizeof(double));
    return &dm->base;
}

Matrix *dense_matrix_trans(const Dense_Matrix *A)
{
    int m = A->base.m;
    int n = A->base.n;
    double *AT_x = (double *) malloc(m * n * sizeof(double));

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            AT_x[j * m + i] = A->x[i * n + j];
        }
    }

    Matrix *result = new_dense_matrix(n, m, AT_x);
    free(AT_x);
    return result;
}
