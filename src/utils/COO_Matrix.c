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
#include "utils/COO_Matrix.h"
#include <stdlib.h>
#include <string.h>

COO_Matrix *new_coo_matrix(const CSR_Matrix *A)
{
    COO_Matrix *coo = (COO_Matrix *) malloc(sizeof(COO_Matrix));
    coo->m = A->m;
    coo->n = A->n;
    coo->nnz = A->nnz;
    coo->rows = (int *) malloc(A->nnz * sizeof(int));
    coo->cols = (int *) malloc(A->nnz * sizeof(int));
    coo->x = (double *) malloc(A->nnz * sizeof(double));
    coo->value_map = NULL;

    for (int r = 0; r < A->m; r++)
    {
        for (int k = A->p[r]; k < A->p[r + 1]; k++)
        {
            coo->rows[k] = r;
        }
    }

    memcpy(coo->cols, A->i, A->nnz * sizeof(int));
    memcpy(coo->x, A->x, A->nnz * sizeof(double));

    return coo;
}

COO_Matrix *new_coo_matrix_lower_triangular(const CSR_Matrix *A)
{
    /* Pass 1: count lower-triangular entries (col <= row) */
    int count = 0;
    for (int r = 0; r < A->m; r++)
    {
        for (int k = A->p[r]; k < A->p[r + 1]; k++)
        {
            if (A->i[k] <= r)
            {
                count++;
            }
        }
    }

    COO_Matrix *coo = (COO_Matrix *) malloc(sizeof(COO_Matrix));
    coo->m = A->m;
    coo->n = A->n;
    coo->nnz = count;
    coo->rows = (int *) malloc(count * sizeof(int));
    coo->cols = (int *) malloc(count * sizeof(int));
    coo->x = (double *) malloc(count * sizeof(double));
    coo->value_map = (int *) malloc(count * sizeof(int));

    /* Pass 2: fill arrays */
    int idx = 0;
    for (int r = 0; r < A->m; r++)
    {
        for (int k = A->p[r]; k < A->p[r + 1]; k++)
        {
            if (A->i[k] <= r)
            {
                coo->rows[idx] = r;
                coo->cols[idx] = A->i[k];
                coo->x[idx] = A->x[k];
                coo->value_map[idx] = k;
                idx++;
            }
        }
    }

    return coo;
}

void refresh_lower_triangular_coo(COO_Matrix *coo, const double *vals)
{
    for (int i = 0; i < coo->nnz; i++)
    {
        coo->x[i] = vals[coo->value_map[i]];
    }
}

void free_coo_matrix(COO_Matrix *matrix)
{
    if (matrix)
    {
        free(matrix->rows);
        free(matrix->cols);
        free(matrix->x);
        free(matrix->value_map);
        free(matrix);
    }
}

size_t coo_bytes(const COO_Matrix *A)
{
    if (!A) return 0;
    size_t bytes = (size_t) A->nnz * sizeof(int)       /* rows */
                   + (size_t) A->nnz * sizeof(int)     /* cols */
                   + (size_t) A->nnz * sizeof(double); /* x */
    if (A->value_map) bytes += (size_t) A->nnz * sizeof(int);
    return bytes;
}
