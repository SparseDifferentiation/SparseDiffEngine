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
#include "utils/COO_matrix.h"
#include "utils/tracked_alloc.h"
#include <stdlib.h>
#include <string.h>

COO_matrix *new_COO_matrix(const CSR_matrix *A)
{
    COO_matrix *coo = (COO_matrix *) sp_malloc(sizeof(COO_matrix));
    coo->m = A->m;
    coo->n = A->n;
    coo->nnz = A->nnz;
    coo->rows = (int *) sp_malloc(A->nnz * sizeof(int));
    coo->cols = (int *) sp_malloc(A->nnz * sizeof(int));
    coo->x = (double *) sp_malloc(A->nnz * sizeof(double));
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

COO_matrix *new_COO_matrix_from_pattern(const CSR_matrix *A, const int *rows,
                                        const int *cols, int nnz)
{
    if (nnz != A->nnz) return NULL;

    COO_matrix *coo = (COO_matrix *) sp_malloc(sizeof(COO_matrix));
    coo->m = A->m;
    coo->n = A->n;
    coo->nnz = nnz;
    coo->rows = (int *) sp_malloc(nnz * sizeof(int));
    coo->cols = (int *) sp_malloc(nnz * sizeof(int));
    coo->x = (double *) sp_malloc(nnz * sizeof(double));
    coo->value_map = NULL;

    memcpy(coo->rows, rows, nnz * sizeof(int));
    memcpy(coo->cols, cols, nnz * sizeof(int));
    memcpy(coo->x, A->x, nnz * sizeof(double));

    return coo;
}

COO_matrix *new_COO_matrix_lower_triangular(const CSR_matrix *A)
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

    COO_matrix *coo = (COO_matrix *) sp_malloc(sizeof(COO_matrix));
    coo->m = A->m;
    coo->n = A->n;
    coo->nnz = count;
    coo->rows = (int *) sp_malloc(count * sizeof(int));
    coo->cols = (int *) sp_malloc(count * sizeof(int));
    coo->x = (double *) sp_malloc(count * sizeof(double));
    coo->value_map = (int *) sp_malloc(count * sizeof(int));

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

void refresh_lower_triangular_coo(COO_matrix *coo, const double *vals)
{
    for (int i = 0; i < coo->nnz; i++)
    {
        coo->x[i] = vals[coo->value_map[i]];
    }
}

void free_COO_matrix(COO_matrix *matrix)
{
    if (matrix)
    {
        sp_free(matrix->rows);
        sp_free(matrix->cols);
        sp_free(matrix->x);
        sp_free(matrix->value_map);
        sp_free(matrix);
    }
}
