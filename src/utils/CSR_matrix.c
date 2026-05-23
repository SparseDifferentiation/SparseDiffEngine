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
#include "utils/CSR_matrix.h"
#include "utils/int_double_pair.h"
#include "utils/tracked_alloc.h"
#include "utils/utils.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CSR_matrix *new_CSR_matrix(int m, int n, int nnz)
{
    CSR_matrix *matrix = (CSR_matrix *) sp_malloc(sizeof(CSR_matrix));
    matrix->p = (int *) sp_calloc(m + 1, sizeof(int));
    matrix->i = (int *) sp_calloc(nnz, sizeof(int));
    matrix->x = (double *) sp_malloc(nnz * sizeof(double));
    matrix->m = m;
    matrix->n = n;
    matrix->nnz = nnz;
    return matrix;
}

CSR_matrix *new_csr(const CSR_matrix *A)
{
    CSR_matrix *copy = new_CSR_matrix(A->m, A->n, A->nnz);
    memcpy(copy->p, A->p, (A->m + 1) * sizeof(int));
    memcpy(copy->i, A->i, A->nnz * sizeof(int));
    memcpy(copy->x, A->x, A->nnz * sizeof(double));
    return copy;
}

CSR_matrix *new_csr_copy_sparsity(const CSR_matrix *A)
{
    CSR_matrix *copy = new_CSR_matrix(A->m, A->n, A->nnz);
    memcpy(copy->p, A->p, (A->m + 1) * sizeof(int));
    memcpy(copy->i, A->i, A->nnz * sizeof(int));
    return copy;
}

void free_CSR_matrix(CSR_matrix *matrix)
{
    if (matrix)
    {
        free(matrix->p);
        free(matrix->i);
        free(matrix->x);
        free(matrix);
    }
}

void copy_CSR_matrix(const CSR_matrix *A, CSR_matrix *C)
{
    C->m = A->m;
    C->n = A->n;
    C->nnz = A->nnz;
    memcpy(C->p, A->p, (A->m + 1) * sizeof(int));
    memcpy(C->i, A->i, A->nnz * sizeof(int));
    memcpy(C->x, A->x, A->nnz * sizeof(double));
}

void Ax_csr(const CSR_matrix *A, const double *x, double *y, int col_offset)
{
    for (int row = 0; row < A->m; row++)
    {
        double sum = 0.0;
        for (int j = A->p[row]; j < A->p[row + 1]; j++)
        {
            sum += A->x[j] * x[A->i[j] - col_offset];
        }
        y[row] = sum;
    }
}

int count_nonzero_cols(const CSR_matrix *A, bool *col_nz)
{
    for (int row = 0; row < A->m; row++)
    {
        for (int j = A->p[row]; j < A->p[row + 1]; j++)
        {
            col_nz[A->i[j]] = true;
        }
    }
    int count = 0;
    for (int col = 0; col < A->n; col++)
    {
        if (col_nz[col]) count++;
    }

    return count;
}

void insert_idx(int idx, int *arr, int len)
{
    int j = 0;

    while (j < len && arr[j] < idx)
    {
        j++;
    }

    // move elements to the right
    memmove(arr + (j + 1), arr + j, (len - j) * sizeof(int));
    arr[j] = idx;
}

void DA_fill_values(const double *d, const CSR_matrix *A, CSR_matrix *C)
{
    memcpy(C->x, A->x, A->nnz * sizeof(double));

    for (int row = 0; row < C->m; row++)
    {
        for (int j = C->p[row]; j < C->p[row + 1]; j++)
        {
            C->x[j] *= d[row];
        }
    }
}

CSR_matrix *transpose(const CSR_matrix *A, int *iwork)
{
    CSR_matrix *AT = new_CSR_matrix(A->n, A->m, A->nnz);

    int i, j;
    int *count = iwork;
    memset(count, 0, A->n * sizeof(int));

    // -------------------------------------------------------------------
    //              compute nnz in each column of A
    // -------------------------------------------------------------------
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            count[A->i[j]]++;
        }
    }

    // ------------------------------------------------------------------
    //                      compute row pointers
    // ------------------------------------------------------------------
    AT->p[0] = 0;
    for (i = 0; i < A->n; i++)
    {
        AT->p[i + 1] = AT->p[i] + count[i];
        iwork[i] = AT->p[i];
    }

    // ------------------------------------------------------------------
    //  fill transposed matrix
    // ------------------------------------------------------------------
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; j++)
        {
            AT->x[count[A->i[j]]] = A->x[j];
            AT->i[count[A->i[j]]] = i;
            count[A->i[j]]++;
        }
    }

    return AT;
}

CSR_matrix *AT_alloc(const CSR_matrix *A, int *iwork)
{
    /* Allocate A^T and compute sparsity pattern without filling values */
    CSR_matrix *AT = new_CSR_matrix(A->n, A->m, A->nnz);

    int i, j;
    int *count = iwork;
    memset(count, 0, A->n * sizeof(int));

    // -------------------------------------------------------------------
    //              compute nnz in each column of A
    // -------------------------------------------------------------------
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            count[A->i[j]]++;
        }
    }

    // ------------------------------------------------------------------
    //                  compute row pointers
    // ------------------------------------------------------------------
    AT->p[0] = 0;
    for (i = 0; i < A->n; i++)
    {
        AT->p[i + 1] = AT->p[i] + count[i];
        count[i] = AT->p[i];
    }

    // ------------------------------------------------------------------
    //                fill column indices
    // ------------------------------------------------------------------
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; j++)
        {
            AT->i[count[A->i[j]]] = i;
            count[A->i[j]]++;
        }
    }

    return AT;
}

void AT_fill_values(const CSR_matrix *A, CSR_matrix *AT, int *iwork)
{
    /* Fill values of A^T given sparsity pattern is already computed */
    int i, j;
    int *count = iwork;
    memcpy(count, AT->p, A->n * sizeof(int));

    /* Fill values by placing each element of A into its transposed position */
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; j++)
        {
            AT->x[count[A->i[j]]] = A->x[j];
            count[A->i[j]]++;
        }
    }
}

double csr_get_value(const CSR_matrix *A, int row, int col)
{
    for (int j = A->p[row]; j < A->p[row + 1]; j++)
    {
        if (A->i[j] == col)
        {
            return A->x[j];
        }
    }
    return 0.0;
}

void symmetrize_csr(const int *Ap, const int *Ai, int m, CSR_matrix *C)
{
    int i, j, col;

    /* Count entries per row */
    int *counts = (int *) sp_calloc(m, sizeof(int));
    for (i = 0; i < m; i++)
    {
        for (j = Ap[i]; j < Ap[i + 1]; j++)
        {
            counts[i]++;
            if (Ai[j] != i) counts[Ai[j]]++;
        }
    }

    /* Build row pointers */
    C->p[0] = 0;
    for (i = 0; i < m; i++)
    {
        C->p[i + 1] = C->p[i] + counts[i];
    }

    /* Fill column indices */
    memset(counts, 0, m * sizeof(int));
    for (i = 0; i < m; i++)
    {
        for (j = Ap[i]; j < Ap[i + 1]; j++)
        {
            col = Ai[j];

            /* Add to row i */
            C->i[C->p[i] + counts[i]] = col;
            counts[i]++;

            /* Add symmetric entry to row col */
            if (col != i)
            {
                C->i[C->p[col] + counts[col]] = i;
                counts[col]++;
            }
        }
    }

    free(counts);
}
