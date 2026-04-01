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
#include "old-code/old_CSR.h"
#include "utils/CSR_Matrix.h"
#include <assert.h>
#include <string.h>

CSR_Matrix *block_diag_repeat_csr(const CSR_Matrix *A, int p)
{
    assert(p > 0);

    int m = A->m;
    int n = A->n;
    int nnz = A->nnz;

    CSR_Matrix *A_kron = new_csr_matrix(m * p, n * p, nnz * p, NULL);

    int nnz_cursor = 0;
    for (int block = 0; block < p; block++)
    {
        int row_offset = block * m;
        int col_offset = block * n;

        for (int row = 0; row < m; row++)
        {
            int dest_row = row_offset + row;
            A_kron->p[dest_row] = nnz_cursor;

            for (int j = A->p[row]; j < A->p[row + 1]; j++)
            {
                A_kron->i[nnz_cursor] = A->i[j] + col_offset;
                A_kron->x[nnz_cursor] = A->x[j];
                nnz_cursor++;
            }

            A_kron->p[dest_row + 1] = nnz_cursor;
        }
    }

    return A_kron;
}

CSR_Matrix *kron_identity_csr(const CSR_Matrix *A, int p)
{
    assert(p > 0);

    int m = A->m;
    int n = A->n;
    int nnz = A->nnz;

    CSR_Matrix *A_kron = new_csr_matrix(m * p, n * p, nnz * p, NULL);

    int nnz_cursor = 0;
    for (int row_block = 0; row_block < m; row_block++)
    {
        for (int diag_idx = 0; diag_idx < p; diag_idx++)
        {
            int dest_row = row_block * p + diag_idx;
            A_kron->p[dest_row] = nnz_cursor;

            /* Copy entries from row_block of A, adjusting column indices */
            for (int j = A->p[row_block]; j < A->p[row_block + 1]; j++)
            {
                int col_block = A->i[j];
                /* Column in result: col_block * p + diag_idx */
                A_kron->i[nnz_cursor] = col_block * p + diag_idx;
                A_kron->x[nnz_cursor] = A->x[j];
                nnz_cursor++;
            }

            A_kron->p[dest_row + 1] = nnz_cursor;
        }
    }

    return A_kron;
}

void Ax_csr_fill_values(const CSR_Matrix *AT, const double *z, CSR_Matrix *C)
{
    int A_ncols = AT->m;

    for (int i = 0; i < A_ncols; i++)
    {
        double val = 0;
        for (int j = AT->p[i]; j < AT->p[i + 1]; j++)
        {
            val += z[AT->i[j]] * AT->x[j];
        }

        if (AT->p[i + 1] - AT->p[i] == 0) continue;

        // find position in C
        for (int k = 0; k < C->nnz; k++)
        {
            if (C->i[k] == i)
            {
                C->x[k] = val;
                break;
            }
        }
    }
}

void csr_insert_value(CSR_Matrix *A, int col_idx, double value)
{
    assert(A->m == 1);

    for (int j = 0; j < A->nnz; j++)
    {
        assert(col_idx != A->i[j]);

        if (col_idx < A->i[j])
        {
            /* move the rest of the elements */
            memmove(A->i + (j + 1), A->i + j, (A->nnz - j) * sizeof(int));
            memmove(A->x + (j + 1), A->x + j, (A->nnz - j) * sizeof(double));

            /* insert new value */
            A->i[j] = col_idx;
            A->x[j] = value;
            A->nnz++;
            return;
        }
    }

    /* if we get here it should be inserted in the end */
    A->i[A->nnz] = col_idx;
    A->x[A->nnz] = value;
    A->nnz++;
}

void Ax_csr_wo_offset(const CSR_Matrix *A, const double *x, double *y)
{
    for (int row = 0; row < A->m; row++)
    {
        double sum = 0.0;
        for (int j = A->p[row]; j < A->p[row + 1]; j++)
        {
            sum += A->x[j] * x[A->i[j]];
        }
        y[row] = sum;
    }
}

void diag_csr_mult(const double *d, const CSR_Matrix *A, CSR_Matrix *C)
{
    copy_csr_matrix(A, C);

    for (int row = 0; row < C->m; row++)
    {
        for (int j = C->p[row]; j < C->p[row + 1]; j++)
        {
            C->x[j] *= d[row];
        }
    }
}
