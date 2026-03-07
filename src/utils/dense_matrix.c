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
#include "utils/cblas_wrapper.h"
#include "utils/iVec.h"
#include "utils/matrix.h"
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
       But x and y are stored as p blocks of length n and m respectively
       (i.e. block-interleaved). This is the same as treating them as
       row-major matrices of shape p x n and p x m, so:
       y (p x m) = x (p x n) * A^T (n x m), all row-major.
       cblas with RowMajor: C = alpha * A * B + beta * C
       where A = x (p x n), B = A^T (n x m), C = y (p x m). */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, p, m, n, 1.0, x, n, dm->x,
                n, 0.0, y, m);
}

static CSC_Matrix *dense_block_left_mult_sparsity(const Matrix *A,
                                                  const CSC_Matrix *J, int p)
{
    int m = A->m;
    int n = A->n;
    int i, j, jj, block, block_start, block_end, block_jj_start, row_offset;

    int *Cp = (int *) malloc((J->n + 1) * sizeof(int));
    iVec *Ci = iVec_new(J->n * m);
    Cp[0] = 0;

    /* for each column of J */
    for (j = 0; j < J->n; j++)
    {
        /* if empty we continue */
        if (J->p[j] == J->p[j + 1])
        {
            Cp[j + 1] = Cp[j];
            continue;
        }

        /* process each of p blocks of rows in this column of J */
        jj = J->p[j];
        for (block = 0; block < p; block++)
        {
            // -----------------------------------------------------------------
            //  find start and end indices of rows of J in this block
            // -----------------------------------------------------------------
            block_start = block * n;
            block_end = block_start + n;
            while (jj < J->p[j + 1] && J->i[jj] < block_start)
            {
                jj++;
            }

            block_jj_start = jj;
            while (jj < J->p[j + 1] && J->i[jj] < block_end)
            {
                jj++;
            }

            /* if no entries in this block, continue */
            if (jj == block_jj_start)
            {
                continue;
            }

            /* dense A: all m rows contribute */
            row_offset = block * m;
            for (i = 0; i < m; i++)
            {
                iVec_append(Ci, row_offset + i);
            }
        }
        Cp[j + 1] = Ci->len;
    }

    CSC_Matrix *C = new_csc_matrix(m * p, J->n, Ci->len);
    memcpy(C->p, Cp, (J->n + 1) * sizeof(int));
    memcpy(C->i, Ci->data, Ci->len * sizeof(int));
    free(Cp);
    iVec_free(Ci);

    return C;
}

static void dense_block_left_mult_values(const Matrix *A, const CSC_Matrix *J,
                                         CSC_Matrix *C)
{
    const Dense_Matrix *dm = (const Dense_Matrix *) A;
    int m = dm->base.m;
    int n = dm->base.n;
    int k = J->n;

    int i, j, s, block, block_start, block_end, start, end;

    double *j_dense = dm->work;

    /* for each column of J (and C) */
    for (j = 0; j < k; j++)
    {
        for (i = C->p[j]; i < C->p[j + 1]; i += m)
        {
            block = C->i[i] / m;
            block_start = block * n;
            block_end = block_start + n;

            start = J->p[j];
            end = J->p[j + 1];

            while (start < J->p[j + 1] && J->i[start] < block_start)
            {
                start++;
            }

            while (end > start && J->i[end - 1] >= block_end)
            {
                end--;
            }

            /* scatter sparse J column into dense vector and then compute
               A @ j_dense */
            memset(j_dense, 0, n * sizeof(double));
            for (s = start; s < end; s++)
            {
                j_dense[J->i[s] - block_start] = J->x[s];
            }

            cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, dm->x, n, j_dense, 1,
                        0.0, C->x + i, 1);
        }
    }
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
    dm->base.block_left_mult_sparsity = dense_block_left_mult_sparsity;
    dm->base.block_left_mult_values = dense_block_left_mult_values;
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
