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
#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"
#include "utils/iVec.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline bool has_overlap(const int *a_idx, int a_len, const int *b_idx,
                               int b_len, int b_offset)
{
    int ai = 0, bi = 0;
    while (ai < a_len && bi < b_len)
    {
        if (a_idx[ai] == b_idx[bi] - b_offset) return true;
        if (a_idx[ai] < b_idx[bi] - b_offset)
        {
            ai++;
        }
        else
        {
            bi++;
        }
    }
    return false;
}

/* Unweighted sparse dot product of two sorted index arrays */
static inline double sparse_dot(const double *a_x, const int *a_i, int a_nnz,
                                const double *b_x, const int *b_i, int b_nnz,
                                int b_offset)
{
    int ii = 0;
    int jj = 0;
    double sum = 0.0;

    while (ii < a_nnz && jj < b_nnz)
    {
        if (a_i[ii] == b_i[jj] - b_offset)
        {
            sum += a_x[ii] * b_x[jj];
            ii++;
            jj++;
        }
        else if (a_i[ii] < b_i[jj] - b_offset)
        {
            ii++;
        }
        else
        {
            jj++;
        }
    }

    return sum;
}

static inline double sparse_dot_offset(const double *a_x, const int *a_idx,
                                       int a_nnz, const double *b_x,
                                       const int *b_idx, int b_nnz, int b_offset)
{
    int ii = 0, jj = 0;
    double sum = 0.0;

    while (ii < a_nnz && jj < b_nnz)
    {
        int b_col = b_idx[jj] - b_offset;

        if (a_idx[ii] == b_col)
        {
            sum += a_x[ii] * b_x[jj];
            ii++;
            jj++;
        }
        else if (a_idx[ii] < b_col)
        {
            ii++;
        }
        else
        {
            jj++;
        }
    }

    return sum;
}

CSC_Matrix *block_left_multiply_fill_sparsity(const CSR_Matrix *A,
                                              const CSC_Matrix *J, int p)
{
    /* A is m x n, J is (n*p) x k, C is (m*p) x k */
    int m = A->m;
    int n = A->n;
    int k = J->n;
    int j, jj, block, block_start, block_end, block_jj_start, block_jj_end,
        row_offset;

    /* allocate column pointers and an estimate of row indices */
    int *Cp = (int *) malloc((k + 1) * sizeof(int));
    iVec *Ci = iVec_new(k * m);
    Cp[0] = 0;

    /* for each column of j */
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

            block_jj_end = jj;
            int nnz_in_block = block_jj_end - block_jj_start;
            if (nnz_in_block == 0)
            {
                continue;
            }

            // -----------------------------------------------------------------
            // check which rows of A overlap with the column indices of J in this
            // block
            // -----------------------------------------------------------------
            row_offset = block * m;
            for (int i = 0; i < A->m; i++)
            {
                int a_len = A->p[i + 1] - A->p[i];
                if (has_overlap(A->i + A->p[i], a_len, J->i + block_jj_start,
                                nnz_in_block, block_start))
                {
                    iVec_append(Ci, row_offset + i);
                }
            }
        }
        Cp[j + 1] = Ci->len;
    }

    /* Allocate result matrix C in CSC format */
    CSC_Matrix *C = new_csc_matrix(m * p, k, Ci->len);

    /* Copy column pointers and row indices */
    memcpy(C->p, Cp, (k + 1) * sizeof(int));
    memcpy(C->i, Ci->data, Ci->len * sizeof(int));

    /* Clean up workspace */
    free(Cp);
    iVec_free(Ci);

    return C;
}

void block_left_multiply_fill_values(const CSR_Matrix *A, const CSC_Matrix *J,
                                     CSC_Matrix *C)
{
    /* A is m x n, J is (n*p) x k, C is (m*p) x k */
    int m = A->m;
    int n = A->n;
    int k = J->n;

    int i, j, row_a, block, block_start, block_end, start, end;

    /* to get rid of unitialized warnings */
    block = 0;
    block_start = 0;
    block_end = 0;
    start = 0;
    end = 0;

    /* for each column of J (and C) */
    for (j = 0; j < k; j++)
    {
        int previous_block = -1;

        for (i = C->p[j]; i < C->p[j + 1]; i++)
        {
            /* choose row of A and block of column of J */
            row_a = C->i[i] % m;
            block = C->i[i] / m;

            // -------------------------------------------------------------------------
            //          find the part of the column of J in the current block
            // -------------------------------------------------------------------------
            if (block != previous_block)
            {
                previous_block = block;
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
            }

            // ------------------------------------------------------------------------------
            // compute value as sparse dot product of row of A and column of J in
            // this block
            // ------------------------------------------------------------------------------
            int a_len = A->p[row_a + 1] - A->p[row_a];
            C->x[i] =
                sparse_dot(A->x + A->p[row_a], A->i + A->p[row_a], a_len,
                           J->x + start, J->i + start, end - start, block_start);
        }
    }
}

/* Fill values of C = A @ B where A is CSR, B is CSC. */
void csr_csc_matmul_fill_values(const CSR_Matrix *A, const CSC_Matrix *B,
                                CSR_Matrix *C)
{
    for (int i = 0; i < A->m; i++)
    {
        for (int jj = C->p[i]; jj < C->p[i + 1]; jj++)
        {
            int j = C->i[jj];

            int a_nnz = A->p[i + 1] - A->p[i];
            int b_nnz = B->p[j + 1] - B->p[j];

            /* Compute dot product of row i of A and column j of B */
            double sum = sparse_dot(A->x + A->p[i], A->i + A->p[i], a_nnz,
                                    B->x + B->p[j], B->i + B->p[j], b_nnz, 0);

            C->x[jj] = sum;
        }
    }
}

/* C = A @ B where A is CSR (m x n), B is CSC (n x p). Result C is CSR (m x p)
  with precomputed sparsity pattern */
CSR_Matrix *csr_csc_matmul_alloc(const CSR_Matrix *A, const CSC_Matrix *B)
{
    int m = A->m;
    int p = B->n;

    int len_a, len_b;

    int *Cp = (int *) malloc((m + 1) * sizeof(int));
    iVec *Ci = iVec_new(m);

    Cp[0] = 0;

    // --------------------------------------------------------------
    //            count nnz and fill column indices
    // --------------------------------------------------------------
    int nnz = 0;
    for (int i = 0; i < A->m; i++)
    {
        len_a = A->p[i + 1] - A->p[i];

        for (int j = 0; j < B->n; j++)
        {
            len_b = B->p[j + 1] - B->p[j];

            if (has_overlap(A->i + A->p[i], len_a, B->i + B->p[j], len_b, 0))
            {
                iVec_append(Ci, j);
                nnz++;
            }
        }

        Cp[i + 1] = nnz;
    }

    CSR_Matrix *C = new_csr_matrix(m, p, nnz);
    memcpy(C->p, Cp, (m + 1) * sizeof(int));
    memcpy(C->i, Ci->data, nnz * sizeof(int));
    free(Cp);
    iVec_free(Ci);

    return C;
}

/* Compute block-wise matrix-vector products.
 * y = [A @ x1; A @ x2; ...; A @ xp] where A is m x n and x is (n*p)-length vector.
 * x is split into p blocks of n elements each.
 */
void block_left_multiply_vec(const struct CSR_Matrix *A, const double *x, double *y,
                             int p)
{
    /* For each block */
    for (int block = 0; block < p; block++)
    {
        int block_start = block * A->n;
        int y_offset = block * A->m;

        /* For each row of A */
        for (int i = 0; i < A->m; i++)
        {
            double row_sum = 0.0;
            int row_nnz = A->p[i + 1] - A->p[i];

            /* Compute sparse dot product of A[i,:] with x_block */
            for (int idx = 0; idx < row_nnz; idx++)
            {
                int col = A->i[A->p[i] + idx];
                row_sum += A->x[A->p[i] + idx] * x[block_start + col];
            }

            y[y_offset + i] = row_sum;
        }
    }
}
