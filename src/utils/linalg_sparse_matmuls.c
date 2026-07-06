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
#include "utils/CSC_matrix.h"
#include "utils/CSR_matrix.h"
#include "utils/iVec.h"
#include "utils/tracked_alloc.h"
#include "utils/utils.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* Symbolic phase of Gustavson's sparse matmul: an output row's (or column's)
   sparsity pattern is the union of the operand index lists selected by keys,
   i.e. the lists list_i[list_p[k] .. list_p[k+1]] for each k = keys[t] -
   key_offset. Compute that union into cand, deduplicated with a
   generation-stamped workspace and sorted ascending (the order the downstream
   sparse dot products require). Returns the number of gathered indices. */
static int sorted_pattern_union(const int *keys, int n_keys, int key_offset,
                                const int *list_p, const int *list_i, int *stamp,
                                int *cand, int *gen)
{
    int g = ++(*gen);
    int n_cand = 0;
    for (int t = 0; t < n_keys; t++)
    {
        int k = keys[t] - key_offset;
        for (int q = list_p[k]; q < list_p[k + 1]; q++)
        {
            int i = list_i[q];
            if (stamp[i] != g)
            {
                stamp[i] = g;
                cand[n_cand++] = i;
            }
        }
    }
    sort_int_array(cand, n_cand);
    return n_cand;
}

CSC_matrix *block_left_multiply_fill_sparsity(const CSR_matrix *A,
                                              const CSC_matrix *J, int p)
{
    /* A is m x n, J is (n*p) x k, C is (m*p) x k */
    int m = A->m;
    int n = A->n;
    int j, jj, block, block_start, block_end, block_jj_start, block_jj_end,
        row_offset;

    /* Allocate column pointers and an estimate of row indices. Capacity hint
       based on J->nnz and not on the dense product */
    int *Cp = (int *) sp_malloc((J->n + 1) * sizeof(int));
    iVec *Ci = iVec_new(J->nnz > 0 ? J->nnz : 1);
    Cp[0] = 0;

    /* Output-driven: row i of A intersects a block's child entries iff i
       appears in the CSC column list of one of those entries, so gather those
       columns instead of testing all m rows per (column, block). */
    int *iwork = (int *) sp_malloc((n > 0 ? n : 1) * sizeof(int));
    CSC_matrix *A_csc = csr_to_csc_alloc(A, iwork);
    sp_free(iwork);

    int *stamp = (int *) sp_calloc(m > 0 ? m : 1, sizeof(int)); /* 0 = unseen */
    int *cand = (int *) sp_malloc((m > 0 ? m : 1) * sizeof(int));
    int gen = 0;

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

            block_jj_end = jj;
            if (block_jj_end == block_jj_start)
            {
                continue;
            }

            // -----------------------------------------------------------------
            // gather the rows of A that hit this block's child entries
            // -----------------------------------------------------------------
            int n_cand = sorted_pattern_union(
                J->i + block_jj_start, block_jj_end - block_jj_start, block_start,
                A_csc->p, A_csc->i, stamp, cand, &gen);
            row_offset = block * m;
            for (int t = 0; t < n_cand; t++)
            {
                iVec_append(Ci, row_offset + cand[t]);
            }
        }
        Cp[j + 1] = Ci->len;
    }

    free_CSC_matrix(A_csc);
    sp_free(stamp);
    sp_free(cand);

    CSC_matrix *C = new_CSC_matrix(m * p, J->n, Ci->len);
    memcpy(C->p, Cp, (J->n + 1) * sizeof(int));
    memcpy(C->i, Ci->data, Ci->len * sizeof(int));
    sp_free(Cp);
    iVec_free(Ci);

    return C;
}

void block_left_multiply_fill_values(const CSR_matrix *A, const CSC_matrix *J,
                                     CSC_matrix *C)
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

/* Fill values of C = A @ B where A is CSR_matrix, B is CSC_matrix. */
void csr_csc_matmul_fill_values(const CSR_matrix *A, const CSC_matrix *B,
                                CSR_matrix *C)
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

/* C = A @ B where A is CSR_matrix (m x n), B is CSC_matrix (n x p). Result C is
  CSR_matrix (m x p) with precomputed sparsity pattern */
CSR_matrix *csr_csc_matmul_alloc(const CSR_matrix *A, const CSC_matrix *B)
{
    int m = A->m;
    int p = B->n;

    int *Cp = (int *) sp_malloc((m + 1) * sizeof(int));
    iVec *Ci = iVec_new(m);

    Cp[0] = 0;

    /* Output-driven: column j of B intersects row i of A iff j appears in the
       CSR row list of one of A's row-i column indices, so gather those rows
       instead of testing all p columns per row of A. */
    int *iwork = (int *) sp_malloc((B->m > 0 ? B->m : 1) * sizeof(int));
    CSR_matrix *B_csr = csc_to_csr_alloc(B, iwork);
    sp_free(iwork);

    int *stamp = (int *) sp_calloc(p > 0 ? p : 1, sizeof(int)); /* 0 = unseen */
    int *cand = (int *) sp_malloc((p > 0 ? p : 1) * sizeof(int));
    int gen = 0;

    // --------------------------------------------------------------
    //            count nnz and fill column indices
    // --------------------------------------------------------------
    int nnz = 0;
    for (int i = 0; i < A->m; i++)
    {
        int n_cand = sorted_pattern_union(A->i + A->p[i], A->p[i + 1] - A->p[i], 0,
                                          B_csr->p, B_csr->i, stamp, cand, &gen);
        for (int t = 0; t < n_cand; t++)
        {
            iVec_append(Ci, cand[t]);
        }
        nnz += n_cand;

        Cp[i + 1] = nnz;
    }

    free_CSR_matrix(B_csr);
    sp_free(stamp);
    sp_free(cand);

    CSR_matrix *C = new_CSR_matrix(m, p, nnz);
    memcpy(C->p, Cp, (m + 1) * sizeof(int));
    memcpy(C->i, Ci->data, nnz * sizeof(int));
    sp_free(Cp);
    iVec_free(Ci);

    return C;
}

/* Compute block-wise matrix-vector products.
 * y = [A @ x1; A @ x2; ...; A @ xp] where A is m x n and x is (n*p)-length vector.
 * x is split into p blocks of n elements each.
 */
void block_left_multiply_vec(const struct CSR_matrix *A, const double *x, double *y,
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
