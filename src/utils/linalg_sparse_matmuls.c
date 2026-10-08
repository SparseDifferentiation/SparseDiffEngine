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

/* Numeric phase of Gustavson's matmul, column by column of C: for each block of
   column j, scatter A[:, c] * J[c, j] into a dense accumulator over the rows of
   A for every entry of J in the block, then gather the accumulator into the
   block's entries of C. Cost is the number of multiply-adds, independent of
   the row lengths of A. The previous version took a merge-based sparse dot of
   a whole row of A per entry of C, which is O(m_out * nnz(row)) and quadratic
   for a long dense row (c @ x with c of length n: n^2).

   Each row's terms are added in increasing column order of A (the order of J's
   row indices within the block), the same order as the merge-based dot, so
   the values are bit-identical to it. acc must hold A_csc->m doubles; it need
   not be initialized. */
void block_left_multiply_fill_values_csc(const CSC_matrix *A_csc,
                                         const CSC_matrix *J, CSC_matrix *C,
                                         double *acc)
{
    /* A is m x n, J is (n*p) x k, C is (m*p) x k */
    int m = A_csc->m;
    int n = A_csc->n;

    for (int j = 0; j < J->n; j++)
    {
        int jj = J->p[j];
        int i = C->p[j];
        while (i < C->p[j + 1])
        {
            /* C's row indices are sorted, so one block's entries are contiguous */
            int block = C->i[i] / m;
            int row_offset = block * m;
            int block_start = block * n;
            int block_end = block_start + n;
            int i_end = i;
            while (i_end < C->p[j + 1] && C->i[i_end] < row_offset + m)
            {
                acc[C->i[i_end] - row_offset] = 0.0;
                i_end++;
            }

            /* J's entries of this block (blocks are visited in increasing order) */
            while (jj < J->p[j + 1] && J->i[jj] < block_start)
            {
                jj++;
            }
            for (; jj < J->p[j + 1] && J->i[jj] < block_end; jj++)
            {
                int c = J->i[jj] - block_start;
                double v = J->x[jj];
                for (int q = A_csc->p[c]; q < A_csc->p[c + 1]; q++)
                {
                    acc[A_csc->i[q]] += A_csc->x[q] * v;
                }
            }

            for (; i < i_end; i++)
            {
                C->x[i] = acc[C->i[i] - row_offset];
            }
        }
    }
}

void block_left_multiply_fill_values(const CSR_matrix *A, const CSC_matrix *J,
                                     CSC_matrix *C)
{
    /* One-off convenience form: builds A's CSC mirror per call. Callers that
       fill repeatedly (sparse_matrix) keep the mirror and the accumulator. */
    int *iwork = (int *) sp_malloc((A->n > 0 ? A->n : 1) * sizeof(int));
    CSC_matrix *A_csc = csr_to_csc_alloc(A, iwork);
    csr_to_csc_fill_values(A, A_csc, iwork);
    double *acc = (double *) sp_malloc((A->m > 0 ? A->m : 1) * sizeof(double));
    block_left_multiply_fill_values_csc(A_csc, J, C, acc);
    sp_free(acc);
    sp_free(iwork);
    free_CSC_matrix(A_csc);
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
