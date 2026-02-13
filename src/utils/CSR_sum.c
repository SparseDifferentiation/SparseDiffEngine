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
#include "utils/CSR_sum.h"
#include "utils/CSR_Matrix.h"
#include "utils/int_double_pair.h"
#include "utils/utils.h"
#include <assert.h>
#include <string.h>

void sum_csr_matrices(const CSR_Matrix *A, const CSR_Matrix *B, CSR_Matrix *C)
{
    /* A and B must be different from C */
    assert(A != C && B != C);

    C->nnz = 0;

    for (int row = 0; row < A->m; row++)
    {
        int a_ptr = A->p[row];
        int a_end = A->p[row + 1];
        int b_ptr = B->p[row];
        int b_end = B->p[row + 1];
        C->p[row] = C->nnz;

        /* Merge while both have elements */
        while (a_ptr < a_end && b_ptr < b_end)
        {
            if (A->i[a_ptr] < B->i[b_ptr])
            {
                C->i[C->nnz] = A->i[a_ptr];
                C->x[C->nnz] = A->x[a_ptr];
                a_ptr++;
            }
            else if (B->i[b_ptr] < A->i[a_ptr])
            {
                C->i[C->nnz] = B->i[b_ptr];
                C->x[C->nnz] = B->x[b_ptr];
                b_ptr++;
            }
            else
            {
                C->i[C->nnz] = A->i[a_ptr];
                C->x[C->nnz] = A->x[a_ptr] + B->x[b_ptr];
                a_ptr++;
                b_ptr++;
            }
            C->nnz++;
        }

        /* Copy remaining elements from A */
        if (a_ptr < a_end)
        {
            int a_remaining = a_end - a_ptr;
            memcpy(C->i + C->nnz, A->i + a_ptr, a_remaining * sizeof(int));
            memcpy(C->x + C->nnz, A->x + a_ptr, a_remaining * sizeof(double));
            C->nnz += a_remaining;
        }

        /* Copy remaining elements from B */
        if (b_ptr < b_end)
        {
            int b_remaining = b_end - b_ptr;
            memcpy(C->i + C->nnz, B->i + b_ptr, b_remaining * sizeof(int));
            memcpy(C->x + C->nnz, B->x + b_ptr, b_remaining * sizeof(double));
            C->nnz += b_remaining;
        }
    }

    C->p[A->m] = C->nnz;
}

void sum_csr_matrices_fill_sparsity(const CSR_Matrix *A, const CSR_Matrix *B,
                                    CSR_Matrix *C)
{
    /* A and B must be different from C */
    assert(A != C && B != C);

    C->nnz = 0;

    for (int row = 0; row < A->m; row++)
    {
        int a_ptr = A->p[row];
        int a_end = A->p[row + 1];
        int b_ptr = B->p[row];
        int b_end = B->p[row + 1];
        C->p[row] = C->nnz;

        /* Merge while both have elements (only column indices) */
        while (a_ptr < a_end && b_ptr < b_end)
        {
            if (A->i[a_ptr] < B->i[b_ptr])
            {
                C->i[C->nnz] = A->i[a_ptr];
                a_ptr++;
            }
            else if (B->i[b_ptr] < A->i[a_ptr])
            {
                C->i[C->nnz] = B->i[b_ptr];
                b_ptr++;
            }
            else
            {
                C->i[C->nnz] = A->i[a_ptr];
                a_ptr++;
                b_ptr++;
            }
            C->nnz++;
        }

        /* Copy remaining elements from A */
        if (a_ptr < a_end)
        {
            int a_remaining = a_end - a_ptr;
            memcpy(C->i + C->nnz, A->i + a_ptr, a_remaining * sizeof(int));
            C->nnz += a_remaining;
        }

        /* Copy remaining elements from B */
        if (b_ptr < b_end)
        {
            int b_remaining = b_end - b_ptr;
            memcpy(C->i + C->nnz, B->i + b_ptr, b_remaining * sizeof(int));
            C->nnz += b_remaining;
        }
    }

    C->p[A->m] = C->nnz;
}

void sum_csr_matrices_fill_values(const CSR_Matrix *A, const CSR_Matrix *B,
                                  CSR_Matrix *C)
{
    /* Assumes C->p and C->i already contain the sparsity pattern of A+B.
       Fills only C->x accordingly. */
    for (int row = 0; row < A->m; row++)
    {
        int a_ptr = A->p[row];
        int a_end = A->p[row + 1];
        int b_ptr = B->p[row];
        int b_end = B->p[row + 1];

        for (int c_ptr = C->p[row]; c_ptr < C->p[row + 1]; c_ptr++)
        {
            int col = C->i[c_ptr];
            double val = 0.0;

            if (a_ptr < a_end && A->i[a_ptr] == col)
            {
                val += A->x[a_ptr];
                a_ptr++;
            }

            if (b_ptr < b_end && B->i[b_ptr] == col)
            {
                val += B->x[b_ptr];
                b_ptr++;
            }
            C->x[c_ptr] = val;
        }
    }
}

void sum_scaled_csr_matrices_fill_values(const CSR_Matrix *A, const CSR_Matrix *B,
                                         CSR_Matrix *C, const double *d1,
                                         const double *d2)
{
    /* Assumes C->p and C->i already contain the sparsity pattern of A+B.
       Fills only C->x accordingly with scaling. */
    for (int row = 0; row < A->m; row++)
    {
        int a_ptr = A->p[row];
        int a_end = A->p[row + 1];
        int b_ptr = B->p[row];
        int b_end = B->p[row + 1];

        for (int c_ptr = C->p[row]; c_ptr < C->p[row + 1]; c_ptr++)
        {
            int col = C->i[c_ptr];
            double val = 0.0;

            if (a_ptr < a_end && A->i[a_ptr] == col)
            {
                val += d1[row] * A->x[a_ptr];
                a_ptr++;
            }

            if (b_ptr < b_end && B->i[b_ptr] == col)
            {
                val += d2[row] * B->x[b_ptr];
                b_ptr++;
            }
            C->x[c_ptr] = val;
        }
    }
}

void sum_scaled_csr_matrices(const CSR_Matrix *A, const CSR_Matrix *B, CSR_Matrix *C,
                             const double *d1, const double *d2)
{
    C->nnz = 0;

    for (int row = 0; row < A->m; row++)
    {
        int a_ptr = A->p[row];
        int a_end = A->p[row + 1];
        int b_ptr = B->p[row];
        int b_end = B->p[row + 1];
        C->p[row] = C->nnz;

        /* Merge while both have elements */
        while (a_ptr < a_end && b_ptr < b_end)
        {
            if (A->i[a_ptr] < B->i[b_ptr])
            {
                C->i[C->nnz] = A->i[a_ptr];
                C->x[C->nnz] = d1[row] * A->x[a_ptr];
                a_ptr++;
            }
            else if (B->i[b_ptr] < A->i[a_ptr])
            {
                C->i[C->nnz] = B->i[b_ptr];
                C->x[C->nnz] = d2[row] * B->x[b_ptr];
                b_ptr++;
            }
            else
            {
                C->i[C->nnz] = A->i[a_ptr];
                C->x[C->nnz] = d1[row] * A->x[a_ptr] + d2[row] * B->x[b_ptr];
                a_ptr++;
                b_ptr++;
            }
            C->nnz++;
        }

        /* Copy remaining elements from A */
        if (a_ptr < a_end)
        {
            int a_remaining = a_end - a_ptr;
            for (int j = 0; j < a_remaining; j++)
            {
                C->i[C->nnz + j] = A->i[a_ptr + j];
                C->x[C->nnz + j] = d1[row] * A->x[a_ptr + j];
            }
            C->nnz += a_remaining;
        }

        /* Copy remaining elements from B */
        if (b_ptr < b_end)
        {
            int b_remaining = b_end - b_ptr;
            for (int j = 0; j < b_remaining; j++)
            {
                C->i[C->nnz + j] = B->i[b_ptr + j];
                C->x[C->nnz + j] = d2[row] * B->x[b_ptr + j];
            }
            C->nnz += b_remaining;
        }
    }

    C->p[A->m] = C->nnz;
}

void sum_all_rows_csr(const CSR_Matrix *A, CSR_Matrix *C, int_double_pair *pairs)
{
    assert(C->m == 1);
    C->n = A->n;
    C->p[0] = 0;

    /* copy A's values and column indices into pairs */
    set_int_double_pair_array(pairs, A->i, A->x, A->nnz);

    /* sort so columns are in order */
    sort_int_double_pair_array(pairs, A->nnz);

    /* merge entries with same columns and insert result in C */
    C->nnz = 0;
    for (int j = 0; j < A->nnz;)
    {
        int current_col = pairs[j].col;
        double sum_val = 0.0;

        /* sum all values with the same column */
        while (j < A->nnz && pairs[j].col == current_col)
        {
            sum_val += pairs[j].val;
            j++;
        }

        /* insert into C */
        C->i[C->nnz] = current_col;
        C->x[C->nnz] = sum_val;
        C->nnz++;
    }

    C->p[1] = C->nnz;
}

void sum_block_of_rows_csr(const CSR_Matrix *A, CSR_Matrix *C,
                           int_double_pair *pairs, int row_block_size)
{
    assert(A->m % row_block_size == 0);
    int n_blocks = A->m / row_block_size;
    assert(C->m == n_blocks);
    C->n = A->n;
    C->p[0] = 0;

    C->nnz = 0;
    for (int block = 0; block < n_blocks; block++)
    {
        int start_row = block * row_block_size;
        int end_row = start_row + row_block_size;

        /* copy block rows' values and column indices into pairs */
        int pair_idx = 0;
        for (int row = start_row; row < end_row; row++)
        {
            for (int j = A->p[row]; j < A->p[row + 1]; j++)
            {
                pairs[pair_idx].col = A->i[j];
                pairs[pair_idx].val = A->x[j];
                pair_idx++;
            }
        }

        /* sort so columns are in order */
        sort_int_double_pair_array(pairs, pair_idx);

        /* merge entries with same columns and insert result in C */
        for (int j = 0; j < pair_idx;)
        {
            int current_col = pairs[j].col;
            double sum_val = 0.0;

            /* sum all values with the same column */
            while (j < pair_idx && pairs[j].col == current_col)
            {
                sum_val += pairs[j].val;
                j++;
            }

            /* insert into C */
            C->i[C->nnz] = current_col;
            C->x[C->nnz] = sum_val;
            C->nnz++;
        }

        C->p[block + 1] = C->nnz;
    }
}

/* iwork must have size max(A->n, A->nnz), and idx_map must have size A->nnz */
void sum_block_of_rows_csr_fill_sparsity_and_idx_map(const CSR_Matrix *A,
                                                     CSR_Matrix *C,
                                                     int row_block_size, int *iwork,
                                                     int *idx_map)
{
    assert(A->m % row_block_size == 0);
    int n_blocks = A->m / row_block_size;
    assert(C->m == n_blocks);

    C->n = A->n;
    C->p[0] = 0;
    C->nnz = 0;

    int *cols = iwork;
    int *col_to_pos = iwork;

    for (int block = 0; block < n_blocks; block++)
    {
        int start_row = block * row_block_size;
        int end_row = start_row + row_block_size;

        // -----------------------------------------------------------------
        // Build sparsity pattern of the row resulting from summing
        // the block of rows from A
        // -----------------------------------------------------------------
        C->p[block] = C->nnz;
        int count = 0;
        for (int row = start_row; row < end_row; row++)
        {
            for (int j = A->p[row]; j < A->p[row + 1]; j++)
            {
                cols[count++] = A->i[j];
            }
        }

        /* Sort columns and write unique pattern into C->i */
        sort_int_array(cols, count);

        int unique_nnz = 0;
        int prev_col = -1;
        for (int t = 0; t < count; t++)
        {
            int col = cols[t];
            if (t == 0 || col != prev_col)
            {
                C->i[C->nnz + unique_nnz] = col;
                prev_col = col;
                unique_nnz++;
            }
        }

        C->nnz += unique_nnz;
        C->p[block + 1] = C->nnz;

        // -----------------------------------------------------------------
        //         Build idx_map for all entries in this block
        // -----------------------------------------------------------------
        int row_start = C->p[block];
        for (int idx = 0; idx < unique_nnz; idx++)
        {
            col_to_pos[C->i[row_start + idx]] = row_start + idx;
        }

        for (int row = start_row; row < end_row; row++)
        {
            for (int j = A->p[row]; j < A->p[row + 1]; j++)
            {
                idx_map[j] = col_to_pos[A->i[j]];
            }
        }
    }
}

/*
void sum_block_of_rows_csr_fill_values(const CSR_Matrix *A, CSR_Matrix *C,
                                       const int *idx_map)
{
    memset(C->x, 0, C->nnz * sizeof(double));

    for (int row = 0; row < A->m; row++)
    {
        for (int j = A->p[row]; j < A->p[row + 1]; j++)
        {
            C->x[idx_map[j]] += A->x[j];
        }
    }
}
*/

void sum_evenly_spaced_rows_csr(const CSR_Matrix *A, CSR_Matrix *C,
                                struct int_double_pair *pairs, int row_spacing)
{
    assert(C->m == row_spacing);
    C->n = A->n;
    C->p[0] = 0;
    C->nnz = 0;

    for (int C_row = 0; C_row < C->m; C_row++)
    {
        /* copy evenly spaced rows into pairs */
        int pair_idx = 0;
        for (int row = C_row; row < A->m; row += row_spacing)
        {
            for (int j = A->p[row]; j < A->p[row + 1]; j++)
            {
                pairs[pair_idx].col = A->i[j];
                pairs[pair_idx].val = A->x[j];
                pair_idx++;
            }
        }

        /* sort so columns are in order */
        sort_int_double_pair_array(pairs, pair_idx);

        /* merge entries with same columns and insert result in C */
        for (int j = 0; j < pair_idx;)
        {
            int current_col = pairs[j].col;
            double sum_val = 0.0;

            /* sum all values with the same column */
            while (j < pair_idx && pairs[j].col == current_col)
            {
                sum_val += pairs[j].val;
                j++;
            }

            /* insert into C */
            C->i[C->nnz] = current_col;
            C->x[C->nnz] = sum_val;
            C->nnz++;
        }

        C->p[C_row + 1] = C->nnz;
    }
}

/* iwork must have size max(A->n, A->nnz), and idx_map must have size A->nnz */
void sum_evenly_spaced_rows_csr_fill_sparsity_and_idx_map(const CSR_Matrix *A,
                                                          CSR_Matrix *C,
                                                          int row_spacing,
                                                          int *iwork, int *idx_map)
{
    assert(C->m == row_spacing);
    C->n = A->n;
    C->p[0] = 0;
    C->nnz = 0;

    int *cols = iwork;
    int *col_to_pos = iwork;

    for (int C_row = 0; C_row < C->m; C_row++)
    {
        // -----------------------------------------------------------------
        // Build sparsity pattern of the row resulting from summing
        // evenly spaced rows from A
        // -----------------------------------------------------------------
        C->p[C_row] = C->nnz;
        int count = 0;
        for (int row = C_row; row < A->m; row += row_spacing)
        {
            for (int j = A->p[row]; j < A->p[row + 1]; j++)
            {
                cols[count++] = A->i[j];
            }
        }

        /* Sort columns and write unique pattern into C->i */
        sort_int_array(cols, count);

        int unique_nnz = 0;
        int prev_col = -1;
        for (int t = 0; t < count; t++)
        {
            int col = cols[t];
            if (t == 0 || col != prev_col)
            {
                C->i[C->nnz + unique_nnz] = col;
                prev_col = col;
                unique_nnz++;
            }
        }

        C->nnz += unique_nnz;
        C->p[C_row + 1] = C->nnz;

        // -----------------------------------------------------------------
        //         Build idx_map for all entries in evenly spaced rows
        // -----------------------------------------------------------------
        int row_start = C->p[C_row];
        for (int idx = 0; idx < unique_nnz; idx++)
        {
            col_to_pos[C->i[row_start + idx]] = row_start + idx;
        }

        for (int row = C_row; row < A->m; row += row_spacing)
        {
            for (int j = A->p[row]; j < A->p[row + 1]; j++)
            {
                idx_map[j] = col_to_pos[A->i[j]];
            }
        }
    }
}

void idx_map_accumulator(const CSR_Matrix *A, const int *idx_map,
                         double *accumulator)
{
    /* don't forget to initialze accumulator to 0 before calling this */
    for (int row = 0; row < A->m; row++)
    {
        for (int j = A->p[row]; j < A->p[row + 1]; j++)
        {
            accumulator[idx_map[j]] += A->x[j];
        }
    }
}

void idx_map_accumulator_with_spacing(const CSR_Matrix *A, const int *idx_map,
                                      double *accumulator, int spacing)
{
    /* don't forget to initialze accumulator to 0 before calling this */
    for (int row = 0; row < A->m; row += spacing)
    {
        for (int j = A->p[row]; j < A->p[row + 1]; j++)
        {
            accumulator[idx_map[j]] += A->x[j];
        }
    }
}

void sum_spaced_rows_into_row_csr(const CSR_Matrix *A, CSR_Matrix *C,
                                  struct int_double_pair *pairs, int offset,
                                  int spacing)
{
    assert(C->m == 1);
    C->n = A->n;
    C->p[0] = 0;
    C->nnz = 0;

    /* copy evenly spaced rows starting at offset into pairs */
    int pair_idx = 0;
    for (int row = offset; row < A->m; row += spacing)
    {
        for (int j = A->p[row]; j < A->p[row + 1]; j++)
        {
            pairs[pair_idx].col = A->i[j];
            pairs[pair_idx].val = A->x[j];
            pair_idx++;
        }
    }

    /* sort so columns are in order */
    sort_int_double_pair_array(pairs, pair_idx);

    /* merge entries with same columns and insert result in C */
    for (int j = 0; j < pair_idx;)
    {
        int current_col = pairs[j].col;
        double sum_val = 0.0;

        /* sum all values with the same column */
        while (j < pair_idx && pairs[j].col == current_col)
        {
            sum_val += pairs[j].val;
            j++;
        }

        /* insert into C */
        C->i[C->nnz] = current_col;
        C->x[C->nnz] = sum_val;
        C->nnz++;
    }

    C->p[1] = C->nnz;
}

void sum_all_rows_csr_fill_sparsity_and_idx_map(const CSR_Matrix *A, CSR_Matrix *C,
                                                int *iwork, int *idx_map)
{
    // -------------------------------------------------------------------
    //           Build sparsity pattern of the summed row
    // -------------------------------------------------------------------
    int *cols = iwork;
    memcpy(cols, A->i, A->nnz * sizeof(int));
    sort_int_array(cols, A->nnz);

    int unique_nnz = 0;
    int prev_col = -1;
    for (int j = 0; j < A->nnz; j++)
    {
        if (cols[j] != prev_col)
        {
            C->i[unique_nnz] = cols[j];
            prev_col = cols[j];
            unique_nnz++;
        }
    }

    C->p[0] = 0;
    C->p[1] = unique_nnz;
    C->nnz = unique_nnz;

    // -------------------------------------------------------------------
    //  Map child values to summed-row positions. col_to_pos maps
    //  column indices to positions in C's row.
    // -------------------------------------------------------------------
    int *col_to_pos = iwork;
    for (int idx = 0; idx < unique_nnz; idx++)
    {
        col_to_pos[C->i[idx]] = idx;
    }

    for (int i = 0; i < A->m; i++)
    {
        for (int j = A->p[i]; j < A->p[i + 1]; j++)
        {
            idx_map[j] = col_to_pos[A->i[j]];
        }
    }
}

/*
void sum_all_rows_csr_fill_values(const CSR_Matrix *A, CSR_Matrix *C,
                                  const int *idx_map)
{
    memset(C->x, 0, C->nnz * sizeof(double));

    for (int row = 0; row < A->m; row++)
    {
        for (int j = A->p[row]; j < A->p[row + 1]; j++)
        {
            C->x[idx_map[j]] += A->x[j];
        }
    }
}
*/

/*
 * Sums evenly spaced rows from A into a single row in C and fills an index map.
 * A: input CSR matrix
 * C: output CSR matrix (must have m=1)
 * spacing: row spacing
 * iwork: workspace of size at least max(A->n, A->nnz)
 * idx_map: output index map, size at least A->nnz
 */
void sum_spaced_rows_into_row_csr_fill_sparsity_and_idx_map(const CSR_Matrix *A,
                                                            CSR_Matrix *C,
                                                            int spacing, int *iwork,
                                                            int *idx_map)
{
    assert(C->m == 1);
    C->n = A->n;

    /* gather all column indices from the spaced rows */
    int count = 0;
    for (int row = 0; row < A->m; row += spacing)
    {
        int len = A->p[row + 1] - A->p[row];
        memcpy(iwork + count, A->i + A->p[row], len * sizeof(int));
        count += len;
    }

    /* fill sparsity pattern */
    sort_int_array(iwork, count);
    int unique_nnz = 0;
    int prev_col = -1;
    for (int i = 0; i < count; i++)
    {
        int col = iwork[i];
        if (col != prev_col)
        {
            C->i[unique_nnz++] = col;
            prev_col = col;
        }
    }
    C->nnz = unique_nnz;
    C->p[0] = 0;
    C->p[1] = C->nnz;

    /* fill idx_map */
    int *col_to_pos = iwork;
    for (int idx = 0; idx < unique_nnz; idx++)
    {
        col_to_pos[C->i[idx]] = idx;
    }

    for (int row = 0; row < A->m; row += spacing)
    {
        for (int j = A->p[row]; j < A->p[row + 1]; j++)
        {
            idx_map[j] = col_to_pos[A->i[j]];
        }
    }
}
