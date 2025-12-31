#include "utils/CSR_Matrix.h"
#include "utils/int_double_pair.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
CSR_Matrix *new_csr_matrix(int m, int n, int nnz)
{
    CSR_Matrix *matrix = (CSR_Matrix *) malloc(sizeof(CSR_Matrix));
    matrix->p = (int *) calloc(m + 1, sizeof(int));
    matrix->i = (int *) malloc(nnz * sizeof(int));
    matrix->x = (double *) malloc(nnz * sizeof(double));
    matrix->m = m;
    matrix->n = n;
    matrix->nnz = nnz;
    return matrix;
}

void free_csr_matrix(CSR_Matrix *matrix)
{
    if (matrix)
    {
        free(matrix->p);
        free(matrix->i);
        free(matrix->x);
        free(matrix);
    }
}

void copy_csr_matrix(const CSR_Matrix *A, CSR_Matrix *C)
{
    C->m = A->m;
    C->n = A->n;
    C->nnz = A->nnz;
    memcpy(C->p, A->p, (A->m + 1) * sizeof(int));
    memcpy(C->i, A->i, A->nnz * sizeof(int));
    memcpy(C->x, A->x, A->nnz * sizeof(double));
}

void csr_matvec(const CSR_Matrix *A, const double *x, double *y, int col_offset)
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

int count_nonzero_cols(const CSR_Matrix *A, bool *col_nz)
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

void sum_csr_matrices(const CSR_Matrix *A, const CSR_Matrix *B, CSR_Matrix *C)
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
            memmove(C->i + C->nnz, A->i + a_ptr, a_remaining * sizeof(int));
            memmove(C->x + C->nnz, A->x + a_ptr, a_remaining * sizeof(double));
            C->nnz += a_remaining;
        }

        /* Copy remaining elements from B */
        if (b_ptr < b_end)
        {
            int b_remaining = b_end - b_ptr;
            memmove(C->i + C->nnz, B->i + b_ptr, b_remaining * sizeof(int));
            memmove(C->x + C->nnz, B->x + b_ptr, b_remaining * sizeof(double));
            C->nnz += b_remaining;
        }
    }

    C->p[A->m] = C->nnz;
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

/* Helper function for qsort to compare column indices */
static int compare_cols(const void *a, const void *b)
{
    int col_a = *((int *) a);
    int col_b = *((int *) b);
    return col_a - col_b;
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

CSR_Matrix *transpose(const CSR_Matrix *A, int *iwork)
{
    CSR_Matrix *AT = new_csr_matrix(A->n, A->m, A->nnz);

    int i, j, start;
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
    //  fill transposed matrix (this is a bottleneck)
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

/**/
void csr_matvec_fill_values(const CSR_Matrix *AT, const double *z, CSR_Matrix *C)
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

// void csr_vecmat_sparse(const double *x, const CSR_Matrix *A, CSR_Matrix *C)
//{
//     memset(C->x, 0, C->nnz * sizeof(double));
//     C->nnz = 0;
//     for (int j = 0; j < A->nnz; j++)
//     {
//         C->x[]
//     }
// }

void symmetrize_csr(const int *Ap, const int *Ai, int m, CSR_Matrix *C)
{
    int i, j, col;

    /* Count entries per row */
    int *counts = (int *) calloc(m, sizeof(int));
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
