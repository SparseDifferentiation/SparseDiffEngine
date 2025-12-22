#include "utils/CSR_Matrix.h"
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
