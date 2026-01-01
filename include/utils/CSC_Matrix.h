#ifndef CSC_MATRIX_H
#define CSC_MATRIX_H

#include "CSR_Matrix.h"

/* CSC (Compressed Sparse Column) Matrix Format
 *
 * For an m x n matrix with nnz nonzeros:
 * - p: array of size (n + 1) indicating start of each column
 * - i: array of size nnz containing row indices
 * - x: array of size nnz containing values
 * - m: number of rows
 * - n: number of columns
 * - nnz: number of nonzero entries
 */
typedef struct CSC_Matrix
{
    int *p;
    int *i;
    double *x;
    int m;
    int n;
    int nnz;
} CSC_Matrix;

/* Allocate a new CSC matrix with given dimensions and nnz */
CSC_Matrix *new_csc_matrix(int m, int n, int nnz);

/* Free a CSC matrix */
void free_csc_matrix(CSC_Matrix *matrix);

CSC_Matrix *csr_to_csc(const CSR_Matrix *A);

/* Allocate sparsity pattern for C = A^T D A for diagonal D */
CSR_Matrix *ATA_alloc(const CSC_Matrix *A);

/* Compute values for C = A^T D A
 * C must have precomputed sparsity pattern
 */
void ATDA_values(const CSC_Matrix *A, const double *d, CSR_Matrix *C);

#endif /* CSC_MATRIX_H */
