#ifndef CSC_MATRIX_H
#define CSC_MATRIX_H

#include "CSR_Matrix.h"
#include <stddef.h>

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

/* constructor and destructor.
   If mem is non-NULL, *mem is incremented by the bytes allocated. */
CSC_Matrix *new_csc_matrix(int m, int n, int nnz, size_t *mem);
void free_csc_matrix(CSC_Matrix *matrix);

/* Fill sparsity of C = A^T D A for diagonal D */
CSR_Matrix *ATA_alloc(const CSC_Matrix *A, size_t *mem);

/* Fill sparsity of C = B^T D A for diagonal D */
CSR_Matrix *BTA_alloc(const CSC_Matrix *A, const CSC_Matrix *B, size_t *mem);

/* Fill sparsity of C = BA, where B is symmetric. */
CSC_Matrix *symBA_alloc(const CSR_Matrix *B, const CSC_Matrix *A, size_t *mem);

/* Compute values for C = A^T D A (null d corresponds to D as identity) */
void ATDA_fill_values(const CSC_Matrix *A, const double *d, CSR_Matrix *C);

/* Compute values for C = B^T D A (null d corresonds to D as identity) */
void BTDA_fill_values(const CSC_Matrix *A, const CSC_Matrix *B, const double *d,
                      CSR_Matrix *C);

/* Fill values of C = BA. The matrix B does not have to be symmetric */
void BA_fill_values(const CSR_Matrix *B, const CSC_Matrix *A, CSC_Matrix *C);

/* Fill values of C = x^T A. The matrix C must have filled sparsity. */
void yTA_fill_values(const CSC_Matrix *A, const double *x, CSR_Matrix *C);

/* Count nonzero columns of a CSC matrix */
int count_nonzero_cols_csc(const CSC_Matrix *A);

/* convert from CSR to CSC format */
CSC_Matrix *csr_to_csc_alloc(const CSR_Matrix *A, int *iwork, size_t *mem);
void csr_to_csc_fill_values(const CSR_Matrix *A, CSC_Matrix *C, int *iwork);

/* convert from CSC to CSR format */
CSR_Matrix *csc_to_csr_alloc(const CSC_Matrix *A, int *iwork, size_t *mem);
void csc_to_csr_fill_values(const CSC_Matrix *A, CSR_Matrix *C, int *iwork);

/* Returns total bytes used by p, i, x arrays (0 if A is NULL) */
size_t csc_memory_bytes(const CSC_Matrix *A);

#endif /* CSC_MATRIX_H */
