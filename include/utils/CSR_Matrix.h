#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

/* CSR (Compressed Sparse Row) Matrix Format
 *
 * For an m x n matrix with nnz nonzeros:
 * - p: array of size (m + 1) indicating start of each row
 * - i: array of size nnz containing column indices
 * - x: array of size nnz containing values
 * - m: number of rows
 * - n: number of columns
 * - nnz: number of nonzero entries
 */
typedef struct CSR_Matrix
{
    int *p;
    int *i;
    double *x;
    int m;
    int n;
    int nnz;
} CSR_Matrix;

/* Allocate a new CSR matrix with given dimensions and nnz */
CSR_Matrix *new_csr_matrix(int m, int n, int nnz);

/* Free a CSR matrix */
void free_csr_matrix(CSR_Matrix *matrix);

/* Copy CSR matrix A to C */
void copy_csr_matrix(const CSR_Matrix *A, CSR_Matrix *C);

/* matvec y = Ax, where A indices minus col_offset gives x indices */
void csr_matvec(const CSR_Matrix *A, const double *x, double *y, int col_offset);

/* Compute C = diag(d) * A where d is an array and A, C are CSR matrices
 * d must have length m
 * C must be pre-allocated with same dimensions as A */
void diag_csr_mult(const double *d, const CSR_Matrix *A, CSR_Matrix *C);

/* Compute C = A + B where A, B, C are CSR matrices
 * A and B must have same dimensions
 * C must be pre-allocated with sufficient nnz capacity */
void sum_csr_matrices(const CSR_Matrix *A, const CSR_Matrix *B, CSR_Matrix *C);

#endif /* CSR_MATRIX_H */
