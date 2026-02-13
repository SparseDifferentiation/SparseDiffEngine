#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H
#include <stdbool.h>

/* forward declaration */
struct int_double_pair;

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

/* constructors and destructors */
CSR_Matrix *new_csr_matrix(int m, int n, int nnz);
CSR_Matrix *new_csr(const CSR_Matrix *A);
void free_csr_matrix(CSR_Matrix *matrix);
void copy_csr_matrix(const CSR_Matrix *A, CSR_Matrix *C);

/* transpose functionality (iwork must be of size A->n) */
CSR_Matrix *transpose(const CSR_Matrix *A, int *iwork);
CSR_Matrix *AT_alloc(const CSR_Matrix *A, int *iwork);
void AT_fill_values(const CSR_Matrix *A, CSR_Matrix *AT, int *iwork);

/* Build (I_p kron A) = blkdiag(A, A, ..., A) of size (p*A->m) x (p*A->n) */
CSR_Matrix *block_diag_repeat_csr(const CSR_Matrix *A, int p);

/* Build (A kron I_p) of size (A->m * p) x (A->n * p) with nnz = A->nnz * p. */
CSR_Matrix *kron_identity_csr(const CSR_Matrix *A, int p);

/* y = Ax, where y is returned as dense */
void csr_matvec(const CSR_Matrix *A, const double *x, double *y, int col_offset);
void csr_matvec_wo_offset(const CSR_Matrix *A, const double *x, double *y);

/* Computes values of the row matrix C = z^T A (column indices must have been
   pre-computed) and transposed matrix AT must be provided) */
void csr_matvec_fill_values(const CSR_Matrix *AT, const double *z, CSR_Matrix *C);

/* Insert value into CSR matrix A with just one row at col_idx. Assumes that A
has enough space and that A does not have an element at col_idx. It does update
nnz. */
void csr_insert_value(CSR_Matrix *A, int col_idx, double value);

/* Compute C = diag(d) * A where d is an array and A, C are CSR matrices
 * d must have length m
 * C must be pre-allocated with same dimensions as A */
void diag_csr_mult(const double *d, const CSR_Matrix *A, CSR_Matrix *C);
void diag_csr_mult_fill_values(const double *d, const CSR_Matrix *A, CSR_Matrix *C);

/* Count number of columns with nonzero entries */
int count_nonzero_cols(const CSR_Matrix *A, bool *col_nz);

/* inserts 'idx' into array 'arr' in sorted order, and moves the other elements */
void insert_idx(int idx, int *arr, int len);

double csr_get_value(const CSR_Matrix *A, int row, int col);

/* Expand symmetric CSR matrix A to full matrix C. A is assumed to store
   only upper triangle. C must be pre-allocated with sufficient nnz */
void symmetrize_csr(const int *Ap, const int *Ai, int m, CSR_Matrix *C);

#endif /* CSR_MATRIX_H */
