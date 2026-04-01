#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H
#include <stdbool.h>

/* Precomputed matching pairs for A^T D A fill.
 * For upper-triangle entry e of C, matches are at p[e]..p[e+1]:
 *   A->x[A->p[i] + ai[k]] * A->x[A->p[j] + aj[k]] * d[rows[k]]
 */
typedef struct MatchPairs
{
    int *p;
    int *ai;
    int *aj;
    int *rows;
    int n_entries;
    int n_matches;
} MatchPairs;

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
    MatchPairs *match;
} CSR_Matrix;

/* constructors and destructors */
CSR_Matrix *new_csr_matrix(int m, int n, int nnz);
CSR_Matrix *new_csr(const CSR_Matrix *A);
CSR_Matrix *new_csr_copy_sparsity(const CSR_Matrix *A);
void free_csr_matrix(CSR_Matrix *matrix);
void copy_csr_matrix(const CSR_Matrix *A, CSR_Matrix *C);

/* transpose functionality (iwork must be of size A->n) */
CSR_Matrix *transpose(const CSR_Matrix *A, int *iwork);
CSR_Matrix *AT_alloc(const CSR_Matrix *A, int *iwork);
void AT_fill_values(const CSR_Matrix *A, CSR_Matrix *AT, int *iwork);

/* computes dense y = Ax */
void Ax_csr(const CSR_Matrix *A, const double *x, double *y, int col_offset);

/* fills values of C = diag(d) @ A */
void DA_fill_values(const double *d, const CSR_Matrix *A, CSR_Matrix *C);

/* Count number of columns with nonzero entries in A and marks them in col_nz */
int count_nonzero_cols(const CSR_Matrix *A, bool *col_nz);

/* inserts 'idx' into array 'arr' in sorted order, and moves the other elements */
void insert_idx(int idx, int *arr, int len);

/* get value at position (row, col) in A */
double csr_get_value(const CSR_Matrix *A, int row, int col);

/* Expand symmetric CSR matrix A to full matrix C. A is assumed to store
   only upper triangle. C must be pre-allocated with sufficient nnz */
void symmetrize_csr(const int *Ap, const int *Ai, int m, CSR_Matrix *C);

#endif /* CSR_MATRIX_H */
