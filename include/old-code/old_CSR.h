#ifndef OLD_CSR_H
#define OLD_CSR_H

#include "utils/CSR_Matrix.h"

/* Build (I_p kron A) = blkdiag(A, A, ..., A) of size (p*A->m) x (p*A->n) */
CSR_Matrix *block_diag_repeat_csr(const CSR_Matrix *A, int p);

/* Build (A kron I_p) of size (A->m * p) x (A->n * p) with nnz = A->nnz * p. */
CSR_Matrix *kron_identity_csr(const CSR_Matrix *A, int p);

/* Computes values of the row matrix C = z^T A (column indices must have been
   pre-computed) and transposed matrix AT must be provided) */
void Ax_csr_fill_values(const CSR_Matrix *AT, const double *z, CSR_Matrix *C);

/* Insert value into CSR matrix A with just one row at col_idx. Assumes that A
has enough space and that A does not have an element at col_idx. It does update
nnz. */
void csr_insert_value(CSR_Matrix *A, int col_idx, double value);

/* Compute C = diag(d) * A where d is an array and A, C are CSR matrices
 * d must have length m
 * C must be pre-allocated with same dimensions as A */
void diag_csr_mult(const double *d, const CSR_Matrix *A, CSR_Matrix *C);

/* y = Ax, where y is returned as dense (no column offset) */
void Ax_csr_wo_offset(const CSR_Matrix *A, const double *x, double *y);

#endif /* OLD_CSR_H */
