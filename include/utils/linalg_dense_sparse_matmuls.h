#ifndef LINALG_DENSE_SPARSE_H
#define LINALG_DENSE_SPARSE_H

#include "CSC_Matrix.h"
#include "CSR_Matrix.h"
#include "matrix.h"

/* C = (I_p kron A) @ J via the polymorphic Matrix interface.
 * A is dense m x n, J is (n*p) x k in CSC, C is (m*p) x k in CSC. */
// TODO: maybe we can replace these with I_kron_X functionality?
CSC_Matrix *I_kron_A_alloc(const Matrix *A, const CSC_Matrix *J, int p);
void I_kron_A_fill_vals(const Matrix *A, const CSC_Matrix *J, CSC_Matrix *C);

/* Sparsity and values of C = (Y^T kron I_m) @ J where Y is k x n, J is (m*k) x p,
   and C is (m*n) x p. Y is given in column-major dense format. */
CSR_Matrix *YT_kron_I_alloc(int m, int k, int n, const CSC_Matrix *J);
void YT_kron_I_fill_vals(int m, int k, int n, const double *Y, const CSC_Matrix *J,
                         CSR_Matrix *C);

/* Sparsity and values of C = (I_n kron X) @ J where X is m x k (col-major dense),
   J is (k*n) x p, and C is (m*n) x p. */
CSR_Matrix *I_kron_X_alloc(int m, int k, int n, const CSC_Matrix *J);
void I_kron_X_fill_vals(int m, int k, int n, const double *X, const CSC_Matrix *J,
                        CSR_Matrix *C);

#endif /* LINALG_DENSE_SPARSE_H */
