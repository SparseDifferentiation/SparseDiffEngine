#ifndef LINALG_H
#define LINALG_H

/* Forward declarations */
struct CSR_Matrix;
struct CSC_Matrix;

/* Compute sparsity pattern and values for the matrix-matrix multiplication
   C = (I_p kron A) @ J where A is m x n, J is (n*p) x k, and C is (m*p) x k,
   without relying on generic sparse matrix-matrix multiplication. Specialized
   logic for this is much faster (50-100x) than generic sparse matmul.

    * J is provided in CSC format and is split into p blocks of n rows each
    * C is returned in CSC format
    * Mathematically it corresponds to  C = [A @ J1; A @ J2; ...; A @ Jp],
      where J = [J1; J2; ...; Jp]
*/
struct CSC_Matrix *block_left_multiply_fill_sparsity(const struct CSR_Matrix *A,
                                                     const struct CSC_Matrix *J,
                                                     int p);

void block_left_multiply_fill_values(const struct CSR_Matrix *A,
                                     const struct CSC_Matrix *J,
                                     struct CSC_Matrix *C);

/* Compute y = kron(I_p, A) @ x where A is m x n and x is(n*p)-length vector.
   The output y is m*p-length vector corresponding to
   y = [A @ x1; A @ x2; ...; A @ xp] where x is divided into p blocks of n
   elements.
*/
void block_left_multiply_vec(const struct CSR_Matrix *A, const double *x, double *y,
                             int p);

/* Fill values of C = A @ B where A is CSR, B is CSC.
 * C must have sparsity pattern already computed.
 */
void csr_csc_matmul_fill_values(const struct CSR_Matrix *A,
                                const struct CSC_Matrix *B, struct CSR_Matrix *C);

/* C = A @ B where A is CSR, B is CSC. Result C is CSR.
 * Allocates and precomputes sparsity pattern. No workspace required.
 */
struct CSR_Matrix *csr_csc_matmul_alloc(const struct CSR_Matrix *A,
                                        const struct CSC_Matrix *B);

#endif /* LINALG_H */
