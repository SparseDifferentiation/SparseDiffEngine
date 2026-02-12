#ifndef LINALG_H
#define LINALG_H

/* Forward declarations */
struct CSR_Matrix;
struct CSC_Matrix;

/* Compute sparsity pattern for block left multiplication.
 * C = [A @ J1; A @ J2; ...; A @ Jp] where A is m x n and J is (n*p) x k.
 * J is provided in CSC format and is split into p blocks of n rows each.
 * Result C is returned as CSC format with dimensions (m*p) x k.
 */
struct CSC_Matrix *block_left_multiply_fill_sparsity(const struct CSR_Matrix *A,
                                                     const struct CSC_Matrix *J,
                                                     int p);

/* Fill values for block left multiplication.
 * C must have sparsity pattern already computed for [A @ J1; ...; A @ Jp].
 */
void block_left_multiply_fill_values(const struct CSR_Matrix *A,
                                     const struct CSC_Matrix *J,
                                     struct CSC_Matrix *C);

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

/* Compute block-wise matrix-vector products.
 * y = [A @ x1; A @ x2; ...; A @ xp] where A is m x n and x is divided into p blocks
 * of n elements. A is provided in CSR format. x is (n*p)-length vector, y is
 * (m*p)-length output.
 */
void block_left_multiply_vec(const struct CSR_Matrix *A, const double *x, double *y,
                             int p);

#endif /* LINALG_H */
