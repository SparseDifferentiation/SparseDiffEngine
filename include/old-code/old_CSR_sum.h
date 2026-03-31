#ifndef OLD_CSR_SUM_H
#define OLD_CSR_SUM_H

#include "utils/CSR_Matrix.h"

/* Compute C = A + B where A, B, C are CSR matrices
 * A and B must have same dimensions
 * C must be pre-allocated with sufficient nnz capacity.
 * C must be different from A and B */
void sum_csr_matrices(const CSR_Matrix *A, const CSR_Matrix *B, CSR_Matrix *C);

/* Compute C = diag(d1) * A + diag(d2) * B where A, B, C are CSR matrices */
void sum_scaled_csr_matrices(const CSR_Matrix *A, const CSR_Matrix *B, CSR_Matrix *C,
                             const double *d1, const double *d2);

/* forward declaration */
struct int_double_pair;

/* Sum all rows of A into a single row matrix C */
void sum_all_rows_csr(const CSR_Matrix *A, CSR_Matrix *C,
                      struct int_double_pair *pairs);

/* Sum blocks of rows of A into a matrix C */
void sum_block_of_rows_csr(const CSR_Matrix *A, CSR_Matrix *C,
                           struct int_double_pair *pairs, int row_block_size);

/* Sum evenly spaced rows of A into a matrix C */
void sum_evenly_spaced_rows_csr(const CSR_Matrix *A, CSR_Matrix *C,
                                struct int_double_pair *pairs, int row_spacing);

/* Sum evenly spaced rows of A starting at offset into a row matrix C */
void sum_spaced_rows_into_row_csr(const CSR_Matrix *A, CSR_Matrix *C,
                                  struct int_double_pair *pairs, int offset,
                                  int spacing);

/* Fill values of summed rows using precomputed idx_map and sparsity of C */
void sum_all_rows_csr_fill_values(const CSR_Matrix *A, CSR_Matrix *C,
                                  const int *idx_map);

/* Fill values of summed block rows using precomputed idx_map */
void sum_block_of_rows_csr_fill_values(const CSR_Matrix *A, CSR_Matrix *C,
                                       const int *idx_map);

#endif /* OLD_CSR_SUM_H */
