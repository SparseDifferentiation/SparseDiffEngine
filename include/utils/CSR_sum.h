#ifndef CSR_SUM_H
#define CSR_SUM_H

#include "utils/CSR_Matrix.h"

/* forward declaration */
struct int_double_pair;

/* Compute C = A + B where A, B, C are CSR matrices
 * A and B must have same dimensions
 * C must be pre-allocated with sufficient nnz capacity.
 * C must be different from A and B */
void sum_csr_matrices(const CSR_Matrix *A, const CSR_Matrix *B, CSR_Matrix *C);

/* Compute sparsity pattern of A + B where A, B, C are CSR matrices.
 * Fills C->p, C->i, and C->nnz; does not touch C->x. */
void sum_csr_matrices_fill_sparsity(const CSR_Matrix *A, const CSR_Matrix *B,
                                    CSR_Matrix *C);

/* Fill only the values of C = A + B, assuming C's sparsity pattern (p and i)
 * is already filled and matches the union of A and B per row. Does not modify
 * C->p, C->i, or C->nnz. */
void sum_csr_matrices_fill_values(const CSR_Matrix *A, const CSR_Matrix *B,
                                  CSR_Matrix *C);

/* Compute C = diag(d1) * A + diag(d2) * B where A, B, C are CSR matrices */
void sum_scaled_csr_matrices(const CSR_Matrix *A, const CSR_Matrix *B, CSR_Matrix *C,
                             const double *d1, const double *d2);

/* Fill only the values of C = diag(d1) * A + diag(d2) * B, assuming C's sparsity
 * pattern (p and i) is already filled and matches the union of A and B per row.
 * Does not modify C->p, C->i, or C->nnz. */
void sum_scaled_csr_matrices_fill_values(const CSR_Matrix *A, const CSR_Matrix *B,
                                         CSR_Matrix *C, const double *d1,
                                         const double *d2);

/* Sum all rows of A into a single row matrix C */
void sum_all_rows_csr(const CSR_Matrix *A, CSR_Matrix *C,
                      struct int_double_pair *pairs);

/* iwork must have size max(C->n, A->nnz), and idx_map must have size A->nnz. */
void sum_all_rows_csr_fill_sparsity_and_idx_map(const CSR_Matrix *A, CSR_Matrix *C,
                                                int *iwork, int *idx_map);

/* Fill values of summed rows using precomputed idx_map and sparsity of C */
// void sum_all_rows_csr_fill_values(const CSR_Matrix *A, CSR_Matrix *C,
//                                  const int *idx_map);

/* Fill accumulator for summing rows using precomputed idx_map for each nnz of A.
   Must memset accumulator to zero before calling. */
void idx_map_accumulator(const CSR_Matrix *A, const int *idx_map,
                         double *accumulator);
void idx_map_accumulator_with_spacing(const CSR_Matrix *A, const int *idx_map,
                                      double *accumulator, int spacing);

/* Sum blocks of rows of A into a matrix C */
void sum_block_of_rows_csr(const CSR_Matrix *A, CSR_Matrix *C,
                           struct int_double_pair *pairs, int row_block_size);

/* Build sparsity and index map for summing blocks of rows.
 * iwork must have size max(A->n, A->nnz), and idx_map must have size A->nnz. */
void sum_block_of_rows_csr_fill_sparsity_and_idx_map(const CSR_Matrix *A,
                                                     CSR_Matrix *C,
                                                     int row_block_size, int *iwork,
                                                     int *idx_map);

/* Sum evenly spaced rows of A into a matrix C */
void sum_evenly_spaced_rows_csr(const CSR_Matrix *A, CSR_Matrix *C,
                                struct int_double_pair *pairs, int row_spacing);

/* Build sparsity and index map for summing evenly spaced rows.
 * iwork must have size max(A->n, A->nnz), and idx_map must have size A->nnz. */
void sum_evenly_spaced_rows_csr_fill_sparsity_and_idx_map(const CSR_Matrix *A,
                                                          CSR_Matrix *C,
                                                          int row_spacing,
                                                          int *iwork, int *idx_map);

/* Sum evenly spaced rows of A starting at offset into a row matrix C */
void sum_spaced_rows_into_row_csr(const CSR_Matrix *A, CSR_Matrix *C,
                                  struct int_double_pair *pairs, int offset,
                                  int spacing);

/* Fills the sparsity and index map for summing spaced rows into a row matrix */
void sum_spaced_rows_into_row_csr_fill_sparsity_and_idx_map(const CSR_Matrix *A,
                                                            CSR_Matrix *C,
                                                            int spacing, int *iwork,
                                                            int *idx_map);

#endif /* CSR_SUM_H */
