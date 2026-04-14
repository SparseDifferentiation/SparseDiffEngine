/*
 * Copyright 2026 Daniel Cederberg and William Zhang
 *
 * This file is part of the SparseDiffEngine project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef CSR_SUM_H
#define CSR_SUM_H

#include "utils/CSR_Matrix.h"

/* forward declaration */
struct int_double_pair;

/* Compute sparsity pattern of C = A + B (and sets C->nnz) */
void sum_csr_alloc(const CSR_Matrix *A, const CSR_Matrix *B, CSR_Matrix *C);

/* Fills values of C = A + B (assuming C's sparsity pattern is set) */
void sum_csr_fill_values(const CSR_Matrix *A, const CSR_Matrix *B, CSR_Matrix *C);

/* Fills values of C = diag(d1) * A + diag(d2) * B (assuming C's sparsity is set)*/
void sum_scaled_csr_matrices_fill_values(const CSR_Matrix *A, const CSR_Matrix *B,
                                         CSR_Matrix *C, const double *d1,
                                         const double *d2);

/* The following five functions are used for summing either more than two CSR
   matrices or rows of CSR matrices. To implement the filling of values efficiently,
   we compute an idx_map when we fill the sparsity pattern of the output matrix,
   which maps each nonzero entry in the input matrix to its position in the output
   matrix. This allows us to fill the values with a single pass of the output matrix
   through the input matrices, without needing to search for the position of each
   entry in the output matrix. So each idx_map should have size equal to the number
   of nonzeros in the corresponding input matrix, and idx_map[j] should give the
   index in the output matrix of the entry (in the value array of the output matrix)
   corresponding to the j-th nonzero in the input matrix.

   Output matrix C, input matrix A, iwork->size = max(A->n, A->nnz) for the first
   four functions. The last function allocates the output matrix and returns it. */
// ------------------------------------------------------------------------------------
void sum_all_rows_csr_alloc(const CSR_Matrix *A, CSR_Matrix *C, int *iwork,
                            int *idx_map);

void sum_block_of_rows_csr_alloc(const CSR_Matrix *A, CSR_Matrix *C,
                                 int row_block_size, int *iwork, int *idx_map);

void sum_evenly_spaced_rows_csr_alloc(const CSR_Matrix *A, CSR_Matrix *C,
                                      int row_spacing, int *iwork, int *idx_map);

void sum_spaced_rows_into_row_csr_alloc(const CSR_Matrix *A, CSR_Matrix *C,
                                        int spacing, int *iwork, int *idx_map);

/* Compute sparsity pattern of out = A + B + C + D */
CSR_Matrix *sum_4_csr_alloc(const CSR_Matrix *A, const CSR_Matrix *B,
                            const CSR_Matrix *C, const CSR_Matrix *D,
                            int *idx_maps[4]);
// ------------------------------------------------------------------------------------

/* Accumulates values from A according to map. Must memset to zero before calling. */
void accumulator(const CSR_Matrix *A, const int *idx_map, double *out);

/* Accumulates values from A according to map with spacing. Must memset to zero
 * before calling. */
void accumulator_with_spacing(const CSR_Matrix *A, const int *idx_map, double *out,
                              int spacing);

#endif /* CSR_SUM_H */