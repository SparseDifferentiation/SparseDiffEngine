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
#ifndef CSC_MATRIX_H
#define CSC_MATRIX_H

#include "CSR_matrix.h"

/* CSC_matrix (Compressed Sparse Column) matrix Format
 *
 * For an m x n matrix with nnz nonzeros:
 * - p: array of size (n + 1) indicating start of each column
 * - i: array of size nnz containing row indices
 * - x: array of size nnz containing values
 * - m: number of rows
 * - n: number of columns
 * - nnz: number of nonzero entries
 */
typedef struct CSC_matrix
{
    int *p;
    int *i;
    double *x;
    int m;
    int n;
    int nnz;
} CSC_matrix;

/* constructor and destructor */
CSC_matrix *new_csc_matrix(int m, int n, int nnz);
void free_csc_matrix(CSC_matrix *matrix);

/* Fill sparsity of C = A^T D A for diagonal D */
CSR_matrix *ATA_alloc(const CSC_matrix *A);

/* Fill sparsity of C = B^T D A for diagonal D */
CSR_matrix *BTA_alloc(const CSC_matrix *A, const CSC_matrix *B);

/* Fill sparsity of C = BA, where B is symmetric. */
CSC_matrix *symBA_alloc(const CSR_matrix *B, const CSC_matrix *A);

/* Compute values for C = A^T D A (null d corresponds to D as identity) */
void ATDA_fill_values(const CSC_matrix *A, const double *d, CSR_matrix *C);

/* Compute values for C = B^T D A (null d corresonds to D as identity) */
void BTDA_fill_values(const CSC_matrix *A, const CSC_matrix *B, const double *d,
                      CSR_matrix *C);

/* Fill values of C = BA. The matrix B does not have to be symmetric */
void BA_fill_values(const CSR_matrix *B, const CSC_matrix *A, CSC_matrix *C);

/* Fill values of C = x^T A. The matrix C must have filled sparsity. */
void yTA_fill_values(const CSC_matrix *A, const double *x, CSR_matrix *C);

/* Count nonzero columns of a CSC_matrix matrix */
int count_nonzero_cols_csc(const CSC_matrix *A);

/* convert from CSR_matrix to CSC_matrix format */
CSC_matrix *csr_to_csc_alloc(const CSR_matrix *A, int *iwork);
void csr_to_csc_fill_values(const CSR_matrix *A, CSC_matrix *C, int *iwork);

/* convert from CSC_matrix to CSR_matrix format */
CSR_matrix *csc_to_csr_alloc(const CSC_matrix *A, int *iwork);
void csc_to_csr_fill_values(const CSC_matrix *A, CSR_matrix *C, int *iwork);

#endif /* CSC_MATRIX_H */
