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

#include "CSR_Matrix.h"

/* CSC (Compressed Sparse Column) Matrix Format
 *
 * For an m x n matrix with nnz nonzeros:
 * - p: array of size (n + 1) indicating start of each column
 * - i: array of size nnz containing row indices
 * - x: array of size nnz containing values
 * - m: number of rows
 * - n: number of columns
 * - nnz: number of nonzero entries
 */
typedef struct CSC_Matrix
{
    int *p;
    int *i;
    double *x;
    int m;
    int n;
    int nnz;
} CSC_Matrix;

/* constructor and destructor */
CSC_Matrix *new_csc_matrix(int m, int n, int nnz);
void free_csc_matrix(CSC_Matrix *matrix);

/* Fill sparsity of C = A^T D A for diagonal D */
CSR_Matrix *ATA_alloc(const CSC_Matrix *A);

/* Fill sparsity of C = B^T D A for diagonal D */
CSR_Matrix *BTA_alloc(const CSC_Matrix *A, const CSC_Matrix *B);

/* Fill sparsity of C = BA, where B is symmetric. */
CSC_Matrix *symBA_alloc(const CSR_Matrix *B, const CSC_Matrix *A);

/* Compute values for C = A^T D A (null d corresponds to D as identity) */
void ATDA_fill_values(const CSC_Matrix *A, const double *d, CSR_Matrix *C);

/* Compute values for C = B^T D A (null d corresonds to D as identity) */
void BTDA_fill_values(const CSC_Matrix *A, const CSC_Matrix *B, const double *d,
                      CSR_Matrix *C);

/* Fill values of C = BA. The matrix B does not have to be symmetric */
void BA_fill_values(const CSR_Matrix *B, const CSC_Matrix *A, CSC_Matrix *C);

/* Fill values of C = x^T A. The matrix C must have filled sparsity. */
void yTA_fill_values(const CSC_Matrix *A, const double *x, CSR_Matrix *C);

/* Count nonzero columns of a CSC matrix */
int count_nonzero_cols_csc(const CSC_Matrix *A);

/* convert from CSR to CSC format */
CSC_Matrix *csr_to_csc_alloc(const CSR_Matrix *A, int *iwork);
void csr_to_csc_fill_values(const CSR_Matrix *A, CSC_Matrix *C, int *iwork);

/* convert from CSC to CSR format */
CSR_Matrix *csc_to_csr_alloc(const CSC_Matrix *A, int *iwork);
void csc_to_csr_fill_values(const CSC_Matrix *A, CSR_Matrix *C, int *iwork);

#endif /* CSC_MATRIX_H */