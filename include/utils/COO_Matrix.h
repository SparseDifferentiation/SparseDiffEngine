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
#ifndef COO_MATRIX_H
#define COO_MATRIX_H

#include "CSR_Matrix.h"

/* COO (Coordinate) Sparse Matrix Format
 *
 * For an m x n matrix with nnz nonzeros:
 * - rows: array of size nnz containing row indices
 * - cols: array of size nnz containing column indices
 * - x: array of size nnz containing values
 * - value_map: array of size nnz mapping CSR entries to COO entries (for
 * lower-triangular COO)
 * - m: number of rows
 * - n: number of columns
 * - nnz: number of nonzero entries
 */
typedef struct COO_Matrix
{
    int *rows;
    int *cols;
    double *x;
    int *value_map;
    int m;
    int n;
    int nnz;
} COO_Matrix;

/* Construct a COO matrix from a CSR matrix */
COO_Matrix *new_coo_matrix(const CSR_Matrix *A);

/* Construct a COO matrix containing only the lower-triangular
 * entries (col <= row) of a symmetric CSR matrix. Populates
 * value_map so that refresh_lower_triangular_coo can update
 * values without recomputing structure. */
COO_Matrix *new_coo_matrix_lower_triangular(const CSR_Matrix *A);

/* Refresh COO values from a new CSR value array using value_map */
void refresh_lower_triangular_coo(COO_Matrix *coo, const double *vals);

void free_coo_matrix(COO_Matrix *matrix);

#endif /* COO_MATRIX_H */