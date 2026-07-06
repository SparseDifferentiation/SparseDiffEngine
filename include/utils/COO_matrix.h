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
#ifndef COO_matrix_H
#define COO_matrix_H

#include "CSR_matrix.h"

/* COO (Coordinate) Sparse matrix Format
 *
 * For an m x n matrix with nnz nonzeros:
 * - rows: array of size nnz containing row indices
 * - cols: array of size nnz containing column indices
 * - x: array of size nnz containing values
 * - value_map: array of size nnz mapping CSR_matrix entries to COO entries (for
 * lower-triangular COO)
 * - m: number of rows
 * - n: number of columns
 * - nnz: number of nonzero entries
 */
typedef struct COO_matrix
{
    int *rows;
    int *cols;
    double *x;
    int *value_map;
    int m;
    int n;
    int nnz;
} COO_matrix;

/* COO from CSR */
COO_matrix *new_COO_matrix(const CSR_matrix *A);

/* COO from a caller-provided pattern (copied), values taken from A->x. The
   pattern must be A's own pattern in row-major (CSR) order; the caller
   certifies this, only nnz is validated (returns NULL on mismatch). */
COO_matrix *new_COO_matrix_from_pattern(const CSR_matrix *A, const int *rows,
                                        const int *cols, int nnz);

/* Construct COO containing only the lower-triangular entries (col <= row) of a
   symmetric CSR. Populates value_map so that refresh_lower_triangular_coo can
   update values without recomputing structure. */
COO_matrix *new_COO_matrix_lower_triangular(const CSR_matrix *A);

/* Lower-triangular COO from a caller-provided pattern (copied). value_map is
   still derived from A (evaluation needs it), but the counting pass and the
   row/col writes are skipped. The pattern must be A's own lower-triangular
   pattern in row-major order; only nnz is validated (NULL on mismatch). */
COO_matrix *new_COO_matrix_lower_triangular_from_pattern(const CSR_matrix *A,
                                                         const int *rows,
                                                         const int *cols,
                                                         int nnz);

/* Refresh COO values from a new CSR_matrix value array using value_map */
void refresh_lower_triangular_coo(COO_matrix *coo, const double *vals);

void free_COO_matrix(COO_matrix *matrix);

#endif /* COO_matrix_H */
