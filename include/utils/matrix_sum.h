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
#ifndef MATRIX_SUM_H
#define MATRIX_SUM_H

#include "matrix.h"

/* Polymorphic wrappers over CSR_sum. A, B, and C must all be Sparse_Matrix-
   backed for now; the union sparsity of A+B is general sparse, so a
   Permuted_Dense output is not supported.

   sum_matrices_alloc fills C's sparsity pattern and re-syncs C's base.nnz
   from the underlying CSR (sum_csr_alloc may shrink nnz below the
   over-allocated max). */
void sum_matrices_alloc(Matrix *A, Matrix *B, Matrix *C);

/* Fills C's values; assumes C already has the union sparsity pattern of
   A and B (typically produced by sum_matrices_alloc). */
void sum_matrices_fill_values(Matrix *A, Matrix *B, Matrix *C);

#endif /* MATRIX_SUM_H */
