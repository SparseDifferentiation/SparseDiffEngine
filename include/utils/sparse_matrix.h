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
#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include "matrix.h"

/* Sparse matrix wrapping CSR_matrix. csc_cache is a lazily-built CSC_matrix mirror
   used by the chain-rule ATA / ATDA paths; it's allocated on first need and refilled
   by refresh_csc_values. csc_iwork is the workspace for csr_to_csc. */
typedef struct sparse_matrix
{
    matrix base;
    CSR_matrix *csr;
    CSC_matrix *csc_cache;
    int *csc_iwork;
    int *transpose_iwork; /* sized csr->n; allocated by sparse_transpose_alloc
                             on the output sm and reused by
                             sparse_transpose_fill_values. NULL when this
                             sm wasn't produced by transpose_alloc. */
} sparse_matrix;

/* Constructor. Takes ownership of A; the caller must not free A separately
   (free_matrix on the returned matrix frees A). */
matrix *new_sparse_matrix(CSR_matrix *A);

/* Convenience: allocate a sparse_matrix of shape (m, n) with capacity for
   nnz entries. Equivalent to new_sparse_matrix(new_CSR_matrix(m, n, nnz)).
   Sparsity pattern and values are uninitialized. */
matrix *new_sparse_matrix_alloc(int m, int n, int nnz);

/* Transpose helper */
matrix *sparse_matrix_trans(const sparse_matrix *self, int *iwork);

/* Build the CSC_matrix cache structure if absent. Idempotent; structure-only,
   values are NOT filled (use refresh_csc_values for that). Exposed so the
   bivariate dispatchers in matrix_BTA can prepare sparsity without touching
   uninitialized values during the init phase. */
void sparse_matrix_ensure_csc_cache(sparse_matrix *sm);

#endif /* SPARSE_MATRIX_H */
