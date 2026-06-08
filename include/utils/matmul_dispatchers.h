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
 */
#ifndef MATMUL_DISPATCHERS_H
#define MATMUL_DISPATCHERS_H

#include "matrix.h"
#include "permuted_dense.h"
#include "sparse_matrix.h"
#include "stacked_pd.h"

/* Polymorphic dispatchers for C = BT @ A and C = BT @ diag(d) @ A, where A
   and B are any matrix type (permuted_dense, stacked_pd, or sparse_matrix).

   Output type varies with the operand pair:
     - B is PD                          -> output is permuted_dense.
     - B is stacked_pd                  -> output is stacked_pd.
     - B is sparse_matrix, A is PD      -> output is permuted_dense.
     - B is sparse_matrix, A is spd     -> output is stacked_pd.
     - B is sparse_matrix, A is sparse  -> output is sparse_matrix.

   Contract: for sparse_matrix operands (A or B), the caller is responsible for
   refreshing csc_cache values via refresh_csc_values before
   BTA_matrices_fill_values / BTDA_matrices_fill_values. */
matrix *BTA_matrices_alloc(matrix *A, matrix *B);
void BTA_matrices_fill_values(matrix *A, matrix *B, matrix *C);
void BTDA_matrices_fill_values(matrix *A, const double *d, matrix *B, matrix *C);

/* Polymorphic dispatchers for C = B @ A, where B is permuted_dense and A is any
   matrix type (permuted_dense, stacked_pd, or sparse_matrix). The output C is
   always permuted_dense.

   Contract: for sparse_matrix A, the caller is responsible for refreshing
   sm_A->csc_cache values via refresh_csc_values before
   BA_pd_matrices_fill_values. */
matrix *BA_pd_matrices_alloc(const permuted_dense *B, matrix *A);
void BA_pd_matrices_fill_values(const permuted_dense *B, const matrix *A,
                                permuted_dense *C);

/* Polymorphic dispatchers for C = kron(I_p, A) @ J, where A is a local dense
   matrix and J is any matrix type (permuted_dense, stacked_pd, or
   sparse_matrix). The output C is always stacked_pd.

   Contract 1: A must be a "full" permuted_dense (m0 == base.m, n0 == base.n,
   and row_perm / col_perm are the identity).

   Contract 2: for sparse_matrix J, the caller is responsible for refreshing
   sm_J->csc_cache values via refresh_csc_values before
   BA_dense_kron_matrices_fill_values. */
matrix *BA_dense_kron_matrices_alloc(const permuted_dense *A, int p, matrix *J);
void BA_dense_kron_matrices_fill_values(const permuted_dense *A, int p,
                                        const matrix *J, stacked_pd *C);

#endif /* MATMUL_DISPATCHERS_H */
