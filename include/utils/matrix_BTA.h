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
#ifndef MATRIX_BTA_H
#define MATRIX_BTA_H

#include "matrix.h"
#include "permuted_dense.h"
#include "stacked_pd.h"

/* TODO: clean up documentation of this file.*/

/* Polymorphic dispatchers for C = BT @ A and C = BT @ diag(d) @ A. Each
   operand may be PD, sparse_matrix, or stacked_pd. Operands are reduced
   to an "effective type" (PD or CSC) before dispatch: PD passes through,
   sparse_matrix exposes its csc_cache, stacked_pd is materialized as a
   temporary CSC via to_csr + csr_to_csc_alloc (correctness-first
   fallback; future fused spd kernels can replace this). Output type
   follows the effective-type pair: PD if at least one operand is PD,
   sparse_matrix otherwise. (Here PD = permuted_dense.)

   Contract: for sparse_matrix operands, the caller is still responsible
   for refreshing csc_cache values before BTDA_matrices_fill_values
   (refresh_csc_values). stacked_pd operands need no preparation; their
   csr_cache is refreshed internally on each call. */

/* Allocate sparsity for C = BT @ A. */
matrix *BTA_matrices_alloc(matrix *A, matrix *B);

/* Fill values of C = BT @ diag(d) @ A. */
void BTDA_matrices_fill_values(matrix *A, const double *d, matrix *B, matrix *C);

/* Polymorphic dispatcher for C = B @ A where B is PD and A is any matrix type.
   C is always PD. For the sparse-A branch the dispatcher ensures
   sm_A->csc_cache structure exists at alloc time but before
   BA_pd_matrices_fill_values the caller must have refreshed sm_A->csc_cache
   values (same fill-side contract as BTDA_matrices_fill_values). */
matrix *BA_pd_matrices_alloc(const permuted_dense *B, matrix *A);
void BA_pd_matrices_fill_values(const permuted_dense *B, const matrix *A,
                                permuted_dense *C);

/* Polymorphic dispatcher for C = B^T @ (diag(d) @) A where B is stacked_pd
   and A is any matrix type (PD, stacked_pd, or sparse_matrix). C is always
   stacked_pd (one block per signature group of overlapping c_k's of B).
   For the sparse-A branch the dispatcher ensures sm_A->csc_cache structure
   exists at alloc time; the caller must refresh sm_A->csc_cache values via
   sm_A->refresh_csc_values before calling _fill_values (same fill-side
   contract as BA_pd_matrices_fill_values). */
matrix *BTA_spd_matrices_alloc(const stacked_pd *B, matrix *A);
void BTDA_spd_matrices_fill_values(const stacked_pd *B, const double *d,
                                   const matrix *A, stacked_pd *C);

/* Polymorphic dispatcher: C = kron(I_p, A) @ J as a stacked_pd, where A
   is a permuted_dense and J is any matrix type (permuted_dense,
   stacked_pd, or sparse_matrix). Output shape: (A->m * p) x J->n. For
   the sparse-J branch the dispatcher ensures sm_J->csc_cache structure
   exists at alloc time; the caller must refresh values via
   sm_J->refresh_csc_values before calling _fill_values. */
matrix *BA_pd_kron_matrices_alloc(const permuted_dense *A, int p, matrix *J);
void BA_pd_kron_matrices_fill_values(const permuted_dense *A, int p, const matrix *J,
                                     stacked_pd *C);

#endif /* MATRIX_BTA_H */
