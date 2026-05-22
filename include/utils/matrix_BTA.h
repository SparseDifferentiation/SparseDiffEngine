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
#include "sparse_matrix.h"
#include "stacked_pd.h"

/* Polymorphic dispatchers for C = B^T @ A and C = B^T @ diag(d) @ A. Each
   operand may be permuted_dense, sparse_matrix, or stacked_pd. The
   dispatcher branches on B's type and delegates to a fixed-B dispatcher
   (BTA_pd_matrices, BTA_spd_matrices, or BTA_sparse_matrices), which in
   turn branches on A's type and routes to the appropriate spd-aware
   primitive — no operand is ever materialized as a temporary CSC.

   Output type varies with the operand pair:
     - B is PD                          -> output is permuted_dense.
     - B is stacked_pd                  -> output is stacked_pd.
     - B is sparse_matrix, A is PD      -> output is permuted_dense.
     - B is sparse_matrix, A is spd     -> output is stacked_pd.
     - B is sparse_matrix, A is sparse  -> output is sparse_matrix.
   Callers should not assume a specific output type; use the matrix
   vtable for downstream operations (to_csr, transpose_alloc, etc.) and
   pass the alloc-returned matrix back to _fill_values verbatim.

   Contract: for sparse_matrix operands (A or B), the caller is
   responsible for refreshing csc_cache values via refresh_csc_values
   before BTDA_matrices_fill_values. stacked_pd operands need no
   preparation; their internal caches are refreshed on each call. */

/* Allocate sparsity for C = B^T @ A. */
matrix *BTA_matrices_alloc(matrix *A, matrix *B);

/* Fill values of C = B^T @ diag(d) @ A. */
void BTDA_matrices_fill_values(matrix *A, const double *d, matrix *B, matrix *C);

/* Polymorphic dispatcher for C = B @ A where B is PD and A is any matrix type.
   C is always PD. For the sparse-A branch the dispatcher ensures
   sm_A->csc_cache structure exists at alloc time but before
   BA_pd_matrices_fill_values the caller must have refreshed sm_A->csc_cache
   values (same fill-side contract as BTDA_matrices_fill_values). */
matrix *BA_pd_matrices_alloc(const permuted_dense *B, matrix *A);
void BA_pd_matrices_fill_values(const permuted_dense *B, const matrix *A,
                                permuted_dense *C);

/* Polymorphic dispatcher for C = B^T @ (diag(d) @) A where B is PD and A
   is any matrix type (PD, stacked_pd, or sparse_matrix). C is always PD.
   For the sparse-A branch the dispatcher ensures sm_A->csc_cache structure
   exists at alloc time; the caller must refresh sm_A->csc_cache values via
   sm_A->refresh_csc_values before calling _fill_values (same fill-side
   contract as BA_pd_matrices_fill_values). */
matrix *BTA_pd_matrices_alloc(const permuted_dense *B, matrix *A);
void BTDA_pd_matrices_fill_values(const permuted_dense *B, const double *d,
                                  const matrix *A, permuted_dense *C);

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

/* Polymorphic dispatcher for C = B^T @ (diag(d) @) A where B is sparse_matrix
   and A is any matrix type (PD, stacked_pd, or sparse_matrix). Output type
   varies with A:
     - A is PD             -> output is permuted_dense.
     - A is stacked_pd     -> output is stacked_pd.
     - A is sparse_matrix  -> output is sparse_matrix.
   Caller passes the alloc-returned matrix to _fill_values.

   The dispatcher ensures B's csc_cache structure exists at alloc time, and
   on the sparse-A branch also ensures sm_A->csc_cache structure. Before
   calling _fill_values, caller must refresh both csc_cache value buffers
   via refresh_csc_values (same fill-side contract as the other
   *_matrices_fill_values dispatchers). */
matrix *BTA_sparse_matrices_alloc(const sparse_matrix *B, matrix *A);
void BTDA_sparse_matrices_fill_values(const sparse_matrix *B, const double *d,
                                      const matrix *A, matrix *C);

/* Polymorphic dispatcher for C = B @ A where B is stacked_pd and A is any
   matrix type (PD, stacked_pd, or sparse_matrix). C is always stacked_pd.
   Not used in production today (stacked_pds appear only as derived
   Jacobians, not as constant left operands of B @ A), but kept as the
   natural counterpart to BTA_spd_matrices_alloc and as a sibling of the
   BA_spd_* primitives in stacked_pd_linalg.h.
   For the sparse-A branch the dispatcher ensures sm_A->csc_cache
   structure exists at alloc time; the caller must refresh values via
   sm_A->refresh_csc_values before calling _fill_values. */
matrix *BA_spd_matrices_alloc(const stacked_pd *B, matrix *A);
void BA_spd_matrices_fill_values(const stacked_pd *B, const matrix *A,
                                 stacked_pd *C);

/* Polymorphic dispatcher: C = kron(I_p, A) @ J as a stacked_pd, where A
   is a local dense matrix and J is any matrix type (permuted_dense,
   stacked_pd, or sparse_matrix). Output shape: (A->m * p) x J->n. For
   the sparse-J branch the dispatcher ensures sm_J->csc_cache structure
   exists at alloc time; the caller must refresh values via
   sm_J->refresh_csc_values before calling _fill_values.

   Contract: A must be a "full" permuted_dense — m0 == base.m, n0 ==
   base.n, and row_perm / col_perm are the identity — typically
   constructed via new_permuted_dense_full. The kron helpers index into
   A by local (i, j) and place the resulting nonzeros at kron-expanded
   positions (k*A.m + i, k*A.n + j); a non-full A would produce silently
   wrong sparsity. _alloc asserts this in Debug builds. */
matrix *BA_dense_kron_matrices_alloc(const permuted_dense *A, int p, matrix *J);
void BA_dense_kron_matrices_fill_values(const permuted_dense *A, int p,
                                        const matrix *J, stacked_pd *C);

#endif /* MATRIX_BTA_H */
