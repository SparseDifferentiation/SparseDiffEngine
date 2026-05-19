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
#ifndef STACKED_PD_LINALG_H
#define STACKED_PD_LINALG_H

#include "CSC_matrix.h"
#include "permuted_dense.h"
#include "stacked_pd.h"

/* Allocate C = B @ A where B is spd, A is CSC. Result blocks are produced
   per source block via BA_pd_csc_alloc; blocks whose result has n0 == 0
   (no column of A intersects that block's col_perm) are dropped.
   C->n_blocks may be less than B->n_blocks; the src_block_idx_* map
   records the surviving source indices (exactly one source per output
   block). */
matrix *BA_spd_csc_alloc(const stacked_pd *B, const CSC_matrix *A);

/* Fill values of C = B @ A. The caller must have produced C via
   BA_spd_csc_alloc and must keep A's CSC structure (sparsity) identical
   between alloc and this call. */
void BA_spd_csc_fill_values(const stacked_pd *B, const CSC_matrix *A, stacked_pd *C);

/* Coalesce: take an spd whose blocks may have overlapping row perms but
   pairwise-disjoint cells, and produce an equivalent spd that satisfies
   the disjoint-row invariant. The output is the tightest dense
   representation — no structural zeros inside any output block —
   built by grouping rows by signature (the set of source blocks
   containing each row). Each unique signature becomes one output block
   whose row_perm is its rows and whose col_perm is the sorted union of
   the source col_perms in the signature. Output blocks are ordered by
   their minimum row index.

   Precondition (debug-asserted): for any two source blocks sharing a
   row, their col_perms must be disjoint. The transpose use case
   (step 3) satisfies this automatically. */
matrix *coalesce_spd_alloc(const stacked_pd *src);

/* Fill values of out = coalesce(src). The caller must have produced
   `out` via coalesce_spd_alloc on a `src` with the same structure
   (block count, row_perms, col_perms) as the one passed now. */
void coalesce_spd_fill_values(const stacked_pd *src, stacked_pd *out);

/* Allocate a new spd C with the same sparsity as A.
   TODO: should this be publically available? */
matrix *copy_sparsity_spd_alloc(const stacked_pd *A);

/* Fill values of C = diag(d) @ A, where 'd' is 'global' with length A->m. */
void DA_spd_fill_values(const double *d, const stacked_pd *A, stacked_pd *C);

/* Allocate sparsity for C = B @ A where B is a stacked_pd and A is a
   permuted_dense. Per source block, computes B_k @ A via BA_pd_pd_alloc;
   B-blocks whose contribution is structurally empty (n0 == 0) are
   dropped; src_block_idx_* records the surviving source indices (one
   source per output block). Output is a stacked_pd: row_perms inherited
   from B (pairwise disjoint by the spd invariant). */
matrix *BA_spd_pd_alloc(const stacked_pd *B, const permuted_dense *A);

/* Fill values of C = B @ A. Caller must have produced C via
   BA_spd_pd_alloc with the same structural inputs. Per surviving
   output block, delegates to BA_pd_pd_fill_values. */
void BA_spd_pd_fill_values(const stacked_pd *B, const permuted_dense *A,
                           stacked_pd *C);

/* Allocate sparsity for C = B @ A where B is a permuted_dense and A is
   a stacked_pd. Output is a single permuted_dense: row_perm =
   B->row_perm, col_perm = sorted union of A's block col_perms over
   those A-blocks whose row_perm intersects B->col_perm. If no A-block
   contributes, the output has n0 = 0. */
matrix *BA_pd_spd_alloc(const permuted_dense *B, const stacked_pd *A);

/* Fill values of C = B @ A. Caller must have produced C via
   BA_pd_spd_alloc on the same B, A structure. Per A-block, this does a
   small gather + dgemm and scatters the result into C with += semantics
   on overlapping output cols. */
void BA_pd_spd_fill_values(const permuted_dense *B, const stacked_pd *A,
                           permuted_dense *C);

/* Allocate sparsity for C = B @ A where both B and A are stacked_pds.
   For each B-block, computes its contribution as a single PD via
   BA_pd_spd_alloc. B-blocks whose contribution is empty (n0 == 0) are
   dropped; src_block_idx_* records the surviving source indices (one
   source per output block). Output row_perms inherit from B and are
   pairwise disjoint by the spd invariant on B. */
matrix *BA_spd_spd_alloc(const stacked_pd *B, const stacked_pd *A);

/* Fill values of C = B @ A. */
void BA_spd_spd_fill_values(const stacked_pd *B, const stacked_pd *A, stacked_pd *C);

/* Allocate sparsity for C = A^T A. A^T A decomposes as Σ_k B_k^T B_k,
   where summands with overlapping col_perms (C_k) share cells. The
   output groups cols of A by signature sig_C(c) = {k : c ∈ C_k}; each
   unique signature becomes one output PD with row_perm = group cols
   and col_perm = ⋃ C_k for k in the signature. No structural zeros.
   The output's `work` slot holds per-source scratch PDs (one symmetric
   PD per source block, row_perm = col_perm = C_k) that
   ATDA_spd_fill_values writes into. */
matrix *ATA_spd_alloc(const stacked_pd *A);

/* Fill values of C = A^T diag(d) A. */
void ATDA_spd_fill_values(const stacked_pd *A, const double *d, stacked_pd *C);

/* Allocate out = transpose(src). Implementation: transpose each source
   PD block individually (yielding a raw spd that may have overlapping
   row perms but pairwise-disjoint cells, by the spd invariant), then
   coalesce. The raw spd is stored on `out->work` for reuse by
   `transpose_spd_fill_values`. */
matrix *transpose_spd_alloc(const stacked_pd *src);

/* Fill values of out = transpose(src). The caller must have produced
   `out` via transpose_spd_alloc on a `src` with the same structure
   (block count, row_perms, col_perms) as the one passed now. */
void transpose_spd_fill_values(const stacked_pd *src, stacked_pd *out);

#endif /* STACKED_PD_LINALG_H */
