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
#ifndef STACKED_PD_H
#define STACKED_PD_H

#include "CSC_matrix.h"
#include "matrix.h"
#include "permuted_dense.h"

/* stacked_pd represents a matrix that is the (vertical) union of
   `n_blocks` permuted_dense blocks. Block row permutations are pairwise
   disjoint; column permutations may overlap across blocks.

   `src_block_idx_p` and `src_block_idx` form a CSR-style map (offsets +
   flat data) from each output block to the source block indices that
   contribute to it in the spd-producing op that built this spd. For ops
   with one source per output (e.g. BA_spd_csc) every entry of
   `src_block_idx_p` increments by one. For coalesce a single output
   block can be fed by multiple source blocks. */
typedef struct stacked_pd
{
    matrix base;
    int n_blocks;
    permuted_dense **blocks; /* owned; length n_blocks                      */
    int *src_block_idx_p;    /* owned; length n_blocks + 1                  */
    int *src_block_idx;      /* owned; length src_block_idx_p[n_blocks]     */
    struct stacked_pd *work; /* owned; NULL except when this spd was        */
                             /* produced by an op that needs a pre-coalesce */
                             /* intermediate (currently transpose). When    */
                             /* set, holds the raw spd that                 */
                             /* coalesce_spd_fill_values reads from.        */
} stacked_pd;

/* Constructor.
   - Takes ownership of every block in `blocks` (frees them on destruction).
   - The `blocks` *array* itself is copied internally; the caller retains
     ownership of the outer array and may free or reuse it.
   - `src_block_idx_p` and `src_block_idx`:
       * both NULL -> identity (each output block has itself as its one
         source: src_block_idx_p[k] = k, src_block_idx[k] = k).
       * both non-NULL -> copied internally; src_block_idx_p must have
         length n_blocks + 1 and src_block_idx must have length
         src_block_idx_p[n_blocks].
       * one NULL and not the other -> undefined; debug-asserted.
   - Validates that block row permutations are pairwise disjoint
     (debug build only).
   - `base.nnz` is set to the sum of block nnz; `base.x = NULL`;
     `free_fn` is wired; all other vtable slots are left NULL. */
matrix *new_stacked_pd(int m, int n, int n_blocks, permuted_dense **blocks,
                       const int *src_block_idx_p, const int *src_block_idx);

/* Same as new_stacked_pd but skips the pairwise-disjoint row_perm check.
   Use only when the spd is a transient overlapping-rows intermediate
   (e.g. an input to coalesce); the public type's canonical invariant is
   still "row perms pairwise disjoint". */
matrix *new_stacked_pd_unchecked(int m, int n, int n_blocks, permuted_dense **blocks,
                                 const int *src_block_idx_p,
                                 const int *src_block_idx);

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

/* Allocate a new spd with the same block structure (per-block row_perm,
   col_perm, and shape) as `src` and uninitialized X buffers. Each output
   block has itself as its one source in src_block_idx_*. `work` is NULL.
   Mirrors the role of permuted_dense's `copy_sparsity` vtable slot. */
matrix *copy_sparsity_spd_alloc(const stacked_pd *src);

/* Fill values of C = diag(d) @ A. `d` has length A->base.m. The caller
   must have produced C via copy_sparsity_spd_alloc(A) (or any allocator
   that preserves A's block structure). Per-block delegates to
   DA_pd_fill_values. */
void DA_spd_fill_values(const double *d, const stacked_pd *A, stacked_pd *C);

/* Allocate sparsity for C = A^T A. A^T A decomposes as Σ_k B_k^T B_k,
   where summands with overlapping col_perms (C_k) share cells. The
   output groups cols of A by signature sig_C(c) = {k : c ∈ C_k}; each
   unique signature becomes one output PD with row_perm = group cols
   and col_perm = ⋃ C_k for k in the signature. No structural zeros.
   The output's `work` slot holds per-source scratch PDs (one symmetric
   PD per source block, row_perm = col_perm = C_k) that
   ATDA_spd_fill_values writes into. */
matrix *ATA_spd_alloc(const stacked_pd *A);

/* Fill values of C = A^T diag(d) A. `d` has length A->base.m. The caller
   must have produced C via ATA_spd_alloc(A). Implementation: per source
   block k, compute B_k^T diag(d) B_k into C->work->blocks[k] via the
   existing ATDA_pd_fill_values; then zero C's output X buffers and
   accumulate (+=) each scratch into the appropriate output PDs. */
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

#endif /* STACKED_PD_H */
