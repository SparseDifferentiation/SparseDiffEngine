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

#include "CSR_matrix.h"
#include "matrix.h"
#include "permuted_dense.h"

/* Shape metadata for one would-be source block of a coalesce. Used by
   spd_blockwise_alloc_coalesce to drive symbolic coalesce without
   allocating per-block Gram PDs, and by spd_blockwise_fill_coalesce to
   re-aim a shared scratch PD at each input block's metadata.
   - (m0, n0): block shape.
   - row_perm, col_perm: aliased pointers into operand arrays; not owned.
   - scratch_*_needed: sizes the matching fill op will read from its
     destination PD's kernel_dwork / kernel_iwork. Coalesce ignores these;
     spd_blockwise uses them to size the shared scratch PD once at the
     max-across-blocks. */
typedef struct
{
    int m0;
    int n0;
    /* Both arrays are SP_MALLOC'd by the shape callback and owned by the
       per_block_shapes array (freed by stacked_pd_free). */
    int *row_perm;
    int *col_perm;
    /* Sizes the matching fill op will read from its destination PD's
       kernel_dwork / kernel_iwork. */
    size_t scratch_dwork_needed;
    size_t scratch_iwork_needed;
    /* Optional metadata to seed scratch->kernel_iwork[0..init_len) with
       before invoking the per-block fill op. Used by BTA_pd_spd's fill
       which reads s_max / max_n0_A from iwork[0..1]. */
    int iwork_init[2];
    int iwork_init_len;
} spd_per_block_shape;

/* stacked_pd represents a matrix that is the (vertical) union of 'n_blocks'
   permuted_dense blocks. Two different blocks have disjoint row permutations,
   but the column permutations may overlap across blocks.

   `src_block_idx_p` and `src_block_idx` form a CSR-style map (offsets + flat
   data) from each output block to the source block indices that contribute to
   it in the spd-producing operations that built this spd. For operations with
   one source per output every entry of `src_block_idx_p` increments by one. For
   coalesce a single output block can be fed by multiple source blocks. */
typedef struct stacked_pd
{
    matrix base;
    int n_blocks;
    permuted_dense **blocks;
    int *src_block_idx_p;
    int *src_block_idx;

    /* Some operations on stacked_pds (like transpose) may in an intermediate
       step result in a stacked_pd with blocks that do not have disjoint row
       permutations. Such operations must incorporate a coalesce step that
       merges rows that appear in several blocks into a single block. This pointer
       holds the raw spd before the coalesce step. */
    struct stacked_pd *pre_coalesce;

    /* lazily built CSR view */
    CSR_matrix *csr_cache;

    /* Private permuted_dense scratch owned by the kernel that produced
       this spd. Allocated by the producing _alloc, used (without
       further allocation) by the matching _fill_values, freed by
       stacked_pd_free. Currently used by BA_dense_kron_spd to cache
       the kron-expanded transpose-of-A buffer across fills. NULL when
       not used. */
    permuted_dense *kernel_pd_scratch;

    /* Streaming-fill state for stacked_pds produced by
       spd_blockwise_alloc_coalesce. NULL when not used. All buffers are
       sized at alloc time so the matching fill function does zero
       SP_MALLOC.
       - scratch_X: per-iteration buffer holding one source block's Gram.
         Sized at max over input blocks of (m0_partial * n0_partial).
       - scratch_dwork / scratch_iwork: workspaces the per-block fill op
         reads from its (scratch) destination PD's kernel_dwork /
         kernel_iwork. Sized at max-across-blocks.
       - per_block_shapes: shape (and scratch sizes) for each input block's
         partial. Indexed [0, n_input_blocks). The scratch PD's metadata
         (m0, n0, row_perm, col_perm) is re-aimed from this array each
         iteration.
       - n_input_blocks: number of input blocks (length of per_block_shapes
         and of src_to_outs_p[]-1). Distinct from n_blocks (which counts
         OUTPUT blocks after coalesce).
       - src_to_outs_p, src_to_outs_data: CSR-style inverse of
         src_block_idx_p/_data. src_to_outs_data[src_to_outs_p[k] ..
         src_to_outs_p[k+1]) lists the output block indices that block k
         contributes to. Drives the per-input-block scatter loop in fill. */
    double *scratch_X;
    size_t scratch_X_capacity;
    double *scratch_dwork;
    size_t scratch_dwork_capacity;
    int *scratch_iwork;
    size_t scratch_iwork_capacity;
    /* Re-initialized per iteration from per_block_shapes[k].row_perm /
       col_perm. Sized at (m, n) — the global dimensions. Some fill ops
       (e.g. BTA_pd_spd_fill_values at stacked_pd_linalg.c:477) read
       C->col_inv to look up scatter positions; the scratch must have
       it populated. */
    int *scratch_row_inv;
    int *scratch_col_inv;
    spd_per_block_shape *per_block_shapes;
    int n_input_blocks;
    int *src_to_outs_p;
    int *src_to_outs_data;
} stacked_pd;

/* Constructor for stacked_pd. Takes ownership of every block in 'blocks'. The
   arrays 'src_block_idx_p' and 'src_block_idx' map each block in the new
   stacked_pd (ie. the block in 'blocks') to the source block(s) that
   contributed to it in the spd-producing operations that built this spd. If
   src_block_idx_p and src_block_idx are NULL, then src_block_idx is ignored and
   the constructor produces the identity mapping (each output block has itself
   as its one source). */
matrix *new_stacked_pd(int m, int n, int n_blocks, permuted_dense **blocks,
                       const int *src_block_idx_p, const int *src_block_idx);

/* Same as new_stacked_pd but skips a pairwise-disjoint row_permutation check in
   debug mode. Use only when the input intentionally has blocks with overlapping
   row permutations. */
matrix *new_stacked_pd_unchecked(int m, int n, int n_blocks, permuted_dense **blocks,
                                 const int *src_block_idx_p,
                                 const int *src_block_idx);

/* Like new_stacked_pd_unchecked, but the blocks' X pointers already point
   into the caller-supplied `shared_x` buffer at the correct offsets
   (typically constructed via new_permuted_dense_view). No absorb / memcpy
   is performed. The stacked_pd takes ownership of `shared_x` and frees it
   on destruction; each block must have owns_X == false. */
matrix *new_stacked_pd_borrowed_x(int m, int n, int n_blocks,
                                  permuted_dense **blocks,
                                  const int *src_block_idx_p,
                                  const int *src_block_idx, double *shared_x);

/* Build a stacked_pd from precomputed per-block shapes. Sums total_nnz,
   SP_MALLOCs one shared X buffer, constructs n_blocks view PDs pointing
   into that buffer at sequential offsets, and wraps via
   new_stacked_pd_borrowed_x. No absorb / double-count. Identity
   src_block_idx (one source per output block). Reads only the shape's
   m0 / n0 / row_perm / col_perm fields; per-block kernel_iwork /
   kernel_dwork are the caller's responsibility. */
matrix *new_stacked_pd_from_shapes_unchecked(int m, int n, int n_blocks,
                                             const spd_per_block_shape *shapes);

/* Filter-map over a stacked_pd's blocks. For each block of B, calls op(Bk,
   ctx). Drop blocks whose result has nnz == 0 and assembles the survivors into
   a new stacked_pd (dimensions Cm x Cn) with one source per output block. */
typedef matrix *(*spd_block_op)(permuted_dense *Bk, const void *ctx);
matrix *spd_map_filter_blocks(const stacked_pd *B, int Cm, int Cn, spd_block_op op,
                              const void *ctx);

/* Coalesce takes a stacked_pd whose blocks are disjoint but may have
   overlapping row permuations and produces an equivalent stacked_pd with blocks
   that have disjoint row permutations. This function allocates the output
   stacked_pd. */
matrix *coalesce_spd_alloc(const stacked_pd *A);

/* Same as coalesce_spd_alloc but skips a pairwise-disjoint row_permutation
   check in debug mode. Use only when the input intentionally has blocks with
   overlapping row permutations. */
matrix *coalesce_spd_alloc_unchecked(const stacked_pd *A);

/* Symbolic coalesce driven by a shapes array (no per-block X / PDs needed).
   Returns an output stacked_pd whose blocks are views into one shared X
   buffer that the output owns. The output's src_block_idx_p / src_block_idx
   hold the forward map; callers that need the inverse read those fields
   directly. Equivalent to coalesce_spd_alloc_unchecked but works from shape
   metadata only — used by spd_blockwise_alloc_coalesce to skip the
   per-block Gram materialization. */
matrix *coalesce_spd_alloc_from_shapes_unchecked(const spd_per_block_shape *shapes,
                                                 int n_input_blocks, int m, int n);

/* Fill values of C = coalesce(A). */
void coalesce_spd_fill_values(const stacked_pd *A, stacked_pd *C);

/* Same scatter as coalesce_spd_fill_values but with += instead of =. Caller is
   responsible for zeroing C->base.x first. */
void coalesce_spd_fill_values_accumulate(const stacked_pd *A, stacked_pd *C);

/* Scatter the cells of one source PD into one output PD (using out_k's
   row_inv / col_inv to map global row / col indices to local positions),
   accumulating with += (caller must zero out_k->X before the first call
   in a scatter cycle). Used by the per-input-block streaming fill in
   spd_blockwise_fill_coalesce_accumulate. */
void scatter_one_source_into_one_output_accumulate(const permuted_dense *src_k,
                                                   permuted_dense *out_k);

/* Re-index `idx_map` in place from CSR ordering to spd-native ordering
   (block-major). Useful for atoms that build idx_maps from a `to_csr`
   view at init time and want to read spd values directly at eval time,
   avoiding the per-call csr_cache->x refresh.

   - `spd` and `csr` must correspond (csr is the result of spd->to_csr).
   - `idx_map` has length `csr->nnz` (= `spd->base.nnz`); on entry each
     entry is indexed by CSR position, on exit each entry is indexed by
     `spd->base.x` position. The buffer is reused; the caller's pointer
     stays valid. */
void compose_csr_idx_map_for_spd(const stacked_pd *spd, const CSR_matrix *csr,
                                 int *idx_map);

#endif /* STACKED_PD_H */
