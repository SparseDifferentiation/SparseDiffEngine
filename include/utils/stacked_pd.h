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

/* stacked_pd represents a matrix that is the (vertical) union of `n_blocks`
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
    permuted_dense **blocks; /* owned; length n_blocks                      */
    int *src_block_idx_p;    /* owned; length n_blocks + 1                  */
    int *src_block_idx;      /* owned; length src_block_idx_p[n_blocks]     */
    struct stacked_pd *pre_coalesce;
    /* owned (pre_coalesce); NULL except when this spd was */
    /* produced by an op that needs a pre-coalesce  */
    /* intermediate (currently transpose). When     */
    /* set, holds the raw spd that                  */
    /* coalesce_spd_fill_values reads from. Lives   */
    /* for the spd's full lifetime, not per-call.   */
    CSR_matrix *csr_cache; /* owned; lazy CSR view built on first         */
                           /* to_csr() call. Structure (p, i) is built    */
                           /* once; values (x) are refreshed on every     */
                           /* subsequent call from the block X buffers.   */
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

/* Fill values of C = coalesce(A). */
void coalesce_spd_fill_values(const stacked_pd *A, stacked_pd *C);

/* Same scatter as coalesce_spd_fill_values but with += instead of =. Caller is
   responsible for zeroing C->base.x first. */
void coalesce_spd_fill_values_accumulate(const stacked_pd *A, stacked_pd *C);

/* Compose a CSR-indexed idx_map with the native->csr permutation of a
   stacked_pd, producing a map indexed against spd->base.x (block-major)
   directly. Useful for atoms that build idx_maps from a `to_csr` view at
   init time and want to read spd values directly at eval time, avoiding
   the per-call csr_cache->x refresh.

   - `spd` and `csr` must correspond (csr is the result of spd->to_csr).
   - `csr_idx_map` has length `csr->nnz` (= `spd->base.nnz`); each entry
     gives some output position for the corresponding CSR nonzero.
   - `native_idx_map` has length `spd->base.nnz`; output. Entry j gives
     the output position for `spd->base.x[j]`. */
void compose_csr_idx_map_for_spd(const stacked_pd *spd, const CSR_matrix *csr,
                                 const int *csr_idx_map, int *native_idx_map);

#endif /* STACKED_PD_H */
