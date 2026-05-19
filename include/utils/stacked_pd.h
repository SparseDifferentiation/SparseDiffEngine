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

/* stacked_pd represents a matrix that is the (vertical) union of
   `n_blocks` permuted_dense blocks. Two different blocks have disjoint
   row permutations, but the column permutations may overlap across blocks.



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
    CSR_matrix *csr_cache;   /* owned; lazy CSR view built on first         */
                             /* to_csr() call. Structure (p, i) is built    */
                             /* once; values (x) are refreshed on every     */
                             /* subsequent call from the block X buffers.   */
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

/* Filter-map over a stacked_pd's blocks. For each block of B, calls
   op(Bk, ctx); drops blocks whose result has nnz == 0; assembles the
   survivors into a new stacked_pd (dimensions Cm x Cn) with one source
   per output block. Used by both the vtable adapters in stacked_pd.c
   and the BA_spd_*_alloc kernels in stacked_pd_linalg.c. */
typedef matrix *(*spd_block_op)(permuted_dense *Bk, const void *ctx);

matrix *spd_map_filter_blocks(const stacked_pd *B, int Cm, int Cn, spd_block_op op,
                              const void *ctx);

#endif /* STACKED_PD_H */
