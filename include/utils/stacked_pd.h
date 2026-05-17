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
   disjoint; column permutations may overlap across blocks. */
typedef struct stacked_pd
{
    matrix base;
    int n_blocks;
    permuted_dense **blocks; /* owned; length n_blocks                     */
    int *src_block_idx;      /* owned; length n_blocks. Maps each result   */
                             /* block to its source block index in the spd */
                             /* that produced this one. Identity           */
                             /* (src_block_idx[k] == k) for a freshly      */
                             /* constructed spd.                           */
} stacked_pd;

/* Constructor.
   - Takes ownership of every block in `blocks` (frees them on destruction).
   - The `blocks` *array* itself is copied internally; the caller retains
     ownership of the outer array and may free or reuse it.
   - `src_block_idx`: NULL -> identity (k -> k). Non-NULL -> copied
     internally.
   - Validates that block row permutations are pairwise disjoint
     (debug build only).
   - `base.nnz` is set to the sum of block nnz; `base.x = NULL`;
     `free_fn` is wired; all other vtable slots are left NULL. */
matrix *new_stacked_pd(int m, int n, int n_blocks, permuted_dense **blocks,
                       const int *src_block_idx);

/* Allocate C = B @ A where B is spd and A is CSC. */
matrix *BA_spd_csc_alloc(const stacked_pd *B, const CSC_matrix *A);

/* Fill values of C = B @ A.*/
void BA_spd_csc_fill_values(const stacked_pd *B, const CSC_matrix *A, stacked_pd *C);

#endif /* STACKED_PD_H */
