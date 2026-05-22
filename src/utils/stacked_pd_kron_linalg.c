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
#include "utils/stacked_pd_kron_linalg.h"

#include "utils/permuted_dense.h"
#include "utils/permuted_dense_linalg.h"
#include "utils/stacked_pd.h"
#include "utils/stacked_pd_linalg.h"
#include "utils/tracked_alloc.h"
#include "utils/utils.h"
#include <stdlib.h>

// ----------------------------------------------------------------------------------
// C = kron(I_p, A) @ J as a stacked_pd, without materializing kron(I_p, A).
// Conceptually equivalent to BA_spd_*_alloc(B, J) with B = kron(I_p, A). We reuse
// a single scratch permuted_dense across the p iterations whose X aliases A->X
// and whose perms are mutated to encode block k on each iteration. The per-type
// functions are thin wrappers that pass a per-J-type per-block kernel into the
// shared kron_map_filter_blocks / kron_for_each_output_block helpers.
// ----------------------------------------------------------------------------------

/* Build the scratch "block view" of kron(I_p, A): a permuted_dense with X
   aliasing A->X (owns_X = false). Initial perms encode block 0; callers
   mutate via kron_scratch_set_block. Free with free_matrix when done. */
static permuted_dense *kron_scratch_init(const permuted_dense *A, int p)
{
    int m = A->m0;
    int n = A->n0;

    /* Initial block-0 perms. */
    int *row_perm = (int *) SP_MALLOC(m * sizeof(int));
    int *col_perm = (int *) SP_MALLOC(n * sizeof(int));
    for (int i = 0; i < m; i++) row_perm[i] = i;
    for (int j = 0; j < n; j++) col_perm[j] = j;

    /* X_data=NULL leaves the buffer uninitialized; we drop it and alias A->X. */
    matrix *pd_m = new_permuted_dense(m * p, n * p, m, n, row_perm, col_perm, NULL);
    free(row_perm);
    free(col_perm);

    permuted_dense *pd = (permuted_dense *) pd_m;
    free(pd->X);
    pd->X = A->X;
    pd->base.x = A->X;
    pd->owns_X = false;
    return pd;
}

/* Mutate the scratch to represent block k of kron(I_p, A). last_k is the
   block the scratch currently encodes (0 immediately after _init); the
   call clears last_k's col_inv entries and sets block-k's, plus refreshes
   row_perm and col_perm. O(m + n) work. row_inv is left stale — none of
   the per-block kernels read it. */
static void kron_scratch_set_block(permuted_dense *scratch, int k, int last_k)
{
    int m = scratch->m0;
    int n = scratch->n0;
    for (int i = 0; i < m; i++)
    {
        scratch->row_perm[i] = k * m + i;
    }
    for (int j = 0; j < n; j++)
    {
        scratch->col_inv[last_k * n + j] = -1;
    }
    for (int j = 0; j < n; j++)
    {
        scratch->col_perm[j] = k * n + j;
        scratch->col_inv[k * n + j] = j;
    }
}

/* Kron analog of spd_map_filter_blocks: iterate p kron-blocks of A, call
   `op(scratch, ctx)` per block (which returns a permuted_dense matrix for
   that output block), drop blocks whose result has nnz == 0, and assemble
   the survivors into a stacked_pd of shape (A->m * p, Cn) with one source
   per output block. */
typedef matrix *(*kron_block_alloc_op)(const permuted_dense *scratch,
                                       const void *ctx);

static matrix *kron_map_filter_blocks(const permuted_dense *A, int p, int Cn,
                                      kron_block_alloc_op op, const void *ctx)
{
    int Cm = A->m0 * p;
    if (p == 0)
    {
        return new_stacked_pd(Cm, Cn, 0, NULL, NULL, NULL);
    }

    permuted_dense *scratch = kron_scratch_init(A, p);
    permuted_dense **C_blocks =
        (permuted_dense **) SP_MALLOC(p * sizeof(permuted_dense *));
    int *C_src = (int *) SP_MALLOC(p * sizeof(int));

    int out_nb = 0;
    int last_k = 0;
    for (int k = 0; k < p; k++)
    {
        kron_scratch_set_block(scratch, k, last_k);
        last_k = k;

        matrix *Ck = op(scratch, ctx);
        if (Ck->nnz == 0)
        {
            free_matrix(Ck);
        }
        else
        {
            C_blocks[out_nb] = (permuted_dense *) Ck;
            C_src[out_nb] = k;
            out_nb++;
        }
    }

    int *C_src_p = (int *) SP_MALLOC((out_nb + 1) * sizeof(int));
    for (int k = 0; k <= out_nb; k++)
    {
        C_src_p[k] = k;
    }

    matrix *C = new_stacked_pd(Cm, Cn, out_nb, C_blocks, C_src_p, C_src);
    free(C_blocks);
    free(C_src);
    free(C_src_p);
    free_matrix(&scratch->base);
    return C;
}

/* Iterate C's output blocks. For each output block k, recover the source
   block index q from C's src_block_idx, set the scratch to encode kron-
   block q, then call `op(scratch, ctx, C->blocks[k])` to fill values. */
typedef void (*kron_block_fill_op)(const permuted_dense *scratch, const void *ctx,
                                   permuted_dense *Ck);

static void kron_for_each_output_block(const permuted_dense *A, int p, stacked_pd *C,
                                       kron_block_fill_op op, const void *ctx)
{
    if (p == 0 || C->n_blocks == 0)
    {
        return;
    }

    permuted_dense *scratch = kron_scratch_init(A, p);

    int last_k = 0;
    for (int k = 0; k < C->n_blocks; k++)
    {
        /* q is the source block index in kron(I_p, A) that contributes to Ck */
        int q = C->src_block_idx[C->src_block_idx_p[k]];
        kron_scratch_set_block(scratch, q, last_k);
        last_k = q;

        op(scratch, ctx, C->blocks[k]);
    }

    free_matrix(&scratch->base);
}

// ----------------------------------------------------------------------------------
// BA_dense_kron_csc: C = kron(I_p, A) @ J where J is CSC.
// ----------------------------------------------------------------------------------
static matrix *wrapper_BA_dense_kron_csc_alloc(const permuted_dense *scratch,
                                               const void *ctx)
{
    return BA_pd_csc_alloc(scratch, (const CSC_matrix *) ctx);
}

matrix *BA_dense_kron_csc_alloc(const permuted_dense *A, int p, const CSC_matrix *J)
{
    return kron_map_filter_blocks(A, p, J->n, wrapper_BA_dense_kron_csc_alloc, J);
}

static void wrapper_BA_dense_kron_csc_fill(const permuted_dense *scratch,
                                           const void *ctx, permuted_dense *Ck)
{
    BA_pd_csc_fill_values(scratch->X, scratch->n0, scratch->col_inv,
                          (const CSC_matrix *) ctx, Ck);
}

void BA_dense_kron_csc_fill_values(const permuted_dense *A, int p,
                                   const CSC_matrix *J, stacked_pd *C)
{
    kron_for_each_output_block(A, p, C, wrapper_BA_dense_kron_csc_fill, J);
}

// ----------------------------------------------------------------------------------
// BA_dense_kron_pd: C = kron(I_p, A) @ J where J is permuted_dense.
// ----------------------------------------------------------------------------------
static matrix *wrapper_BA_dense_kron_pd_alloc(const permuted_dense *scratch,
                                              const void *ctx)
{
    return BA_pd_pd_alloc(scratch, (const permuted_dense *) ctx);
}

matrix *BA_dense_kron_pd_alloc(const permuted_dense *A, int p,
                               const permuted_dense *J)
{
    return kron_map_filter_blocks(A, p, J->base.n, wrapper_BA_dense_kron_pd_alloc,
                                  J);
}

static void wrapper_BA_dense_kron_pd_fill(const permuted_dense *scratch,
                                          const void *ctx, permuted_dense *Ck)
{
    const permuted_dense *J = (const permuted_dense *) ctx;
    /* BA_pd_pd_fill_values reads scratch->kernel_dwork on the slow path. At alloc
       time BA_pd_pd_alloc sized it, but on a scratch we've since freed —
       ensure the fresh scratch has dwork before the first fill. Idempotent
       after the first block. */
    int s_max = MIN(scratch->n0, J->m0);
    permuted_dense_ensure_kernel_dwork(scratch, (size_t) s_max * scratch->m0);
    BA_pd_pd_fill_values(scratch, J, Ck);
}

void BA_dense_kron_pd_fill_values(const permuted_dense *A, int p,
                                  const permuted_dense *J, stacked_pd *C)
{
    kron_for_each_output_block(A, p, C, wrapper_BA_dense_kron_pd_fill, J);
}

// ----------------------------------------------------------------------------------
// BA_dense_kron_spd: C = kron(I_p, A) @ J where J is stacked_pd.
//
// Bypasses the BA_pd_spd_* wrapper (and its transpose_cache pattern) because
// the kron pattern's scratch B mutates row_perm / col_perm between iterations,
// which would invalidate the cache. Instead we manage BT explicitly: allocate
// one BT for the whole call (the transpose-of-A buffer is invariant since
// each iteration's scratch.X aliases A.X), rewrite BT's perms cheaply per
// iteration via kron_BT_set_block, and call BTA_pd_spd_* directly. Also
// drops the scratch PD entirely — BT alone encodes the current kron block
// for the BTA kernel.
//
// To respect the "no allocation in _fill_values" convention, BT is allocated
// inside _alloc and stashed on C->kernel_pd_scratch. _fill_values reads the
// cached BT, refreshes its X via transpose_pd_fill_values(A, BT) (alloc-free),
// and runs the loop. BT is released when C is freed (stacked_pd_free handles
// the kernel_pd_scratch slot).
// ----------------------------------------------------------------------------------

/* Allocate BT with the kron-expanded global shape (transposed):
   shape (A.n0 * p, A.m0 * p), inner block (A.n0, A.m0). X buffer is
   uninitialized (caller fills if needed). Initial perm values don't
   matter — kron_BT_set_block overwrites them before BT is read. */
static permuted_dense *kron_BT_alloc(const permuted_dense *A, int p)
{
    int n = A->n0;
    int m = A->m0;
    int *row_perm = (int *) SP_MALLOC(n * sizeof(int));
    int *col_perm = (int *) SP_MALLOC(m * sizeof(int));
    for (int j = 0; j < n; j++) row_perm[j] = j;
    for (int i = 0; i < m; i++) col_perm[i] = i;
    matrix *BT_m = new_permuted_dense(n * p, m * p, n, m, row_perm, col_perm, NULL);
    free(row_perm);
    free(col_perm);
    return (permuted_dense *) BT_m;
}

/* Rewrite BT's row_perm / col_perm to represent kron block k.
   BTA_pd_spd_* never reads B's col_inv or row_inv, so we leave them stale
   (and skip the clear-last-k machinery that kron_scratch_set_block needs). */
static void kron_BT_set_block(permuted_dense *BT, int k)
{
    /* BT.m0 = A.n0, BT.n0 = A.m0 (transposed inner block). */
    int n = BT->m0;
    int m = BT->n0;
    for (int j = 0; j < n; j++)
    {
        BT->row_perm[j] = k * n + j;
    }
    for (int i = 0; i < m; i++)
    {
        BT->col_perm[i] = k * m + i;
    }
}

matrix *BA_dense_kron_spd_alloc(const permuted_dense *A, int p, const stacked_pd *J)
{
    int Cn = J->base.n;
    int Cm = A->m0 * p;
    if (p == 0)
    {
        return new_stacked_pd(Cm, Cn, 0, NULL, NULL, NULL);
    }

    /* One BT for the whole call. Initial perm values don't matter —
       kron_BT_set_block overwrites them each iter before BT is read.
       X is uninitialized; BTA_pd_spd_alloc doesn't read it. We do NOT
       free BT at the end of this function: ownership is transferred
       to C->kernel_pd_scratch below, so _fill_values can reuse it
       without allocating. */
    permuted_dense *BT = kron_BT_alloc(A, p);

    permuted_dense **C_blocks =
        (permuted_dense **) SP_MALLOC(p * sizeof(permuted_dense *));
    int *C_src = (int *) SP_MALLOC(p * sizeof(int));

    int out_nb = 0;
    for (int k = 0; k < p; k++)
    {
        kron_BT_set_block(BT, k);

        matrix *Ck = BTA_pd_spd_alloc(BT, J);
        if (Ck->nnz == 0)
        {
            free_matrix(Ck);
        }
        else
        {
            C_blocks[out_nb] = (permuted_dense *) Ck;
            C_src[out_nb] = k;
            out_nb++;
        }
    }

    int *C_src_p = (int *) SP_MALLOC((out_nb + 1) * sizeof(int));
    for (int k = 0; k <= out_nb; k++)
    {
        C_src_p[k] = k;
    }

    matrix *C = new_stacked_pd(Cm, Cn, out_nb, C_blocks, C_src_p, C_src);
    free(C_blocks);
    free(C_src);
    free(C_src_p);

    /* Hand BT to C for reuse at fill time. stacked_pd_free releases it. */
    ((stacked_pd *) C)->kernel_pd_scratch = BT;
    return C;
}

void BA_dense_kron_spd_fill_values(const permuted_dense *A, int p,
                                   const stacked_pd *J, stacked_pd *C)
{
    if (p == 0 || C->n_blocks == 0)
    {
        return;
    }

    /* BT was allocated by BA_dense_kron_spd_alloc and stashed on C.
       Refresh BT.X (A.X may have changed between alloc and fill, or
       between fills, via parameter refresh) — single in-place
       transpose, no allocation. */
    permuted_dense *BT = C->kernel_pd_scratch;
    transpose_pd_fill_values(A, BT);

    for (int k = 0; k < C->n_blocks; k++)
    {
        int q = C->src_block_idx[C->src_block_idx_p[k]];
        kron_BT_set_block(BT, q);
        BTA_pd_spd_fill_values(BT, J, C->blocks[k]);
    }
}
