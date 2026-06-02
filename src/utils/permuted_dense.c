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
#include "utils/permuted_dense.h"

#include "utils/cblas_wrapper.h"
#include "utils/linalg_dense_sparse_matmuls.h"
#include "utils/permuted_dense_linalg.h"
#include "utils/tracked_alloc.h"
#include "utils/utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void permuted_dense_free(matrix *self)
{
    permuted_dense *pd = (permuted_dense *) self;
    sp_free(pd->row_perm);
    sp_free(pd->col_perm);
    sp_free(pd->col_inv);
    sp_free(pd->row_inv);
    /* csr_cache->x aliases pd->X (set in permuted_dense_to_csr_alloc); NULL it
       so free_CSR_matrix doesn't double-free the shared buffer. */
    if (pd->csr_cache != NULL)
    {
        pd->csr_cache->x = NULL;
    }
    free_CSR_matrix(pd->csr_cache);
    if (pd->owns_X)
    {
        sp_free(pd->X);
    }
    sp_free(pd->kernel_dwork);
    sp_free(pd->kernel_iwork);
    free_matrix((matrix *) pd->transpose_cache);
    sp_free(pd);
}

/* permuted_dense has no CSC_matrix mirror; chain-rule kernels operate on X directly.
 */
static void permuted_dense_refresh_csc_values(matrix *self)
{
    (void) self;
}

/* Vtable adapters — each delegates to the existing permuted_dense_* kernel. */
static matrix *permuted_dense_vtable_copy_sparsity(const matrix *self)
{
    return copy_sparsity_pd_alloc((const permuted_dense *) self);
}

static void permuted_dense_vtable_DA_fill_values(const double *d, const matrix *self,
                                                 matrix *out)
{
    DA_pd_fill_values(d, (const permuted_dense *) self, (permuted_dense *) out);
}

static matrix *permuted_dense_vtable_ATA_alloc(matrix *self)
{
    return ATA_pd_alloc((const permuted_dense *) self);
}

static void permuted_dense_vtable_ATDA_fill_values(const matrix *self,
                                                   const double *d, matrix *out)
{
    ATDA_pd_fill_values((const permuted_dense *) self, d, (permuted_dense *) out);
}

/* Forward decl; definition lower in the file. */
static CSR_matrix *permuted_dense_to_csr_alloc(const permuted_dense *A);

/* Lazy CSR_matrix view: allocate structure on first call, then return the cache.
   The cache's x array aliases pd->X (see permuted_dense_to_csr_alloc), so
   values are always live without a per-call refresh. */
static CSR_matrix *permuted_dense_to_csr(matrix *self)
{
    permuted_dense *pd = (permuted_dense *) self;
    if (pd->csr_cache == NULL)
    {
        pd->csr_cache = permuted_dense_to_csr_alloc(pd);
    }
    return pd->csr_cache;
}

static matrix *permuted_dense_vtable_transpose_alloc(const matrix *self)
{
    return transpose_pd_alloc((const permuted_dense *) self);
}

static void permuted_dense_vtable_transpose_fill_values(const matrix *self,
                                                        matrix *out)
{
    transpose_pd_fill_values((const permuted_dense *) self, (permuted_dense *) out);
}

matrix *index_pd_alloc(const permuted_dense *A, const int *indices, int n_idxs)
{
    /* Scan indices: which output positions i hit a row in A->row_perm? */
    int *new_row_perm = (int *) sp_malloc(n_idxs * sizeof(int));
    int new_m0 = 0;
    for (int i = 0; i < n_idxs; i++)
    {
        if (A->row_inv[indices[i]] >= 0)
        {
            new_row_perm[new_m0++] = i;
        }
    }

    matrix *out = new_permuted_dense(n_idxs, A->base.n, new_m0, A->n0, new_row_perm,
                                     A->col_perm, NULL);
    sp_free(new_row_perm);
    return out;
}

void index_pd_fill_values(const permuted_dense *A, const int *indices, int n_idxs,
                          permuted_dense *C)
{
    (void) n_idxs;
    int n0 = A->n0;
    for (int k = 0; k < C->m0; k++)
    {
        int i = C->row_perm[k];
        int old_ii = A->row_inv[indices[i]];
        memcpy(C->X + k * n0, A->X + old_ii * n0, n0 * sizeof(double));
    }
}

static matrix *permuted_dense_vtable_index_alloc(matrix *self, const int *indices,
                                                 int n_idxs)
{
    return index_pd_alloc((const permuted_dense *) self, indices, n_idxs);
}

static void permuted_dense_vtable_index_fill_values(matrix *self, const int *indices,
                                                    int n_idxs, matrix *out)
{
    index_pd_fill_values((const permuted_dense *) self, indices, n_idxs,
                         (permuted_dense *) out);
}

matrix *promote_pd_alloc(const permuted_dense *A, int size)
{
    assert(A->m0 <= 1);

    /* does this actually happen? if so, when?
    actually, what even is m0 here, we should have more documentation */
    if (A->m0 == 0)
    {
        /* source row is all-zero; output is also structurally all-zero. */
        return new_permuted_dense(size, A->base.n, 0, A->n0, NULL, A->col_perm,
                                  NULL);
    }

    /* does it make sense to use the row_permutation of the original pd matrix A?
    it seems like the broadcast atom does that. */
    int *new_row_perm = (int *) sp_malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        new_row_perm[i] = i;
    }
    matrix *out = new_permuted_dense(size, A->base.n, size, A->n0, new_row_perm,
                                     A->col_perm, NULL);
    sp_free(new_row_perm);
    return out;
}

void promote_pd_fill_values(const permuted_dense *A, permuted_dense *C)
{
    if (A->m0 == 0) return;
    int n0 = A->n0;
    for (int k = 0; k < C->m0; k++)
    {
        memcpy(C->X + k * n0, A->X, n0 * sizeof(double));
    }
}

static matrix *permuted_dense_vtable_promote_alloc(matrix *self, int size)
{
    return promote_pd_alloc((const permuted_dense *) self, size);
}

static void permuted_dense_vtable_promote_fill_values(matrix *self, matrix *out)
{
    promote_pd_fill_values((const permuted_dense *) self, (permuted_dense *) out);
}

matrix *broadcast_pd_alloc(const permuted_dense *A, broadcast_type type, int d1,
                           int d2)
{
    int out_m = d1 * d2;

    int new_m0;
    if (type == BROADCAST_SCALAR)
    {
        new_m0 = (A->m0 == 0) ? 0 : out_m;
    }
    else if (type == BROADCAST_ROW)
    {
        new_m0 = d1 * A->m0;
    }
    else /* BROADCAST_COL */
    {
        new_m0 = d2 * A->m0;
    }

    if (new_m0 == 0)
    {
        return new_permuted_dense(out_m, A->base.n, 0, A->n0, NULL, A->col_perm,
                                  NULL);
    }

    int *new_row_perm = (int *) sp_malloc(new_m0 * sizeof(int));
    int k = 0;
    if (type == BROADCAST_SCALAR)
    {
        for (int i = 0; i < out_m; i++)
        {
            new_row_perm[k++] = i;
        }
    }
    else if (type == BROADCAST_ROW)
    {
        for (int j_ii = 0; j_ii < A->m0; j_ii++)
        {
            int j_old = A->row_perm[j_ii];
            for (int i = 0; i < d1; i++)
            {
                new_row_perm[k++] = j_old * d1 + i;
            }
        }
    }
    else /* BROADCAST_COL */
    {
        for (int j = 0; j < d2; j++)
        {
            for (int ii_old = 0; ii_old < A->m0; ii_old++)
            {
                new_row_perm[k++] = j * d1 + A->row_perm[ii_old];
            }
        }
    }

    matrix *out = new_permuted_dense(out_m, A->base.n, new_m0, A->n0, new_row_perm,
                                     A->col_perm, NULL);
    sp_free(new_row_perm);
    return out;
}

void broadcast_pd_fill_values(const permuted_dense *A, broadcast_type type, int d1,
                              int d2, permuted_dense *C)
{
    if (A->m0 == 0)
    {
        return;
    }
    int n0 = A->n0;

    if (type == BROADCAST_SCALAR)
    {
        for (int k = 0; k < C->m0; k++)
        {
            memcpy(C->X + k * n0, A->X, n0 * sizeof(double));
        }
    }
    else if (type == BROADCAST_ROW)
    {
        /* output row k corresponds to child dense row (k / d1). */
        (void) d2;
        for (int k = 0; k < C->m0; k++)
        {
            memcpy(C->X + k * n0, A->X + (k / d1) * n0, n0 * sizeof(double));
        }
    }
    else /* BROADCAST_COL */
    {
        (void) d1;
        size_t child_block = A->m0 * n0;
        for (int j = 0; j < d2; j++)
        {
            memcpy(C->X + j * child_block, A->X, child_block * sizeof(double));
        }
    }
}

static matrix *permuted_dense_vtable_broadcast_alloc(matrix *self,
                                                     broadcast_type type, int d1,
                                                     int d2)
{
    return broadcast_pd_alloc((const permuted_dense *) self, type, d1, d2);
}

static void permuted_dense_vtable_broadcast_fill_values(matrix *self,
                                                        broadcast_type type, int d1,
                                                        int d2, matrix *out)
{
    broadcast_pd_fill_values((const permuted_dense *) self, type, d1, d2,
                             (permuted_dense *) out);
}

matrix *diag_vec_pd_alloc(const permuted_dense *A)
{
    int n = A->base.m;
    int out_m = n * n;

    if (A->m0 == 0)
    {
        return new_permuted_dense(out_m, A->base.n, 0, A->n0, NULL, A->col_perm,
                                  NULL);
    }

    int *new_row_perm = (int *) sp_malloc(A->m0 * sizeof(int));
    for (int ii = 0; ii < A->m0; ii++)
    {
        new_row_perm[ii] = A->row_perm[ii] * (n + 1);
    }
    matrix *out = new_permuted_dense(out_m, A->base.n, A->m0, A->n0, new_row_perm,
                                     A->col_perm, NULL);
    sp_free(new_row_perm);
    return out;
}

void diag_vec_pd_fill_values(const permuted_dense *A, permuted_dense *C)
{
    if (A->m0 == 0)
    {
        return;
    }
    memcpy(C->X, A->X, A->m0 * A->n0 * sizeof(double));
}

static matrix *permuted_dense_vtable_diag_vec_alloc(matrix *self)
{
    return diag_vec_pd_alloc((const permuted_dense *) self);
}

static void permuted_dense_vtable_diag_vec_fill_values(matrix *self, matrix *out)
{
    diag_vec_pd_fill_values((const permuted_dense *) self, (permuted_dense *) out);
}

/* ===== Operator-role adapters: PD acting as the constant left operand of
   left_matmul. Currently restricted to full-block PDs (m0 == m, n0 == n,
   identity perms) — the only operator shape any caller needs today. */

static void permuted_dense_vtable_block_left_mult_vec(const matrix *A,
                                                      const double *x, double *y,
                                                      int p)
{
    /* Full-block precondition: A->x is a single contiguous row-major m x n
       block (perms are identity). For a non-trivial PD, A->x still points
       at pd->X but X only stores the values at the permuted positions; the
       layout below assumes a full m x n matrix, hence the assert. */
    assert(((const permuted_dense *) A)->m0 == A->m &&
           ((const permuted_dense *) A)->n0 == A->n);

    /* y = kron(I_p, A) @ x via a single dgemm.
       Input x is p blocks of length n (block-interleaved); output y is p
       blocks of length m. That's identical in memory to row-major matrices
       of shape (p, n) and (p, m) respectively, so we can compute
           y (p x m) = x (p x n) * A^T (n x m)
       in one shot. CblasRowMajor + CblasNoTrans on x + CblasTrans on A
       gives exactly that. */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, p, A->m, A->n, 1.0, x, A->n,
                A->x, A->n, 0.0, y, A->m);
}

static CSC_matrix *
permuted_dense_vtable_block_left_mult_sparsity(const matrix *A, const CSC_matrix *J,
                                               int p)
{
    const permuted_dense *pd = (const permuted_dense *) A;
    assert(pd->m0 == A->m && pd->n0 == A->n);
    /* Pre-size dwork for the subsequent block_left_mult_values fill, which
       densifies a sparse column of J (size A->n) before applying A. Honors
       the no-alloc-in-fill rule. */
    permuted_dense_ensure_kernel_dwork(pd, A->n);
    return I_kron_A_alloc(A, J, p);
}

static void permuted_dense_vtable_block_left_mult_values(const matrix *A,
                                                         const CSC_matrix *J,
                                                         CSC_matrix *C)
{
    const permuted_dense *pd = (const permuted_dense *) A;
    assert(pd->m0 == A->m && pd->n0 == A->n);
    I_kron_A_fill_values(A, J, C, pd->kernel_dwork);
}

/* C = sum-all-rows of A. */
static matrix *sum_all_rows_pd_alloc(matrix *A, int *idx_map)
{
    permuted_dense *pd = (permuted_dense *) A;

    /* allocate C */
    int zero = 0;
    matrix *C = new_permuted_dense(1, A->n, 1, pd->n0, &zero, pd->col_perm, NULL);

    /* fill idx_map */
    for (int i = 0; i < pd->m0; i++)
    {
        int *idx_base = idx_map + i * pd->n0;
        for (int j = 0; j < pd->n0; j++)
        {
            idx_base[j] = j;
        }
    }
    return C;
}

/* C = block-sum of A's rows in consecutive groups of d1. */
static matrix *sum_block_of_rows_pd_alloc(matrix *A_matrix, int d1, int *idx_map)
{
    permuted_dense *A = (permuted_dense *) A_matrix;
    int C_m0 = 0;
    int last_bucket = -1;
    int *C_row_perm = (int *) sp_malloc(A->m0 * sizeof(int));

    /* per input dense row ii, the index of its bucket within C_row_perm */
    int *row_to_out = (int *) sp_malloc(A->m0 * sizeof(int));

    // ---------------------------------------------------------------------------
    //                          determine C's row_perm
    // ---------------------------------------------------------------------------

    /* for every row in the dense block */
    for (int ii = 0; ii < A->m0; ii++)
    {
        /* find the bucket to which this row belongs */
        int bucket = A->row_perm[ii] / d1;

        /* add the bucket to C if it's new */
        if (bucket != last_bucket)
        {
            C_row_perm[C_m0++] = bucket;
            last_bucket = bucket;
        }

        /* map the input row of A to its row in C */
        row_to_out[ii] = C_m0 - 1;
    }

    matrix *C = new_permuted_dense(A_matrix->m / d1, A_matrix->n, C_m0, A->n0,
                                   C_row_perm, A->col_perm, NULL);

    // ---------------------------------------------------------------------------
    //                          fill idx_map
    // ---------------------------------------------------------------------------
    for (int ii = 0; ii < A->m0; ii++)
    {
        int offset = row_to_out[ii] * A->n0;
        int *idx_base = idx_map + ii * A->n0;
        for (int jj = 0; jj < A->n0; jj++)
        {
            idx_base[jj] = offset + jj;
        }
    }

    sp_free(row_to_out);
    sp_free(C_row_perm);
    return C;
}

/* C = stride-sum of A's rows at modular spacing d1. C has shape (d1, A->n);
   C[j, :] = sum_{i : i % d1 == j} A[i, :]. */
static matrix *sum_evenly_spaced_rows_pd_alloc(matrix *self, int d1, int *idx_map)
{
    permuted_dense *A = (permuted_dense *) self;

    // ---------------------------------------------------------------------------
    //          which buckets of [0, d1) are hit by A->row_perm?
    // ---------------------------------------------------------------------------
    bool *seen = (bool *) sp_calloc(d1, sizeof(bool));
    for (int ii = 0; ii < A->m0; ii++)
    {
        seen[A->row_perm[ii] % d1] = true;
    }

    // ---------------------------------------------------------------------------
    //              determine C's row_perm (guarantees sorted order)
    // ---------------------------------------------------------------------------
    int *C_row_perm = (int *) sp_malloc(A->m0 * sizeof(int));
    int *bucket_to_out_idx = (int *) sp_malloc(d1 * sizeof(int));
    int C_m0 = 0;
    for (int ii = 0; ii < d1; ii++)
    {
        if (seen[ii])
        {
            bucket_to_out_idx[ii] = C_m0;
            C_row_perm[C_m0++] = ii;
        }
    }
    sp_free(seen);

    matrix *C =
        new_permuted_dense(d1, self->n, C_m0, A->n0, C_row_perm, A->col_perm, NULL);

    // ---------------------------------------------------------------------------
    //                          fill idx_map
    // ---------------------------------------------------------------------------
    for (int ii = 0; ii < A->m0; ii++)
    {
        int base = bucket_to_out_idx[A->row_perm[ii] % d1] * A->n0;
        int *idx_base = idx_map + ii * A->n0;
        for (int jj = 0; jj < A->n0; jj++)
        {
            idx_base[jj] = base + jj;
        }
    }

    sp_free(bucket_to_out_idx);
    sp_free(C_row_perm);
    return C;
}

static matrix *permuted_dense_vtable_sum_row_partition_alloc(matrix *self, int axis,
                                                             int d1, int *idx_map)
{
    if (axis == -1)
    {
        return sum_all_rows_pd_alloc(self, idx_map);
    }

    if (axis == 0)
    {
        return sum_block_of_rows_pd_alloc(self, d1, idx_map);
    }

    return sum_evenly_spaced_rows_pd_alloc(self, d1, idx_map); /* axis == 1 */
}

static void wire_vtable(permuted_dense *pd)
{
    pd->base.is_permuted_dense = true;
    pd->base.free_fn = permuted_dense_free;
    pd->base.block_left_mult_vec = permuted_dense_vtable_block_left_mult_vec;
    pd->base.block_left_mult_sparsity =
        permuted_dense_vtable_block_left_mult_sparsity;
    pd->base.block_left_mult_values = permuted_dense_vtable_block_left_mult_values;
    pd->base.copy_sparsity = permuted_dense_vtable_copy_sparsity;
    pd->base.DA_fill_values = permuted_dense_vtable_DA_fill_values;
    pd->base.ATA_alloc = permuted_dense_vtable_ATA_alloc;
    pd->base.ATDA_fill_values = permuted_dense_vtable_ATDA_fill_values;
    pd->base.to_csr = permuted_dense_to_csr;
    pd->base.transpose_alloc = permuted_dense_vtable_transpose_alloc;
    pd->base.transpose_fill_values = permuted_dense_vtable_transpose_fill_values;
    pd->base.index_alloc = permuted_dense_vtable_index_alloc;
    pd->base.index_fill_values = permuted_dense_vtable_index_fill_values;
    pd->base.promote_alloc = permuted_dense_vtable_promote_alloc;
    pd->base.promote_fill_values = permuted_dense_vtable_promote_fill_values;
    pd->base.broadcast_alloc = permuted_dense_vtable_broadcast_alloc;
    pd->base.broadcast_fill_values = permuted_dense_vtable_broadcast_fill_values;
    pd->base.diag_vec_alloc = permuted_dense_vtable_diag_vec_alloc;
    pd->base.diag_vec_fill_values = permuted_dense_vtable_diag_vec_fill_values;
    pd->base.sum_row_partition_alloc = permuted_dense_vtable_sum_row_partition_alloc;
    pd->base.refresh_csc_values = permuted_dense_refresh_csc_values;
}

matrix *new_permuted_dense(int m, int n, int m0, int n0, const int *row_perm,
                           const int *col_perm, const double *X_data)
{
    /* Validate sorted invariants. */
    for (int ii = 1; ii < m0; ii++)
    {
        assert(row_perm[ii] > row_perm[ii - 1]);
    }
    for (int jj = 1; jj < n0; jj++)
    {
        assert(col_perm[jj] > col_perm[jj - 1]);
    }
    if (m0 > 0)
    {
        assert(row_perm[0] >= 0 && row_perm[m0 - 1] < m);
    }
    if (n0 > 0)
    {
        assert(col_perm[0] >= 0 && col_perm[n0 - 1] < n);
    }

    permuted_dense *pd = (permuted_dense *) sp_calloc(1, sizeof(permuted_dense));
    pd->base.m = m;
    pd->base.n = n;
    pd->base.nnz = m0 * n0;
    wire_vtable(pd);

    pd->m0 = m0;
    pd->n0 = n0;

    int sz = m0 * n0;
    pd->row_perm = (int *) sp_malloc(m0 * sizeof(int));
    pd->col_perm = (int *) sp_malloc(n0 * sizeof(int));
    pd->X = (double *) sp_malloc(sz * sizeof(double));
    pd->base.x = pd->X;
    pd->owns_X = true;
    pd->col_inv = (int *) sp_malloc(n * sizeof(int));
    pd->row_inv = (int *) sp_malloc(m * sizeof(int));

    if (m0 > 0)
    {
        memcpy(pd->row_perm, row_perm, m0 * sizeof(int));
    }
    if (n0 > 0)
    {
        memcpy(pd->col_perm, col_perm, n0 * sizeof(int));
    }

    for (int j = 0; j < n; j++)
    {
        pd->col_inv[j] = -1;
    }
    for (int jj = 0; jj < n0; jj++)
    {
        pd->col_inv[col_perm[jj]] = jj;
    }

    for (int i = 0; i < m; i++)
    {
        pd->row_inv[i] = -1;
    }
    for (int ii = 0; ii < m0; ii++)
    {
        pd->row_inv[row_perm[ii]] = ii;
    }

    if (X_data != NULL && sz > 0)
    {
        memcpy(pd->X, X_data, sz * sizeof(double));
    }

    return &pd->base;
}

matrix *new_permuted_dense_full(int m, int n, const double *data)
{
    int *row_perm = (int *) sp_malloc(m * sizeof(int));
    int *col_perm = (int *) sp_malloc(n * sizeof(int));
    for (int i = 0; i < m; i++) row_perm[i] = i;
    for (int j = 0; j < n; j++) col_perm[j] = j;
    matrix *out = new_permuted_dense(m, n, m, n, row_perm, col_perm, data);
    sp_free(row_perm);
    sp_free(col_perm);
    return out;
}

static CSR_matrix *permuted_dense_to_csr_alloc(const permuted_dense *A)
{
    int m0 = A->m0;
    int n0 = A->n0;
    int m = A->base.m;
    CSR_matrix *C = new_CSR_matrix(m, A->base.n, m0 * n0);

    /* Alias C->x to A->X: the dense block layout already matches what the
       CSR_matrix view's value array would hold, so values are always live with no
       memcpy needed. The PD owns the buffer; permuted_dense_free nulls
       C->x before free_CSR_matrix to avoid double-free. */
    sp_free(C->x);
    C->x = A->X;

    /* fill column indices (each dense row contributes a copy of col_perm) */
    for (int ii = 0; ii < m0; ii++)
    {
        memcpy(C->i + ii * n0, A->col_perm, n0 * sizeof(int));
    }

    /* set row pointers via count and then cumulative sum  */
    memset(C->p, 0, (m + 1) * sizeof(int));
    for (int ii = 0; ii < m0; ii++)
    {
        C->p[A->row_perm[ii] + 1] = n0;
    }

    cumsum(C->p, m);

    return C;
}

void permuted_dense_ensure_kernel_dwork(const permuted_dense *pd_const, size_t size)
{
    /* TODO: refactor this - maybe pass in kernel and size instead? Then this is
     * a general function, not necessarily tied to permuted_dense*/
    permuted_dense *pd = (permuted_dense *) pd_const;
    if (pd->kernel_dwork_size >= size) return;
    sp_free(pd->kernel_dwork);
    pd->kernel_dwork = (double *) sp_malloc(size * sizeof(double));
    pd->kernel_dwork_size = size;
}
