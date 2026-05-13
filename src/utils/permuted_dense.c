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
#include "utils/iVec.h"
#include "utils/linalg_dense_sparse_matmuls.h"
#include "utils/tracked_alloc.h"
#include "utils/utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void permuted_dense_free(matrix *self)
{
    permuted_dense *pd = (permuted_dense *) self;
    free(pd->row_perm);
    free(pd->col_perm);
    free(pd->col_inv);
    free(pd->row_inv);
    /* csr_cache->x aliases pd->X (set in permuted_dense_to_csr_alloc); NULL it
       so free_CSR_matrix doesn't double-free the shared buffer. */
    if (pd->csr_cache != NULL)
    {
        pd->csr_cache->x = NULL;
    }
    free_CSR_matrix(pd->csr_cache);
    free(pd->X);
    free(pd->dwork);
    free(pd->iwork);
    free(pd);
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
    const permuted_dense *pd = (const permuted_dense *) self;
    return new_permuted_dense(pd->base.m, pd->base.n, pd->m0, pd->n0, pd->row_perm,
                              pd->col_perm, NULL);
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

/* Forward decls; definitions lower in the file. */
static CSR_matrix *permuted_dense_to_csr_alloc(const permuted_dense *A);
static void permuted_dense_ensure_dwork(const permuted_dense *pd_const, size_t size);

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
    const permuted_dense *pd = (const permuted_dense *) self;
    /* Swap (m, n), (m0, n0), and (row_perm, col_perm). The constructor
       asserts strict increase of both perms, which holds by construction. */
    return new_permuted_dense(pd->base.n, pd->base.m, pd->n0, pd->m0, pd->col_perm,
                              pd->row_perm, NULL);
}

static void permuted_dense_vtable_transpose_fill_values(const matrix *self,
                                                        matrix *out)
{
    const permuted_dense *pd_in = (const permuted_dense *) self;
    permuted_dense *pd_out = (permuted_dense *) out;
    int m0 = pd_in->m0;
    int n0 = pd_in->n0;
    /* pd_out has shape (n0, m0); transpose pd_in->X into pd_out->X. */
    for (int ii = 0; ii < m0; ii++)
    {
        for (int jj = 0; jj < n0; jj++)
        {
            pd_out->X[jj * m0 + ii] = pd_in->X[ii * n0 + jj];
        }
    }
}

static matrix *permuted_dense_vtable_index_alloc(matrix *self, const int *indices,
                                                 int n_idxs)
{
    const permuted_dense *pd = (const permuted_dense *) self;

    /* Scan indices: which output positions i hit a row in pd->row_perm? */
    int *new_row_perm = (int *) SP_MALLOC(n_idxs * sizeof(int));
    int new_m0 = 0;
    for (int i = 0; i < n_idxs; i++)
    {
        if (pd->row_inv[indices[i]] >= 0)
        {
            new_row_perm[new_m0++] = i;
        }
    }

    matrix *out = new_permuted_dense(n_idxs, pd->base.n, new_m0, pd->n0,
                                     new_row_perm, pd->col_perm, NULL);
    free(new_row_perm);
    return out;
}

static void permuted_dense_vtable_index_fill_values(matrix *self, const int *indices,
                                                    int n_idxs, matrix *out)
{
    (void) n_idxs;
    const permuted_dense *pd = (const permuted_dense *) self;
    permuted_dense *out_pd = (permuted_dense *) out;
    int n0 = pd->n0;
    for (int k = 0; k < out_pd->m0; k++)
    {
        int i = out_pd->row_perm[k];
        int old_ii = pd->row_inv[indices[i]];
        memcpy(out_pd->X + k * n0, pd->X + old_ii * n0, n0 * sizeof(double));
    }
}

static matrix *permuted_dense_vtable_promote_alloc(matrix *self, int size)
{
    const permuted_dense *pd = (const permuted_dense *) self;
    assert(pd->m0 <= 1);

    if (pd->m0 == 0)
    {
        /* source row is all-zero; output is also structurally all-zero. */
        return new_permuted_dense(size, pd->base.n, 0, pd->n0, NULL, pd->col_perm,
                                  NULL);
    }

    int *new_row_perm = (int *) SP_MALLOC(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        new_row_perm[i] = i;
    }
    matrix *out = new_permuted_dense(size, pd->base.n, size, pd->n0, new_row_perm,
                                     pd->col_perm, NULL);
    free(new_row_perm);
    return out;
}

static void permuted_dense_vtable_promote_fill_values(matrix *self, matrix *out)
{
    const permuted_dense *pd = (const permuted_dense *) self;
    permuted_dense *out_pd = (permuted_dense *) out;
    if (pd->m0 == 0) return;
    int n0 = pd->n0;
    for (int k = 0; k < out_pd->m0; k++)
    {
        memcpy(out_pd->X + k * n0, pd->X, n0 * sizeof(double));
    }
}

static matrix *permuted_dense_vtable_broadcast_alloc(matrix *self,
                                                     broadcast_type type, int d1,
                                                     int d2)
{
    const permuted_dense *pd = (const permuted_dense *) self;
    int out_m = d1 * d2;

    int new_m0;
    if (type == BROADCAST_SCALAR)
    {
        new_m0 = (pd->m0 == 0) ? 0 : out_m;
    }
    else if (type == BROADCAST_ROW)
    {
        new_m0 = d1 * pd->m0;
    }
    else /* BROADCAST_COL */
    {
        new_m0 = d2 * pd->m0;
    }

    if (new_m0 == 0)
    {
        return new_permuted_dense(out_m, pd->base.n, 0, pd->n0, NULL, pd->col_perm,
                                  NULL);
    }

    int *new_row_perm = (int *) SP_MALLOC(new_m0 * sizeof(int));
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
        for (int j_ii = 0; j_ii < pd->m0; j_ii++)
        {
            int j_old = pd->row_perm[j_ii];
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
            for (int ii_old = 0; ii_old < pd->m0; ii_old++)
            {
                new_row_perm[k++] = j * d1 + pd->row_perm[ii_old];
            }
        }
    }

    matrix *out = new_permuted_dense(out_m, pd->base.n, new_m0, pd->n0, new_row_perm,
                                     pd->col_perm, NULL);
    free(new_row_perm);
    return out;
}

static void permuted_dense_vtable_broadcast_fill_values(matrix *self,
                                                        broadcast_type type, int d1,
                                                        int d2, matrix *out)
{
    const permuted_dense *pd = (const permuted_dense *) self;
    permuted_dense *out_pd = (permuted_dense *) out;
    if (pd->m0 == 0)
    {
        return;
    }
    int n0 = pd->n0;

    if (type == BROADCAST_SCALAR)
    {
        for (int k = 0; k < out_pd->m0; k++)
        {
            memcpy(out_pd->X + k * n0, pd->X, n0 * sizeof(double));
        }
    }
    else if (type == BROADCAST_ROW)
    {
        /* output row k corresponds to child dense row (k / d1). */
        (void) d2;
        for (int k = 0; k < out_pd->m0; k++)
        {
            memcpy(out_pd->X + k * n0, pd->X + (k / d1) * n0, n0 * sizeof(double));
        }
    }
    else /* BROADCAST_COL */
    {
        (void) d1;
        size_t child_block = pd->m0 * n0;
        for (int j = 0; j < d2; j++)
        {
            memcpy(out_pd->X + j * child_block, pd->X, child_block * sizeof(double));
        }
    }
}

static matrix *permuted_dense_vtable_diag_vec_alloc(matrix *self)
{
    const permuted_dense *pd = (const permuted_dense *) self;
    int n = pd->base.m;
    int out_m = n * n;

    if (pd->m0 == 0)
    {
        return new_permuted_dense(out_m, pd->base.n, 0, pd->n0, NULL, pd->col_perm,
                                  NULL);
    }

    int *new_row_perm = (int *) SP_MALLOC(pd->m0 * sizeof(int));
    for (int ii = 0; ii < pd->m0; ii++)
    {
        new_row_perm[ii] = pd->row_perm[ii] * (n + 1);
    }
    matrix *out = new_permuted_dense(out_m, pd->base.n, pd->m0, pd->n0, new_row_perm,
                                     pd->col_perm, NULL);
    free(new_row_perm);
    return out;
}

static void permuted_dense_vtable_diag_vec_fill_values(matrix *self, matrix *out)
{
    const permuted_dense *pd = (const permuted_dense *) self;
    permuted_dense *out_pd = (permuted_dense *) out;
    if (pd->m0 == 0)
    {
        return;
    }
    memcpy(out_pd->X, pd->X, pd->m0 * pd->n0 * sizeof(double));
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
    permuted_dense_ensure_dwork(pd, (size_t) A->n);
    return I_kron_A_alloc(A, J, p);
}

static void permuted_dense_vtable_block_left_mult_values(const matrix *A,
                                                         const CSC_matrix *J,
                                                         CSC_matrix *C)
{
    const permuted_dense *pd = (const permuted_dense *) A;
    assert(pd->m0 == A->m && pd->n0 == A->n);
    I_kron_A_fill_values(A, J, C, pd->dwork);
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

    permuted_dense *pd = (permuted_dense *) SP_CALLOC(1, sizeof(permuted_dense));
    pd->base.m = m;
    pd->base.n = n;
    pd->base.nnz = m0 * n0;
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
    pd->base.is_permuted_dense = true;
    pd->base.index_alloc = permuted_dense_vtable_index_alloc;
    pd->base.index_fill_values = permuted_dense_vtable_index_fill_values;
    pd->base.promote_alloc = permuted_dense_vtable_promote_alloc;
    pd->base.promote_fill_values = permuted_dense_vtable_promote_fill_values;
    pd->base.broadcast_alloc = permuted_dense_vtable_broadcast_alloc;
    pd->base.broadcast_fill_values = permuted_dense_vtable_broadcast_fill_values;
    pd->base.diag_vec_alloc = permuted_dense_vtable_diag_vec_alloc;
    pd->base.diag_vec_fill_values = permuted_dense_vtable_diag_vec_fill_values;
    pd->base.refresh_csc_values = permuted_dense_refresh_csc_values;
    pd->base.free_fn = permuted_dense_free;

    pd->m0 = m0;
    pd->n0 = n0;

    int sz = m0 * n0;
    pd->row_perm = (int *) SP_MALLOC(m0 * sizeof(int));
    pd->col_perm = (int *) SP_MALLOC(n0 * sizeof(int));
    pd->X = (double *) SP_MALLOC(sz * sizeof(double));
    pd->base.x = pd->X;
    /* dwork is allocated lazily by kernels via permuted_dense_ensure_dwork.
       SP_CALLOC above already zeroed dwork / dwork_size, but make it
       explicit. */
    pd->dwork = NULL;
    pd->dwork_size = 0;
    pd->col_inv = (int *) SP_MALLOC(n * sizeof(int));
    pd->row_inv = (int *) SP_MALLOC(m * sizeof(int));

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
    int *row_perm = (int *) SP_MALLOC(m * sizeof(int));
    int *col_perm = (int *) SP_MALLOC(n * sizeof(int));
    for (int i = 0; i < m; i++) row_perm[i] = i;
    for (int j = 0; j < n; j++) col_perm[j] = j;
    matrix *out = new_permuted_dense(m, n, m, n, row_perm, col_perm, data);
    free(row_perm);
    free(col_perm);
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
    free(C->x);
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

    for (int i = 0; i < m; i++)
    {
        C->p[i + 1] += C->p[i];
    }

    return C;
}

void DA_pd_fill_values(const double *d, const permuted_dense *A, permuted_dense *C)
{
    int m0 = A->m0;
    int n0 = A->n0;
    cblas_dcopy(m0 * n0, A->X, 1, C->X, 1);
    for (int ii = 0; ii < m0; ii++)
    {
        cblas_dscal(n0, d[A->row_perm[ii]], C->X + ii * n0, 1);
    }
}

/* Ensure pd->dwork is sized at least `size` doubles. Grows in place;
   contents are NOT preserved. Called from allocator functions so that the
   corresponding fill kernels never need to allocate. Takes a const pointer
   and casts internally — this matches the dwork contract (header) that
   dwork is mutable through a const permuted_dense *. */
static void permuted_dense_ensure_dwork(const permuted_dense *pd_const, size_t size)
{
    permuted_dense *pd = (permuted_dense *) pd_const;
    if (pd->dwork_size >= size) return;
    free(pd->dwork);
    pd->dwork = (double *) SP_MALLOC(size * sizeof(double));
    pd->dwork_size = size;
}

matrix *ATA_pd_alloc(const permuted_dense *A)
{
    int n = A->base.n;
    /* C = AT @ A has a dense block of size n0 x n0, with row and column index
       sets given by A's col_perm. (This follows from Cij = ai^T aj where
       ai and aj are columns of A. Here, ai and aj always have overlapping entries,
       so Cij != 0 for (i, j) in A->col_perm x A->col_perm) */

    /* Pre-size A's dwork for the ATDA fill (Y-buffer = diag(d_perm) X). */
    permuted_dense_ensure_dwork(A, (size_t) A->m0 * A->n0);

    return new_permuted_dense(n, n, A->n0, A->n0, A->col_perm, A->col_perm, NULL);
}

void ATDA_pd_fill_values(const permuted_dense *A, const double *d, permuted_dense *C)
{
    int m0 = A->m0;
    int n0 = A->n0;

    /* dwork = diag(d_perm) @ X, where d_perm[ii] = d[row_perm[ii]]. */
    cblas_dcopy(m0 * n0, A->X, 1, A->dwork, 1);
    for (int ii = 0; ii < m0; ii++)
    {
        cblas_dscal(n0, d[A->row_perm[ii]], A->dwork + ii * n0, 1);
    }

    /* C  = XT @ dwork = XT @ diag(d_perm) @ X */
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n0, n0, m0, 1.0, A->X, n0,
                A->dwork, n0, 0.0, C->X, n0);
}

matrix *BTA_pd_pd_alloc(const permuted_dense *B, const permuted_dense *A)
{
    /* if A and B have no overlapping rows, then C = BT @ A is empty */
    if (!has_overlap(A->row_perm, A->m0, B->row_perm, B->m0, 0))
    {
        return new_permuted_dense(B->base.n, A->base.n, 0, 0, NULL, NULL, NULL);
    }

    /* otherwise C has a dense block of size B->n0 x A->n0, with row and column
       index sets given by B->col_perm and A->col_perm, respectively */
    matrix *C = new_permuted_dense(B->base.n, A->base.n, B->n0, A->n0, B->col_perm,
                                   A->col_perm, NULL);

    /* Pre-size A's and B's dwork for the BTA fill slow path (gathered row
       buffers). Each operand needs s_max rows of its own n0 doubles, where
       s_max = MIN(A->m0, B->m0) bounds the intersection of row_perms. */
    int s_max = MIN(A->m0, B->m0);
    permuted_dense_ensure_dwork(A, (size_t) s_max * A->n0);
    permuted_dense_ensure_dwork(B, (size_t) s_max * B->n0);

    /* Pre-allocate C->iwork for idx_A + idx_B in BTA / BTDA_pd_pd slow paths
       (each needs at most s_max ints; we store both arrays back-to-back
       in iwork, hence 2 * s_max). */
    permuted_dense *C_pd = (permuted_dense *) C;
    C_pd->iwork_size = (size_t) 2 * s_max;
    C_pd->iwork = (int *) SP_MALLOC(C_pd->iwork_size * sizeof(int));

    return C;
}

/* Return 1 iff arrays a and b of length n are element-wise equal. */
static int int_arrays_equal(const int *a, const int *b, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

/* Find intersection of two sorted, ascending int arrays. For each pair of positions
   (ii, jj) where a[ii] == b[jj], write ii into idx_a and jj into idx_b. Returns the
   count of matches. Buffers idx_a and idx_b must have capacity >= min(a_len, b_len);
   no allocation is performed. */
static inline int sorted_intersect_indices(const int *a, int a_len, const int *b,
                                           int b_len, int *idx_a, int *idx_b)
{
    int s = 0;
    int ii = 0, jj = 0;
    while (ii < a_len && jj < b_len)
    {
        int ra = a[ii];
        int rb = b[jj];
        if (ra == rb)
        {
            idx_a[s] = ii;
            idx_b[s] = jj;
            s++;
            ii++;
            jj++;
        }
        else if (ra < rb)
        {
            ii++;
        }
        else
        {
            jj++;
        }
    }
    return s;
}

void BTA_pd_pd_fill_values(const permuted_dense *B, const permuted_dense *A,
                           permuted_dense *C)
{
    /* C may be empty if there is no overlap in row permutations */
    if (C->base.nnz == 0)
    {
        return;
    }

    /* if B and A have identical row_perms, one matmul suffices */
    if (A->m0 == B->m0 && int_arrays_equal(A->row_perm, B->row_perm, A->m0))
    {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, B->n0, A->n0, A->m0,
                    1.0, B->X, B->n0, A->X, A->n0, 0.0, C->X, A->n0);
        return;
    }

    // -----------------------------------------------------------------------
    // find intersection of row permutations. C->iwork was pre-sized by
    // BTA_pd_pd_alloc to 2 * MIN(A->m0, B->m0) ints (idx_A | idx_B back-
    // to-back), so no allocation here.
    // -----------------------------------------------------------------------
    int s_max = MIN(A->m0, B->m0);
    int *idx_A = C->iwork;
    int *idx_B = C->iwork + s_max;
    int s = sorted_intersect_indices(A->row_perm, A->m0, B->row_perm, B->m0, idx_A,
                                     idx_B);
    assert(s > 0);

    // ------------------------------------------------------------------------
    // Gather the matching rows into A->dwork and B->dwork. dwork is pre-sized
    // by BTA_pd_pd_alloc (one ensure_dwork call per operand at alloc time).
    // ------------------------------------------------------------------------
    for (int k = 0; k < s; k++)
    {
        memcpy(A->dwork + k * A->n0, A->X + idx_A[k] * A->n0,
               A->n0 * sizeof(double));
        memcpy(B->dwork + k * B->n0, B->X + idx_B[k] * B->n0,
               B->n0 * sizeof(double));
    }

    /* matmul on the gathered rows */
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, B->n0, A->n0, s, 1.0,
                B->dwork, B->n0, A->dwork, A->n0, 0.0, C->X, A->n0);
}

void BTDA_pd_pd_fill_values(const permuted_dense *B, const double *d,
                            const permuted_dense *A, permuted_dense *C)
{
    /* C may be empty if there is no overlap in row permutations of A and B */
    if (C->base.nnz == 0)
    {
        return;
    }

    /* TODO: must remove this allocation. Very important. The DA
       intermediate PD is allocated and freed on every Hessian iteration
       — violates the no-alloc-in-fill policy. Fix is to fold diag(d)
       directly into BTA_pd_pd_fill_values's gather/dgemm (either via a
       shared internal helper that takes an optional d, or by rewriting
       this kernel inline using pre-sized A->dwork). */
    /* C = BT @ (DA) */
    permuted_dense *DA = (permuted_dense *) A->base.copy_sparsity(&A->base);
    DA_pd_fill_values(d, A, DA);
    BTA_pd_pd_fill_values(B, DA, C);
    free_matrix(&DA->base);
}

/* The CSR-flavored kernels for (B=Sparse, A=PD) live in src/old-code; the
   production path uses BTA_csc_pd_alloc / BTDA_csc_pd_fill_values defined
   further below, which delegate to BTA_pd_csc via the (A^T B)^T identity. */

/* Return true if any of the 'len' integers in 'indices' exist in the set
   marked by 'inv' (inv[k] != -1 iff k is in the set). */
static inline bool idxs_hits_set(const int *idxs, int len, const int *inv)
{
    for (int ii = 0; ii < len; ii++)
    {
        if (inv[idxs[ii]] != -1)
        {
            return true;
        }
    }
    return false;
}

/* Inner product of a sparse vector (vals[0..len) at positions idxs[0..len))
   with a dense vector, where inv maps each idxs value to a position in
   'dense' (inv[k] == -1 means skip that entry). */
static inline double sparse_dot_dense(const double *vals, const int *idxs, int len,
                                      const int *inv, const double *dense)
{
    double sum = 0.0;
    for (int e = 0; e < len; e++)
    {
        int kk = inv[idxs[e]];
        if (kk == -1)
        {
            continue;
        }
        sum += vals[e] * dense[kk];
    }
    return sum;
}

matrix *BA_pd_csc_alloc(const permuted_dense *B, const CSC_matrix *A)
{
    /* Cij != 0 if row i of B overlaps with column j of A. So we loop through
    the columns of A. For each column of A, we check if it has any nonzeros in
    rows that are in B's col_perm. If yes, column j of C will have a nonzero
    block corresponding to the rows of B */
    iVec *col_perm_C = iVec_new(10);
    for (int j = 0; j < A->n; j++)
    {
        int start = A->p[j];
        int len = A->p[j + 1] - start;
        if (idxs_hits_set(A->i + start, len, B->col_inv))
        {
            iVec_append(col_perm_C, j);
        }
    }

    matrix *C = new_permuted_dense(B->base.m, A->n, B->m0, col_perm_C->len,
                                   B->row_perm, col_perm_C->data, NULL);
    iVec_free(col_perm_C);
    return C;
}

void BA_pd_csc_fill_values(const double *B, int n0_B, const int *inv,
                           const CSC_matrix *A, permuted_dense *C)
{
    /* C[i, j] = bi^T @ ajj, where bi is the ith row of B_X (length n0_B,
       row stride n0_B) and ajj is the jjth column of A's sparse block
       (column jj = C->col_perm[j]). inv maps A's row indices to positions
       in B_X (entries with inv[r] == -1 are skipped). */

    /* row i of C */
    for (int i = 0; i < C->m0; i++)
    {
        double *ci = C->X + i * C->n0;

        /* col j of C  */
        for (int j = 0; j < C->n0; j++)
        {

            int jj = C->col_perm[j];
            int start = A->p[jj];
            int len = A->p[jj + 1] - start;
            /* we compute entry C[i, j] */
            ci[j] =
                sparse_dot_dense(A->x + start, A->i + start, len, inv, B + i * n0_B);
        }
    }
}

matrix *BA_pd_pd_alloc(const permuted_dense *B, const permuted_dense *A)
{
    /* if B's columns don't overlap with A's rows, C = B @ A is empty */
    if (!has_overlap(B->col_perm, B->n0, A->row_perm, A->m0, 0))
    {
        return new_permuted_dense(B->base.m, A->base.n, 0, 0, NULL, NULL, NULL);
    }

    /* otherwise C has a dense block of size B->m0 x A->n0, with row index
       set B->row_perm and column index set A->col_perm. */
    matrix *C = new_permuted_dense(B->base.m, A->base.n, B->m0, A->n0, B->row_perm,
                                   A->col_perm, NULL);

    int s_max = MIN(B->n0, A->m0);

    /* Pre-size B's and A's dwork for the gathers in fill. Worst-case
       intersection size is s_max; B_sub is (m0, s) and A_sub is (s, n0). */
    permuted_dense_ensure_dwork(A, (size_t) s_max * A->n0);
    permuted_dense_ensure_dwork(B, (size_t) s_max * B->m0);

    /* Pre-allocate C->iwork for idx_B + idx_A back-to-back (2 * s_max ints),
       same idiom as BTA_pd_pd_alloc. */
    permuted_dense *C_pd = (permuted_dense *) C;
    C_pd->iwork_size = (size_t) 2 * s_max;
    C_pd->iwork = (int *) SP_MALLOC(C_pd->iwork_size * sizeof(int));

    return C;
}

/* TODO: do we want to reuse BTA_pd_pd_fill_values? */
void BA_pd_pd_fill_values(const permuted_dense *B, const permuted_dense *A,
                          permuted_dense *C)
{
    /* C may be empty when B->col_perm and A->row_perm don't overlap. */
    if (C->base.nnz == 0)
    {
        return;
    }

    /* if B's col_perm and A's row_perm are identical, one matmul suffices */
    if (B->n0 == A->m0 && int_arrays_equal(B->col_perm, A->row_perm, B->n0))
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B->m0, A->n0, B->n0,
                    1.0, B->X, B->n0, A->X, A->n0, 0.0, C->X, A->n0);
        return;
    }

    // -----------------------------------------------------------------------
    // find intersection of B's col_perm and A's row_perm. C->iwork was
    // pre-sized by BA_pd_pd_alloc to 2 * MIN(B->n0, A->m0) ints (idx_B |
    // idx_A back-to-back), so no allocation here.
    // -----------------------------------------------------------------------
    int s_max = MIN(B->n0, A->m0);
    int *idx_B = C->iwork;
    int *idx_A = C->iwork + s_max;
    int s = sorted_intersect_indices(B->col_perm, B->n0, A->row_perm, A->m0, idx_B,
                                     idx_A);
    assert(s > 0);

    // ------------------------------------------------------------------------
    // Gather the matching slices into B->dwork (column gather) and A->dwork
    // (row gather). dwork is pre-sized by BA_pd_pd_alloc (one ensure_dwork
    // call per operand at alloc time).
    // ------------------------------------------------------------------------
    /* B_sub shape (B->m0, s) row-major: B_sub[ii, kk] = B->X[ii, idx_B[kk]]. */
    for (int ii = 0; ii < B->m0; ii++)
    {
        for (int kk = 0; kk < s; kk++)
        {
            B->dwork[ii * s + kk] = B->X[ii * B->n0 + idx_B[kk]];
        }
    }
    /* A_sub shape (s, A->n0) row-major: A_sub[kk, :] = A->X[idx_A[kk], :]. */
    for (int kk = 0; kk < s; kk++)
    {
        memcpy(A->dwork + kk * A->n0, A->X + idx_A[kk] * A->n0,
               A->n0 * sizeof(double));
    }

    /* matmul on the gathered slices */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B->m0, A->n0, s, 1.0,
                B->dwork, s, A->dwork, A->n0, 0.0, C->X, A->n0);
}

matrix *BTA_pd_csc_alloc(const permuted_dense *B, const CSC_matrix *A)
{
    /* Cij != 0 if column i of B overlaps with column j of A. So we loop
    through the columns of A. For each column of A, we check if it has any
    nonzeros in rows that are in B's row_perm. If yes, column j of C will
    have a nonzero block corresponding to the columns of B */
    iVec *col_active = iVec_new(8);
    for (int j = 0; j < A->n; j++)
    {
        int start = A->p[j];
        int len = A->p[j + 1] - start;
        if (idxs_hits_set(A->i + start, len, B->row_inv))
        {
            iVec_append(col_active, j);
        }
    }

    matrix *C = new_permuted_dense(B->base.n, A->n, B->n0, col_active->len,
                                   B->col_perm, col_active->data, NULL);
    iVec_free(col_active);

    /* Pre-size B's dwork for the BTDA fill (holds (diag(d) B)^T). */
    permuted_dense_ensure_dwork(B, (size_t) B->m0 * B->n0);

    return C;
}

/* C = B^T diag(d) A = (diag (d) B)^T A */
void BTDA_pd_csc_fill_values(const permuted_dense *B, const double *d,
                             const CSC_matrix *A, permuted_dense *C)
{
    /* C may be empty */
    if (C->base.nnz == 0)
    {
        return;
    }

    int m0 = B->m0;
    int n0 = B->n0;

    /* conpute B->dwork = (diag(d) B)^T */
    for (int kk = 0; kk < m0; kk++)
    {
        double dk = d[B->row_perm[kk]];
        for (int ii = 0; ii < n0; ii++)
        {
            B->dwork[ii * m0 + kk] = dk * B->X[kk * n0 + ii];
        }
    }

    BA_pd_csc_fill_values(B->dwork, m0, B->row_inv, A, C);
}

matrix *BTA_csc_pd_alloc(const CSC_matrix *B, const permuted_dense *A)
{
    /* Cij != 0 if column i of B overlaps with row j of A. So we loop through the
       columns of B. For each column of B, we check if it has any nonzeros in rows
       that are in A->row_perm. If yes, column i of C will have a nonzero block
       corresponding to the columns of A */

    iVec *row_active = iVec_new(10);
    for (int i = 0; i < B->n; i++)
    {
        int start = B->p[i];
        int len = B->p[i + 1] - start;
        if (idxs_hits_set(B->i + start, len, A->row_inv))
        {
            iVec_append(row_active, i);
        }
    }

    matrix *C = new_permuted_dense(B->n, A->base.n, row_active->len, A->n0,
                                   row_active->data, A->col_perm, NULL);
    iVec_free(row_active);

    /* Pre-size A's dwork for the BTDA fill (holds (diag(d_perm) X_A)^T). */
    permuted_dense_ensure_dwork(A, (size_t) A->m0 * A->n0);

    return C;
}

/* Internal helper for BTDA_csc_pd_fill_values: C = B^T @ A where B is CSC
   and the right operand A is supplied as a transposed-layout raw buffer
   (row j of A_T = m0_A contiguous doubles = the j-th column of A's dense
   block). Transposed-output sibling of BA_pd_csc_fill_values. */
static void BTA_csc_pd_fill_values(const CSC_matrix *B, const double *A_T, int m0_A,
                                   const int *inv, permuted_dense *C)
{
    /* C[i_C, j_C] = dot(col C->row_perm[i_C] of B, row j_C of A_T). */
    for (int i_C = 0; i_C < C->m0; i_C++)
    {
        int B_col = C->row_perm[i_C];
        int start = B->p[B_col];
        int len = B->p[B_col + 1] - start;
        double *ci = C->X + i_C * C->n0;
        for (int j_C = 0; j_C < C->n0; j_C++)
        {
            ci[j_C] = sparse_dot_dense(B->x + start, B->i + start, len, inv,
                                       A_T + j_C * m0_A);
        }
    }
}

/* C = B^T diag(d) A. Folds diag(d) into A's dense block (writing
   (diag(d_perm) X_A)^T into A->dwork) and delegates to BTA_csc_pd_fill_values.
   Mirrors how BTDA_pd_csc_fill_values wraps BA_pd_csc_fill_values. */
void BTDA_csc_pd_fill_values(const CSC_matrix *B, const double *d,
                             const permuted_dense *A, permuted_dense *C)
{
    if (C->base.nnz == 0)
    {
        return;
    }

    int m0_A = A->m0;
    int n0_A = A->n0;

    /* A->dwork = (diag(d_perm) X_A)^T, row-major shape (n0_A, m0_A).
       Pre-sized by BTA_csc_pd_alloc; no allocation in fill.
       Column j of (diag(d) X_A) lives contiguously in dwork as row j —
       which is exactly the layout BTA_csc_pd_fill_values wants. */
    for (int kk = 0; kk < m0_A; kk++)
    {
        double dk = d[A->row_perm[kk]];
        for (int jj = 0; jj < n0_A; jj++)
        {
            A->dwork[jj * m0_A + kk] = dk * A->X[kk * n0_A + jj];
        }
    }

    BTA_csc_pd_fill_values(B, A->dwork, m0_A, A->row_inv, C);
}

/* Original transpose-via-Cprime implementation of BTDA_csc_pd_fill_values.
   No longer linked; preserved here as in-file reference for the math
   identity C = (A^T diag(d) B)^T and the BA_pd_csc_fill_values delegation. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((unused))
#endif
static void
BTDA_csc_pd_fill_values_via_transpose_dead(const CSC_matrix *B, const double *d,
                                           const permuted_dense *A,
                                           permuted_dense *C)
{
    if (C->base.nnz == 0)
    {
        return;
    }

    /* Cprime has shape (A->n0, |row_active|) — i.e. C transposed. */
    matrix *Cprime_m = BTA_pd_csc_alloc(A, B);
    permuted_dense *Cprime = (permuted_dense *) Cprime_m;
    BTDA_pd_csc_fill_values(A, d, B, Cprime);

    /* C->X = Cprime->X^T. Cprime has dims (C->n0, C->m0). */
    int m0 = C->m0;
    int n0 = C->n0;
    for (int i = 0; i < m0; i++)
    {
        for (int j = 0; j < n0; j++)
        {
            C->X[i * n0 + j] = Cprime->X[j * m0 + i];
        }
    }

    free_matrix(Cprime_m);
}
