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
    permuted_dense_DA_fill_values(d, (const permuted_dense *) self,
                                  (permuted_dense *) out);
}

static matrix *permuted_dense_vtable_ATA_alloc(matrix *self)
{
    return permuted_dense_ATA_alloc((const permuted_dense *) self);
}

static void permuted_dense_vtable_ATDA_fill_values(const matrix *self,
                                                   const double *d, matrix *out)
{
    permuted_dense_ATDA_fill_values((const permuted_dense *) self, d,
                                    (permuted_dense *) out);
}

/* Forward decl; definition lower in the file. */
static CSR_matrix *permuted_dense_to_csr_alloc(const permuted_dense *A);

/* Lazy CSR_matrix view: allocate structure on first call, then return the cache.
   The cache's x array aliases pd->X (see permuted_dense_to_csr_alloc), so
   values are always live without a per-call refresh. */
static struct permuted_dense *permuted_dense_as_permuted_dense(matrix *self)
{
    return (permuted_dense *) self;
}

static CSR_matrix *permuted_dense_to_csr(matrix *self)
{
    permuted_dense *pd = (permuted_dense *) self;
    if (pd->csr_cache == NULL)
    {
        pd->csr_cache = permuted_dense_to_csr_alloc(pd);
    }
    return pd->csr_cache;
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
    pd->base.copy_sparsity = permuted_dense_vtable_copy_sparsity;
    pd->base.DA_fill_values = permuted_dense_vtable_DA_fill_values;
    pd->base.ATA_alloc = permuted_dense_vtable_ATA_alloc;
    pd->base.ATDA_fill_values = permuted_dense_vtable_ATDA_fill_values;
    pd->base.to_csr = permuted_dense_to_csr;
    pd->base.as_permuted_dense = permuted_dense_as_permuted_dense;
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
    /* `dwork` sized for the Y-buffer role (Y = diag(d_perm) X) used by ATDA /
       BTDA_pd_pd. BTA_pd_csr_alloc / BTA_csr_pd_alloc upgrade this to a
       larger gather buffer when their output PD will instead play that role. */
    pd->dwork_size = sz;
    pd->dwork = (double *) SP_MALLOC(pd->dwork_size * sizeof(double));
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

void permuted_dense_DA_fill_values(const double *d, const permuted_dense *A,
                                   permuted_dense *C)
{
    int m0 = A->m0;
    int n0 = A->n0;
    cblas_dcopy(m0 * n0, A->X, 1, C->X, 1);
    for (int ii = 0; ii < m0; ii++)
    {
        cblas_dscal(n0, d[A->row_perm[ii]], C->X + ii * n0, 1);
    }
}

matrix *permuted_dense_ATA_alloc(const permuted_dense *A)
{
    int n = A->base.n;
    /* C = AT @ A has a dense block of size n0 x n0, with row and column index
       sets given by A's col_perm. (This follows from Cij = ai^T aj where
       ai and aj are columns of A. Here, ai and aj always have overlapping entries,
       so Cij != 0 for (i, j) in A->col_perm x A->col_perm) */
    return new_permuted_dense(n, n, A->n0, A->n0, A->col_perm, A->col_perm, NULL);
}

void permuted_dense_ATDA_fill_values(const permuted_dense *A, const double *d,
                                     permuted_dense *C)
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

    /* Pre-allocate C->iwork for idx_A + idx_B in BTA / BTDA_pd_pd slow paths
       (each needs at most max_s = MIN(A->m0, B->m0) ints; we store both
       arrays back-to-back in iwork, hence 2 * max_s). */
    permuted_dense *C_pd = (permuted_dense *) C;
    C_pd->iwork_size = (size_t) 2 * MIN(A->m0, B->m0);
    if (C_pd->iwork_size > 0)
    {
        C_pd->iwork = (int *) SP_MALLOC(C_pd->iwork_size * sizeof(int));
    }
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
    // find intersection of row permutations. We use C->iwork as the storage
    // for idx_A | idx_B (back-to-back) and grow it in place if too small
    // -----------------------------------------------------------------------
    int max_s = MIN(A->m0, B->m0);
    size_t needed = 2 * (size_t) max_s;
    if (C->iwork_size < needed)
    {
        free(C->iwork);
        C->iwork = (int *) SP_MALLOC(needed * sizeof(int));
        C->iwork_size = needed;
    }
    int *idx_A = C->iwork;
    int *idx_B = C->iwork + max_s;
    int s = sorted_intersect_indices(A->row_perm, A->m0, B->row_perm, B->m0, idx_A,
                                     idx_B);
    assert(s > 0);

    // ------------------------------------------------------------------------
    // Gather the matching rows into A->dwork and B->dwork (space is sufficient
    // since A->dwork has at least space for A's full block, and we only need
    // part of it. Same comment applies to B->dwork).
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

    /* d == NULL means plain BT @ A */
    if (d == NULL)
    {
        BTA_pd_pd_fill_values(B, A, C);
        return;
    }

    /* C = BT @ (DA) */
    permuted_dense *DA = (permuted_dense *) A->base.copy_sparsity(&A->base);
    permuted_dense_DA_fill_values(d, A, DA);
    BTA_pd_pd_fill_values(B, DA, C);
    free_matrix(&DA->base);
}

matrix *BTA_csr_pd_alloc(const CSR_matrix *B_csr, const permuted_dense *A)
{
    /* Gather the union of columns appearing in B's rows at positions
       row_perm_A. Bitmap of size B_csr->n for O(nnz) collection. */
    int q = B_csr->n;
    char *seen = (char *) SP_CALLOC(q, sizeof(char));
    int r_B = 0;
    for (int kk = 0; kk < A->m0; kk++)
    {
        int row = A->row_perm[kk];
        for (int e = B_csr->p[row]; e < B_csr->p[row + 1]; e++)
        {
            int i = B_csr->i[e];
            if (!seen[i])
            {
                seen[i] = 1;
                r_B++;
            }
        }
    }

    int *row_active = (int *) SP_MALLOC((r_B > 0 ? r_B : 1) * sizeof(int));
    int idx = 0;
    for (int i = 0; i < q; i++)
    {
        if (seen[i])
        {
            row_active[idx++] = i;
        }
    }

    matrix *C =
        new_permuted_dense(q, A->base.n, r_B, A->n0, row_active, A->col_perm, NULL);
    free(row_active);
    free(seen);

    /* Upgrade `dwork` (currently sized for the Y-role at m0_C * n0_C = r_B *
       A->n0) to fit the gather buffer B_sub_dense used by BTA_csr_pd /
       BTDA_csr_pd_fill_values: shape (A->m0, r_B) row-major. */
    permuted_dense *C_pd = (permuted_dense *) C;
    size_t gather_size = A->m0 * r_B;
    if (gather_size > C_pd->dwork_size)
    {
        free(C_pd->dwork);
        C_pd->dwork_size = gather_size;
        C_pd->dwork = (double *) SP_CALLOC(gather_size, sizeof(double));
    }
    return C;
}

/* BTDA variant of BTA_csr_pd: C->X = B_sub_dense^T diag(d) X_A. Folds d
   into the scatter step. */
void BTDA_csr_pd_fill_values(const CSR_matrix *B_csr, const double *d,
                             const permuted_dense *A, permuted_dense *C)
{
    int m0 = A->m0;
    int dn_A = A->n0;
    int r_B = C->m0;

    if (r_B == 0 || m0 == 0)
    {
        return;
    }

    double *B_sub_dense = C->dwork;
    size_t used = m0 * r_B;
    memset(B_sub_dense, 0, used * sizeof(double));

    for (int kk = 0; kk < m0; kk++)
    {
        int row = A->row_perm[kk];
        double dk = d ? d[row] : 1.0;
        for (int e = B_csr->p[row]; e < B_csr->p[row + 1]; e++)
        {
            int i = B_csr->i[e];
            int ii = C->row_inv[i];
            if (ii >= 0)
            {
                B_sub_dense[kk * r_B + ii] = dk * B_csr->x[e];
            }
        }
    }

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, r_B, dn_A, m0, 1.0,
                B_sub_dense, r_B, A->X, dn_A, 0.0, C->X, dn_A);
}

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
    return C;
}

/* C = B^T diag(d) A = (diag (d) B)^T A */
void BTDA_pd_csc_fill_values(const permuted_dense *B, const double *d,
                             const CSC_matrix *A, permuted_dense *C)
{
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
