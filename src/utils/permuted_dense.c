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
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void permuted_dense_free(Matrix *self)
{
    Permuted_Dense *pd = (Permuted_Dense *) self;
    free(pd->row_perm);
    free(pd->col_perm);
    free(pd->Y_scratch);
    free(pd->col_inv);
    free(pd->row_inv);
    /* csr_cache->x aliases pd->X (set in permuted_dense_to_csr_alloc); NULL it
       so free_csr_matrix doesn't double-free the shared buffer. */
    if (pd->csr_cache != NULL)
    {
        pd->csr_cache->x = NULL;
    }
    free_csr_matrix(pd->csr_cache);
    free(pd->X);
    free(pd->gather_X_scratch);
    free(pd);
}

/* Permuted_Dense has no CSC mirror; chain-rule kernels operate on X directly. */
static void permuted_dense_refresh_csc_values(Matrix *self)
{
    (void) self;
}

/* Vtable adapters — each delegates to the existing permuted_dense_* kernel. */
static Matrix *permuted_dense_vtable_copy_sparsity(const Matrix *self)
{
    const Permuted_Dense *pd = (const Permuted_Dense *) self;
    return new_permuted_dense(pd->base.m, pd->base.n, pd->dense_m, pd->dense_n,
                              pd->row_perm, pd->col_perm, NULL);
}

static void permuted_dense_vtable_DA_fill_values(const double *d, const Matrix *self,
                                                 Matrix *out)
{
    permuted_dense_DA_fill_values(d, (const Permuted_Dense *) self,
                                  (Permuted_Dense *) out);
}

static Matrix *permuted_dense_vtable_ATA_alloc(Matrix *self)
{
    return permuted_dense_ATA_alloc((const Permuted_Dense *) self);
}

static void permuted_dense_vtable_ATDA_fill_values(const Matrix *self,
                                                   const double *d, Matrix *out)
{
    permuted_dense_ATDA_fill_values((const Permuted_Dense *) self, d,
                                    (Permuted_Dense *) out);
}

/* Forward decl; definition lower in the file. */
static CSR_Matrix *permuted_dense_to_csr_alloc(const Permuted_Dense *self);

/* Lazy CSR view: allocate structure on first call, then return the cache.
   The cache's x array aliases pd->X (see permuted_dense_to_csr_alloc), so
   values are always live without a per-call refresh. */
static struct Permuted_Dense *permuted_dense_as_permuted_dense(Matrix *self)
{
    return (Permuted_Dense *) self;
}

static CSR_Matrix *permuted_dense_to_csr(Matrix *self)
{
    Permuted_Dense *pd = (Permuted_Dense *) self;
    if (pd->csr_cache == NULL)
    {
        pd->csr_cache = permuted_dense_to_csr_alloc(pd);
    }
    return pd->csr_cache;
}

static Matrix *permuted_dense_vtable_index_alloc(Matrix *self, const int *indices,
                                                 int n_idxs)
{
    const Permuted_Dense *pd = (const Permuted_Dense *) self;

    /* Scan indices: which output positions i hit a row in pd->row_perm? */
    int *new_row_perm = (int *) SP_MALLOC(n_idxs * sizeof(int));
    int new_dense_m = 0;
    for (int i = 0; i < n_idxs; i++)
    {
        if (pd->row_inv[indices[i]] >= 0)
        {
            new_row_perm[new_dense_m++] = i;
        }
    }

    Matrix *out = new_permuted_dense(n_idxs, pd->base.n, new_dense_m,
                                     pd->dense_n, new_row_perm, pd->col_perm,
                                     NULL);
    free(new_row_perm);
    return out;
}

static void permuted_dense_vtable_index_fill_values(Matrix *self,
                                                    const int *indices, int n_idxs,
                                                    Matrix *out)
{
    (void) n_idxs;
    const Permuted_Dense *pd = (const Permuted_Dense *) self;
    Permuted_Dense *out_pd = (Permuted_Dense *) out;
    int dense_n = pd->dense_n;
    for (int k = 0; k < out_pd->dense_m; k++)
    {
        int i = out_pd->row_perm[k];
        int old_ii = pd->row_inv[indices[i]];
        memcpy(out_pd->X + (size_t) k * dense_n, pd->X + (size_t) old_ii * dense_n,
               dense_n * sizeof(double));
    }
}

static Matrix *permuted_dense_vtable_promote_alloc(Matrix *self, int size)
{
    const Permuted_Dense *pd = (const Permuted_Dense *) self;
    assert(pd->dense_m <= 1);

    if (pd->dense_m == 0)
    {
        /* source row is all-zero; output is also structurally all-zero. */
        return new_permuted_dense(size, pd->base.n, 0, pd->dense_n, NULL,
                                  pd->col_perm, NULL);
    }

    int *new_row_perm = (int *) SP_MALLOC(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        new_row_perm[i] = i;
    }
    Matrix *out = new_permuted_dense(size, pd->base.n, size, pd->dense_n,
                                     new_row_perm, pd->col_perm, NULL);
    free(new_row_perm);
    return out;
}

static void permuted_dense_vtable_promote_fill_values(Matrix *self, Matrix *out)
{
    const Permuted_Dense *pd = (const Permuted_Dense *) self;
    Permuted_Dense *out_pd = (Permuted_Dense *) out;
    if (pd->dense_m == 0) return;
    int dense_n = pd->dense_n;
    for (int k = 0; k < out_pd->dense_m; k++)
    {
        memcpy(out_pd->X + (size_t) k * dense_n, pd->X,
               dense_n * sizeof(double));
    }
}

static Matrix *permuted_dense_vtable_broadcast_alloc(Matrix *self,
                                                     broadcast_type type, int d1,
                                                     int d2)
{
    const Permuted_Dense *pd = (const Permuted_Dense *) self;
    int out_m = d1 * d2;

    int new_dense_m;
    if (type == BROADCAST_SCALAR)
    {
        new_dense_m = (pd->dense_m == 0) ? 0 : out_m;
    }
    else if (type == BROADCAST_ROW)
    {
        new_dense_m = d1 * pd->dense_m;
    }
    else /* BROADCAST_COL */
    {
        new_dense_m = d2 * pd->dense_m;
    }

    if (new_dense_m == 0)
    {
        return new_permuted_dense(out_m, pd->base.n, 0, pd->dense_n, NULL,
                                  pd->col_perm, NULL);
    }

    int *new_row_perm = (int *) SP_MALLOC(new_dense_m * sizeof(int));
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
        for (int j_ii = 0; j_ii < pd->dense_m; j_ii++)
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
            for (int ii_old = 0; ii_old < pd->dense_m; ii_old++)
            {
                new_row_perm[k++] = j * d1 + pd->row_perm[ii_old];
            }
        }
    }

    Matrix *out = new_permuted_dense(out_m, pd->base.n, new_dense_m,
                                     pd->dense_n, new_row_perm, pd->col_perm,
                                     NULL);
    free(new_row_perm);
    return out;
}

static void permuted_dense_vtable_broadcast_fill_values(Matrix *self,
                                                        broadcast_type type, int d1,
                                                        int d2, Matrix *out)
{
    const Permuted_Dense *pd = (const Permuted_Dense *) self;
    Permuted_Dense *out_pd = (Permuted_Dense *) out;
    if (pd->dense_m == 0)
    {
        return;
    }
    int dense_n = pd->dense_n;

    if (type == BROADCAST_SCALAR)
    {
        for (int k = 0; k < out_pd->dense_m; k++)
        {
            memcpy(out_pd->X + (size_t) k * dense_n, pd->X,
                   dense_n * sizeof(double));
        }
    }
    else if (type == BROADCAST_ROW)
    {
        /* output row k corresponds to child dense row (k / d1). */
        (void) d2;
        for (int k = 0; k < out_pd->dense_m; k++)
        {
            memcpy(out_pd->X + (size_t) k * dense_n,
                   pd->X + (size_t) (k / d1) * dense_n,
                   dense_n * sizeof(double));
        }
    }
    else /* BROADCAST_COL */
    {
        (void) d1;
        size_t child_block = (size_t) pd->dense_m * (size_t) dense_n;
        for (int j = 0; j < d2; j++)
        {
            memcpy(out_pd->X + (size_t) j * child_block, pd->X,
                   child_block * sizeof(double));
        }
    }
}

static Matrix *permuted_dense_vtable_diag_vec_alloc(Matrix *self)
{
    const Permuted_Dense *pd = (const Permuted_Dense *) self;
    int n = pd->base.m;
    int out_m = n * n;

    if (pd->dense_m == 0)
    {
        return new_permuted_dense(out_m, pd->base.n, 0, pd->dense_n, NULL,
                                  pd->col_perm, NULL);
    }

    int *new_row_perm = (int *) SP_MALLOC(pd->dense_m * sizeof(int));
    for (int ii = 0; ii < pd->dense_m; ii++)
    {
        new_row_perm[ii] = pd->row_perm[ii] * (n + 1);
    }
    Matrix *out = new_permuted_dense(out_m, pd->base.n, pd->dense_m, pd->dense_n,
                                     new_row_perm, pd->col_perm, NULL);
    free(new_row_perm);
    return out;
}

static void permuted_dense_vtable_diag_vec_fill_values(Matrix *self, Matrix *out)
{
    const Permuted_Dense *pd = (const Permuted_Dense *) self;
    Permuted_Dense *out_pd = (Permuted_Dense *) out;
    if (pd->dense_m == 0)
    {
        return;
    }
    memcpy(out_pd->X, pd->X,
           (size_t) pd->dense_m * (size_t) pd->dense_n * sizeof(double));
}

Matrix *new_permuted_dense(int m, int n, int dense_m, int dense_n,
                           const int *row_perm, const int *col_perm,
                           const double *X_data)
{
    /* Validate sorted invariants. */
    for (int ii = 1; ii < dense_m; ii++)
    {
        assert(row_perm[ii] > row_perm[ii - 1]);
    }
    for (int jj = 1; jj < dense_n; jj++)
    {
        assert(col_perm[jj] > col_perm[jj - 1]);
    }
    if (dense_m > 0)
    {
        assert(row_perm[0] >= 0 && row_perm[dense_m - 1] < m);
    }
    if (dense_n > 0)
    {
        assert(col_perm[0] >= 0 && col_perm[dense_n - 1] < n);
    }

    Permuted_Dense *pd = (Permuted_Dense *) SP_CALLOC(1, sizeof(Permuted_Dense));
    pd->base.m = m;
    pd->base.n = n;
    pd->base.nnz = dense_m * dense_n;
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

    pd->dense_m = dense_m;
    pd->dense_n = dense_n;

    int sz = dense_m * dense_n;
    pd->row_perm = (int *) SP_MALLOC(dense_m * sizeof(int));
    pd->col_perm = (int *) SP_MALLOC(dense_n * sizeof(int));
    pd->X = (double *) SP_MALLOC(sz * sizeof(double));
    pd->base.x = pd->X;
    pd->Y_scratch = (double *) SP_MALLOC(sz * sizeof(double));
    pd->col_inv = (int *) SP_MALLOC(n * sizeof(int));
    pd->row_inv = (int *) SP_MALLOC(m * sizeof(int));

    if (dense_m > 0)
    {
        memcpy(pd->row_perm, row_perm, dense_m * sizeof(int));
    }
    if (dense_n > 0)
    {
        memcpy(pd->col_perm, col_perm, dense_n * sizeof(int));
    }

    for (int j = 0; j < n; j++)
    {
        pd->col_inv[j] = -1;
    }
    for (int jj = 0; jj < dense_n; jj++)
    {
        pd->col_inv[col_perm[jj]] = jj;
    }

    for (int i = 0; i < m; i++)
    {
        pd->row_inv[i] = -1;
    }
    for (int ii = 0; ii < dense_m; ii++)
    {
        pd->row_inv[row_perm[ii]] = ii;
    }

    if (X_data != NULL && sz > 0)
    {
        memcpy(pd->X, X_data, sz * sizeof(double));
    }

    return &pd->base;
}

static CSR_Matrix *permuted_dense_to_csr_alloc(const Permuted_Dense *self)
{
    int dense_m = self->dense_m;
    int dense_n = self->dense_n;
    int m = self->base.m;
    CSR_Matrix *C = new_csr_matrix(m, self->base.n, dense_m * dense_n);

    /* Alias C->x to self->X: the dense block layout already matches what the
       CSR view's value array would hold, so values are always live with no
       memcpy needed. The PD owns the buffer; permuted_dense_free nulls
       C->x before free_csr_matrix to avoid double-free. */
    free(C->x);
    C->x = self->X;

    /* fill column indices (each dense row contributes a copy of col_perm) */
    for (int ii = 0; ii < dense_m; ii++)
    {
        memcpy(C->i + ii * dense_n, self->col_perm, dense_n * sizeof(int));
    }

    /* set row pointers via count and then cumulative sum  */
    memset(C->p, 0, (m + 1) * sizeof(int));
    for (int ii = 0; ii < dense_m; ii++)
    {
        C->p[self->row_perm[ii] + 1] = dense_n;
    }

    for (int i = 0; i < m; i++)
    {
        C->p[i + 1] += C->p[i];
    }

    return C;
}

void permuted_dense_DA_fill_values(const double *d, const Permuted_Dense *self,
                                   Permuted_Dense *out)
{
    int dense_m = self->dense_m;
    int dense_n = self->dense_n;
    cblas_dcopy(dense_m * dense_n, self->X, 1, out->X, 1);
    for (int ii = 0; ii < dense_m; ii++)
    {
        cblas_dscal(dense_n, d[self->row_perm[ii]], out->X + ii * dense_n, 1);
    }
}

Matrix *permuted_dense_ATA_alloc(const Permuted_Dense *self)
{
    int n = self->base.n;
    int dense_n = self->dense_n;
    return new_permuted_dense(n, n, dense_n, dense_n, self->col_perm, self->col_perm,
                              NULL);
}

void permuted_dense_ATDA_fill_values(const Permuted_Dense *self, const double *d,
                                     Permuted_Dense *out)
{
    int dense_m = self->dense_m;
    int dense_n = self->dense_n;

    /* Y_scratch = diag(d_perm) X, where d_perm[kk] = d[row_perm[kk]]. */
    cblas_dcopy(dense_m * dense_n, self->X, 1, self->Y_scratch, 1);
    for (int ii = 0; ii < dense_m; ii++)
    {
        cblas_dscal(dense_n, d[self->row_perm[ii]], self->Y_scratch + ii * dense_n,
                    1);
    }

    /* out.X = X^T Y_scratch. */
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dense_n, dense_n, dense_m,
                1.0, self->X, dense_n, self->Y_scratch, dense_n, 0.0, out->X,
                dense_n);
}

Matrix *permuted_dense_BTA_alloc(const Permuted_Dense *A, const Permuted_Dense *B)
{
    return new_permuted_dense(B->base.n, A->base.n, B->dense_n, A->dense_n,
                              B->col_perm, A->col_perm, NULL);
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

void permuted_dense_BTA_fill_values(const Permuted_Dense *A, const Permuted_Dense *B,
                                    Permuted_Dense *out)
{
    int dn_A = A->dense_n;
    int dn_B = B->dense_n;

    /* Fast path: matching row_perms (the common case). One dgemm on the
       full X buffers. */
    if (A->dense_m == B->dense_m &&
        int_arrays_equal(A->row_perm, B->row_perm, A->dense_m))
    {
        int s = A->dense_m;
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dn_B, dn_A, s, 1.0,
                    B->X, dn_B, A->X, dn_A, 0.0, out->X, dn_A);
        return;
    }

    /* General path: intersect row_perm_A and row_perm_B via sorted merge,
       gather the matching rows into contiguous scratch buffers, dgemm. */
    int max_s = A->dense_m < B->dense_m ? A->dense_m : B->dense_m;
    int *idx_A = (int *) SP_MALLOC((size_t) max_s * sizeof(int));
    int *idx_B = (int *) SP_MALLOC((size_t) max_s * sizeof(int));
    int s = 0;
    int ii = 0, jj = 0;
    while (ii < A->dense_m && jj < B->dense_m)
    {
        int ra = A->row_perm[ii];
        int rb = B->row_perm[jj];
        if (ra == rb)
        {
            idx_A[s] = ii;
            idx_B[s] = jj;
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

    if (s == 0)
    {
        memset(out->X, 0,
               (size_t) out->dense_m * (size_t) out->dense_n * sizeof(double));
        free(idx_A);
        free(idx_B);
        return;
    }

    double *XA_sub = (double *) SP_MALLOC((size_t) s * (size_t) dn_A * sizeof(double));
    double *XB_sub = (double *) SP_MALLOC((size_t) s * (size_t) dn_B * sizeof(double));
    for (int k = 0; k < s; k++)
    {
        memcpy(XA_sub + (size_t) k * dn_A, A->X + (size_t) idx_A[k] * dn_A,
               (size_t) dn_A * sizeof(double));
        memcpy(XB_sub + (size_t) k * dn_B, B->X + (size_t) idx_B[k] * dn_B,
               (size_t) dn_B * sizeof(double));
    }

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dn_B, dn_A, s, 1.0,
                XB_sub, dn_B, XA_sub, dn_A, 0.0, out->X, dn_A);

    free(XA_sub);
    free(XB_sub);
    free(idx_A);
    free(idx_B);
}

Matrix *BTA_csr_pd_alloc(const CSR_Matrix *A_csr, const Permuted_Dense *B)
{
    /* Gather the union of columns appearing in A's rows at positions
       row_perm_B. Use a bitmap of size A_csr->n for O(nnz) collection. */
    int p = A_csr->n;
    char *seen = (char *) SP_CALLOC((size_t) p, sizeof(char));
    int s_A = 0;
    for (int kk = 0; kk < B->dense_m; kk++)
    {
        int row = B->row_perm[kk];
        for (int e = A_csr->p[row]; e < A_csr->p[row + 1]; e++)
        {
            int j = A_csr->i[e];
            if (!seen[j])
            {
                seen[j] = 1;
                s_A++;
            }
        }
    }

    int *col_active = (int *) SP_MALLOC((size_t) (s_A > 0 ? s_A : 1) * sizeof(int));
    int idx = 0;
    for (int j = 0; j < p; j++)
    {
        if (seen[j])
        {
            col_active[idx++] = j;
        }
    }

    Matrix *out = new_permuted_dense(B->base.n, p, B->dense_n, s_A,
                                     B->col_perm, col_active, NULL);
    free(col_active);
    free(seen);

    /* Persistent scratch for BTA_csr_pd_fill_values / BTDA_csr_pd_fill_values:
       A_sub_dense (B->dense_m x s_A row-major doubles). Pre-zeroed; each fill
       memsets only the slots it touches by re-zeroing the whole buffer. */
    Permuted_Dense *out_pd = (Permuted_Dense *) out;
    out_pd->gather_X_size = (size_t) B->dense_m * (size_t) s_A;
    if (out_pd->gather_X_size > 0)
    {
        out_pd->gather_X_scratch =
            (double *) SP_CALLOC(out_pd->gather_X_size, sizeof(double));
    }
    return out;
}

/* Note: when A_csr is a leaf-variable Jacobian (each row has a single entry
   at column var_id + k, value 1), A_sub_dense is a permuted identity and
   the dgemm reduces to X_C = X_B^T — a pure transpose with no multiplication
   needed. A fast path can detect this and skip the dgemm; deferred until a
   workload shows the savings matter. */
void BTA_csr_pd_fill_values(const CSR_Matrix *A_csr, const Permuted_Dense *B,
                            Permuted_Dense *out)
{
    int dense_m = B->dense_m;
    int dn_B = B->dense_n;
    int s_A = out->dense_n;

    if (s_A == 0 || dense_m == 0)
    {
        /* Output dense block is empty; nothing to fill. */
        return;
    }

    /* Use out->col_inv (pre-built by new_permuted_dense) as col_inv_out and
       out->gather_X_scratch as A_sub_dense; both are owned by out. */
    double *A_sub_dense = out->gather_X_scratch;
    memset(A_sub_dense, 0, out->gather_X_size * sizeof(double));

    for (int kk = 0; kk < dense_m; kk++)
    {
        int row = B->row_perm[kk];
        for (int e = A_csr->p[row]; e < A_csr->p[row + 1]; e++)
        {
            int j = A_csr->i[e];
            int jj = out->col_inv[j];
            /* jj should always be valid (we built col_perm from these entries),
               but guard against asymmetry between alloc and fill calls. */
            if (jj >= 0)
            {
                A_sub_dense[(size_t) kk * s_A + jj] = A_csr->x[e];
            }
        }
    }

    /* out->X = X_B^T @ A_sub_dense */
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dn_B, s_A, dense_m,
                1.0, B->X, dn_B, A_sub_dense, s_A, 0.0, out->X, s_A);
}

Matrix *BTA_pd_csr_alloc(const Permuted_Dense *A, const CSR_Matrix *B_csr)
{
    /* Gather the union of columns appearing in B's rows at positions
       row_perm_A. Bitmap of size B_csr->n for O(nnz) collection. */
    int q = B_csr->n;
    char *seen = (char *) SP_CALLOC((size_t) q, sizeof(char));
    int r_B = 0;
    for (int kk = 0; kk < A->dense_m; kk++)
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

    int *row_active = (int *) SP_MALLOC((size_t) (r_B > 0 ? r_B : 1) * sizeof(int));
    int idx = 0;
    for (int i = 0; i < q; i++)
    {
        if (seen[i])
        {
            row_active[idx++] = i;
        }
    }

    Matrix *out = new_permuted_dense(q, A->base.n, r_B, A->dense_n,
                                     row_active, A->col_perm, NULL);
    free(row_active);
    free(seen);

    /* Persistent scratch for BTA_pd_csr_fill_values / BTDA_pd_csr_fill_values:
       B_sub_dense (A->dense_m x r_B row-major doubles). */
    Permuted_Dense *out_pd = (Permuted_Dense *) out;
    out_pd->gather_X_size = (size_t) A->dense_m * (size_t) r_B;
    if (out_pd->gather_X_size > 0)
    {
        out_pd->gather_X_scratch =
            (double *) SP_CALLOC(out_pd->gather_X_size, sizeof(double));
    }
    return out;
}

/* Note: when B_csr is a leaf-variable Jacobian (each row has a single entry
   at column var_id + k, value 1), B_sub_dense is an identity matrix and
   the dgemm reduces to X_C = X_A — a pure copy with no multiplication
   needed. A fast path can detect this and skip the dgemm; deferred until a
   workload shows the savings matter. */
void BTA_pd_csr_fill_values(const Permuted_Dense *A, const CSR_Matrix *B_csr,
                            Permuted_Dense *out)
{
    int dense_m = A->dense_m;
    int dn_A = A->dense_n;
    int r_B = out->dense_m;

    if (r_B == 0 || dense_m == 0)
    {
        /* Output dense block is empty; nothing to fill. */
        return;
    }

    /* Use out->row_inv (pre-built by new_permuted_dense) as row_inv_out and
       out->gather_X_scratch as B_sub_dense; both are owned by out. */
    double *B_sub_dense = out->gather_X_scratch;
    memset(B_sub_dense, 0, out->gather_X_size * sizeof(double));

    for (int kk = 0; kk < dense_m; kk++)
    {
        int row = A->row_perm[kk];
        for (int e = B_csr->p[row]; e < B_csr->p[row + 1]; e++)
        {
            int i = B_csr->i[e];
            int ii = out->row_inv[i];
            if (ii >= 0)
            {
                B_sub_dense[(size_t) kk * r_B + ii] = B_csr->x[e];
            }
        }
    }

    /* out->X = B_sub_dense^T @ X_A */
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, r_B, dn_A, dense_m,
                1.0, B_sub_dense, r_B, A->X, dn_A, 0.0, out->X, dn_A);
}

/* BTDA variant of BTA_csr_pd: out->X = X_B^T diag(d) A_sub_dense. Folds d
   into the scatter step. */
void BTDA_csr_pd_fill_values(const CSR_Matrix *A_csr, const double *d,
                             const Permuted_Dense *B, Permuted_Dense *out)
{
    int dense_m = B->dense_m;
    int dn_B = B->dense_n;
    int s_A = out->dense_n;

    if (s_A == 0 || dense_m == 0)
    {
        return;
    }

    double *A_sub_dense = out->gather_X_scratch;
    memset(A_sub_dense, 0, out->gather_X_size * sizeof(double));

    for (int kk = 0; kk < dense_m; kk++)
    {
        int row = B->row_perm[kk];
        double dk = d ? d[row] : 1.0;
        for (int e = A_csr->p[row]; e < A_csr->p[row + 1]; e++)
        {
            int j = A_csr->i[e];
            int jj = out->col_inv[j];
            if (jj >= 0)
            {
                A_sub_dense[(size_t) kk * s_A + jj] = dk * A_csr->x[e];
            }
        }
    }

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dn_B, s_A, dense_m,
                1.0, B->X, dn_B, A_sub_dense, s_A, 0.0, out->X, s_A);
}

/* BTDA variant of BTA_pd_csr: out->X = B_sub_dense^T diag(d) X_A. Folds d
   into the scatter step. */
void BTDA_pd_csr_fill_values(const Permuted_Dense *A, const double *d,
                             const CSR_Matrix *B_csr, Permuted_Dense *out)
{
    int dense_m = A->dense_m;
    int dn_A = A->dense_n;
    int r_B = out->dense_m;

    if (r_B == 0 || dense_m == 0)
    {
        return;
    }

    double *B_sub_dense = out->gather_X_scratch;
    memset(B_sub_dense, 0, out->gather_X_size * sizeof(double));

    for (int kk = 0; kk < dense_m; kk++)
    {
        int row = A->row_perm[kk];
        double dk = d ? d[row] : 1.0;
        for (int e = B_csr->p[row]; e < B_csr->p[row + 1]; e++)
        {
            int i = B_csr->i[e];
            int ii = out->row_inv[i];
            if (ii >= 0)
            {
                B_sub_dense[(size_t) kk * r_B + ii] = dk * B_csr->x[e];
            }
        }
    }

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, r_B, dn_A, dense_m,
                1.0, B_sub_dense, r_B, A->X, dn_A, 0.0, out->X, dn_A);
}

/* BTDA(PD, PD): out->X = X_B^T diag(d) X_A, restricted to row_perm_A ∩
   row_perm_B. When d == NULL, this is just permuted_dense_BTA_fill_values.
   Otherwise we first row-scale A's X into A's Y_scratch by d, then run the
   same intersect-and-gather logic as the BTA case using Y_scratch in place
   of X_A. */
void BTDA_pd_pd_fill_values(const Permuted_Dense *A, const double *d,
                            const Permuted_Dense *B, Permuted_Dense *out)
{
    if (d == NULL)
    {
        permuted_dense_BTA_fill_values(A, B, out);
        return;
    }

    int dn_A = A->dense_n;
    int dn_B = B->dense_n;
    int dense_m_A = A->dense_m;

    /* Build Y = diag(d_perm_A) X_A in A's Y_scratch (mutates only the
       Y_scratch buffer, so const A is preserved in spirit). */
    cblas_dcopy(dense_m_A * dn_A, A->X, 1, A->Y_scratch, 1);
    for (int kk = 0; kk < dense_m_A; kk++)
    {
        cblas_dscal(dn_A, d[A->row_perm[kk]], A->Y_scratch + kk * dn_A, 1);
    }

    /* Fast path: matching row_perms. One dgemm using Y_scratch as A. */
    int match = (A->dense_m == B->dense_m);
    if (match)
    {
        for (int kk = 0; kk < A->dense_m; kk++)
        {
            if (A->row_perm[kk] != B->row_perm[kk])
            {
                match = 0;
                break;
            }
        }
    }
    if (match)
    {
        int s = A->dense_m;
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dn_B, dn_A, s, 1.0,
                    B->X, dn_B, A->Y_scratch, dn_A, 0.0, out->X, dn_A);
        return;
    }

    /* General path: intersect row_perm_A and row_perm_B, gather Y_scratch's
       and B's matched rows, then dgemm. */
    int max_s = A->dense_m < B->dense_m ? A->dense_m : B->dense_m;
    int *idx_A = (int *) SP_MALLOC((size_t) max_s * sizeof(int));
    int *idx_B = (int *) SP_MALLOC((size_t) max_s * sizeof(int));
    int s = 0;
    int ii = 0, jj = 0;
    while (ii < A->dense_m && jj < B->dense_m)
    {
        int ra = A->row_perm[ii];
        int rb = B->row_perm[jj];
        if (ra == rb)
        {
            idx_A[s] = ii;
            idx_B[s] = jj;
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

    if (s == 0)
    {
        memset(out->X, 0,
               (size_t) out->dense_m * (size_t) out->dense_n * sizeof(double));
        free(idx_A);
        free(idx_B);
        return;
    }

    double *YA_sub = (double *) SP_MALLOC((size_t) s * (size_t) dn_A * sizeof(double));
    double *XB_sub = (double *) SP_MALLOC((size_t) s * (size_t) dn_B * sizeof(double));
    for (int k = 0; k < s; k++)
    {
        memcpy(YA_sub + (size_t) k * dn_A, A->Y_scratch + (size_t) idx_A[k] * dn_A,
               (size_t) dn_A * sizeof(double));
        memcpy(XB_sub + (size_t) k * dn_B, B->X + (size_t) idx_B[k] * dn_B,
               (size_t) dn_B * sizeof(double));
    }

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dn_B, dn_A, s, 1.0,
                XB_sub, dn_B, YA_sub, dn_A, 0.0, out->X, dn_A);

    free(YA_sub);
    free(XB_sub);
    free(idx_A);
    free(idx_B);
}

Matrix *permuted_dense_times_csc_alloc(const Permuted_Dense *self,
                                       const CSC_Matrix *J)
{
    /* Active columns: those with at least one structural nonzero in a row
       of col_perm_self. col_inv[r] != -1 iff r is in col_perm_self. */
    iVec *col_perm_out = iVec_new(8);
    for (int j = 0; j < J->n; j++)
    {
        for (int e = J->p[j]; e < J->p[j + 1]; e++)
        {
            if (self->col_inv[J->i[e]] != -1)
            {
                iVec_append(col_perm_out, j);
                break;
            }
        }
    }

    Matrix *M_out =
        new_permuted_dense(self->base.m, J->n, self->dense_m, col_perm_out->len,
                           self->row_perm, col_perm_out->data, NULL);
    iVec_free(col_perm_out);
    return M_out;
}

void permuted_dense_times_csc_fill_values(const Permuted_Dense *self,
                                          const CSC_Matrix *J, Permuted_Dense *out)
{
    int dense_m = self->dense_m;
    int dense_n_self = self->dense_n;
    int dense_n_out = out->dense_n;

    /* Each entry (r, val) of J in active columns with r in col_perm_self
       contributes val * self.X[:, kk] to out.X[:, jj], where kk = col_inv[r]
       and jj is the position of the column in col_perm_out. Columns of X
       and out.X are accessed via row-major strides. */
    memset(out->X, 0, dense_m * dense_n_out * sizeof(double));
    for (int jj = 0; jj < dense_n_out; jj++)
    {
        int col = out->col_perm[jj];
        for (int e = J->p[col]; e < J->p[col + 1]; e++)
        {
            int kk = self->col_inv[J->i[e]];
            if (kk == -1) continue;
            cblas_daxpy(dense_m, J->x[e], self->X + kk, dense_n_self, out->X + jj,
                        dense_n_out);
        }
    }
}
