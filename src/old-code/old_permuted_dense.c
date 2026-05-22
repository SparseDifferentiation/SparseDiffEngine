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
#include "old-code/old_permuted_dense.h"

#include "utils/cblas_wrapper.h"
#include "utils/tracked_alloc.h"
#include <stdlib.h>
#include <string.h>

matrix *BTA_pd_csr_alloc(const permuted_dense *B, const CSR_matrix *A)
{
    /* Cij != 0 only if i is in B's column permutation and column j of A
      overlaps with column i of B. */

    /* Gather the union of columns appearing in A's rows at positions
       row_perm_B. Use a bitmap of size A->n for O(nnz) collection. */
    int p = A->n;
    char *seen = (char *) SP_CALLOC(p, sizeof(char));
    int s_A = 0;
    for (int kk = 0; kk < B->m0; kk++)
    {
        int row = B->row_perm[kk];
        for (int e = A->p[row]; e < A->p[row + 1]; e++)
        {
            int j = A->i[e];
            if (!seen[j])
            {
                seen[j] = 1;
                s_A++;
            }
        }
    }

    int *col_active = (int *) SP_MALLOC((s_A > 0 ? s_A : 1) * sizeof(int));
    int idx = 0;
    for (int j = 0; j < p; j++)
    {
        if (seen[j])
        {
            col_active[idx++] = j;
        }
    }

    matrix *C =
        new_permuted_dense(B->base.n, p, B->n0, s_A, B->col_perm, col_active, NULL);
    free(col_active);
    free(seen);

    /* Upgrade `dwork` (currently sized for the Y-role at m0_C * n0_C = B->n0 *
       s_A) to fit the gather buffer A_sub_dense used by BTA_csr_pd /
       BTDA_pd_csr_fill_values: shape (B->m0, s_A) row-major. The dgemm
       reads it as (B->m0, s_A), so size B->m0 * s_A doubles suffices. */
    permuted_dense *C_pd = (permuted_dense *) C;
    size_t gather_size = B->m0 * s_A;
    if (gather_size > C_pd->kernel_dwork_size)
    {
        free(C_pd->kernel_dwork);
        C_pd->kernel_dwork_size = gather_size;
        C_pd->kernel_dwork = (double *) SP_CALLOC(gather_size, sizeof(double));
    }
    return C;
}

void BTA_pd_csr_fill_values(const permuted_dense *B, const CSR_matrix *A_csr,
                            permuted_dense *C)
{
    int m0 = B->m0;
    int dn_B = B->n0;
    int s_A = C->n0;

    if (s_A == 0 || m0 == 0)
    {
        /* Output dense block is empty; nothing to fill. */
        return;
    }

    /* Use C->col_inv (pre-built by new_permuted_dense) as col_inv_out and
       C->kernel_dwork as A_sub_dense; both are owned by C. dwork is sized at alloc
       time to cover m0 * s_A; only that prefix is touched. */
    double *A_sub_dense = C->kernel_dwork;
    size_t used = m0 * s_A;
    memset(A_sub_dense, 0, used * sizeof(double));

    for (int kk = 0; kk < m0; kk++)
    {
        int row = B->row_perm[kk];
        for (int e = A_csr->p[row]; e < A_csr->p[row + 1]; e++)
        {
            int j = A_csr->i[e];
            int jj = C->col_inv[j];
            /* jj should always be valid (we built col_perm from these entries),
               but guard against asymmetry between alloc and fill calls. */
            if (jj >= 0)
            {
                A_sub_dense[kk * s_A + jj] = A_csr->x[e];
            }
        }
    }

    /* C->X = X_B^T @ A_sub_dense */
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dn_B, s_A, m0, 1.0, B->X,
                dn_B, A_sub_dense, s_A, 0.0, C->X, s_A);
}

/* BTDA variant of BTA_csr_pd: C->X = X_B^T diag(d) A_sub_dense. Folds d
   into the scatter step. */
void BTDA_pd_csr_fill_values(const permuted_dense *B, const double *d,
                             const CSR_matrix *A_csr, permuted_dense *C)
{
    int m0 = B->m0;
    int dn_B = B->n0;
    int s_A = C->n0;

    if (s_A == 0 || m0 == 0)
    {
        return;
    }

    double *A_sub_dense = C->kernel_dwork;
    size_t used = m0 * s_A;
    memset(A_sub_dense, 0, used * sizeof(double));

    for (int kk = 0; kk < m0; kk++)
    {
        int row = B->row_perm[kk];
        double dk = d ? d[row] : 1.0;
        for (int e = A_csr->p[row]; e < A_csr->p[row + 1]; e++)
        {
            int j = A_csr->i[e];
            int jj = C->col_inv[j];
            if (jj >= 0)
            {
                A_sub_dense[kk * s_A + jj] = dk * A_csr->x[e];
            }
        }
    }

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dn_B, s_A, m0, 1.0, B->X,
                dn_B, A_sub_dense, s_A, 0.0, C->X, s_A);
}

/* Legacy CSR-pd kernels (B=CSR, A=PD), formerly in src/utils/permuted_dense.c.
   Production now goes through BTA_csc_pd_alloc / BTDA_csc_pd_fill_values;
   these are kept here for reference + direct unit tests. */

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
    if (gather_size > C_pd->kernel_dwork_size)
    {
        free(C_pd->kernel_dwork);
        C_pd->kernel_dwork_size = gather_size;
        C_pd->kernel_dwork = (double *) SP_CALLOC(gather_size, sizeof(double));
    }
    return C;
}

/* No-d BTA fill for the legacy CSR-pd kernel. */
void BTA_csr_pd_fill_values(const CSR_matrix *B_csr, const permuted_dense *A,
                            permuted_dense *C)
{
    int m0 = A->m0;
    int dn_A = A->n0;
    int r_B = C->m0;

    if (r_B == 0 || m0 == 0)
    {
        /* Output dense block is empty; nothing to fill. */
        return;
    }

    /* Use C->row_inv (pre-built by new_permuted_dense) as row_inv_out and
       C->kernel_dwork as B_sub_dense; both are owned by C. dwork is sized at alloc
       time to cover m0 * r_B; only that prefix is touched. */
    double *B_sub_dense = C->kernel_dwork;
    size_t used = m0 * r_B;
    memset(B_sub_dense, 0, used * sizeof(double));

    for (int kk = 0; kk < m0; kk++)
    {
        int row = A->row_perm[kk];
        for (int e = B_csr->p[row]; e < B_csr->p[row + 1]; e++)
        {
            int i = B_csr->i[e];
            int ii = C->row_inv[i];
            if (ii >= 0)
            {
                B_sub_dense[kk * r_B + ii] = B_csr->x[e];
            }
        }
    }

    /* C->X = B_sub_dense^T @ X_A */
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, r_B, dn_A, m0, 1.0,
                B_sub_dense, r_B, A->X, dn_A, 0.0, C->X, dn_A);
}

/* BTDA variant: C->X = B_sub_dense^T diag(d) X_A. Folds d into the scatter
   step. d may be NULL (treated as identity). */
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

    double *B_sub_dense = C->kernel_dwork;
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
