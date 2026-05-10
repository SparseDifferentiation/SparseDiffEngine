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
    free(pd->X);
    free(pd->Y_scratch);
    free(pd->col_inv);
    free(pd);
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
    pd->base.free_fn = permuted_dense_free;
    /* Other vtable slots are wired up in later steps as the operations
       they dispatch to are implemented. */

    pd->dense_m = dense_m;
    pd->dense_n = dense_n;

    int sz = dense_m * dense_n;
    pd->row_perm = (int *) SP_MALLOC(dense_m * sizeof(int));
    pd->col_perm = (int *) SP_MALLOC(dense_n * sizeof(int));
    pd->X = (double *) SP_MALLOC(sz * sizeof(double));
    pd->Y_scratch = (double *) SP_MALLOC(sz * sizeof(double));
    pd->col_inv = (int *) SP_MALLOC(n * sizeof(int));

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

    if (X_data != NULL && sz > 0)
    {
        memcpy(pd->X, X_data, sz * sizeof(double));
    }

    return &pd->base;
}

CSR_Matrix *permuted_dense_to_csr_alloc(const Permuted_Dense *self)
{
    int dense_m = self->dense_m;
    int dense_n = self->dense_n;
    int m = self->base.m;
    CSR_Matrix *C = new_csr_matrix(m, self->base.n, dense_m * dense_n);

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

void permuted_dense_to_csr_fill_values(const Permuted_Dense *self, CSR_Matrix *out)
{
    memcpy(out->x, self->X, self->dense_m * self->dense_n * sizeof(double));
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
