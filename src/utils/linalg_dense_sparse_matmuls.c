/*
 * Copyright 2026 Daniel Cederberg and William Zhang
 *
 * This file is part of the DNLP-differentiation-engine project.
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
#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"
#include "utils/cblas_wrapper.h"
#include "utils/dense_matrix.h"
#include "utils/iVec.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------
 * C = (I_p kron A) @ J  via the polymorphic Matrix interface.
 * A is dense m x n, J is (n*p) x k CSC, C is (m*p) x k CSC.
 * --------------------------------------------------------------- */
CSC_Matrix *I_kron_A_alloc(const Matrix *A, const CSC_Matrix *J, int p)
{
    int m = A->m;
    int n = A->n;
    int i, j, jj, block, block_start, block_end, block_jj_start, row_offset;

    int *Cp = (int *) malloc((J->n + 1) * sizeof(int));
    iVec *Ci = iVec_new(J->n * m);
    Cp[0] = 0;

    /* for each column of J */
    for (j = 0; j < J->n; j++)
    {
        /* if empty we continue */
        if (J->p[j] == J->p[j + 1])
        {
            Cp[j + 1] = Cp[j];
            continue;
        }

        /* process each of p blocks of rows in this column of J */
        jj = J->p[j];
        for (block = 0; block < p; block++)
        {
            block_start = block * n;
            block_end = block_start + n;
            while (jj < J->p[j + 1] && J->i[jj] < block_start)
            {
                jj++;
            }

            block_jj_start = jj;
            while (jj < J->p[j + 1] && J->i[jj] < block_end)
            {
                jj++;
            }

            /* if no entries in this block, continue */
            if (jj == block_jj_start)
            {
                continue;
            }

            /* dense A: all m rows contribute */
            row_offset = block * m;
            for (i = 0; i < m; i++)
            {
                iVec_append(Ci, row_offset + i);
            }
        }
        Cp[j + 1] = Ci->len;
    }

    CSC_Matrix *C = new_csc_matrix(m * p, J->n, Ci->len, NULL);
    memcpy(C->p, Cp, (J->n + 1) * sizeof(int));
    memcpy(C->i, Ci->data, Ci->len * sizeof(int));
    free(Cp);
    iVec_free(Ci);

    return C;
}

void I_kron_A_fill_values(const Matrix *A, const CSC_Matrix *J, CSC_Matrix *C)
{
    const Dense_Matrix *dm = (const Dense_Matrix *) A;
    int m = dm->base.m;
    int n = dm->base.n;
    int k = J->n;

    int i, j, s, block, block_start, block_end, start, end;

    double *j_dense = dm->work;

    /* for each column of J (and C) */
    for (j = 0; j < k; j++)
    {
        for (i = C->p[j]; i < C->p[j + 1]; i += m)
        {
            block = C->i[i] / m;
            block_start = block * n;
            block_end = block_start + n;

            start = J->p[j];
            end = J->p[j + 1];

            while (start < J->p[j + 1] && J->i[start] < block_start)
            {
                start++;
            }

            while (end > start && J->i[end - 1] >= block_end)
            {
                end--;
            }

            int count = end - start;

            if (count == 1)
            {
                /* Fast path: C column segment = val * A[:, row_in_block] */
                int row_in_block = J->i[start] - block_start;
                double val = J->x[start];
                cblas_dcopy(m, dm->x + row_in_block, n, C->x + i, 1);
                if (val != 1.0)
                {
                    cblas_dscal(m, val, C->x + i, 1);
                }
            }
            else
            {
                /* scatter sparse J col into dense vector and then
                 * compute A @ j_dense */
                memset(j_dense, 0, n * sizeof(double));
                for (s = start; s < end; s++)
                {
                    j_dense[J->i[s] - block_start] = J->x[s];
                }

                cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, dm->x, n,
                            j_dense, 1, 0.0, C->x + i, 1);
            }
        }
    }
}

/* ---------------------------------------------------------------
 * C = (Y^T kron I_m) @ J
 * Y is k x n (col-major), J is (m*k) x p CSC, C is (m*n) x p CSR
 * --------------------------------------------------------------- */
CSR_Matrix *YT_kron_I_alloc(int m, int k, int n, const CSC_Matrix *J, size_t *mem)
{
    (void) k;
    /* C has n blocks of m rows.  All rows at the same position within
       their block (same blk_row) share the same column sparsity.

       Step 1: for each blk_row, find which columns of J contribute.
               Column j contributes iff J[:,j] has any nonzero r
               with r % m == blk_row.
       Step 2: replicate each blk_row's pattern across all n blocks. */

    int i, j, ii, blk_row, total_nnz;

    // ---------------------------------------------------------------
    //           build sparsity pattern per blk_row
    // ---------------------------------------------------------------
    iVec **pattern = (iVec **) malloc(m * sizeof(iVec *));
    total_nnz = 0;
    for (blk_row = 0; blk_row < m; blk_row++)
    {
        pattern[blk_row] = iVec_new(J->n);

        /* check each column of J */
        for (j = 0; j < J->n; j++)
        {
            for (ii = J->p[j]; ii < J->p[j + 1]; ii++)
            {
                if (J->i[ii] % m == blk_row)
                {
                    iVec_append(pattern[blk_row], j);
                    break;
                }
            }
        }
        total_nnz += pattern[blk_row]->len * n;
    }

    // ---------------------------------------------------------------
    //           replicate sparsity pattern across blocks
    // ---------------------------------------------------------------
    CSR_Matrix *C = new_csr_matrix(m * n, J->n, total_nnz, mem);
    int idx = 0;
    for (i = 0; i < m * n; i++)
    {
        blk_row = i % m;
        C->p[i] = idx;
        int len = pattern[blk_row]->len;
        memcpy(C->i + idx, pattern[blk_row]->data, len * sizeof(int));
        idx += len;
    }
    C->p[m * n] = idx;
    assert(idx == total_nnz);

    for (blk_row = 0; blk_row < m; blk_row++)
    {
        iVec_free(pattern[blk_row]);
    }
    free(pattern);
    return C;
}

void YT_kron_I_fill_values(int m, int k, int n, const double *Y, const CSC_Matrix *J,
                           CSR_Matrix *C)
{
    (void) n;
    assert(C->m == m * n);
    /* C[i, j] = sum_l Y[l, blk] * J[blk_row + l*m, j]
     * where blk_row = i % m, blk = i / m */

    int i, j, ii, jj, blk, blk_row;
    double sum;

    /* for each row i of C */
    for (i = 0; i < C->m; i++)
    {
        blk = i / m;                       /* which block of C */
        blk_row = i % m;                   /* which row within block */
        const double *Y_col = Y + blk * k; /* column blk of Y */

        /* for each column j in row i of C */
        for (jj = C->p[i]; jj < C->p[i + 1]; jj++)
        {
            j = C->i[jj];
            sum = 0.0;

            /* matching J nonzeros in this column */
            for (ii = J->p[j]; ii < J->p[j + 1]; ii++)
            {
                if (J->i[ii] % m == blk_row)
                {
                    sum += Y_col[J->i[ii] / m] * J->x[ii];
                }
            }
            C->x[jj] = sum;
        }
    }
}

CSR_Matrix *I_kron_X_alloc(int m, int k, int n, const CSC_Matrix *J, size_t *mem)
{
    /* Step 1: for each block, find which columns of J have any
     *         nonzero in row range [blk*k, blk*k + k). */
    int i, j, ii, blk;

    iVec **pattern = (iVec **) malloc(n * sizeof(iVec *));
    int total_nnz = 0;
    for (blk = 0; blk < n; blk++)
    {
        int blk_start = blk * k;
        int blk_end = blk_start + k;
        pattern[blk] = iVec_new(J->n);

        /* check each column of J */
        for (j = 0; j < J->n; j++)
        {
            for (ii = J->p[j]; ii < J->p[j + 1]; ii++)
            {
                if (J->i[ii] >= blk_start && J->i[ii] < blk_end)
                {
                    iVec_append(pattern[blk], j);
                    break;
                }
            }
        }
        total_nnz += (int) pattern[blk]->len * m;
    }

    /* Step 2: replicate each block's pattern for all m rows
     *         within that block. */
    CSR_Matrix *C = new_csr_matrix(m * n, J->n, total_nnz, mem);
    int idx = 0;
    for (i = 0; i < m * n; i++)
    {
        blk = i / m;
        C->p[i] = idx;
        int len = (int) pattern[blk]->len;
        memcpy(C->i + idx, pattern[blk]->data, len * sizeof(int));
        idx += len;
    }
    C->p[m * n] = idx;
    assert(idx == total_nnz);

    for (blk = 0; blk < n; blk++)
    {
        iVec_free(pattern[blk]);
    }
    free(pattern);
    return C;
}

void I_kron_X_fill_values(int m, int k, int n, const double *X, const CSC_Matrix *J,
                          CSR_Matrix *C)
{
    (void) n;
    assert(C->m == m * n);
    /* C[i, j] = sum_l X[blk_row + l*m] * J[blk*k + l, j]
     * where blk = i / m, blk_row = i % m */

    int i, j, ii, jj, blk, blk_row;
    double sum;

    /* for each row i of C */
    for (i = 0; i < C->m; i++)
    {
        blk = i / m;     /* which block of C */
        blk_row = i % m; /* which row within block */
        int blk_start = blk * k;
        int blk_end = blk_start + k;

        /* for each column j in row i of C */
        for (jj = C->p[i]; jj < C->p[i + 1]; jj++)
        {
            j = C->i[jj];
            sum = 0.0;

            /* J nonzeros in column j within this block's row range */
            for (ii = J->p[j]; ii < J->p[j + 1]; ii++)
            {
                int r = J->i[ii];
                if (r >= blk_start && r < blk_end)
                {
                    int l = r - blk_start;
                    sum += X[blk_row + l * m] * J->x[ii];
                }
            }
            C->x[jj] = sum;
        }
    }
}
