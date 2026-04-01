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
#include "utils/Timer.h"
#include "utils/iVec.h"
#include "utils/utils.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

CSC_Matrix *new_csc_matrix(int m, int n, int nnz)
{
    CSC_Matrix *matrix = (CSC_Matrix *) malloc(sizeof(CSC_Matrix));
    if (!matrix) return NULL;

    matrix->p = (int *) malloc((n + 1) * sizeof(int));
    matrix->i = (int *) malloc(nnz * sizeof(int));
    matrix->x = (double *) malloc(nnz * sizeof(double));

    if (!matrix->p || !matrix->i || !matrix->x)
    {
        free(matrix->p);
        free(matrix->i);
        free(matrix->x);
        free(matrix);
        return NULL;
    }

    matrix->m = m;
    matrix->n = n;
    matrix->nnz = nnz;

    return matrix;
}

void free_csc_matrix(CSC_Matrix *matrix)
{
    if (matrix)
    {
        free(matrix->p);
        free(matrix->i);
        free(matrix->x);
        free(matrix);
    }
}

CSR_Matrix *ATA_alloc(const CSC_Matrix *A)
{
    /* A is m x n, A^T A is n x n */
    int n = A->n;
    int m = A->m;
    int nnz = 0;
    int i, j, ii, jj;

    /* row ptr and column idxs for upper triangular part of C = A^T A */
    int *Cp = (int *) malloc((n + 1) * sizeof(int));
    iVec *Ci = iVec_new(m);
    Cp[0] = 0;

    /* compute sparsity pattern, only storing upper triangular part */
    for (i = 0; i < n; i++)
    {
        /* check if Cij != 0 */
        for (j = i; j < n; j++)
        {
            ii = A->p[i];
            jj = A->p[j];

            while (ii < A->p[i + 1] && jj < A->p[j + 1])
            {
                if (A->i[ii] == A->i[jj])
                {
                    nnz += (j != i) ? 2 : 1;
                    iVec_append(Ci, j);
                    break;
                }
                else if (A->i[ii] < A->i[jj])
                {
                    ii++;
                }
                else
                {
                    jj++;
                }
            }
        }
        Cp[i + 1] = Ci->len;
    }

    /* Allocate C and symmetrize it */
    CSR_Matrix *C = new_csr_matrix(n, n, nnz);
    symmetrize_csr(Cp, Ci->data, n, C);

    /* free workspace */
    free(Cp);
    iVec_free(Ci);

    /* fill matching pairs */
    ATA_fill_matching_pairs(A, C);

    return C;
}

static inline double sparse_dot(const double *a_x, const int *a_i, int a_nnz,
                                const double *b_x, const int *b_i, int b_nnz)
{
    int ii = 0;
    int jj = 0;
    double sum = 0.0;

    while (ii < a_nnz && jj < b_nnz)
    {
        if (a_i[ii] == b_i[jj])
        {
            sum += a_x[ii] * b_x[jj];
            ii++;
            jj++;
        }
        else if (a_i[ii] < b_i[jj])
        {
            ii++;
        }
        else
        {
            jj++;
        }
    }

    return sum;
}

static inline double sparse_wdot(const double *a_x, const int *a_i, int a_nnz,
                                 const double *b_x, const int *b_i, int b_nnz,
                                 const double *d)
{
    int ii = 0;
    int jj = 0;
    double sum = 0.0;

    while (ii < a_nnz && jj < b_nnz)
    {
        if (a_i[ii] == b_i[jj])
        {
            sum += a_x[ii] * b_x[jj] * d[a_i[ii]];
            ii++;
            jj++;
        }
        else if (a_i[ii] < b_i[jj])
        {
            ii++;
        }
        else
        {
            jj++;
        }
    }

    return sum;
}

static void ATDA_copy_symmetric(CSR_Matrix *C, int i, int jj, int j, int *cursor)
{
    while (C->i[cursor[j]] != i)
    {
        cursor[j]++;
    }
    C->x[jj] = C->x[cursor[j]];
    cursor[j]++;
}

void ATDA_fill_values(const CSC_Matrix *A, const double *d, CSR_Matrix *C)
{
    int *cursor = (int *) malloc(C->m * sizeof(int));
    memcpy(cursor, C->p, C->m * sizeof(int));

    for (int i = 0; i < C->m; i++)
    {
        for (int jj = C->p[i]; jj < C->p[i + 1]; jj++)
        {
            int j = C->i[jj];

            if (j < i)
            {
                ATDA_copy_symmetric(C, i, jj, j, cursor);
            }
            else
            {
                int nnz_ai = A->p[i + 1] - A->p[i];
                int nnz_aj = A->p[j + 1] - A->p[j];

                if (d != NULL)
                {
                    C->x[jj] =
                        sparse_wdot(A->x + A->p[i], A->i + A->p[i], nnz_ai,
                                    A->x + A->p[j], A->i + A->p[j], nnz_aj, d);
                }
                else
                {
                    C->x[jj] = sparse_dot(A->x + A->p[i], A->i + A->p[i], nnz_ai,
                                          A->x + A->p[j], A->i + A->p[j], nnz_aj);
                }
            }
        }
    }
    free(cursor);
}

void ATA_fill_matching_pairs(const CSC_Matrix *A, CSR_Matrix *C)
{
    iVec *ai_offsets = iVec_new(C->nnz);
    iVec *aj_offsets = iVec_new(C->nnz);
    iVec *row_indices = iVec_new(C->nnz);
    iVec *entry_ptrs = iVec_new(C->nnz / 2 + 1);
    iVec_append(entry_ptrs, 0);

    int i, j, ii, jj, kk;

    for (i = 0; i < C->m; i++)
    {
        for (jj = C->p[i]; jj < C->p[i + 1]; jj++)
        {
            j = C->i[jj];
            if (j < i) continue;

            ii = A->p[i];
            kk = A->p[j];
            while (ii < A->p[i + 1] && kk < A->p[j + 1])
            {
                if (A->i[ii] == A->i[kk])
                {
                    iVec_append(ai_offsets, ii - A->p[i]);
                    iVec_append(aj_offsets, kk - A->p[j]);
                    iVec_append(row_indices, A->i[ii]);
                    ii++;
                    kk++;
                }
                else if (A->i[ii] < A->i[kk])
                {
                    ii++;
                }
                else
                {
                    kk++;
                }
            }
            iVec_append(entry_ptrs, ai_offsets->len);
        }
    }

    MatchPairs *mp = (MatchPairs *) malloc(sizeof(MatchPairs));
    mp->n_entries = entry_ptrs->len - 1;
    mp->n_matches = ai_offsets->len;
    mp->p = entry_ptrs->data;
    mp->ai = ai_offsets->data;
    mp->aj = aj_offsets->data;
    mp->rows = row_indices->data;

    C->match = mp;

    free(entry_ptrs);
    free(ai_offsets);
    free(aj_offsets);
    free(row_indices);
}

void ATDA_fill_values_matching_pairs(const CSC_Matrix *A, const double *d,
                                     CSR_Matrix *C)
{
    MatchPairs *mp = C->match;

    /* todo: get rid of this malloc */
    int *cursor = (int *) malloc(C->m * sizeof(int));
    memcpy(cursor, C->p, C->m * sizeof(int));

    int i, j, jj;
    int e = 0;

    for (i = 0; i < C->m; i++)
    {
        for (jj = C->p[i]; jj < C->p[i + 1]; jj++)
        {
            j = C->i[jj];

            if (j < i)
            {
                /* TODO: should profile this and the memset */
                while (C->i[cursor[j]] != i)
                {
                    cursor[j]++;
                }
                C->x[jj] = C->x[cursor[j]];
                cursor[j]++;
            }
            else
            {
                double sum = 0.0;
                const double *xi = A->x + A->p[i];
                const double *xj = A->x + A->p[j];
                for (int k = mp->p[e]; k < mp->p[e + 1]; k++)
                {
                    sum += xi[mp->ai[k]] * xj[mp->aj[k]] * d[mp->rows[k]];
                }
                C->x[jj] = sum;
                e++;
            }
        }
    }

    free(cursor);
}

CSC_Matrix *csr_to_csc_alloc(const CSR_Matrix *A, int *iwork)
{
    CSC_Matrix *C = new_csc_matrix(A->m, A->n, A->nnz);

    int i, j;
    int *count = iwork;
    memset(count, 0, A->n * sizeof(int));

    // -------------------------------------------------------------------
    //              compute nnz in each column of A
    // -------------------------------------------------------------------
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            count[A->i[j]]++;
        }
    }

    // ------------------------------------------------------------------
    //                      compute column pointers
    // ------------------------------------------------------------------
    C->p[0] = 0;
    for (i = 0; i < A->n; ++i)
    {
        C->p[i + 1] = C->p[i] + count[i];
        count[i] = C->p[i];
    }

    // ------------------------------------------------------------------
    //                         fill row indices
    // ------------------------------------------------------------------
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            C->i[count[A->i[j]]] = i;
            count[A->i[j]]++;
        }
    }

    return C;
}

void csr_to_csc_fill_values(const CSR_Matrix *A, CSC_Matrix *C, int *iwork)
{
    int i, j;
    int *count = iwork;
    memcpy(count, C->p, A->n * sizeof(int));

    // ------------------------------------------------------------------
    //                         fill values
    // ------------------------------------------------------------------
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            C->x[count[A->i[j]]] = A->x[j];
            count[A->i[j]]++;
        }
    }
}

CSR_Matrix *csc_to_csr_alloc(const CSC_Matrix *A, int *iwork)
{
    CSR_Matrix *C = new_csr_matrix(A->m, A->n, A->nnz);

    int i, j;
    int *count = iwork;
    memset(count, 0, A->m * sizeof(int));

    // -------------------------------------------------------------------
    //              compute nnz in each row of A, store in count
    // -------------------------------------------------------------------
    for (i = 0; i < A->n; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            assert(A->i[j] < A->m);
            count[A->i[j]]++;
        }
    }

    // ------------------------------------------------------------------
    //                      compute row pointers
    // ------------------------------------------------------------------
    C->p[0] = 0;
    for (i = 0; i < A->m; ++i)
    {
        C->p[i + 1] = C->p[i] + count[i];
        count[i] = C->p[i];
    }

    // ------------------------------------------------------------------
    //                         fill column indices
    // ------------------------------------------------------------------
    for (i = 0; i < A->n; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            assert(A->i[j] < A->m);
            C->i[count[A->i[j]]] = i;
            count[A->i[j]]++;
        }
    }

    return C;
}

void csc_to_csr_fill_values(const CSC_Matrix *A, CSR_Matrix *C, int *iwork)
{
    int i, j;
    int *count = iwork;
    memcpy(count, C->p, A->m * sizeof(int));

    // ------------------------------------------------------------------
    //                         fill values
    // ------------------------------------------------------------------
    for (i = 0; i < A->n; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            assert(A->i[j] < A->m);
            C->x[count[A->i[j]]] = A->x[j];
            count[A->i[j]]++;
        }
    }
}

CSR_Matrix *BTA_alloc(const CSC_Matrix *A, const CSC_Matrix *B)
{
    /* A is m x n, B is m x p, C = B^T A is p x n */
    int n = A->n;
    int p = B->n;
    int nnz = 0;
    int i, j, ii, jj;

    /* row ptr and column idxs for C = B^T A */
    int *Cp = (int *) malloc((p + 1) * sizeof(int));
    iVec *Ci = iVec_new(n);
    Cp[0] = 0;

    /* compute sparsity pattern */
    for (i = 0; i < p; i++)
    {
        /* check if Cij != 0 for each column j of A */
        for (j = 0; j < n; j++)
        {
            ii = B->p[i];
            jj = A->p[j];

            /* check if row i of B^T (column i of B) has common row with column j of
             * A */
            while (ii < B->p[i + 1] && jj < A->p[j + 1])
            {
                if (B->i[ii] == A->i[jj])
                {
                    nnz++;
                    iVec_append(Ci, j);
                    break;
                }
                else if (B->i[ii] < A->i[jj])
                {
                    ii++;
                }
                else
                {
                    jj++;
                }
            }
        }
        Cp[i + 1] = Ci->len;
    }

    /* Allocate C */
    CSR_Matrix *C = new_csr_matrix(p, n, nnz);
    memcpy(C->p, Cp, (p + 1) * sizeof(int));
    memcpy(C->i, Ci->data, nnz * sizeof(int));

    /* free workspace */
    free(Cp);
    iVec_free(Ci);

    return C;
}

void yTA_fill_values(const CSC_Matrix *A, const double *y, CSR_Matrix *C)
{
    for (int col = 0; col < A->n; col++)
    {
        double val = 0;
        for (int j = A->p[col]; j < A->p[col + 1]; j++)
        {
            val += y[A->i[j]] * A->x[j];
        }

        if (A->p[col + 1] - A->p[col] == 0) continue;

        /* find position in C and fill value */
        for (int k = 0; k < C->nnz; k++)
        {
            if (C->i[k] == col)
            {
                C->x[k] = val;
                break;
            }
        }
    }
}

/* computes C = B^T * D * A in CSR */
void BTDA_fill_values(const CSC_Matrix *A, const CSC_Matrix *B, const double *d,
                      CSR_Matrix *C)
{
    int i, j, jj;
    for (i = 0; i < C->m; i++)
    {
        for (jj = C->p[i]; jj < C->p[i + 1]; jj++)
        {
            j = C->i[jj];

            int nnz_bi = B->p[i + 1] - B->p[i];
            int nnz_aj = A->p[j + 1] - A->p[j];

            if (d != NULL)
            {
                C->x[jj] = sparse_wdot(B->x + B->p[i], B->i + B->p[i], nnz_bi,
                                       A->x + A->p[j], A->i + A->p[j], nnz_aj, d);
            }
            else
            {
                C->x[jj] = sparse_dot(B->x + B->p[i], B->i + B->p[i], nnz_bi,
                                      A->x + A->p[j], A->i + A->p[j], nnz_aj);
            }
        }
    }
}

/* NOTE: an alternative marker-based approach (scatter A_{k,j} * Q[k,:]
 * into column j of C using a marker array for position lookup) may be
 * faster when Q is dense, since it touches each Q entry exactly once.
 * The sparse_dot approach below is simpler but redundantly scans
 * column j of A for each nonzero row of C. */
void BA_fill_values(const CSR_Matrix *Q, const CSC_Matrix *A, CSC_Matrix *C)
{
    /* fill values of C = Q * A, given the sparsity pattern of C. */
    int i, j, ii;

    /* for each column j of C */
    for (j = 0; j < C->n; j++)
    {
        for (ii = C->p[j]; ii < C->p[j + 1]; ii++)
        {
            i = C->i[ii];
            int nnz_q = Q->p[i + 1] - Q->p[i];
            int nnz_a = A->p[j + 1] - A->p[j];

            /* inner product between row i of Q and column j of A */
            C->x[ii] = sparse_dot(Q->x + Q->p[i], Q->i + Q->p[i], nnz_q,
                                  A->x + A->p[j], A->i + A->p[j], nnz_a);
        }
    }
}

CSC_Matrix *symBA_alloc(const CSR_Matrix *B, const CSC_Matrix *A)
{
    /* Allocate C = B * A (sparsity only). B must be symmetric.
     * B is CSR (m x m), A is CSC (m x n), C is CSC (m x n).
     *
     * Column j of C is B * a_j = sum_k A_{k,j} B[:, k], so the nonzero
     * rows of column j of C are the union of the nonzero rows of B[:, k].
     *
     * Since B is symmetric, we can find the nonzero rows of B[:, k] by
     * finding the nonzero columns of B in row k.
     *
     * We use a marker array to avoid duplicates: marker[l] stores the
     * last column j that registered l as nonzero, so checking
     * marker[l] != j avoids duplicates. */

    int m = B->m;
    int n = A->n;
    int i, j, k, jj, ii, ell;

    /* marker[row] = last column j that registered row as nonzero */
    int *marker = (int *) malloc(m * sizeof(int));
    for (i = 0; i < m; i++)
    {
        marker[i] = -1;
    }

    int *Cp = (int *) malloc((n + 1) * sizeof(int));
    iVec *Ci = iVec_new(A->nnz);
    Cp[0] = 0;

    /* for each column j of C */
    for (j = 0; j < n; j++)
    {
        int col_nnz = 0;

        /* iterate over nonzero rows k in column j of A */
        for (ii = A->p[j]; ii < A->p[j + 1]; ii++)
        {
            k = A->i[ii];

            /* find nonzero rows ell of column k of B */
            for (jj = B->p[k]; jj < B->p[k + 1]; jj++)
            {
                ell = B->i[jj];
                if (marker[ell] != j)
                {
                    marker[ell] = j;
                    iVec_append(Ci, ell);
                    col_nnz++;
                }
            }
        }

        Cp[j + 1] = Cp[j] + col_nnz;
    }

    /* allocate C and copy the computed structure */
    int total_nnz = Cp[n];
    CSC_Matrix *C = new_csc_matrix(m, n, total_nnz);
    memcpy(C->p, Cp, (n + 1) * sizeof(int));
    memcpy(C->i, Ci->data, total_nnz * sizeof(int));

    free(marker);
    free(Cp);
    iVec_free(Ci);

    return C;
}

int count_nonzero_cols_csc(const CSC_Matrix *A)
{
    int count = 0;
    for (int j = 0; j < A->n; j++)
    {
        if (A->p[j + 1] > A->p[j])
        {
            count++;
        }
    }
    return count;
}
