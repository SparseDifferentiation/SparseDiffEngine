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
#include "bivariate.h"
#include "subexpr.h"
#include "utils/blas_compat.h"
#include "utils/linalg_sparse_matmuls.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DGEMM_BATCH_SIZE 256

/* This file implements 'dense_left_matmul' for the operation y = A @ f(x),
   where A is a DENSE matrix (stored row-major) and f(x) is an arbitrary
   expression. The dimensions are A - m x n, f(x) - n x p, y - m x p.

   This is a performance-optimized variant of left_matmul that avoids
   sparse overhead when A is dense:
   - Forward pass uses cblas_dgemv instead of sparse matvec
   - Jacobian sparsity: if ANY entry in a block column of J_child is nonzero,
     ALL m rows get entries (since A is dense), avoiding per-row overlap checks
   - Jacobian values use cblas_dgemv per block column instead of sparse dot products
   - Weighted sum Hessian uses cblas_dgemv with AT for A^T @ w
*/

static void forward(expr *node, const double *u)
{
    dense_left_matmul_expr *dn = (dense_left_matmul_expr *) node;

    /* child's forward pass */
    node->left->forward(node->left, u);

    /* y = (I_p ⊗ A) @ vec(f(x)), computed as p independent A @ x_block */
    for (int b = 0; b < dn->n_blocks; b++)
    {
        cblas_dgemv(CblasRowMajor, CblasNoTrans, dn->m, dn->n, 1.0, dn->A_dense,
                    dn->n, node->left->value + b * dn->n, 1, 0.0,
                    node->value + b * dn->m, 1);
    }
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    dense_left_matmul_expr *dn = (dense_left_matmul_expr *) node;
    free(dn->A_dense);
    free(dn->AT_dense);
    free_csc_matrix(dn->Jchild_CSC);
    free_csc_matrix(dn->J_CSC);
    free(dn->csc_to_csr_workspace);
    free(dn->J_block_work);
    free(dn->C_block_work);
    free(dn->col_work);
    dn->A_dense = NULL;
    dn->AT_dense = NULL;
    dn->Jchild_CSC = NULL;
    dn->J_CSC = NULL;
    dn->csc_to_csr_workspace = NULL;
    dn->J_block_work = NULL;
    dn->C_block_work = NULL;
    dn->col_work = NULL;
}

/* Compute sparsity pattern of C = (I_p ⊗ A_dense) @ J where A_dense is dense m×n.
   Since A is dense, if ANY entry in a block of J's column is nonzero,
   ALL m rows in the corresponding block of C get entries.
   Two-pass approach: count nnz first, then fill — avoids O(m*p*k) temp buffer. */
static CSC_Matrix *dense_block_left_multiply_fill_sparsity(const CSC_Matrix *J,
                                                           int m, int n, int p)
{
    int k = J->n;
    int *Cp = (int *) malloc((k + 1) * sizeof(int));
    Cp[0] = 0;

    /* Pass 1: count nnz per column */
    for (int j = 0; j < k; j++)
    {
        if (J->p[j] == J->p[j + 1])
        {
            Cp[j + 1] = Cp[j];
            continue;
        }

        int col_nnz = 0;
        int jj = J->p[j];
        for (int block = 0; block < p; block++)
        {
            int block_start = block * n;
            int block_end = block_start + n;

            while (jj < J->p[j + 1] && J->i[jj] < block_start) jj++;
            bool has_entry = (jj < J->p[j + 1] && J->i[jj] < block_end);
            while (jj < J->p[j + 1] && J->i[jj] < block_end) jj++;

            if (has_entry) col_nnz += m;
        }
        Cp[j + 1] = Cp[j] + col_nnz;
    }

    int total_nnz = Cp[k];
    CSC_Matrix *C = new_csc_matrix(m * p, k, total_nnz);
    memcpy(C->p, Cp, (k + 1) * sizeof(int));
    free(Cp);

    /* Pass 2: fill row indices directly into C */
    for (int j = 0; j < k; j++)
    {
        if (C->p[j] == C->p[j + 1]) continue;

        int ci = C->p[j];
        int jj = J->p[j];
        for (int block = 0; block < p; block++)
        {
            int block_start = block * n;
            int block_end = block_start + n;

            while (jj < J->p[j + 1] && J->i[jj] < block_start) jj++;
            bool has_entry = (jj < J->p[j + 1] && J->i[jj] < block_end);
            while (jj < J->p[j + 1] && J->i[jj] < block_end) jj++;

            if (has_entry)
            {
                int row_offset = block * m;
                for (int i = 0; i < m; i++)
                {
                    C->i[ci++] = row_offset + i;
                }
            }
        }
    }

    return C;
}

/* Fill values of C = (I_p ⊗ A_dense) @ J using batched BLAS dgemm.
   For each block, gathers all active columns into a dense matrix and
   uses a single dgemm instead of per-column dgemv calls.
   Columns with a single entry in the block use a fast scaled-column-copy
   path (O(m) vs O(m*n) for dgemm), which is critical when child J is
   an identity or permutation matrix. */
static void dense_block_left_multiply_fill_values(
    const double *A_dense, const CSC_Matrix *J, CSC_Matrix *C, int m, int n, int p,
    double *J_block, double *C_block, int *col_indices, int *col_C_offsets)
{
    int k = J->n;

    for (int block = 0; block < p; block++)
    {
        int block_start = block * n;
        int block_end = block_start + n;
        int row_offset = block * m;

        /* Gather columns that have entries in this block, classify as
           single-entry (fast path) or multi-entry (dgemm path). */
        int n_gemm = 0; /* columns needing dgemm */
        for (int j = 0; j < k; j++)
        {
            if (C->p[j] == C->p[j + 1]) continue;

            /* Find offset of this block in C's column j.
               Each present block contributes exactly m consecutive entries. */
            int ci = C->p[j];
            while (ci < C->p[j + 1] && C->i[ci] < row_offset) ci += m;
            if (ci >= C->p[j + 1] || C->i[ci] != row_offset) continue;

            /* Count entries in this block */
            int count = 0;
            int single_row = -1;
            double single_val = 0.0;
            for (int jj = J->p[j]; jj < J->p[j + 1]; jj++)
            {
                int row = J->i[jj];
                if (row >= block_start && row < block_end)
                {
                    count++;
                    single_row = row - block_start;
                    single_val = J->x[jj];
                }
            }

            if (count == 1)
            {
                /* Fast path: C[:, j] = A[:, single_row] * single_val.
                   A is row-major so column access has stride n. */
                cblas_dcopy(m, A_dense + single_row, n, C->x + ci, 1);
                if (single_val != 1.0)
                {
                    cblas_dscal(m, single_val, C->x + ci, 1);
                }
            }
            else
            {
                /* Queue for dgemm batch */
                col_indices[n_gemm] = j;
                col_C_offsets[n_gemm] = ci;
                n_gemm++;
            }
        }

        /* Process multi-entry columns in batches via dgemm */
        for (int bs = 0; bs < n_gemm; bs += DGEMM_BATCH_SIZE)
        {
            int batch = n_gemm - bs;
            if (batch > DGEMM_BATCH_SIZE) batch = DGEMM_BATCH_SIZE;

            /* Zero J_block (n × batch, column-major) */
            memset(J_block, 0, (size_t) n * batch * sizeof(double));

            /* Scatter J entries into J_block */
            for (int bi = 0; bi < batch; bi++)
            {
                int j = col_indices[bs + bi];
                for (int jj = J->p[j]; jj < J->p[j + 1]; jj++)
                {
                    int row = J->i[jj];
                    if (row >= block_start && row < block_end)
                    {
                        J_block[(row - block_start) + (size_t) bi * n] = J->x[jj];
                    }
                }
            }

            /* C_block(m × batch) = A(m×n) × J_block(n × batch)
               A_dense is row-major m×n = col-major n×m, so CblasTrans gives m×n. */
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, batch, n, 1.0,
                        A_dense, n, J_block, n, 0.0, C_block, m);

            /* Scatter C_block → C->x */
            for (int bi = 0; bi < batch; bi++)
            {
                memcpy(C->x + col_C_offsets[bs + bi], C_block + (size_t) bi * m,
                       m * sizeof(double));
            }
        }
    }
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    dense_left_matmul_expr *dn = (dense_left_matmul_expr *) node;

    /* initialize child's jacobian and convert to CSC */
    x->jacobian_init(x);
    dn->Jchild_CSC = csr_to_csc_fill_sparsity(x->jacobian, node->iwork);

    /* compute sparsity of this node's jacobian in CSC and CSR */
    dn->J_CSC = dense_block_left_multiply_fill_sparsity(dn->Jchild_CSC, dn->m, dn->n,
                                                        dn->n_blocks);
    node->jacobian = csc_to_csr_fill_sparsity(dn->J_CSC, dn->csc_to_csr_workspace);

    /* Allocate dgemm batch workspaces */
    int batch = DGEMM_BATCH_SIZE;
    dn->J_block_work = (double *) malloc((size_t) dn->n * batch * sizeof(double));
    dn->C_block_work = (double *) malloc((size_t) dn->m * batch * sizeof(double));
    dn->col_work = (int *) malloc(2 * node->n_vars * sizeof(int));

    /* For affine children, the Jacobian is constant — compute once and cache. */
    if (x->is_affine(x))
    {
        x->eval_jacobian(x);
        csr_to_csc_fill_values(x->jacobian, dn->Jchild_CSC, node->iwork);
        dense_block_left_multiply_fill_values(
            dn->A_dense, dn->Jchild_CSC, dn->J_CSC, dn->m, dn->n, dn->n_blocks,
            dn->J_block_work, dn->C_block_work, dn->col_work,
            dn->col_work + node->n_vars);
        csc_to_csr_fill_values(dn->J_CSC, node->jacobian, dn->csc_to_csr_workspace);
        dn->affine_cached = true;
    }
}

static void eval_jacobian(expr *node)
{
    dense_left_matmul_expr *dn = (dense_left_matmul_expr *) node;

    /* Affine children have constant Jacobians — already computed in jacobian_init.
     */
    if (dn->affine_cached) return;

    expr *x = node->left;

    /* evaluate child's jacobian and convert to CSC */
    x->eval_jacobian(x);
    csr_to_csc_fill_values(x->jacobian, dn->Jchild_CSC, node->iwork);

    /* compute this node's jacobian using batched dgemm */
    dense_block_left_multiply_fill_values(dn->A_dense, dn->Jchild_CSC, dn->J_CSC,
                                          dn->m, dn->n, dn->n_blocks,
                                          dn->J_block_work, dn->C_block_work,
                                          dn->col_work, dn->col_work + node->n_vars);
    csc_to_csr_fill_values(dn->J_CSC, node->jacobian, dn->csc_to_csr_workspace);
}

static void wsum_hess_init(expr *node)
{
    /* initialize child's hessian */
    expr *x = node->left;
    x->wsum_hess_init(x);

    /* allocate this node's hessian with the same sparsity as child's */
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, x->wsum_hess->nnz);
    memcpy(node->wsum_hess->p, x->wsum_hess->p, (node->n_vars + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, x->wsum_hess->i, x->wsum_hess->nnz * sizeof(int));

    /* node->dwork is already allocated in constructor (size n * n_blocks) */
}

static void eval_wsum_hess(expr *node, const double *w)
{
    dense_left_matmul_expr *dn = (dense_left_matmul_expr *) node;

    /* compute A^T @ w, block-wise: for each block b,
       dwork[b*n .. (b+1)*n-1] = AT @ w[b*m .. (b+1)*m-1] */
    for (int b = 0; b < dn->n_blocks; b++)
    {
        cblas_dgemv(CblasRowMajor, CblasNoTrans, dn->n, dn->m, 1.0, dn->AT_dense,
                    dn->m, w + b * dn->m, 1, 0.0, node->dwork + b * dn->n, 1);
    }

    node->left->eval_wsum_hess(node->left, node->dwork);
    memcpy(node->wsum_hess->x, node->left->wsum_hess->x,
           node->wsum_hess->nnz * sizeof(double));
}

expr *new_dense_left_matmul(expr *child, const double *A_dense, int m, int n)
{
    /* Dimension logic — same as left_matmul */
    int d1, d2, n_blocks;
    if (child->d1 == n)
    {
        d1 = m;
        d2 = child->d2;
        n_blocks = child->d2;
    }
    else if (child->d2 == n && child->d1 == 1)
    {
        d1 = 1;
        d2 = m;
        n_blocks = 1;
    }
    else
    {
        fprintf(stderr, "Error in new_dense_left_matmul: dimension mismatch\n");
        exit(1);
    }

    /* Allocate the type-specific struct */
    dense_left_matmul_expr *dn =
        (dense_left_matmul_expr *) calloc(1, sizeof(dense_left_matmul_expr));
    expr *node = &dn->base;
    init_expr(node, d1, d2, child->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, free_type_data);
    node->left = child;
    expr_retain(child);

    /* Store dense A (row-major m x n) and AT (row-major n x m) */
    dn->m = m;
    dn->n = n;
    dn->n_blocks = n_blocks;

    dn->A_dense = (double *) malloc(m * n * sizeof(double));
    memcpy(dn->A_dense, A_dense, m * n * sizeof(double));

    dn->AT_dense = (double *) malloc(n * m * sizeof(double));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            dn->AT_dense[j * m + i] = A_dense[i * n + j];
        }
    }

    /* Allocate workspaces */
    node->iwork = (int *) malloc(node->n_vars * sizeof(int));
    dn->csc_to_csr_workspace = (int *) malloc(node->size * sizeof(int));

    /* dwork: used for AT @ w in hess eval. Needs n * n_blocks doubles. */
    int dwork_size = n * n_blocks;
    if (dwork_size < n) dwork_size = n;
    node->dwork = (double *) malloc(dwork_size * sizeof(double));

    return node;
}
