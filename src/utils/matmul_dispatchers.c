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
 */
#include "utils/matmul_dispatchers.h"

#include "utils/CSC_matrix.h"
#include "utils/CSR_matrix.h"
#include "utils/permuted_dense.h"
#include "utils/permuted_dense_linalg.h"
#include "utils/sparse_matrix.h"
#include "utils/stacked_pd.h"
#include "utils/stacked_pd_kron_linalg.h"
#include "utils/stacked_pd_linalg.h"
#include <assert.h>

/* Forward declarations of the fixed-B dispatchers used internally by
   BTA_matrices_alloc / BTDA_matrices_fill_values. These were public API
   in earlier revisions; demoted to static now that nothing outside this
   file calls them in production. The definitions follow below. */
static matrix *BTA_pd_matrices_alloc(const permuted_dense *B, matrix *A);
static void BTDA_pd_matrices_fill_values(const permuted_dense *B, const double *d,
                                         const matrix *A, permuted_dense *C);
static matrix *BTA_spd_matrices_alloc(const stacked_pd *B, matrix *A);
static void BTDA_spd_matrices_fill_values(const stacked_pd *B, const double *d,
                                          const matrix *A, stacked_pd *C);
static matrix *BTA_sparse_matrices_alloc(const sparse_matrix *B, matrix *A);
static void BTDA_sparse_matrices_fill_values(const sparse_matrix *B, const double *d,
                                             const matrix *A, matrix *C);

/* Thin 3-branch dispatch on B's type. Each branch delegates to a fixed-B
   dispatcher (declared above), which handles A's branching internally
   and routes to the appropriate spd-aware kernel. */
matrix *BTA_matrices_alloc(matrix *A, matrix *B)
{
    if (B->is_permuted_dense)
    {
        return BTA_pd_matrices_alloc((const permuted_dense *) B, A);
    }
    if (B->is_stacked_pd)
    {
        return BTA_spd_matrices_alloc((const stacked_pd *) B, A);
    }
    /* B is sparse_matrix */
    return BTA_sparse_matrices_alloc((const sparse_matrix *) B, A);
}

void BTDA_matrices_fill_values(matrix *A, const double *d, matrix *B, matrix *C)
{
    if (B->is_permuted_dense)
    {
        BTDA_pd_matrices_fill_values((const permuted_dense *) B, d, A,
                                     (permuted_dense *) C);
        return;
    }
    if (B->is_stacked_pd)
    {
        BTDA_spd_matrices_fill_values((const stacked_pd *) B, d, A,
                                      (stacked_pd *) C);
        return;
    }
    /* B is sparse_matrix */
    BTDA_sparse_matrices_fill_values((const sparse_matrix *) B, d, A, C);
}

matrix *BA_pd_matrices_alloc(const permuted_dense *B, matrix *A)
{
    if (A->is_permuted_dense)
    {
        return BA_pd_pd_alloc(B, (const permuted_dense *) A);
    }
    if (A->is_stacked_pd)
    {
        return BA_pd_spd_alloc(B, (const stacked_pd *) A);
    }

    /* A is sparse */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    sparse_matrix_ensure_csc_cache(sm_A);
    return BA_pd_csc_alloc(B, sm_A->csc_cache);
}

void BA_pd_matrices_fill_values(const permuted_dense *B, const matrix *A,
                                permuted_dense *C)
{
    if (A->is_permuted_dense)
    {
        BA_pd_pd_fill_values(B, (const permuted_dense *) A, C);
        return;
    }
    if (A->is_stacked_pd)
    {
        BA_pd_spd_fill_values(B, (const stacked_pd *) A, C);
        return;
    }

    /* A is sparse */
    const sparse_matrix *sm_A = (const sparse_matrix *) A;
    BA_pd_csc_fill_values(B->X, B->n0, B->col_inv, sm_A->csc_cache, C);
}

static matrix *BTA_pd_matrices_alloc(const permuted_dense *B, matrix *A)
{
    if (A->is_permuted_dense)
    {
        return BTA_pd_pd_alloc(B, (const permuted_dense *) A);
    }
    if (A->is_stacked_pd)
    {
        return BTA_pd_spd_alloc(B, (const stacked_pd *) A);
    }

    /* A is sparse */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    sparse_matrix_ensure_csc_cache(sm_A);
    return BTA_pd_csc_alloc(B, sm_A->csc_cache);
}

static void BTDA_pd_matrices_fill_values(const permuted_dense *B, const double *d,
                                         const matrix *A, permuted_dense *C)
{
    if (A->is_permuted_dense)
    {
        BTDA_pd_pd_fill_values(B, d, (const permuted_dense *) A, C);
        return;
    }
    if (A->is_stacked_pd)
    {
        BTDA_pd_spd_fill_values(B, d, (const stacked_pd *) A, C);
        return;
    }

    /* A is sparse */
    const sparse_matrix *sm_A = (const sparse_matrix *) A;
    BTDA_pd_csc_fill_values(B, d, sm_A->csc_cache, C);
}

static matrix *BTA_spd_matrices_alloc(const stacked_pd *B, matrix *A)
{
    if (A->is_permuted_dense)
    {
        return BTA_spd_pd_alloc(B, (const permuted_dense *) A);
    }
    if (A->is_stacked_pd)
    {
        return BTA_spd_spd_alloc(B, (const stacked_pd *) A);
    }

    /* A is sparse */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    sparse_matrix_ensure_csc_cache(sm_A);
    return BTA_spd_csc_alloc(B, sm_A->csc_cache);
}

static void BTDA_spd_matrices_fill_values(const stacked_pd *B, const double *d,
                                          const matrix *A, stacked_pd *C)
{
    if (A->is_permuted_dense)
    {
        BTDA_spd_pd_fill_values(B, d, (const permuted_dense *) A, C);
        return;
    }
    if (A->is_stacked_pd)
    {
        BTDA_spd_spd_fill_values(B, d, (const stacked_pd *) A, C);
        return;
    }

    /* A is sparse */
    const sparse_matrix *sm_A = (const sparse_matrix *) A;
    BTDA_spd_csc_fill_values(B, d, sm_A->csc_cache, C);
}

static matrix *BTA_sparse_matrices_alloc(const sparse_matrix *B, matrix *A)
{
    /* Ensure B's csc_cache structure exists. */
    sparse_matrix_ensure_csc_cache((sparse_matrix *) B);

    if (A->is_permuted_dense)
    {
        return BTA_csc_pd_alloc(B->csc_cache, (const permuted_dense *) A);
    }
    if (A->is_stacked_pd)
    {
        return BTA_csc_spd_alloc(B->csc_cache, (const stacked_pd *) A);
    }

    /* A is sparse */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    sparse_matrix_ensure_csc_cache(sm_A);
    /* BTA_alloc takes (A_csc, B_csc) — computes B^T A and returns CSR. */
    CSR_matrix *C_csr = BTA_alloc(sm_A->csc_cache, B->csc_cache);
    return new_sparse_matrix(C_csr);
}

static void BTDA_sparse_matrices_fill_values(const sparse_matrix *B, const double *d,
                                             const matrix *A, matrix *C)
{
    if (A->is_permuted_dense)
    {
        BTDA_csc_pd_fill_values(B->csc_cache, d, (const permuted_dense *) A,
                                (permuted_dense *) C);
        return;
    }
    if (A->is_stacked_pd)
    {
        BTDA_csc_spd_fill_values(B->csc_cache, d, (const stacked_pd *) A,
                                 (stacked_pd *) C);
        return;
    }

    /* A is sparse */
    const sparse_matrix *sm_A = (const sparse_matrix *) A;
    BTDA_fill_values(sm_A->csc_cache, B->csc_cache, d, ((sparse_matrix *) C)->csr);
}

/* Debug-only check that A is the "full" permuted_dense shape the kron
   helpers require: m0 == base.m, n0 == base.n, identity inner perms.
   Called once at alloc; not repeated at fill (caller would have to
   reuse the same A). */
static void assert_dense_kron_A_is_full(const permuted_dense *A)
{
    assert(A->m0 == A->base.m);
    assert(A->n0 == A->base.n);
    for (int i = 0; i < A->m0; i++) assert(A->row_perm[i] == i);
    for (int j = 0; j < A->n0; j++) assert(A->col_perm[j] == j);
}

matrix *BA_dense_kron_matrices_alloc(const permuted_dense *A, int p, matrix *J)
{
    assert_dense_kron_A_is_full(A);

    if (J->is_permuted_dense)
    {
        return BA_dense_kron_pd_alloc(A, p, (const permuted_dense *) J);
    }
    if (J->is_stacked_pd)
    {
        return BA_dense_kron_spd_alloc(A, p, (const stacked_pd *) J);
    }
    /* J is sparse */
    sparse_matrix *sm_J = (sparse_matrix *) J;
    sparse_matrix_ensure_csc_cache(sm_J);
    return BA_dense_kron_csc_alloc(A, p, sm_J->csc_cache);
}

void BA_dense_kron_matrices_fill_values(const permuted_dense *A, int p,
                                        const matrix *J, stacked_pd *C)
{
    if (J->is_permuted_dense)
    {
        BA_dense_kron_pd_fill_values(A, p, (const permuted_dense *) J, C);
        return;
    }
    if (J->is_stacked_pd)
    {
        BA_dense_kron_spd_fill_values(A, p, (const stacked_pd *) J, C);
        return;
    }
    /* J is sparse */
    const sparse_matrix *sm_J = (const sparse_matrix *) J;
    BA_dense_kron_csc_fill_values(A, p, sm_J->csc_cache, C);
}
