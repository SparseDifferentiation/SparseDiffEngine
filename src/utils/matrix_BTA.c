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
#include "utils/matrix_BTA.h"

#include "utils/CSC_matrix.h"
#include "utils/CSR_matrix.h"
#include "utils/permuted_dense.h"
#include "utils/permuted_dense_linalg.h"
#include "utils/sparse_matrix.h"
#include "utils/stacked_pd.h"
#include "utils/stacked_pd_kron_linalg.h"
#include "utils/stacked_pd_linalg.h"
#include "utils/tracked_alloc.h"

/* Effective-type view of a BTA/BTDA dispatcher operand. Exactly one of
   `pd` or `csc` is non-NULL: the operand reduces to either a PD or a CSC.
   `owned_csc` / `owned_iwork` are non-NULL only on the spd path, where we
   materialized a temp CSC from the spd's csr_cache; the caller must
   release them with release_operand. */
typedef struct
{
    // doesn't this have spd?
    permuted_dense *pd;
    CSC_matrix *csc;
    CSC_matrix *owned_csc;
    int *owned_iwork;
} operand_view;

static operand_view resolve_operand(matrix *X)
{
    operand_view v;
    v.pd = NULL;
    v.csc = NULL;
    v.owned_csc = NULL;
    v.owned_iwork = NULL;

    if (X->is_permuted_dense)
    {
        v.pd = (permuted_dense *) X;
    }
    else if (X->is_stacked_pd)
    {
        // memory allocation in every iteration????? what is this?
        /* to_csr refreshes csr_cache values from block X buffers. */
        CSR_matrix *csr = X->to_csr(X);
        v.owned_iwork = (int *) SP_MALLOC(csr->n * sizeof(int));
        v.owned_csc = csr_to_csc_alloc(csr, v.owned_iwork);
        csr_to_csc_fill_values(csr, v.owned_csc, v.owned_iwork);
        v.csc = v.owned_csc;
    }
    else
    {
        sparse_matrix *sm = (sparse_matrix *) X;
        sparse_matrix_ensure_csc_cache(sm);
        v.csc = sm->csc_cache;
    }
    return v;
}

static void release_operand(operand_view *v)
{
    if (v->owned_csc != NULL)
    {
        free_CSC_matrix(v->owned_csc);
    }
    free(v->owned_iwork);
}

matrix *BTA_matrices_alloc(matrix *A, matrix *B)
{
    operand_view va = resolve_operand(A);
    operand_view vb = resolve_operand(B);
    matrix *C;

    if (va.pd != NULL && vb.pd != NULL)
    {
        C = BTA_pd_pd_alloc(vb.pd, va.pd);
    }
    else if (vb.pd != NULL)
    {
        C = BTA_pd_csc_alloc(vb.pd, va.csc);
    }
    else if (va.pd != NULL)
    {
        C = BTA_csc_pd_alloc(vb.csc, va.pd);
    }
    else
    {
        CSR_matrix *C_csr = BTA_alloc(va.csc, vb.csc);
        C = new_sparse_matrix(C_csr);
    }

    release_operand(&va);
    release_operand(&vb);
    return C;
}

void BTDA_matrices_fill_values(matrix *A, const double *d, matrix *B, matrix *C)
{
    operand_view va = resolve_operand(A);
    operand_view vb = resolve_operand(B);

    if (va.pd != NULL && vb.pd != NULL)
    {
        BTDA_pd_pd_fill_values(vb.pd, d, va.pd, (permuted_dense *) C);
    }
    else if (vb.pd != NULL)
    {
        BTDA_pd_csc_fill_values(vb.pd, d, va.csc, (permuted_dense *) C);
    }
    else if (va.pd != NULL)
    {
        BTDA_csc_pd_fill_values(vb.csc, d, va.pd, (permuted_dense *) C);
    }
    else
    {
        BTDA_fill_values(va.csc, vb.csc, d, ((sparse_matrix *) C)->csr);
    }

    release_operand(&va);
    release_operand(&vb);
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

matrix *BTA_spd_matrices_alloc(const stacked_pd *B, matrix *A)
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

void BTDA_spd_matrices_fill_values(const stacked_pd *B, const double *d,
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

matrix *BA_pd_kron_matrices_alloc(const permuted_dense *A, int p, matrix *J)
{
    if (J->is_permuted_dense)
    {
        return BA_pd_kron_pd_alloc(A, p, (const permuted_dense *) J);
    }
    if (J->is_stacked_pd)
    {
        return BA_pd_kron_spd_alloc(A, p, (const stacked_pd *) J);
    }
    /* J is sparse */
    sparse_matrix *sm_J = (sparse_matrix *) J;
    sparse_matrix_ensure_csc_cache(sm_J);
    return BA_pd_kron_csc_alloc(A, p, sm_J->csc_cache);
}

void BA_pd_kron_matrices_fill_values(const permuted_dense *A, int p, const matrix *J,
                                     stacked_pd *C)
{
    if (J->is_permuted_dense)
    {
        BA_pd_kron_pd_fill_values(A, p, (const permuted_dense *) J, C);
        return;
    }
    if (J->is_stacked_pd)
    {
        BA_pd_kron_spd_fill_values(A, p, (const stacked_pd *) J, C);
        return;
    }
    /* J is sparse */
    const sparse_matrix *sm_J = (const sparse_matrix *) J;
    BA_pd_kron_csc_fill_values(A, p, sm_J->csc_cache, C);
}
