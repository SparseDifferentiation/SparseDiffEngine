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
#include "utils/sparse_matrix.h"

matrix *BTA_matrices_alloc(matrix *A, matrix *B)
{
    permuted_dense *pd_A = A->as_permuted_dense(A);
    permuted_dense *pd_B = B->as_permuted_dense(B);

    if (pd_A && pd_B)
    {
        return BTA_pd_pd_alloc(pd_B, pd_A);
    }
    if (pd_B)
    {
        /* A is Sparse, B is PD — CSC kernel (see permuted_dense.{h,c}). */
        sparse_matrix *sm_A = (sparse_matrix *) A;
        A->refresh_csc_values(A);
        return BTA_pd_csc_alloc(pd_B, sm_A->csc_cache);
    }
    if (pd_A)
    {
        /* A is PD, B is Sparse — CSC kernel (see permuted_dense.{h,c}). */
        sparse_matrix *sm_B = (sparse_matrix *) B;
        B->refresh_csc_values(B);
        return BTA_csc_pd_alloc(sm_B->csc_cache, pd_A);
    }

    /* Both Sparse: delegate to CSC_matrix BTA. Caller must ensure caches are fresh.
     */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    sparse_matrix *sm_B = (sparse_matrix *) B;
    A->refresh_csc_values(A);
    B->refresh_csc_values(B);
    CSR_matrix *C_csr = BTA_alloc(sm_A->csc_cache, sm_B->csc_cache);
    return new_sparse_matrix(C_csr);
}

void BTDA_matrices_fill_values(matrix *A, const double *d, matrix *B, matrix *C)
{
    permuted_dense *pd_A = A->as_permuted_dense(A);
    permuted_dense *pd_B = B->as_permuted_dense(B);

    if (pd_A && pd_B)
    {
        BTDA_pd_pd_fill_values(pd_B, d, pd_A, (permuted_dense *) C);
        return;
    }
    if (pd_B)
    {
        sparse_matrix *sm_A = (sparse_matrix *) A;
        A->refresh_csc_values(A);
        BTDA_pd_csc_fill_values(pd_B, d, sm_A->csc_cache, (permuted_dense *) C);
        return;
    }
    if (pd_A)
    {
        sparse_matrix *sm_B = (sparse_matrix *) B;
        B->refresh_csc_values(B);
        BTDA_csc_pd_fill_values(sm_B->csc_cache, d, pd_A, (permuted_dense *) C);
        return;
    }

    /* Both Sparse: delegate to CSC_matrix BTDA. */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    sparse_matrix *sm_B = (sparse_matrix *) B;
    sparse_matrix *sm_C = (sparse_matrix *) C;
    BTDA_fill_values(sm_A->csc_cache, sm_B->csc_cache, d, sm_C->csr);
}
