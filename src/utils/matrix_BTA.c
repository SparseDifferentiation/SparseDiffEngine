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

#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"
#include "utils/permuted_dense.h"

Matrix *BTA_matrices_alloc(Matrix *A, Matrix *B)
{
    Permuted_Dense *pd_A = A->as_permuted_dense(A);
    Permuted_Dense *pd_B = B->as_permuted_dense(B);

    if (pd_A && pd_B)
    {
        return permuted_dense_BTA_alloc(pd_A, pd_B);
    }
    if (pd_B)
    {
        /* A is Sparse, B is PD */
        CSR_Matrix *A_csr = A->to_csr(A);
        return BTA_csr_pd_alloc(A_csr, pd_B);
    }
    if (pd_A)
    {
        /* A is PD, B is Sparse */
        CSR_Matrix *B_csr = B->to_csr(B);
        return BTA_pd_csr_alloc(pd_A, B_csr);
    }

    /* Both Sparse: delegate to CSC BTA. Caller must ensure caches are fresh. */
    Sparse_Matrix *sm_A = (Sparse_Matrix *) A;
    Sparse_Matrix *sm_B = (Sparse_Matrix *) B;
    A->refresh_csc_values(A);
    B->refresh_csc_values(B);
    CSR_Matrix *C_csr = BTA_alloc(sm_A->csc_cache, sm_B->csc_cache);
    return new_sparse_matrix(C_csr);
}

void BTDA_matrices_fill_values(Matrix *A, const double *d, Matrix *B, Matrix *C)
{
    Permuted_Dense *pd_A = A->as_permuted_dense(A);
    Permuted_Dense *pd_B = B->as_permuted_dense(B);

    if (pd_A && pd_B)
    {
        BTDA_pd_pd_fill_values(pd_A, d, pd_B, (Permuted_Dense *) C);
        return;
    }
    if (pd_B)
    {
        CSR_Matrix *A_csr = A->to_csr(A);
        BTDA_csr_pd_fill_values(A_csr, d, pd_B, (Permuted_Dense *) C);
        return;
    }
    if (pd_A)
    {
        CSR_Matrix *B_csr = B->to_csr(B);
        BTDA_pd_csr_fill_values(pd_A, d, B_csr, (Permuted_Dense *) C);
        return;
    }

    /* Both Sparse: delegate to CSC BTDA. */
    Sparse_Matrix *sm_A = (Sparse_Matrix *) A;
    Sparse_Matrix *sm_B = (Sparse_Matrix *) B;
    Sparse_Matrix *sm_C = (Sparse_Matrix *) C;
    BTDA_fill_values(sm_A->csc_cache, sm_B->csc_cache, d, sm_C->csr);
}
