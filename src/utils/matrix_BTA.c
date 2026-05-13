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
    if (A->is_permuted_dense && B->is_permuted_dense)
    {
        return BTA_pd_pd_alloc((permuted_dense *) B, (permuted_dense *) A);
    }
    if (B->is_permuted_dense)
    {
        sparse_matrix *sm_A = (sparse_matrix *) A;
        return BTA_pd_csc_alloc((permuted_dense *) B, sm_A->csc_cache);
    }
    if (A->is_permuted_dense)
    {
        sparse_matrix *sm_B = (sparse_matrix *) B;
        return BTA_csc_pd_alloc(sm_B->csc_cache, (permuted_dense *) A);
    }

    /* both sparse */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    sparse_matrix *sm_B = (sparse_matrix *) B;
    CSR_matrix *C_csr = BTA_alloc(sm_A->csc_cache, sm_B->csc_cache);
    return new_sparse_matrix(C_csr);
}

void BTDA_matrices_fill_values(matrix *A, const double *d, matrix *B, matrix *C)
{
    if (A->is_permuted_dense && B->is_permuted_dense)
    {
        BTDA_pd_pd_fill_values((permuted_dense *) B, d, (permuted_dense *) A,
                               (permuted_dense *) C);
        return;
    }
    if (B->is_permuted_dense)
    {
        sparse_matrix *sm_A = (sparse_matrix *) A;
        BTDA_pd_csc_fill_values((permuted_dense *) B, d, sm_A->csc_cache,
                                (permuted_dense *) C);
        return;
    }
    if (A->is_permuted_dense)
    {
        sparse_matrix *sm_B = (sparse_matrix *) B;
        BTDA_csc_pd_fill_values(sm_B->csc_cache, d, (permuted_dense *) A,
                                (permuted_dense *) C);
        return;
    }

    /* both sparse */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    sparse_matrix *sm_B = (sparse_matrix *) B;
    sparse_matrix *sm_C = (sparse_matrix *) C;
    BTDA_fill_values(sm_A->csc_cache, sm_B->csc_cache, d, sm_C->csr);
}

matrix *BA_pd_matrices_alloc(const permuted_dense *B, const matrix *A)
{
    if (A->is_permuted_dense)
    {
        return BA_pd_pd_alloc(B, (const permuted_dense *) A);
    }
    /* A is sparse — use the existing BA_pd_csc_* kernels. Ensure the
       csc_cache structure exists at alloc time. */
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
    /* A is sparse — caller must have refreshed sm_A->csc_cache values. */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    BA_pd_csc_fill_values(B->X, B->n0, B->col_inv, sm_A->csc_cache, C);
}
