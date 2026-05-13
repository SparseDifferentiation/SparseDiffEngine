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
