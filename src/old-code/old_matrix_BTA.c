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
#include "old-code/old_matrix_BTA.h"

#include "old-code/old_stacked_pd_linalg.h"
#include "utils/permuted_dense.h"
#include "utils/sparse_matrix.h"
#include "utils/stacked_pd.h"

matrix *BA_spd_matrices_alloc(const stacked_pd *B, matrix *A)
{
    if (A->is_stacked_pd)
    {
        return BA_spd_spd_alloc(B, (const stacked_pd *) A);
    }

    if (A->is_permuted_dense)
    {
        return BA_spd_pd_alloc(B, (const permuted_dense *) A);
    }

    /* A is sparse */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    sparse_matrix_ensure_csc_cache(sm_A);
    return BA_spd_csc_alloc(B, sm_A->csc_cache);
}

void BA_spd_matrices_fill_values(const stacked_pd *B, const matrix *A, stacked_pd *C)
{
    if (A->is_stacked_pd)
    {
        BA_spd_spd_fill_values(B, (const stacked_pd *) A, C);
        return;
    }
    if (A->is_permuted_dense)
    {
        BA_spd_pd_fill_values(B, (const permuted_dense *) A, C);
        return;
    }
    /* A is sparse */
    const sparse_matrix *sm_A = (const sparse_matrix *) A;
    BA_spd_csc_fill_values(B, sm_A->csc_cache, C);
}
