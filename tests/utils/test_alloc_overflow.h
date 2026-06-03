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
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TEST_ALLOC_OVERFLOW_H
#define TEST_ALLOC_OVERFLOW_H

#include "minunit.h"
#include "utils/CSR_matrix.h"
#include "utils/matrix.h"
#include "utils/sparse_matrix.h"
#include "utils/utils.h"
#include <limits.h>
#include <stdio.h>
#include <string.h>

/* sat_mul_int returns the exact product when it fits and clamps to INT_MAX on
   overflow (never wraps to a negative or small-positive value). */
const char *test_sat_mul_int_clamps_on_overflow(void)
{
    mu_assert("exact product", sat_mul_int(40000, 40000) == 1600000000);
    mu_assert("zero operand", sat_mul_int(0, 12345) == 0);
    mu_assert("negative operand", sat_mul_int(-1, 12345) == 0);
    /* 250000 * 250000 = 6.25e10 overflows int32; must clamp, not wrap negative. */
    mu_assert("overflow clamps to INT_MAX", sat_mul_int(250000, 250000) == INT_MAX);
    mu_assert("boundary still positive", sat_mul_int(46341, 46341) == INT_MAX);
    return 0;
}

/* sparse_index_alloc sized its allocation as MIN(Jx->nnz, n_idxs * self->n). For a
   large jacobian the dense product n_idxs * self->n overflows int (e.g. a transpose
   of a 250000-variable matrix), wrapping negative so MIN selected it -> calloc
   overflow -> SIGSEGV. This builds an index op whose product overflows and checks
   the result is a valid CSR with the true (subset) nnz. */
const char *test_sparse_index_alloc_no_int_overflow(void)
{
    /* n_idxs * self->n = 25000 * 100000 = 2.5e9 overflows int32. */
    const int n_idxs = 25000;
    const int n_cols = 100000;

    /* identity-like CSR: n_idxs rows, one nonzero per row */
    CSR_matrix *Jx = new_CSR_matrix(n_idxs, n_cols, n_idxs);
    for (int i = 0; i < n_idxs; i++)
    {
        Jx->p[i] = i;
        Jx->i[i] = i;
        Jx->x[i] = 1.0;
    }
    Jx->p[n_idxs] = n_idxs;

    matrix *mat = new_sparse_matrix(Jx);

    int *indices = (int *) malloc(n_idxs * sizeof(int));
    for (int i = 0; i < n_idxs; i++) indices[i] = i;

    /* pre-fix: crashes here in new_CSR_matrix(... negative nnz) */
    matrix *out = mat->index_alloc(mat, indices, n_idxs);
    CSR_matrix *out_csr = out->to_csr(out);

    mu_assert("nnz must equal selected-row total", out_csr->nnz == n_idxs);
    mu_assert("nnz must be non-negative", out_csr->nnz >= 0);

    free(indices);
    free_matrix(out);
    free_matrix(mat);
    return 0;
}

#endif /* TEST_ALLOC_OVERFLOW_H */
