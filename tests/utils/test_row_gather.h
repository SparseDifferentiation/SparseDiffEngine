#ifndef TEST_ROW_GATHER_H
#define TEST_ROW_GATHER_H

#include "minunit.h"
#include "test_helpers.h"
#include "utils/CSR_matrix.h"
#include "utils/permuted_dense.h"
#include "utils/sparse_matrix.h"
#include "utils/stacked_pd.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* row_gather_alloc / row_gather_fill_values across the three matrix kinds.
   The map contract under test: map[i] in [0, A->m) copies source row map[i],
   map[i] == -1 leaves output row i structurally empty, repeats are allowed,
   and the map is bound to the result at alloc time (the caller's copy is
   dead afterwards). */

/* Shared 4x5 CSR source with an empty row 2:
     row 0: (0: 1.0) (3: 2.0)
     row 1: (1: 3.0) (4: 4.0)
     row 2: empty
     row 3: (0: 5.0) (2: 6.0) (4: 7.0) */
static matrix *row_gather_sparse_fixture(void)
{
    CSR_matrix *A = new_CSR_matrix(4, 5, 7);
    int p[5] = {0, 2, 4, 4, 7};
    int i[7] = {0, 3, 1, 4, 0, 2, 4};
    double x[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    memcpy(A->p, p, 5 * sizeof(int));
    memcpy(A->i, i, 7 * sizeof(int));
    memcpy(A->x, x, 7 * sizeof(double));
    return new_sparse_matrix(A);
}

/* 7x5 stacked_pd with three blocks; block 2 is never hit by the maps below so
   the gather must drop it:
     block 0: rows {0, 4}, cols {0, 2}, X = [[1, 2], [3, 4]]
     block 1: rows {2, 3}, cols {1, 2}, X = [[5, 6], [7, 8]]
     block 2: rows {6},    cols {4},    X = [[9]]
   Row 1 and row 5 belong to no block. */
static matrix *row_gather_spd_fixture(void)
{
    int rp0[2] = {0, 4}, cp0[2] = {0, 2};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    int rp1[2] = {2, 3}, cp1[2] = {1, 2};
    double X1[4] = {5.0, 6.0, 7.0, 8.0};
    int rp2[1] = {6}, cp2[1] = {4};
    double X2[1] = {9.0};
    permuted_dense *blocks[3] = {
        (permuted_dense *) new_permuted_dense(7, 5, 2, 2, rp0, cp0, X0),
        (permuted_dense *) new_permuted_dense(7, 5, 2, 2, rp1, cp1, X1),
        (permuted_dense *) new_permuted_dense(7, 5, 1, 1, rp2, cp2, X2)};
    return new_stacked_pd(7, 5, 3, blocks, NULL, NULL);
}

static void row_gather_poison(matrix *M)
{
    for (int k = 0; k < M->nnz; k++) M->x[k] = (double) NAN;
}

/* Deep-copy any matrix's CSR view into a fresh sparse_matrix. */
static matrix *row_gather_sparse_twin(matrix *M)
{
    CSR_matrix *src = M->to_csr(M);
    CSR_matrix *dst = new_CSR_matrix(src->m, src->n, src->nnz);
    memcpy(dst->p, src->p, (size_t) (src->m + 1) * sizeof(int));
    memcpy(dst->i, src->i, (size_t) src->nnz * sizeof(int));
    memcpy(dst->x, src->x, (size_t) src->nnz * sizeof(double));
    dst->nnz = src->nnz;
    return new_sparse_matrix(dst);
}

/* Structure + values equality through the CSR views. */
static int row_gather_same_csr(matrix *X, matrix *Y)
{
    CSR_matrix *a = X->to_csr(X);
    CSR_matrix *b = Y->to_csr(Y);
    if (a->m != b->m || a->n != b->n || a->nnz != b->nnz) return 0;
    return cmp_int_array(a->p, b->p, a->m + 1) &&
           cmp_int_array(a->i, b->i, a->nnz) && cmp_double_array(a->x, b->x, a->nnz);
}

/* map = [3, -1, 0, 3, 2, 1]: a permutation, a repeat (row 3 twice), a -1, and
   the empty source row 2. Output is 6x5 with nnz 3+0+2+3+0+2 = 10. */
const char *test_row_gather_sparse(void)
{
    matrix *A = row_gather_sparse_fixture();
    int map[6] = {3, -1, 0, 3, 2, 1};
    matrix *C = A->row_gather_alloc(A, map, 6);

    int exp_p[7] = {0, 3, 3, 5, 8, 8, 10};
    int exp_i[10] = {0, 2, 4, 0, 3, 0, 2, 4, 1, 4};
    mu_assert("shape", C->m == 6 && C->n == 5);
    mu_assert("sparsity", cmp_sparsity(C, exp_p, exp_i, 6, 10));

    /* the map is bound at alloc time: clobbering the caller's copy is fine */
    for (int k = 0; k < 6; k++) map[k] = -7;

    row_gather_poison(C);
    A->row_gather_fill_values(A, C);
    double exp_x[10] = {5.0, 6.0, 7.0, 1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0};
    mu_assert("values", cmp_values(C, exp_x, 10));

    /* fill reads the source's current values, not a snapshot */
    for (int k = 0; k < A->nnz; k++) A->x[k] *= 10.0;
    for (int k = 0; k < 10; k++) exp_x[k] *= 10.0;
    A->row_gather_fill_values(A, C);
    mu_assert("values after refill", cmp_values(C, exp_x, 10));

    free_matrix(C);
    free_matrix(A);
    return 0;
}

const char *test_row_gather_sparse_all_empty(void)
{
    matrix *A = row_gather_sparse_fixture();
    int map[3] = {-1, -1, -1};
    matrix *C = A->row_gather_alloc(A, map, 3);
    int exp_p[4] = {0, 0, 0, 0};
    mu_assert("empty structure", cmp_sparsity(C, exp_p, NULL, 3, 0));
    A->row_gather_fill_values(A, C); /* nothing to write; must not touch A */
    free_matrix(C);
    free_matrix(A);
    return 0;
}

/* pd 6x5 with rows {1, 3, 4} x cols {0, 2}. The map hits a row outside
   row_perm (0 and 5), repeats row 3, and contains a -1. Result must stay a pd
   and agree entrywise with the same gather on a sparse twin. */
const char *test_row_gather_pd_vs_sparse_twin(void)
{
    int row_perm[3] = {1, 3, 4};
    int col_perm[2] = {0, 2};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    matrix *A = new_permuted_dense(6, 5, 3, 2, row_perm, col_perm, X);
    matrix *A_tw = row_gather_sparse_twin(A);

    int map[7] = {0, 3, -1, 1, 3, 5, 4};
    matrix *C = A->row_gather_alloc(A, map, 7);
    matrix *C_tw = A_tw->row_gather_alloc(A_tw, map, 7);
    mu_assert("kind preserved", C->is_permuted_dense);
    mu_assert("m0", ((permuted_dense *) C)->m0 == 4);
    mu_assert("nnz", C->nnz == 8);

    row_gather_poison(C);
    A->row_gather_fill_values(A, C);
    A_tw->row_gather_fill_values(A_tw, C_tw);
    matrix_values_changed(C);
    mu_assert("twin match", row_gather_same_csr(C, C_tw));

    /* order-independent mutation of both sources, then refill */
    for (int k = 0; k < A->nnz; k++) A->x[k] *= 2.0;
    for (int k = 0; k < A_tw->nnz; k++) A_tw->x[k] *= 2.0;
    matrix_values_changed(A);
    matrix_values_changed(A_tw);
    A->row_gather_fill_values(A, C);
    A_tw->row_gather_fill_values(A_tw, C_tw);
    matrix_values_changed(C);
    mu_assert("twin match after refill", row_gather_same_csr(C, C_tw));

    free_matrix(C_tw);
    free_matrix(C);
    free_matrix(A_tw);
    free_matrix(A);
    return 0;
}

/* spd fixture above; map = [4, 2, -1, 4, 0, 3, 1, 2]:
     pos 0, 3   -> row 4 (block 0, twice)
     pos 4      -> row 0 (block 0)
     pos 1, 7   -> row 2 (block 1, twice)
     pos 5      -> row 3 (block 1)
     pos 2      -> -1
     pos 6      -> row 1 (no block)
   Block 2 gets no hits and must be dropped; output blocks stay row-disjoint. */
const char *test_row_gather_spd_vs_sparse_twin(void)
{
    matrix *A = row_gather_spd_fixture();
    matrix *A_tw = row_gather_sparse_twin(A);

    int map[8] = {4, 2, -1, 4, 0, 3, 1, 2};
    matrix *C = A->row_gather_alloc(A, map, 8);
    matrix *C_tw = A_tw->row_gather_alloc(A_tw, map, 8);
    mu_assert("kind preserved", C->is_stacked_pd);
    stacked_pd *C_spd = (stacked_pd *) C;
    mu_assert("empty block dropped", C_spd->n_blocks == 2);
    mu_assert("src_block_idx",
              C_spd->src_block_idx[0] == 0 && C_spd->src_block_idx[1] == 1);
    int exp_rp0[3] = {0, 3, 4};
    int exp_rp1[3] = {1, 5, 7};
    mu_assert("block 0 row_perm",
              C_spd->blocks[0]->m0 == 3 &&
                  cmp_int_array(C_spd->blocks[0]->row_perm, exp_rp0, 3));
    mu_assert("block 1 row_perm",
              C_spd->blocks[1]->m0 == 3 &&
                  cmp_int_array(C_spd->blocks[1]->row_perm, exp_rp1, 3));

    row_gather_poison(C);
    A->row_gather_fill_values(A, C);
    A_tw->row_gather_fill_values(A_tw, C_tw);
    matrix_values_changed(C);
    mu_assert("twin match", row_gather_same_csr(C, C_tw));

    for (int k = 0; k < A->nnz; k++) A->x[k] *= 2.0;
    for (int k = 0; k < A_tw->nnz; k++) A_tw->x[k] *= 2.0;
    matrix_values_changed(A);
    matrix_values_changed(A_tw);
    A->row_gather_fill_values(A, C);
    A_tw->row_gather_fill_values(A_tw, C_tw);
    matrix_values_changed(C);
    mu_assert("twin match after refill", row_gather_same_csr(C, C_tw));

    free_matrix(C_tw);
    free_matrix(C);
    free_matrix(A_tw);
    free_matrix(A);
    return 0;
}

#ifdef SP_TRACK_MEMORY
typedef struct
{
    matrix *A;
    matrix *C;
} row_gather_fill_args;

static void run_row_gather_fill(const void *ctx)
{
    const row_gather_fill_args *a = (const row_gather_fill_args *) ctx;
    a->A->row_gather_fill_values(a->A, a->C);
}

/* The spd fill dispatches through src_block_idx into per-block pd fills; none
   of it may allocate once the alloc phase is done. */
const char *test_row_gather_spd_fill_no_transient_alloc(void)
{
    matrix *A = row_gather_spd_fixture();
    int map[8] = {4, 2, -1, 4, 0, 3, 1, 2};
    matrix *C = A->row_gather_alloc(A, map, 8);
    row_gather_fill_args args = {A, C};
    run_row_gather_fill(&args); /* warm-up */
    mu_assert("spd row_gather fill must not allocate",
              fill_is_alloc_free(run_row_gather_fill, &args));
    free_matrix(C);
    free_matrix(A);
    return 0;
}
#endif

#endif /* TEST_ROW_GATHER_H */
