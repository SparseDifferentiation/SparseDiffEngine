#ifndef TEST_ROW_REDUCE_H
#define TEST_ROW_REDUCE_H

#include "minunit.h"
#include "test_helpers.h"
#include "utils/CSR_matrix.h"
#include "utils/permuted_dense.h"
#include "utils/sparse_matrix.h"
#include "utils/stacked_pd.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* row_reduce_alloc / row_reduce_fill_values across the three matrix kinds:
   C[j, :] = sum of rows i of A with group[i] == j. The reduction map is bound to
   the result at alloc time; the fill zeroes C and accumulates from A's current
   values. */

/* 4x5 CSR source with an empty row 2:
     row 0: (0: 1.0) (3: 2.0)
     row 1: (1: 3.0) (4: 4.0)
     row 2: empty
     row 3: (0: 5.0) (2: 6.0) (4: 7.0) */
static matrix *row_reduce_sparse_fixture(void)
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

/* 4x4 stacked_pd whose blocks share column 2:
     block 0: rows {0, 1}, cols {0, 2}, X = [[1, 2], [3, 4]]
     block 1: rows {2, 3}, cols {1, 2}, X = [[5, 6], [7, 8]] */
static matrix *row_reduce_spd_fixture(void)
{
    int rp0[2] = {0, 1}, cp0[2] = {0, 2};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    int rp1[2] = {2, 3}, cp1[2] = {1, 2};
    double X1[4] = {5.0, 6.0, 7.0, 8.0};
    permuted_dense *blocks[2] = {
        (permuted_dense *) new_permuted_dense(4, 4, 2, 2, rp0, cp0, X0),
        (permuted_dense *) new_permuted_dense(4, 4, 2, 2, rp1, cp1, X1)};
    return new_stacked_pd(4, 4, 2, blocks, NULL, NULL);
}

static void row_reduce_poison(matrix *M)
{
    for (int k = 0; k < M->nnz; k++) M->x[k] = (double) NAN;
}

/* Deep-copy any matrix's CSR view into a fresh sparse_matrix. */
static matrix *row_reduce_sparse_twin(matrix *M)
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
static int row_reduce_same_csr(matrix *X, matrix *Y)
{
    CSR_matrix *a = X->to_csr(X);
    CSR_matrix *b = Y->to_csr(Y);
    if (a->m != b->m || a->n != b->n || a->nnz != b->nnz) return 0;
    return cmp_int_array(a->p, b->p, a->m + 1) &&
           cmp_int_array(a->i, b->i, a->nnz) && cmp_double_array(a->x, b->x, a->nnz);
}

/* Reduce A and its sparse twin with the same group, compare through to_csr
   after a poisoned fill and again after scaling both sources. Returns the
   failing assertion message or NULL. */
static const char *row_reduce_check_against_twin(matrix *A, const int *group,
                                                 int m_out, matrix **C_out)
{
    matrix *A_tw = row_reduce_sparse_twin(A);
    matrix *C = A->row_reduce_alloc(A, group, m_out);
    matrix *C_tw = A_tw->row_reduce_alloc(A_tw, group, m_out);

    row_reduce_poison(C);
    A->row_reduce_fill_values(A, C);
    A_tw->row_reduce_fill_values(A_tw, C_tw);
    matrix_values_changed(C);
    int ok = row_reduce_same_csr(C, C_tw);

    for (int k = 0; k < A->nnz; k++) A->x[k] *= 2.0;
    for (int k = 0; k < A_tw->nnz; k++) A_tw->x[k] *= 2.0;
    matrix_values_changed(A);
    A->row_reduce_fill_values(A, C);
    A_tw->row_reduce_fill_values(A_tw, C_tw);
    matrix_values_changed(C);
    int ok_refill = row_reduce_same_csr(C, C_tw);

    free_matrix(C_tw);
    free_matrix(A_tw);
    *C_out = C;
    if (!ok) return "twin match";
    if (!ok_refill) return "twin match after refill";
    return NULL;
}

/* group = [1, 0, 1, 0], m_out = 3: output row 0 = rows 1 + 3 (columns merge
   on 4), output row 1 = rows 0 + 2 (row 2 is empty), output row 2 has no
   members. */
const char *test_row_reduce_sparse(void)
{
    matrix *A = row_reduce_sparse_fixture();
    int group[4] = {1, 0, 1, 0};
    matrix *C = A->row_reduce_alloc(A, group, 3);

    int exp_p[4] = {0, 4, 6, 6};
    int exp_i[6] = {0, 1, 2, 4, 0, 3};
    mu_assert("shape", C->m == 3 && C->n == 5);
    mu_assert("sparsity", cmp_sparsity(C, exp_p, exp_i, 3, 6));

    /* the group is bound at alloc time: clobbering the caller's copy is fine */
    for (int k = 0; k < 4; k++) group[k] = -7;

    /* poisoned output proves the fill zeroes before accumulating */
    row_reduce_poison(C);
    A->row_reduce_fill_values(A, C);
    double exp_x[6] = {5.0, 3.0, 6.0, 11.0, 1.0, 2.0};
    mu_assert("values", cmp_values(C, exp_x, 6));

    for (int k = 0; k < A->nnz; k++) A->x[k] *= 10.0;
    for (int k = 0; k < 6; k++) exp_x[k] *= 10.0;
    A->row_reduce_fill_values(A, C);
    mu_assert("values after refill", cmp_values(C, exp_x, 6));

    free_matrix(C);
    free_matrix(A);
    return 0;
}

/* (6, 4) pd with dense rows {0, 3, 4} x cols {1, 3}; group = [0, 0, 1, 0, 1, 0]
   sends dense rows 0 and 3 to output 0 and dense row 4 to output 1
   (non-monotone in the row index, two rows collapse). */
const char *test_row_reduce_pd(void)
{
    int row_perm[3] = {0, 3, 4};
    int col_perm[2] = {1, 3};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    matrix *A = new_permuted_dense(6, 4, 3, 2, row_perm, col_perm, X);

    int group[6] = {0, 0, 1, 0, 1, 0};
    matrix *C;
    const char *msg = row_reduce_check_against_twin(A, group, 2, &C);
    mu_assert(msg ? msg : "", msg == NULL);

    mu_assert("kind preserved", C->is_permuted_dense);
    permuted_dense *pd = (permuted_dense *) C;
    mu_assert("shape", C->m == 2 && C->n == 4 && pd->m0 == 2 && pd->n0 == 2);
    int exp_row_perm[2] = {0, 1};
    int exp_col_perm[2] = {1, 3};
    int exp_bound[3] = {0, 0, 1};
    mu_assert("row_perm", cmp_int_array(pd->row_perm, exp_row_perm, 2));
    mu_assert("col_perm", cmp_int_array(pd->col_perm, exp_col_perm, 2));
    mu_assert("bound_iwork", cmp_int_array(pd->bound_iwork, exp_bound, 3));
    /* sources were scaled by 2 inside the twin check */
    double exp_X[4] = {8.0, 12.0, 10.0, 12.0};
    mu_assert("X", cmp_double_array(pd->X, exp_X, 4));

    free_matrix(C);
    free_matrix(A);
    return 0;
}

/* group = [0, 1, 0, 1]: every output row draws one row from each block, so
   the blocks' shared column 2 must be summed across blocks. One output block
   with the column union {0, 1, 2}. */
const char *test_row_reduce_spd_cross_block_accumulate(void)
{
    matrix *A = row_reduce_spd_fixture();
    int group[4] = {0, 1, 0, 1};
    matrix *C;
    const char *msg = row_reduce_check_against_twin(A, group, 2, &C);
    mu_assert(msg ? msg : "", msg == NULL);

    mu_assert("kind preserved", C->is_stacked_pd);
    stacked_pd *spd = (stacked_pd *) C;
    mu_assert("one output block", spd->n_blocks == 1);
    int exp_row_perm[2] = {0, 1};
    int exp_col_perm[3] = {0, 1, 2};
    mu_assert("row_perm", cmp_int_array(spd->blocks[0]->row_perm, exp_row_perm, 2));
    mu_assert("col_perm", cmp_int_array(spd->blocks[0]->col_perm, exp_col_perm, 3));
    /* sources were scaled by 2: row 0 = [1, 5, 2 + 6], row 1 = [3, 7, 4 + 8] */
    double exp_X[6] = {2.0, 10.0, 16.0, 6.0, 14.0, 24.0};
    mu_assert("X", cmp_double_array(spd->blocks[0]->X, exp_X, 6));

    free_matrix(C);
    free_matrix(A);
    return 0;
}

/* group = [0, 0, 1, 1]: each output row is fed by a single block, so the
   result keeps two blocks with the source column footprints. */
const char *test_row_reduce_spd_within_block(void)
{
    matrix *A = row_reduce_spd_fixture();
    int group[4] = {0, 0, 1, 1};
    matrix *C;
    const char *msg = row_reduce_check_against_twin(A, group, 2, &C);
    mu_assert(msg ? msg : "", msg == NULL);

    mu_assert("kind preserved", C->is_stacked_pd);
    stacked_pd *spd = (stacked_pd *) C;
    mu_assert("two output blocks", spd->n_blocks == 2);
    int exp_cp0[2] = {0, 2};
    int exp_cp1[2] = {1, 2};
    mu_assert("block 0", spd->blocks[0]->m0 == 1 &&
                             spd->blocks[0]->row_perm[0] == 0 &&
                             cmp_int_array(spd->blocks[0]->col_perm, exp_cp0, 2));
    mu_assert("block 1", spd->blocks[1]->m0 == 1 &&
                             spd->blocks[1]->row_perm[0] == 1 &&
                             cmp_int_array(spd->blocks[1]->col_perm, exp_cp1, 2));
    double exp_X0[2] = {8.0, 12.0};  /* 2 * ([1, 2] + [3, 4]) */
    double exp_X1[2] = {24.0, 28.0}; /* 2 * ([5, 6] + [7, 8]) */
    mu_assert("X0", cmp_double_array(spd->blocks[0]->X, exp_X0, 2));
    mu_assert("X1", cmp_double_array(spd->blocks[1]->X, exp_X1, 2));

    free_matrix(C);
    free_matrix(A);
    return 0;
}

/* All rows into one: a single 1-row block over the column union. */
const char *test_row_reduce_spd_all_to_one(void)
{
    matrix *A = row_reduce_spd_fixture();
    int group[4] = {0, 0, 0, 0};
    matrix *C;
    const char *msg = row_reduce_check_against_twin(A, group, 1, &C);
    mu_assert(msg ? msg : "", msg == NULL);

    mu_assert("kind preserved", C->is_stacked_pd);
    stacked_pd *spd = (stacked_pd *) C;
    mu_assert("one output block", spd->n_blocks == 1 && spd->blocks[0]->m0 == 1);
    int exp_col_perm[3] = {0, 1, 2};
    mu_assert("col_perm", cmp_int_array(spd->blocks[0]->col_perm, exp_col_perm, 3));
    double exp_X[3] = {8.0, 24.0, 40.0}; /* 2 * [1 + 3, 5 + 7, 2 + 4 + 6 + 8] */
    mu_assert("X", cmp_double_array(spd->blocks[0]->X, exp_X, 3));

    free_matrix(C);
    free_matrix(A);
    return 0;
}

#ifdef SP_TRACK_MEMORY
typedef struct
{
    matrix *A;
    matrix *C;
} row_reduce_fill_args;

static void run_row_reduce_fill(const void *ctx)
{
    const row_reduce_fill_args *a = (const row_reduce_fill_args *) ctx;
    a->A->row_reduce_fill_values(a->A, a->C);
}

/* The spd fill reduces per block into the raw spd, then coalesce-accumulates;
   none of it may allocate once the alloc phase is done. */
const char *test_row_reduce_spd_fill_no_transient_alloc(void)
{
    matrix *A = row_reduce_spd_fixture();
    int group[4] = {0, 1, 0, 1};
    matrix *C = A->row_reduce_alloc(A, group, 2);
    row_reduce_fill_args args = {A, C};
    run_row_reduce_fill(&args); /* warm-up */
    mu_assert("spd row_reduce fill must not allocate",
              fill_is_alloc_free(run_row_reduce_fill, &args));
    free_matrix(C);
    free_matrix(A);
    return 0;
}
#endif

#endif /* TEST_ROW_REDUCE_H */
