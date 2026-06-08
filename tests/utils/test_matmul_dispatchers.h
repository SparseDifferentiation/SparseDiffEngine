#ifndef TEST_MATMUL_DISPATCHERS_H
#define TEST_MATMUL_DISPATCHERS_H

#include "minunit.h"
#include "old-code/old_permuted_dense.h"
#include "test_helpers.h"
#include "utils/CSC_matrix.h"
#include "utils/CSR_matrix.h"
#include "utils/matmul_dispatchers.h"
#include "utils/permuted_dense.h"
#include "utils/permuted_dense_linalg.h"
#include "utils/sparse_matrix.h"
#include "utils/stacked_pd.h"
#include "utils/stacked_pd_linalg.h"
#include "utils/utils.h"
#include <stdlib.h>
#include <string.h>

/* Wrapper dispatch sanity: (PD, PD). Compare against direct
   BTDA_pd_pd_fill_values. */
const char *test_BTDA_matrices_pd_pd(void)
{
    int row_perm[2] = {0, 1};
    int col_perm_A[2] = {0, 2};
    int col_perm_B[2] = {1, 3};
    double XA[4] = {1.0, 2.0, 3.0, 4.0};
    double XB[4] = {5.0, 6.0, 7.0, 8.0};
    double d[2] = {2.0, -1.5};

    matrix *A_m = new_permuted_dense(2, 4, 2, 2, row_perm, col_perm_A, XA);
    matrix *B_m = new_permuted_dense(2, 4, 2, 2, row_perm, col_perm_B, XB);

    /* Wrapper path. */
    matrix *C_m = BTA_matrices_alloc(A_m, B_m);
    BTDA_matrices_fill_values(A_m, d, B_m, C_m);

    /* Direct primitive path on independent operands. */
    matrix *A2 = new_permuted_dense(2, 4, 2, 2, row_perm, col_perm_A, XA);
    matrix *B2 = new_permuted_dense(2, 4, 2, 2, row_perm, col_perm_B, XB);
    matrix *C2 = BTA_pd_pd_alloc((permuted_dense *) B2, (permuted_dense *) A2);
    BTDA_pd_pd_fill_values((permuted_dense *) B2, d, (permuted_dense *) A2,
                           (permuted_dense *) C2);

    mu_assert("values", cmp_double_array(C_m->x, C2->x, C_m->nnz));

    free_matrix(C_m);
    free_matrix(B_m);
    free_matrix(A_m);
    free_matrix(C2);
    free_matrix(B2);
    free_matrix(A2);
    return 0;
}

/* Wrapper dispatch sanity: (CSR_matrix, PD). Compare against direct
   BTDA_pd_csr_fill_values. */
const char *test_BTDA_matrices_csr_pd(void)
{
    /* A: 4x5 CSR_matrix */
    CSR_matrix *A = new_CSR_matrix(4, 5, 5);
    A->p[0] = 0;
    A->p[1] = 2;
    A->p[2] = 3;
    A->p[3] = 4;
    A->p[4] = 5;
    int Ai[5] = {0, 3, 2, 1, 4};
    double Ax[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    memcpy(A->i, Ai, sizeof Ai);
    memcpy(A->x, Ax, sizeof Ax);
    matrix *A_m = new_sparse_matrix(A);

    /* B: 4x4 PD, row_perm = [1, 3], col_perm = [0, 2]. */
    int row_perm_B[2] = {1, 3};
    int col_perm_B[2] = {0, 2};
    double XB[4] = {10.0, 20.0, 30.0, 40.0};
    matrix *B_m = new_permuted_dense(4, 4, 2, 2, row_perm_B, col_perm_B, XB);

    double d[4] = {1.0, -2.0, 0.5, 3.0};

    /* Wrapper path. Dispatchers don't touch sparse_matrix internals — caller
       owns csc_cache structure and values. */
    sparse_matrix_ensure_csc_cache((sparse_matrix *) A_m);
    matrix *C_m = BTA_matrices_alloc(A_m, B_m);
    A_m->refresh_csc_values(A_m);
    BTDA_matrices_fill_values(A_m, d, B_m, C_m);

    /* Direct primitive path. */
    CSR_matrix *A2 = new_CSR_matrix(4, 5, 5);
    A2->p[0] = 0;
    A2->p[1] = 2;
    A2->p[2] = 3;
    A2->p[3] = 4;
    A2->p[4] = 5;
    memcpy(A2->i, Ai, sizeof Ai);
    memcpy(A2->x, Ax, sizeof Ax);
    matrix *B2_m = new_permuted_dense(4, 4, 2, 2, row_perm_B, col_perm_B, XB);
    permuted_dense *B2 = (permuted_dense *) B2_m;
    matrix *C2 = BTA_pd_csr_alloc(B2, A2);
    BTDA_pd_csr_fill_values(B2, d, A2, (permuted_dense *) C2);

    mu_assert("values", cmp_double_array(C_m->x, C2->x, C_m->nnz));

    free_matrix(C_m);
    free_matrix(B_m);
    free_matrix(A_m);
    free_matrix(C2);
    free_matrix(B2_m);
    free_CSR_matrix(A2);
    return 0;
}

/* Wrapper dispatch sanity: (PD, CSR_matrix). Compare against direct
   BTDA_csc_pd_fill_values. */
const char *test_BTDA_matrices_pd_csr(void)
{
    /* A: 4x5 PD, row_perm = [1, 3], col_perm = [0, 2]. */
    int row_perm_A[2] = {1, 3};
    int col_perm_A[2] = {0, 2};
    double XA[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *A_m = new_permuted_dense(4, 5, 2, 2, row_perm_A, col_perm_A, XA);

    /* B: 4x4 CSR_matrix. */
    CSR_matrix *B = new_CSR_matrix(4, 4, 5);
    B->p[0] = 0;
    B->p[1] = 2;
    B->p[2] = 3;
    B->p[3] = 4;
    B->p[4] = 5;
    int Bi[5] = {0, 2, 1, 0, 3};
    double Bx[5] = {10.0, 20.0, 30.0, 40.0, 50.0};
    memcpy(B->i, Bi, sizeof Bi);
    memcpy(B->x, Bx, sizeof Bx);
    matrix *B_m = new_sparse_matrix(B);

    double d[4] = {1.0, -2.0, 0.5, 3.0};

    /* Wrapper path. Dispatchers don't touch sparse_matrix internals — caller
       owns csc_cache structure and values. */
    sparse_matrix_ensure_csc_cache((sparse_matrix *) B_m);
    matrix *C_m = BTA_matrices_alloc(A_m, B_m);
    B_m->refresh_csc_values(B_m);
    BTDA_matrices_fill_values(A_m, d, B_m, C_m);

    /* Direct primitive path: production now dispatches the (PD, Sparse)
       branch through CSC-pd kernels. Build a CSC view of B and call
       BTA_csc_pd_alloc + BTDA_csc_pd_fill_values to match. */
    matrix *A2_m = new_permuted_dense(4, 5, 2, 2, row_perm_A, col_perm_A, XA);
    permuted_dense *A2 = (permuted_dense *) A2_m;
    CSR_matrix *B2_csr = new_CSR_matrix(4, 4, 5);
    B2_csr->p[0] = 0;
    B2_csr->p[1] = 2;
    B2_csr->p[2] = 3;
    B2_csr->p[3] = 4;
    B2_csr->p[4] = 5;
    memcpy(B2_csr->i, Bi, sizeof Bi);
    memcpy(B2_csr->x, Bx, sizeof Bx);
    int *iwork = (int *) malloc(MAX(B2_csr->m, B2_csr->n) * sizeof(int));
    CSC_matrix *B2_csc = csr_to_csc_alloc(B2_csr, iwork);
    csr_to_csc_fill_values(B2_csr, B2_csc, iwork);
    matrix *C2 = BTA_csc_pd_alloc(B2_csc, A2);
    BTDA_csc_pd_fill_values(B2_csc, d, A2, (permuted_dense *) C2);

    mu_assert("values", cmp_double_array(C_m->x, C2->x, C_m->nnz));

    free_matrix(C_m);
    free_matrix(B_m);
    free_matrix(A_m);
    free_matrix(C2);
    free_CSC_matrix(B2_csc);
    free_CSR_matrix(B2_csr);
    free(iwork);
    free_matrix(A2_m);
    return 0;
}

/* ---------------------------------------------------------------- */
/* spd to_csr fallback in the BTA/BTDA dispatcher: when an operand is */
/* spd, the dispatcher must materialize it as a temp CSC via to_csr  */
/* and produce results identical to dispatching with the same matrix */
/* expressed directly as a sparse_matrix.                            */
/* ---------------------------------------------------------------- */

/* Deep-copy an spd's csr_cache into a fresh sparse_matrix (taking the
   csr by value so we don't share buffers). Used as the "reference"
   route in the equivalence tests below. */
static matrix *spd_to_sparse_matrix_copy(matrix *spd_m)
{
    CSR_matrix *src = spd_m->to_csr(spd_m);
    CSR_matrix *dst = new_CSR_matrix(src->m, src->n, src->nnz);
    memcpy(dst->p, src->p, (size_t) (src->m + 1) * sizeof(int));
    memcpy(dst->i, src->i, (size_t) src->nnz * sizeof(int));
    memcpy(dst->x, src->x, (size_t) src->nnz * sizeof(double));
    dst->nnz = src->nnz;
    return new_sparse_matrix(dst);
}

/* Primitive BTA_csc_pd kernel (no diagonal): C = B^T @ A, B is CSC, A is PD.
   Route 1 calls BTA_csc_pd_alloc + BTA_csc_pd_fill_values directly. The oracle
   flattens BOTH operands to sparse_matrix and runs the dispatcher with d = ones,
   which routes through the csc/csc sparse_dot path — sharing no code with the
   csc_pd kernel. Both compared via to_csr. */
const char *test_BTA_csc_pd_basic(void)
{
    /* A: 4x5 PD, row_perm = [1,3], col_perm = [0,2] (both permuted, so row_inv
       genuinely permutes and the transpose matters). */
    int row_perm_A[2] = {1, 3};
    int col_perm_A[2] = {0, 2};
    double XA[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *A_m = new_permuted_dense(4, 5, 2, 2, row_perm_A, col_perm_A, XA);

    /* B: 4x4 CSR -> CSC. Rows: 0:{0,2} 1:{1} 2:{0} 3:{3}. */
    int Bp[5] = {0, 2, 3, 4, 5};
    int Bi[5] = {0, 2, 1, 0, 3};
    double Bx[5] = {10.0, 20.0, 30.0, 40.0, 50.0};

    CSR_matrix *B_csr = new_CSR_matrix(4, 4, 5);
    memcpy(B_csr->p, Bp, sizeof Bp);
    memcpy(B_csr->i, Bi, sizeof Bi);
    memcpy(B_csr->x, Bx, sizeof Bx);
    int *iwork = (int *) malloc(MAX(B_csr->m, B_csr->n) * sizeof(int));
    CSC_matrix *B_csc = csr_to_csc_alloc(B_csr, iwork);
    csr_to_csc_fill_values(B_csr, B_csc, iwork);

    /* Route 1: our kernel. */
    matrix *C_ours = BTA_csc_pd_alloc(B_csc, (permuted_dense *) A_m);
    BTA_csc_pd_fill_values(B_csc, (permuted_dense *) A_m, (permuted_dense *) C_ours);

    /* Oracle: both operands sparse, d = ones, csc/csc dispatch. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_m);
    CSR_matrix *B_csr2 = new_CSR_matrix(4, 4, 5);
    memcpy(B_csr2->p, Bp, sizeof Bp);
    memcpy(B_csr2->i, Bi, sizeof Bi);
    memcpy(B_csr2->x, Bx, sizeof Bx);
    matrix *B_sparse = new_sparse_matrix(B_csr2);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    A_sparse->refresh_csc_values(A_sparse);
    B_sparse->refresh_csc_values(B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_CSC_matrix(B_csc);
    free_CSR_matrix(B_csr);
    free(iwork);
    free_matrix(A_m);
    return 0;
}

/* BTA_csc_pd corner case: B columns that (a) hit no A-row -> excluded from C's
   row_perm by BTA_csc_pd_alloc; (b) partially overlap A's rows (some B->i map to
   row_inv == -1) -> exercises sparse_dot_dense's -1 filtering; (c) fully land in
   A's rows. A->row_perm = {1,3}. B columns by source rows: col0:{0,2} (excluded),
   col1:{1} (clean), col2:{0,1,3} (partial), col3:{3} (clean). */
const char *test_BTA_csc_pd_partial_and_excluded_cols(void)
{
    int row_perm_A[2] = {1, 3};
    int col_perm_A[2] = {0, 2};
    double XA[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *A_m = new_permuted_dense(4, 5, 2, 2, row_perm_A, col_perm_A, XA);

    /* B: 4x4. Rows: 0:{0,2} 1:{1,2} 2:{0} 3:{2,3}. */
    int Bp[5] = {0, 2, 4, 5, 7};
    int Bi[7] = {0, 2, 1, 2, 0, 2, 3};
    double Bx[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    CSR_matrix *B_csr = new_CSR_matrix(4, 4, 7);
    memcpy(B_csr->p, Bp, sizeof Bp);
    memcpy(B_csr->i, Bi, sizeof Bi);
    memcpy(B_csr->x, Bx, sizeof Bx);
    int *iwork = (int *) malloc(MAX(B_csr->m, B_csr->n) * sizeof(int));
    CSC_matrix *B_csc = csr_to_csc_alloc(B_csr, iwork);
    csr_to_csc_fill_values(B_csr, B_csc, iwork);

    matrix *C_ours = BTA_csc_pd_alloc(B_csc, (permuted_dense *) A_m);
    BTA_csc_pd_fill_values(B_csc, (permuted_dense *) A_m, (permuted_dense *) C_ours);

    matrix *A_sparse = spd_to_sparse_matrix_copy(A_m);
    CSR_matrix *B_csr2 = new_CSR_matrix(4, 4, 7);
    memcpy(B_csr2->p, Bp, sizeof Bp);
    memcpy(B_csr2->i, Bi, sizeof Bi);
    memcpy(B_csr2->x, Bx, sizeof Bx);
    matrix *B_sparse = new_sparse_matrix(B_csr2);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    A_sparse->refresh_csc_values(A_sparse);
    B_sparse->refresh_csc_values(B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_CSC_matrix(B_csc);
    free_CSR_matrix(B_csr);
    free(iwork);
    free_matrix(A_m);
    return 0;
}

/* BTA_csc_pd corner case: every B column misses A's rows -> C->base.nnz == 0
   early-out. A->row_perm = {1,3}; B has nonzeros only in rows {0,2}. */
const char *test_BTA_csc_pd_empty(void)
{
    int row_perm_A[2] = {1, 3};
    int col_perm_A[2] = {0, 2};
    double XA[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *A_m = new_permuted_dense(4, 5, 2, 2, row_perm_A, col_perm_A, XA);

    /* B: 4x4, nonzeros only in rows 0 and 2. Rows: 0:{0,1} 2:{2,3}. */
    int Bp[5] = {0, 2, 2, 4, 4};
    int Bi[4] = {0, 1, 2, 3};
    double Bx[4] = {1.0, 2.0, 3.0, 4.0};

    CSR_matrix *B_csr = new_CSR_matrix(4, 4, 4);
    memcpy(B_csr->p, Bp, sizeof Bp);
    memcpy(B_csr->i, Bi, sizeof Bi);
    memcpy(B_csr->x, Bx, sizeof Bx);
    int *iwork = (int *) malloc(MAX(B_csr->m, B_csr->n) * sizeof(int));
    CSC_matrix *B_csc = csr_to_csc_alloc(B_csr, iwork);
    csr_to_csc_fill_values(B_csr, B_csc, iwork);

    matrix *C_ours = BTA_csc_pd_alloc(B_csc, (permuted_dense *) A_m);
    BTA_csc_pd_fill_values(B_csc, (permuted_dense *) A_m, (permuted_dense *) C_ours);

    matrix *A_sparse = spd_to_sparse_matrix_copy(A_m);
    CSR_matrix *B_csr2 = new_CSR_matrix(4, 4, 4);
    memcpy(B_csr2->p, Bp, sizeof Bp);
    memcpy(B_csr2->i, Bi, sizeof Bi);
    memcpy(B_csr2->x, Bx, sizeof Bx);
    matrix *B_sparse = new_sparse_matrix(B_csr2);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    A_sparse->refresh_csc_values(A_sparse);
    B_sparse->refresh_csc_values(B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("nnz zero", csr_ours->nnz == 0);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_CSC_matrix(B_csc);
    free_CSR_matrix(B_csr);
    free(iwork);
    free_matrix(A_m);
    return 0;
}

/* Wrapper dispatch: (A=spd, B=PD). The spd-fallback path goes through
   BTA_csc_pd_alloc / BTDA_csc_pd_fill_values; result must match the
   equivalent (A=sparse, B=PD) dispatch on the same matrix. */
const char *test_BTDA_matrices_spd_pd(void)
{
    /* A: 4x3 spd, two blocks.
       blk0: rows {0,1}, cols {0,2}, X = [[1,2],[3,4]]
       blk1: rows {2,3}, cols {1,2}, X = [[5,6],[7,8]]                     */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    /* B: 4x5 PD. */
    int B_rp[3] = {0, 1, 2};
    int B_cp[3] = {1, 3, 4};
    double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *B = new_permuted_dense(4, 5, 3, 3, B_rp, B_cp, BX);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: dispatcher with spd operand. */
    matrix *C_spd = BTA_matrices_alloc(A_spd, B);
    BTDA_matrices_fill_values(A_spd, d, B, C_spd);

    /* Route 2: dispatcher with the same matrix expressed as sparse. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B);
    A_sparse->refresh_csc_values(A_sparse);
    BTDA_matrices_fill_values(A_sparse, d, B, C_ref);

    /* Both routes produce a PD (B is PD). Compare X buffers. */
    permuted_dense *C_spd_pd = (permuted_dense *) C_spd;
    permuted_dense *C_ref_pd = (permuted_dense *) C_ref;
    mu_assert("m", C_spd->m == C_ref->m);
    mu_assert("n", C_spd->n == C_ref->n);
    mu_assert("m0", C_spd_pd->m0 == C_ref_pd->m0);
    mu_assert("n0", C_spd_pd->n0 == C_ref_pd->n0);
    mu_assert("row_perm",
              cmp_int_array(C_spd_pd->row_perm, C_ref_pd->row_perm, C_spd_pd->m0));
    mu_assert("col_perm",
              cmp_int_array(C_spd_pd->col_perm, C_ref_pd->col_perm, C_spd_pd->n0));
    mu_assert("X", cmp_double_array(C_spd_pd->X, C_ref_pd->X,
                                    (size_t) C_spd_pd->m0 * C_spd_pd->n0));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_spd);
    free_matrix(B);
    free_matrix(A_spd);
    return 0;
}

/* Wrapper dispatch: (A=PD, B=spd). Goes through BTA_pd_csc_alloc /
   BTDA_pd_csc_fill_values; equivalence with sparse-B. */
const char *test_BTDA_matrices_pd_spd(void)
{
    /* A: 4x5 PD. */
    int A_rp[3] = {0, 1, 2};
    int A_cp[3] = {1, 3, 4};
    double AX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *A = new_permuted_dense(4, 5, 3, 3, A_rp, A_cp, AX);

    /* B: 4x3 spd, same as test_BTDA_matrices_spd_pd. */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    matrix *C_spd = BTA_matrices_alloc(A, B_spd);
    BTDA_matrices_fill_values(A, d, B_spd, C_spd);

    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A, B_sparse);
    B_sparse->refresh_csc_values(B_sparse);
    BTDA_matrices_fill_values(A, d, B_sparse, C_ref);

    /* Output types now differ: the spd-B route returns stacked_pd (via
       BTA_spd_matrices), the sparse-B route returns permuted_dense (via
       BTA_sparse_matrices' PD branch). Compare via to_csr. */
    CSR_matrix *csr_spd = C_spd->to_csr(C_spd);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_spd->m == csr_ref->m);
    mu_assert("n", csr_spd->n == csr_ref->n);
    mu_assert("nnz", csr_spd->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_spd->p, csr_ref->p, csr_spd->m + 1));
    mu_assert("i", cmp_int_array(csr_spd->i, csr_ref->i, csr_spd->nnz));
    mu_assert("x", cmp_double_array(csr_spd->x, csr_ref->x, csr_spd->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_spd);
    free_matrix(B_spd);
    free_matrix(A);
    return 0;
}

/* Wrapper dispatch: (A=spd, B=spd). The spd-spd route returns stacked_pd
   (via BTA_spd_matrices); the sparse-sparse fallback returns sparse_matrix.
   Compare via to_csr. */
const char *test_BTDA_matrices_spd_spd(void)
{
    /* A: 4x3 spd. */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *Ablk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *Ablk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) Ablk0,
                                   (permuted_dense *) Ablk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    /* B: 4x4 spd. */
    int B0_rp[2] = {0, 2};
    int B0_cp[2] = {0, 3};
    double B0X[4] = {9, 8, 7, 6};
    matrix *Bblk0 = new_permuted_dense(4, 4, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {1, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 4, 3, 2};
    matrix *Bblk1 = new_permuted_dense(4, 4, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) Bblk0,
                                   (permuted_dense *) Bblk1};
    matrix *B_spd = new_stacked_pd(4, 4, 2, B_blocks, NULL, NULL);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    matrix *C_spd = BTA_matrices_alloc(A_spd, B_spd);
    BTDA_matrices_fill_values(A_spd, d, B_spd, C_spd);

    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    A_sparse->refresh_csc_values(A_sparse);
    B_sparse->refresh_csc_values(B_sparse);
    BTDA_matrices_fill_values(A_sparse, d, B_sparse, C_ref);

    /* Output types differ: spd-spd route returns stacked_pd, sparse-sparse
       returns sparse_matrix. Compare via to_csr. */
    CSR_matrix *csr_spd = C_spd->to_csr(C_spd);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_spd->m == csr_ref->m);
    mu_assert("n", csr_spd->n == csr_ref->n);
    mu_assert("nnz", csr_spd->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_spd->p, csr_ref->p, csr_spd->m + 1));
    mu_assert("i", cmp_int_array(csr_spd->i, csr_ref->i, csr_spd->nnz));
    mu_assert("x", cmp_double_array(csr_spd->x, csr_ref->x, csr_spd->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_spd);
    free_matrix(B_spd);
    free_matrix(A_spd);
    return 0;
}

/* Primitive BTA_pd_spd kernel (transpose + BA_pd_spd implementation):
   C = B^T @ A with B a permuted_dense and A a 2-block stacked_pd whose
   col_perms overlap (share col 2), exercising the scatter-accumulate
   path inside BA_pd_spd. Reference is the production dispatcher path
   with A flattened to a sparse_matrix and d = ones (BTDA with all-ones
   d is equivalent to BTA). */
const char *test_BTA_pd_spd_two_blocks_both_kept(void)
{
    /* A: 4x3 spd, two blocks.
       blk0: rows {0,1}, cols {0,2}, X = [[1,2],[3,4]]
       blk1: rows {2,3}, cols {1,2}, X = [[5,6],[7,8]]                     */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    /* B: 4x5 PD with row_perm = {0,1,2}, col_perm = {1,3,4}. */
    int B_rp[3] = {0, 1, 2};
    int B_cp[3] = {1, 3, 4};
    double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *B = new_permuted_dense(4, 5, 3, 3, B_rp, B_cp, BX);

    /* Route 1: our new BTA_pd_spd_* primitives. */
    matrix *C_ours = BTA_pd_spd_alloc((permuted_dense *) B, (stacked_pd *) A_spd);
    BTA_pd_spd_fill_values((permuted_dense *) B, (stacked_pd *) A_spd,
                           (permuted_dense *) C_ours);

    /* Route 2: dispatcher with A flattened to sparse_matrix, d = ones. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B);
    A_sparse->refresh_csc_values(A_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B, C_ref);

    /* Both routes produce a PD with row_perm = B->col_perm and col_perm
       = union of contributing A_k->col_perms. Compare shape, perms, X. */
    permuted_dense *C_ours_pd = (permuted_dense *) C_ours;
    permuted_dense *C_ref_pd = (permuted_dense *) C_ref;
    mu_assert("m", C_ours->m == C_ref->m);
    mu_assert("n", C_ours->n == C_ref->n);
    mu_assert("m0", C_ours_pd->m0 == C_ref_pd->m0);
    mu_assert("n0", C_ours_pd->n0 == C_ref_pd->n0);
    mu_assert("row_perm",
              cmp_int_array(C_ours_pd->row_perm, C_ref_pd->row_perm, C_ours_pd->m0));
    mu_assert("col_perm",
              cmp_int_array(C_ours_pd->col_perm, C_ref_pd->col_perm, C_ours_pd->n0));
    mu_assert("X", cmp_double_array(C_ours_pd->X, C_ref_pd->X,
                                    (size_t) C_ours_pd->m0 * C_ours_pd->n0));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(B);
    free_matrix(A_spd);
    return 0;
}

/* Primitive BTDA_pd_spd kernel (temp-DA composition):
   C = B^T @ diag(d) @ A. Same B and A layout as
   test_BTA_pd_spd_two_blocks_both_kept, but with a non-trivial d so
   BTDA differs from BTA. Reference is the production dispatcher path
   with A flattened to a sparse_matrix. */
const char *test_BTDA_pd_spd_two_blocks_both_kept(void)
{
    /* A: 4x3 spd, two blocks.
       blk0: rows {0,1}, cols {0,2}, X = [[1,2],[3,4]]
       blk1: rows {2,3}, cols {1,2}, X = [[5,6],[7,8]]                     */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    /* B: 4x5 PD with row_perm = {0,1,2}, col_perm = {1,3,4}. */
    int B_rp[3] = {0, 1, 2};
    int B_cp[3] = {1, 3, 4};
    double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *B = new_permuted_dense(4, 5, 3, 3, B_rp, B_cp, BX);

    /* Non-trivial d so BTDA != BTA. */
    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: our new BTDA_pd_spd_fill_values. C structure comes from
       BTA_pd_spd_alloc (BTDA shares sparsity with BTA). */
    matrix *C_ours = BTA_pd_spd_alloc((permuted_dense *) B, (stacked_pd *) A_spd);
    BTDA_pd_spd_fill_values((permuted_dense *) B, d, (stacked_pd *) A_spd,
                            (permuted_dense *) C_ours);

    /* Route 2: dispatcher with A flattened to sparse_matrix. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B);
    A_sparse->refresh_csc_values(A_sparse);
    BTDA_matrices_fill_values(A_sparse, d, B, C_ref);

    /* Both routes produce a PD with row_perm = B->col_perm and col_perm
       = union of contributing A_k->col_perms. Compare shape, perms, X. */
    permuted_dense *C_ours_pd = (permuted_dense *) C_ours;
    permuted_dense *C_ref_pd = (permuted_dense *) C_ref;
    mu_assert("m", C_ours->m == C_ref->m);
    mu_assert("n", C_ours->n == C_ref->n);
    mu_assert("m0", C_ours_pd->m0 == C_ref_pd->m0);
    mu_assert("n0", C_ours_pd->n0 == C_ref_pd->n0);
    mu_assert("row_perm",
              cmp_int_array(C_ours_pd->row_perm, C_ref_pd->row_perm, C_ours_pd->m0));
    mu_assert("col_perm",
              cmp_int_array(C_ours_pd->col_perm, C_ref_pd->col_perm, C_ours_pd->n0));
    mu_assert("X", cmp_double_array(C_ours_pd->X, C_ref_pd->X,
                                    (size_t) C_ours_pd->m0 * C_ours_pd->n0));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(B);
    free_matrix(A_spd);
    return 0;
}

/* Primitive BTDA_spd_pd kernel (ATA-style direct: per-block BTDA_pd_pd
   + accumulating coalesce). C = B^T @ diag(d) @ A with B a 2-block
   stacked_pd whose col_perms share column 2 — exercises the
   accumulating coalesce path (multiple source blocks contribute to the
   same output row). Reference is the production dispatcher with B
   flattened to a sparse_matrix; outputs differ in storage (stacked_pd
   vs permuted_dense), so compare via to_csr. */
const char *test_BTDA_spd_pd_overlapping_cp(void)
{
    /* B: 4x3 spd, two blocks with overlapping col_perms (share col 2).
       blk0: rows {0,1}, cols {0,2}, X = [[1,2],[3,4]]
       blk1: rows {2,3}, cols {1,2}, X = [[5,6],[7,8]]                     */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    /* A: 4x3 PD, full row_perm, full col_perm. */
    int A_rp[4] = {0, 1, 2, 3};
    int A_cp[3] = {0, 1, 2};
    double AX[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    matrix *A = new_permuted_dense(4, 3, 4, 3, A_rp, A_cp, AX);

    /* Non-trivial d so BTDA != BTA. */
    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: our new BTDA_spd_pd_fill_values. */
    matrix *C_ours = BTA_spd_pd_alloc((stacked_pd *) B_spd, (permuted_dense *) A);
    BTDA_spd_pd_fill_values((stacked_pd *) B_spd, d, (permuted_dense *) A,
                            (stacked_pd *) C_ours);

    /* Route 2: dispatcher with B flattened to sparse_matrix. Note
       BTA_matrices_alloc(A, B) computes B^T @ A, so A goes first. */
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A, B_sparse);
    B_sparse->refresh_csc_values(B_sparse);
    BTDA_matrices_fill_values(A, d, B_sparse, C_ref);

    /* C_ours is stacked_pd, C_ref is permuted_dense — compare via to_csr.
       Both represent the same global matrix; rows are emitted in sorted
       order, and the spd's per-signature blocks share col_perm = A->col_perm,
       so the CSR layouts match exactly. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_matrix(A);
    free_matrix(B_spd);
    return 0;
}

/* Primitive BTA_spd_pd kernel (no diagonal). Same matrices as the BTDA
   variant above. Reference is the production dispatcher with B flattened to a
   sparse_matrix and d = ones, which equals plain B^T @ A. */
const char *test_BTA_spd_pd_overlapping_cp(void)
{
    /* B: 4x3 spd, two blocks with overlapping col_perms (share col 2).
       blk0: rows {0,1}, cols {0,2}, X = [[1,2],[3,4]]
       blk1: rows {2,3}, cols {1,2}, X = [[5,6],[7,8]]                     */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    /* A: 4x3 PD, full row_perm, full col_perm. */
    int A_rp[4] = {0, 1, 2, 3};
    int A_cp[3] = {0, 1, 2};
    double AX[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    matrix *A = new_permuted_dense(4, 3, 4, 3, A_rp, A_cp, AX);

    /* Route 1: our new BTA_spd_pd_fill_values. */
    matrix *C_ours = BTA_spd_pd_alloc((stacked_pd *) B_spd, (permuted_dense *) A);
    BTA_spd_pd_fill_values((stacked_pd *) B_spd, (permuted_dense *) A,
                           (stacked_pd *) C_ours);

    /* Route 2: dispatcher with B flattened to sparse_matrix and d = ones, so
       BTDA == BTA. Note BTA_matrices_alloc(A, B) computes B^T @ A, so A goes
       first. */
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A, B_sparse);
    B_sparse->refresh_csc_values(B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A, d_ones, B_sparse, C_ref);

    /* C_ours is stacked_pd, C_ref is permuted_dense — compare via to_csr. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_matrix(A);
    free_matrix(B_spd);
    return 0;
}

/* Primitive BTDA_spd_csc kernel (ATA-style direct: per-block BTDA_pd_csc
   + accumulating coalesce). C = B^T @ diag(d) @ A with B a 2-block
   stacked_pd whose col_perms share column 2, and A a sparse_matrix
   wrapping a small CSR. Reference is the production dispatcher with B
   flattened to a sparse_matrix; outputs differ in storage (stacked_pd
   vs sparse_matrix), so compare via to_csr. */
const char *test_BTDA_spd_csc_overlapping_cp(void)
{
    /* B: 4x3 spd, two blocks with overlapping col_perms (share col 2).
       Same layout as test_BTDA_spd_pd_overlapping_cp.
       blk0: rows {0,1}, cols {0,2}, X = [[1,2],[3,4]]
       blk1: rows {2,3}, cols {1,2}, X = [[5,6],[7,8]]                     */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    /* A: 4x3 sparse_matrix wrapping a CSR with arbitrary nonzeros.
       Row 0: col 0 = 1, col 2 = 2
       Row 1: col 1 = 3
       Row 2: col 0 = 4, col 1 = 5, col 2 = 6
       Row 3: col 2 = 7
       p = [0, 2, 3, 6, 7], i = [0,2, 1, 0,1,2, 2], x = [1,2, 3, 4,5,6, 7]. */
    CSR_matrix *A_csr = new_CSR_matrix(4, 3, 7);
    int Ap[5] = {0, 2, 3, 6, 7};
    int Ai[7] = {0, 2, 1, 0, 1, 2, 2};
    double Ax[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A_sm = new_sparse_matrix(A_csr);

    /* Non-trivial d so BTDA != BTA. */
    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: our new BTDA_spd_csc_fill_values. Need A's csc_cache. */
    sparse_matrix_ensure_csc_cache((sparse_matrix *) A_sm);
    A_sm->refresh_csc_values(A_sm);
    matrix *C_ours =
        BTA_spd_csc_alloc((stacked_pd *) B_spd, ((sparse_matrix *) A_sm)->csc_cache);
    BTDA_spd_csc_fill_values((stacked_pd *) B_spd, d,
                             ((sparse_matrix *) A_sm)->csc_cache,
                             (stacked_pd *) C_ours);

    /* Route 2: dispatcher with B flattened to sparse_matrix. Note
       BTA_matrices_alloc(A, B) computes B^T @ A, so A goes first. */
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sm, B_sparse);
    B_sparse->refresh_csc_values(B_sparse);
    BTDA_matrices_fill_values(A_sm, d, B_sparse, C_ref);

    /* C_ours is stacked_pd, C_ref is sparse_matrix — compare via to_csr. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_matrix(A_sm);
    free_matrix(B_spd);
    return 0;
}

/* Primitive BTDA_spd_spd kernel (per-block BTDA_pd_spd + accumulating coalesce).
   C = B^T @ diag(d) @ A with both B and A 2-block stacked_pds. B's c_k's
   share col 2 (outer coalesce-accumulate path), and A's col_perms also
   share at least one column (inner BTA_pd_spd accumulate path). Reference
   is the production dispatcher with both operands flattened to sparse_matrix.
   Output types differ (stacked_pd vs sparse_matrix); compare via to_csr. */
const char *test_BTDA_spd_spd_overlapping(void)
{
    /* B: 4x3 spd, two blocks with overlapping c_k(B) (share col 2). */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *B_blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *B_blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) B_blk0,
                                   (permuted_dense *) B_blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    /* A: 4x3 spd, two blocks. A_0's rows overlap B_0's, A_1's rows overlap
       B_1's. A_0 and A_1 share col 1 → inner BTA_pd_spd accumulate fires. */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 1};
    double A0X[4] = {10, 11, 12, 13};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {20, 21, 22, 23};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: new kernel. */
    matrix *C_ours = BTA_spd_spd_alloc((stacked_pd *) B_spd, (stacked_pd *) A_spd);
    BTDA_spd_spd_fill_values((stacked_pd *) B_spd, d, (stacked_pd *) A_spd,
                             (stacked_pd *) C_ours);

    /* Route 2: dispatcher with both flattened to sparse_matrix.
       BTA_matrices_alloc(A, B) computes B^T @ A, so A goes first. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    A_sparse->refresh_csc_values(A_sparse);
    B_sparse->refresh_csc_values(B_sparse);
    BTDA_matrices_fill_values(A_sparse, d, B_sparse, C_ref);

    /* C_ours is stacked_pd, C_ref is sparse_matrix — compare via to_csr. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A_spd);
    free_matrix(B_spd);
    return 0;
}

/* Primitive BTA_spd_spd kernel (no diagonal). Same matrices as the BTDA
   variant above. Reference is the production dispatcher with both operands
   flattened to sparse_matrix and d = ones, which equals plain B^T @ A. */
const char *test_BTA_spd_spd_overlapping(void)
{
    /* B: 4x3 spd, two blocks with overlapping c_k(B) (share col 2). */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *B_blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *B_blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) B_blk0,
                                   (permuted_dense *) B_blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    /* A: 4x3 spd, two blocks. A_0's rows overlap B_0's, A_1's rows overlap
       B_1's. A_0 and A_1 share col 1. */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 1};
    double A0X[4] = {10, 11, 12, 13};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {20, 21, 22, 23};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    /* Route 1: new kernel. */
    matrix *C_ours = BTA_spd_spd_alloc((stacked_pd *) B_spd, (stacked_pd *) A_spd);
    BTA_spd_spd_fill_values((stacked_pd *) B_spd, (stacked_pd *) A_spd,
                            (stacked_pd *) C_ours);

    /* Route 2: dispatcher with both flattened to sparse_matrix and d = ones, so
       BTDA == BTA. BTA_matrices_alloc(A, B) computes B^T @ A, so A goes first. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    A_sparse->refresh_csc_values(A_sparse);
    B_sparse->refresh_csc_values(B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sparse, C_ref);

    /* C_ours is stacked_pd, C_ref is sparse_matrix — compare via to_csr. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A_spd);
    free_matrix(B_spd);
    return 0;
}

/* BTA_spd_spd corner case: a single B-block whose rows span MULTIPLE A-blocks,
   forcing BTA_pd_spd_fill_values' inner += accumulation over A-blocks into one
   partial, with overlapping output columns (A_0 and A_1 share col 1). Not
   exercised by test_BTA_spd_spd_overlapping (there each B-block overlaps exactly
   one A-block). */
const char *test_BTA_spd_spd_multi_A_per_block(void)
{
    /* B: 4x2, one block spanning all four contraction rows. */
    int B0_rp[4] = {0, 1, 2, 3};
    int B0_cp[2] = {0, 1};
    double B0X[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    matrix *B_blk0 = new_permuted_dense(4, 2, 4, 2, B0_rp, B0_cp, B0X);
    permuted_dense *B_blocks[1] = {(permuted_dense *) B_blk0};
    matrix *B_spd = new_stacked_pd(4, 2, 1, B_blocks, NULL, NULL);

    /* A: 4x3, two blocks with disjoint rows but overlapping cols (share col 1).
       Both overlap B_0's rows, so both contribute to B_0's single partial. */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 1};
    double A0X[4] = {10, 11, 12, 13};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {20, 21, 22, 23};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    matrix *C_ours = BTA_spd_spd_alloc((stacked_pd *) B_spd, (stacked_pd *) A_spd);
    BTA_spd_spd_fill_values((stacked_pd *) B_spd, (stacked_pd *) A_spd,
                            (stacked_pd *) C_ours);

    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    A_sparse->refresh_csc_values(A_sparse);
    B_sparse->refresh_csc_values(B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A_spd);
    free_matrix(B_spd);
    return 0;
}

/* BTA_spd_spd corner case: a B-block that overlaps NO A-block, producing an
   empty partial (n0 = 0) that coalesces to a zero-column output block (output
   row 1, all zeros). Exercises the partial-level nnz == 0 early-out in
   BTA_pd_spd_fill_values and the empty-block path through coalesce/to_csr. */
const char *test_BTA_spd_spd_nonoverlapping_block(void)
{
    /* B: 4x2, two blocks. B_0 rows {0,1}; B_1 rows {2,3}. */
    int B0_rp[2] = {0, 1};
    int B0_cp[1] = {0};
    double B0X[2] = {1, 2};
    matrix *B_blk0 = new_permuted_dense(4, 2, 2, 1, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[1] = {1};
    double B1X[2] = {3, 4};
    matrix *B_blk1 = new_permuted_dense(4, 2, 2, 1, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) B_blk0,
                                   (permuted_dense *) B_blk1};
    matrix *B_spd = new_stacked_pd(4, 2, 2, B_blocks, NULL, NULL);

    /* A: 4x2, one block on rows {0,1}. B_0 overlaps it; B_1 does not. */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 1};
    double A0X[4] = {10, 11, 12, 13};
    matrix *A_blk0 = new_permuted_dense(4, 2, 2, 2, A0_rp, A0_cp, A0X);
    permuted_dense *A_blocks[1] = {(permuted_dense *) A_blk0};
    matrix *A_spd = new_stacked_pd(4, 2, 1, A_blocks, NULL, NULL);

    matrix *C_ours = BTA_spd_spd_alloc((stacked_pd *) B_spd, (stacked_pd *) A_spd);
    BTA_spd_spd_fill_values((stacked_pd *) B_spd, (stacked_pd *) A_spd,
                            (stacked_pd *) C_ours);

    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    A_sparse->refresh_csc_values(A_sparse);
    B_sparse->refresh_csc_values(B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A_spd);
    free_matrix(B_spd);
    return 0;
}

/* BTA_spd_matrices dispatcher: A is permuted_dense. Verifies the
   PD-branch routes to BTA_spd_pd / BTDA_spd_pd. */
const char *test_BTA_spd_matrices_pd_A(void)
{
    /* Same inputs as test_BTDA_spd_pd_overlapping_cp. */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *B_blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *B_blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) B_blk0,
                                   (permuted_dense *) B_blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    int A_rp[4] = {0, 1, 2, 3};
    int A_cp[3] = {0, 1, 2};
    double AX[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    matrix *A = new_permuted_dense(4, 3, 4, 3, A_rp, A_cp, AX);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: public dispatcher with B in its native spd type — routes
       internally through the spd-B branch. */
    matrix *C_ours = BTA_matrices_alloc(A, B_spd);
    BTDA_matrices_fill_values(A, d, B_spd, C_ours);

    /* Route 2: same public dispatcher with B flattened to sparse — routes
       through a different internal branch (csc-PD path). */
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A, B_sparse);
    B_sparse->refresh_csc_values(B_sparse);
    BTDA_matrices_fill_values(A, d, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_matrix(A);
    free_matrix(B_spd);
    return 0;
}

/* Public dispatcher equivalence: B is spd and A is sparse_matrix. The
   spd-B internal branch is exercised in Route 1; Route 2 flattens B to
   sparse and exercises the csc-csc branch. */
const char *test_BTA_spd_matrices_csc_A(void)
{
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *B_blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *B_blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) B_blk0,
                                   (permuted_dense *) B_blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    /* A as sparse_matrix wrapping a CSR (same shape/values as
       test_BTDA_spd_csc_overlapping_cp). */
    CSR_matrix *A_csr = new_CSR_matrix(4, 3, 7);
    int Ap[5] = {0, 2, 3, 6, 7};
    int Ai[7] = {0, 2, 1, 0, 1, 2, 2};
    double Ax[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A_sm = new_sparse_matrix(A_csr);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: public dispatcher with B native spd — routes through
       the spd-B branch (BTA_spd_csc internally). */
    matrix *C_ours = BTA_matrices_alloc(A_sm, B_spd);
    A_sm->refresh_csc_values(A_sm);
    BTDA_matrices_fill_values(A_sm, d, B_spd, C_ours);

    /* Route 2: same public dispatcher with B flattened to sparse —
       routes through the csc-csc branch. */
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sm, B_sparse);
    A_sm->refresh_csc_values(A_sm);
    B_sparse->refresh_csc_values(B_sparse);
    BTDA_matrices_fill_values(A_sm, d, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_matrix(A_sm);
    free_matrix(B_spd);
    return 0;
}

/* BTA_spd_matrices dispatcher: A is stacked_pd. Verifies the
   spd-branch routes to BTA_spd_spd / BTDA_spd_spd. */
const char *test_BTA_spd_matrices_spd_A(void)
{
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *B_blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *B_blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) B_blk0,
                                   (permuted_dense *) B_blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 1};
    double A0X[4] = {10, 11, 12, 13};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {20, 21, 22, 23};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: public dispatcher with both operands as native spd —
       routes through the spd-spd branch (BTA_spd_spd internally). */
    matrix *C_ours = BTA_matrices_alloc(A_spd, B_spd);
    BTDA_matrices_fill_values(A_spd, d, B_spd, C_ours);

    /* Route 2: same public dispatcher with both flattened to sparse —
       routes through the csc-csc branch. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    A_sparse->refresh_csc_values(A_sparse);
    B_sparse->refresh_csc_values(B_sparse);
    BTDA_matrices_fill_values(A_sparse, d, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A_spd);
    free_matrix(B_spd);
    return 0;
}

/* BA_spd_matrices dispatcher tests previously lived here. The dispatcher
   is now static inside matmul_dispatchers.c (no production caller, no public
   API). The underlying primitives BA_spd_csc_*, BA_spd_pd_*, BA_spd_spd_*
   are tested directly in tests/utils/test_stacked_pd.h. */

/* BTA_pd_matrices dispatcher: A is permuted_dense. Verifies the PD-branch
   routes to BTA_pd_pd / BTDA_pd_pd. Both routes produce a PD; compare X
   buffers directly. */
const char *test_BTA_pd_matrices_pd_A(void)
{
    /* B: 2x4 PD with row_perm = {0,1}, col_perm = {1,3}. */
    int B_rp[2] = {0, 1};
    int B_cp[2] = {1, 3};
    double BX[4] = {1, 2, 3, 4};
    matrix *B = new_permuted_dense(2, 4, 2, 2, B_rp, B_cp, BX);

    /* A: 2x4 PD with row_perm = {0,1}, col_perm = {0,2}. */
    int A_rp[2] = {0, 1};
    int A_cp[2] = {0, 2};
    double AX[4] = {5, 6, 7, 8};
    matrix *A = new_permuted_dense(2, 4, 2, 2, A_rp, A_cp, AX);

    double d[2] = {2.0, -1.5};

    /* Route 1: public dispatcher with both operands native PD —
       routes through BTA_pd_matrices -> BTA_pd_pd internally. */
    matrix *C_ours = BTA_matrices_alloc(A, B);
    BTDA_matrices_fill_values(A, d, B, C_ours);

    /* Route 2: same dispatcher with A flattened to sparse — routes
       through BTA_pd_matrices -> BTA_pd_csc, a different kernel. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B);
    A_sparse->refresh_csc_values(A_sparse);
    BTDA_matrices_fill_values(A_sparse, d, B, C_ref);

    /* Both routes produce PD (B is PD). Compare X/perms directly. */
    permuted_dense *C_ours_pd = (permuted_dense *) C_ours;
    permuted_dense *C_ref_pd = (permuted_dense *) C_ref;
    mu_assert("m", C_ours->m == C_ref->m);
    mu_assert("n", C_ours->n == C_ref->n);
    mu_assert("m0", C_ours_pd->m0 == C_ref_pd->m0);
    mu_assert("n0", C_ours_pd->n0 == C_ref_pd->n0);
    mu_assert("row_perm",
              cmp_int_array(C_ours_pd->row_perm, C_ref_pd->row_perm, C_ours_pd->m0));
    mu_assert("col_perm",
              cmp_int_array(C_ours_pd->col_perm, C_ref_pd->col_perm, C_ours_pd->n0));
    mu_assert("X", cmp_double_array(C_ours_pd->X, C_ref_pd->X,
                                    (size_t) C_ours_pd->m0 * C_ours_pd->n0));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* BTA_pd_matrices dispatcher: A is sparse_matrix. Verifies the sparse-branch
   ensures csc_cache and routes to BTA_pd_csc / BTDA_pd_csc. */
const char *test_BTA_pd_matrices_csc_A(void)
{
    /* B: 4x5 PD. */
    int B_rp[3] = {0, 1, 2};
    int B_cp[3] = {1, 3, 4};
    double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *B = new_permuted_dense(4, 5, 3, 3, B_rp, B_cp, BX);

    /* A: 4x3 sparse_matrix wrapping a CSR. */
    CSR_matrix *A_csr = new_CSR_matrix(4, 3, 7);
    int Ap[5] = {0, 2, 3, 6, 7};
    int Ai[7] = {0, 2, 1, 0, 1, 2, 2};
    double Ax[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A_sm = new_sparse_matrix(A_csr);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: public dispatcher with B native PD, A native sparse —
       routes through BTA_pd_matrices -> BTA_pd_csc internally. */
    matrix *C_ours = BTA_matrices_alloc(A_sm, B);
    A_sm->refresh_csc_values(A_sm);
    BTDA_matrices_fill_values(A_sm, d, B, C_ours);

    /* Route 2: same dispatcher with B flattened to sparse — routes
       through BTA_sparse_matrices -> BTA_alloc (csc-csc), a different
       internal kernel. Output type differs (sparse_matrix vs PD), so
       compare via to_csr. */
    matrix *B_sparse = spd_to_sparse_matrix_copy(B);
    matrix *C_ref = BTA_matrices_alloc(A_sm, B_sparse);
    A_sm->refresh_csc_values(A_sm);
    B_sparse->refresh_csc_values(B_sparse);
    BTDA_matrices_fill_values(A_sm, d, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_matrix(A_sm);
    free_matrix(B);
    return 0;
}

/* BTA_pd_matrices dispatcher: A is stacked_pd. Verifies the spd-branch
   routes to BTA_pd_spd / BTDA_pd_spd. */
const char *test_BTA_pd_matrices_spd_A(void)
{
    /* B: 4x5 PD. */
    int B_rp[3] = {0, 1, 2};
    int B_cp[3] = {1, 3, 4};
    double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *B = new_permuted_dense(4, 5, 3, 3, B_rp, B_cp, BX);

    /* A: 4x3 spd, two blocks with shared col 2 (exercises BTA_pd_spd
       scatter-accumulate). */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: public dispatcher with A native spd, B native PD —
       routes through BTA_pd_matrices -> BTA_pd_spd internally. */
    matrix *C_ours = BTA_matrices_alloc(A_spd, B);
    BTDA_matrices_fill_values(A_spd, d, B, C_ours);

    /* Route 2: same public dispatcher with A flattened to sparse —
       routes through BTA_pd_matrices -> BTA_pd_csc, a different
       internal kernel. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B);
    A_sparse->refresh_csc_values(A_sparse);
    BTDA_matrices_fill_values(A_sparse, d, B, C_ref);

    /* Both routes produce PD (B is PD). Compare X/perms directly. */
    permuted_dense *C_ours_pd = (permuted_dense *) C_ours;
    permuted_dense *C_ref_pd = (permuted_dense *) C_ref;
    mu_assert("m", C_ours->m == C_ref->m);
    mu_assert("n", C_ours->n == C_ref->n);
    mu_assert("m0", C_ours_pd->m0 == C_ref_pd->m0);
    mu_assert("n0", C_ours_pd->n0 == C_ref_pd->n0);
    mu_assert("row_perm",
              cmp_int_array(C_ours_pd->row_perm, C_ref_pd->row_perm, C_ours_pd->m0));
    mu_assert("col_perm",
              cmp_int_array(C_ours_pd->col_perm, C_ref_pd->col_perm, C_ours_pd->n0));
    mu_assert("X", cmp_double_array(C_ours_pd->X, C_ref_pd->X,
                                    (size_t) C_ours_pd->m0 * C_ours_pd->n0));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A_spd);
    free_matrix(B);
    return 0;
}

/* Primitive BTDA_csc_spd kernel: C = B^T @ diag(d) @ A with B a CSC and A a
   2-block stacked_pd whose col_perms share at least one column (so the
   accumulating coalesce path fires). Reference is the production
   dispatcher with A flattened to a sparse_matrix; outputs differ in
   storage (stacked_pd vs sparse_matrix), so compare via to_csr. */
const char *test_BTDA_csc_spd_overlapping(void)
{
    /* A: 4x3 spd, two blocks with overlapping col_perms (share col 2). */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    /* B: 4x3 sparse_matrix wrapping a CSR. */
    CSR_matrix *B_csr = new_CSR_matrix(4, 3, 7);
    int Bp[5] = {0, 2, 3, 6, 7};
    int Bi[7] = {0, 2, 1, 0, 1, 2, 2};
    double Bx[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(B_csr->p, Bp, sizeof(Bp));
    memcpy(B_csr->i, Bi, sizeof(Bi));
    memcpy(B_csr->x, Bx, sizeof(Bx));
    matrix *B_sm = new_sparse_matrix(B_csr);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: new kernel. Ensure B's csc_cache and refresh values. */
    sparse_matrix_ensure_csc_cache((sparse_matrix *) B_sm);
    B_sm->refresh_csc_values(B_sm);
    matrix *C_ours =
        BTA_csc_spd_alloc(((sparse_matrix *) B_sm)->csc_cache, (stacked_pd *) A_spd);
    BTDA_csc_spd_fill_values(((sparse_matrix *) B_sm)->csc_cache, d,
                             (stacked_pd *) A_spd, (stacked_pd *) C_ours);

    /* Route 2: production dispatcher with A flattened to sparse_matrix.
       BTA_matrices_alloc(A, B) computes B^T @ A, so A goes first. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sm);
    A_sparse->refresh_csc_values(A_sparse);
    BTDA_matrices_fill_values(A_sparse, d, B_sm, C_ref);

    /* C_ours is stacked_pd, C_ref is sparse_matrix — compare via to_csr. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(B_sm);
    free_matrix(A_spd);
    return 0;
}

/* BTA_sparse_matrices dispatcher: A is permuted_dense. Both routes produce
   PD; compare X / perms directly. */
const char *test_BTA_sparse_matrices_pd_A(void)
{
    /* B: 4x3 sparse_matrix. */
    CSR_matrix *B_csr = new_CSR_matrix(4, 3, 7);
    int Bp[5] = {0, 2, 3, 6, 7};
    int Bi[7] = {0, 2, 1, 0, 1, 2, 2};
    double Bx[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(B_csr->p, Bp, sizeof(Bp));
    memcpy(B_csr->i, Bi, sizeof(Bi));
    memcpy(B_csr->x, Bx, sizeof(Bx));
    matrix *B_sm = new_sparse_matrix(B_csr);

    /* A: 4x3 PD. */
    int A_rp[4] = {0, 1, 2, 3};
    int A_cp[3] = {0, 1, 2};
    double AX[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    matrix *A = new_permuted_dense(4, 3, 4, 3, A_rp, A_cp, AX);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: public dispatcher with B native sparse, A native PD —
       routes through BTA_sparse_matrices -> BTA_csc_pd internally. */
    matrix *C_ours = BTA_matrices_alloc(A, B_sm);
    B_sm->refresh_csc_values(B_sm);
    BTDA_matrices_fill_values(A, d, B_sm, C_ours);

    /* Route 2: same dispatcher with A flattened to sparse — routes
       through BTA_sparse_matrices -> BTA_alloc (csc-csc), a different
       kernel. Output type differs (sparse_matrix vs PD), so compare
       via to_csr. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sm);
    A_sparse->refresh_csc_values(A_sparse);
    B_sm->refresh_csc_values(B_sm);
    BTDA_matrices_fill_values(A_sparse, d, B_sm, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A);
    free_matrix(B_sm);
    return 0;
}

/* Public dispatcher: B and A both sparse_matrix. Both routes go
   through the same csc-csc kernel, so the reference is a hand-computed
   dense matrix product. */
const char *test_BTA_sparse_matrices_csc_A(void)
{
    /* B: 4x3 sparse_matrix. */
    CSR_matrix *B_csr = new_CSR_matrix(4, 3, 7);
    int Bp[5] = {0, 2, 3, 6, 7};
    int Bi[7] = {0, 2, 1, 0, 1, 2, 2};
    double Bx[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(B_csr->p, Bp, sizeof(Bp));
    memcpy(B_csr->i, Bi, sizeof(Bi));
    memcpy(B_csr->x, Bx, sizeof(Bx));
    matrix *B_sm = new_sparse_matrix(B_csr);

    /* A: 4x3 sparse_matrix. */
    CSR_matrix *A_csr = new_CSR_matrix(4, 3, 6);
    int Ap[5] = {0, 1, 3, 5, 6};
    int Ai[6] = {1, 0, 2, 1, 2, 0};
    double Ax[6] = {10, 20, 30, 40, 50, 60};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A_sm = new_sparse_matrix(A_csr);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1 (only route): public dispatcher with both operands native
       sparse — routes through BTA_sparse_matrices -> BTA_alloc (csc-csc). */
    matrix *C_ours = BTA_matrices_alloc(A_sm, B_sm);
    A_sm->refresh_csc_values(A_sm);
    B_sm->refresh_csc_values(B_sm);
    BTDA_matrices_fill_values(A_sm, d, B_sm, C_ours);

    /* Reference: hand-compute C = B^T @ diag(d) @ A as a dense 3x3 buffer.
       Both operands are 4x3 sparse — too small to be worth a second
       dispatcher route (nothing to flatten). */
    double B_dense[4 * 3] = {
        1, 0, 2, /* row 0 */
        0, 3, 0, /* row 1 */
        4, 5, 6, /* row 2 */
        0, 0, 7  /* row 3 */
    };
    double A_dense[4 * 3] = {
        0,  10, 0,  /* row 0 */
        20, 0,  30, /* row 1 */
        0,  40, 50, /* row 2 */
        60, 0,  0   /* row 3 */
    };
    double C_dense_ref[3 * 3] = {0};
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            double s = 0.0;
            for (int l = 0; l < 4; l++)
            {
                s += B_dense[l * 3 + i] * d[l] * A_dense[l * 3 + j];
            }
            C_dense_ref[i * 3 + j] = s;
        }
    }

    /* Scatter dispatcher's CSR output into a dense buffer for comparison. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    mu_assert("m", csr_ours->m == 3);
    mu_assert("n", csr_ours->n == 3);
    double C_dense_ours[3 * 3] = {0};
    for (int i = 0; i < csr_ours->m; i++)
    {
        for (int jj = csr_ours->p[i]; jj < csr_ours->p[i + 1]; jj++)
        {
            C_dense_ours[i * 3 + csr_ours->i[jj]] = csr_ours->x[jj];
        }
    }
    mu_assert("C dense", cmp_double_array(C_dense_ours, C_dense_ref, 9));

    free_matrix(C_ours);
    free_matrix(A_sm);
    free_matrix(B_sm);
    return 0;
}

/* BTA_sparse_matrices dispatcher: A is stacked_pd. Our route returns
   stacked_pd (via BTA_csc_spd_alloc); the production route returns
   sparse_matrix (it materializes A->csc inside resolve_operand, then
   takes the csc-csc branch). Compare via to_csr. */
const char *test_BTA_sparse_matrices_spd_A(void)
{
    /* B: 4x3 sparse_matrix. */
    CSR_matrix *B_csr = new_CSR_matrix(4, 3, 7);
    int Bp[5] = {0, 2, 3, 6, 7};
    int Bi[7] = {0, 2, 1, 0, 1, 2, 2};
    double Bx[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(B_csr->p, Bp, sizeof(Bp));
    memcpy(B_csr->i, Bi, sizeof(Bi));
    memcpy(B_csr->x, Bx, sizeof(Bx));
    matrix *B_sm = new_sparse_matrix(B_csr);

    /* A: 4x3 spd, two blocks with overlapping col_perm (share col 2). */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: public dispatcher with B native sparse, A native spd —
       routes through BTA_sparse_matrices -> BTA_csc_spd internally. */
    matrix *C_ours = BTA_matrices_alloc(A_spd, B_sm);
    B_sm->refresh_csc_values(B_sm);
    BTDA_matrices_fill_values(A_spd, d, B_sm, C_ours);

    /* Route 2: same dispatcher with A flattened to sparse — routes
       through BTA_sparse_matrices -> BTA_alloc (csc-csc). */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sm);
    A_sparse->refresh_csc_values(A_sparse);
    B_sm->refresh_csc_values(B_sm);
    BTDA_matrices_fill_values(A_sparse, d, B_sm, C_ref);

    /* Routes produce different output types — compare via to_csr. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A_spd);
    free_matrix(B_sm);
    return 0;
}

/* Regression test for the kron-scratch / transpose-cache crash. Exercises
   BA_dense_kron_spd_fill_values with p > 1: the underlying kron_for_each_
   output_block reuses a single scratch PD across iterations, mutating
   its row_perm / col_perm in place. A prior implementation of
   BA_pd_spd_fill_values cached B's transpose on B->transpose_cache,
   which (a) was never populated for the fill-path scratch, and (b)
   would have held stale perms even if populated. Now BA_pd_spd does
   per-call transpose alloc/free so this scenario is correct. The
   reference path runs the same dispatch with J flattened to
   sparse_matrix (which routes via BA_dense_kron_csc, no transpose-cache
   involvement); we compare via to_csr. */
const char *test_BA_pd_kron_spd_no_cache_staleness(void)
{
    /* A: 2x2 PD (kron-block size). */
    int A_rp[2] = {0, 1};
    int A_cp[2] = {0, 1};
    double AX[4] = {1, 2, 3, 4};
    matrix *A = new_permuted_dense(2, 2, 2, 2, A_rp, A_cp, AX);

    /* J: 6x3 spd with three blocks, one block per kron-stride so each
       iteration's scratch perms differ. p = 3 -> kron(I_3, A) is 6x6,
       J is 6x3, output is 6x3. */
    int J0_rp[2] = {0, 1};
    int J0_cp[2] = {0, 1};
    double J0X[4] = {1, 2, 3, 4};
    matrix *Jblk0 = new_permuted_dense(6, 3, 2, 2, J0_rp, J0_cp, J0X);
    int J1_rp[2] = {2, 3};
    int J1_cp[2] = {1, 2};
    double J1X[4] = {5, 6, 7, 8};
    matrix *Jblk1 = new_permuted_dense(6, 3, 2, 2, J1_rp, J1_cp, J1X);
    int J2_rp[2] = {4, 5};
    int J2_cp[2] = {0, 2};
    double J2X[4] = {9, 10, 11, 12};
    matrix *Jblk2 = new_permuted_dense(6, 3, 2, 2, J2_rp, J2_cp, J2X);
    permuted_dense *J_blocks[3] = {(permuted_dense *) Jblk0,
                                   (permuted_dense *) Jblk1,
                                   (permuted_dense *) Jblk2};
    matrix *J_spd = new_stacked_pd(6, 3, 3, J_blocks, NULL, NULL);

    int p = 3;

    /* Route 1: spd path through BA_pd_kron_spd. */
    matrix *C_spd = BA_dense_kron_matrices_alloc((permuted_dense *) A, p, J_spd);
    BA_dense_kron_matrices_fill_values((permuted_dense *) A, p, J_spd,
                                       (stacked_pd *) C_spd);

    /* Route 2: sparse path via spd_to_sparse_matrix_copy + BA_dense_kron_csc. */
    matrix *J_sparse = spd_to_sparse_matrix_copy(J_spd);
    matrix *C_ref = BA_dense_kron_matrices_alloc((permuted_dense *) A, p, J_sparse);
    J_sparse->refresh_csc_values(J_sparse);
    BA_dense_kron_matrices_fill_values((permuted_dense *) A, p, J_sparse,
                                       (stacked_pd *) C_ref);

    /* Compare via to_csr — output structures may differ. */
    CSR_matrix *csr_spd = C_spd->to_csr(C_spd);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_spd->m == csr_ref->m);
    mu_assert("n", csr_spd->n == csr_ref->n);
    mu_assert("nnz", csr_spd->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_spd->p, csr_ref->p, csr_spd->m + 1));
    mu_assert("i", cmp_int_array(csr_spd->i, csr_ref->i, csr_spd->nnz));
    mu_assert("x", cmp_double_array(csr_spd->x, csr_ref->x, csr_spd->nnz));

    free_matrix(C_ref);
    free_matrix(J_sparse);
    free_matrix(C_spd);
    free_matrix(J_spd);
    free_matrix(A);
    return 0;
}

#endif /* TEST_MATMUL_DISPATCHERS_H */
