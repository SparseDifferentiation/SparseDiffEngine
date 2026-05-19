#ifndef TEST_MATRIX_BTA_H
#define TEST_MATRIX_BTA_H

#include "minunit.h"
#include "old-code/old_permuted_dense.h"
#include "test_helpers.h"
#include "utils/CSC_matrix.h"
#include "utils/CSR_matrix.h"
#include "utils/matrix_BTA.h"
#include "utils/permuted_dense.h"
#include "utils/permuted_dense_linalg.h"
#include "utils/sparse_matrix.h"
#include "utils/stacked_pd.h"
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
    free_matrix(B_sparse);
    free_matrix(C_spd);
    free_matrix(B_spd);
    free_matrix(A);
    return 0;
}

/* Wrapper dispatch: (A=spd, B=spd). Both operands materialize to CSC ->
   BTA_alloc / BTDA_fill_values; result is sparse_matrix. */
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

    /* Both outputs are sparse_matrix. Compare CSRs structurally + by value. */
    CSR_matrix *csr_spd = ((sparse_matrix *) C_spd)->csr;
    CSR_matrix *csr_ref = ((sparse_matrix *) C_ref)->csr;
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

#endif /* TEST_MATRIX_BTA_H */
