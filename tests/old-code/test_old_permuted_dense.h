#ifndef TEST_OLD_PERMUTED_DENSE_H
#define TEST_OLD_PERMUTED_DENSE_H

#include "minunit.h"
#include "old-code/old_permuted_dense.h"
#include "sparse_linalg/CSC_matrix.h"
#include "sparse_linalg/CSR_matrix.h"
#include "sparse_linalg/matmul_dispatchers.h"
#include "sparse_linalg/permuted_dense.h"
#include "sparse_linalg/permuted_dense_linalg.h"
#include "sparse_linalg/sparse_matrix.h"
#include "sparse_linalg/utils.h"
#include "test_helpers.h"
#include <stdlib.h>
#include <string.h>

/* Direct unit tests for the legacy CSR-pd BTA kernels in old-code. They no
   longer sit on a production path (matrix_BTA dispatcher hard-wires the
   CSC variants), but the kernels remain as reference implementations and
   serve as the oracle for test_BTA_pd_csc_matches_csr and
   test_BTDA_matrices_csr_pd below. Run by all_tests only. */

const char *test_BTA_pd_csr_basic(void)
{
    /* CSR_matrix A: m=4, n=5, with nonzeros:
       row 0: cols {1, 4}
       row 1: cols {0, 2}
       row 2: cols {2}
       row 3: cols {1, 4} */
    CSR_matrix *A = new_CSR_matrix(4, 5, 7);
    A->p[0] = 0;
    A->p[1] = 2;
    A->p[2] = 4;
    A->p[3] = 5;
    A->p[4] = 7;
    int Ai[7] = {1, 4, 0, 2, 2, 1, 4};
    double Ax[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    memcpy(A->i, Ai, sizeof Ai);
    memcpy(A->x, Ax, sizeof Ax);

    /* PD B: m=4, n=4, row_perm = [1, 3], col_perm = [0, 2], X = [[10, 20], [30,
     * 40]]. */
    int row_perm_B[2] = {1, 3};
    int col_perm_B[2] = {0, 2};
    double XB[4] = {10.0, 20.0, 30.0, 40.0};
    matrix *B_m = new_permuted_dense(4, 4, 2, 2, row_perm_B, col_perm_B, XB);
    permuted_dense *B = (permuted_dense *) B_m;

    matrix *out_m = BTA_pd_csr_alloc(B, A);
    permuted_dense *out = (permuted_dense *) out_m;

    /* Expected col_active: union of A's columns in rows 1 and 3
       = {0, 2} ∪ {1, 4} = {0, 1, 2, 4}, size 4. */
    int expected_col_perm[4] = {0, 1, 2, 4};
    mu_assert("out m", out_m->m == 4); /* B.n */
    mu_assert("out n", out_m->n == 5); /* A.n */
    mu_assert("m0", out->m0 == 2);
    mu_assert("n0", out->n0 == 4);
    mu_assert("row_perm", cmp_int_array(out->row_perm, col_perm_B, 2));
    mu_assert("col_perm", cmp_int_array(out->col_perm, expected_col_perm, 4));

    BTA_pd_csr_fill_values(B, A, out);

    /* Reference: scatter A and B to dense 4x{5,4}, compute B^T A, extract
       block at (col_perm_B × out->col_perm). Scatter inlined locally to
       avoid coupling to the static helpers in tests/utils/test_permuted_dense.h. */
    double *A_d = (double *) calloc(4 * 5, sizeof(double));
    double *B_d = (double *) calloc(4 * 4, sizeof(double));
    for (int i = 0; i < A->m; i++)
        for (int e = A->p[i]; e < A->p[i + 1]; e++) A_d[i * 5 + A->i[e]] = A->x[e];
    for (int kk = 0; kk < B->m0; kk++)
        for (int jj = 0; jj < B->n0; jj++)
            B_d[B->row_perm[kk] * 4 + B->col_perm[jj]] = B->X[kk * B->n0 + jj];

    double C_ref[4 * 5];
    memset(C_ref, 0, sizeof C_ref);
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            double s = 0.0;
            for (int k = 0; k < 4; k++)
            {
                s += B_d[k * 4 + i] * A_d[k * 5 + j];
            }
            C_ref[i * 5 + j] = s;
        }
    }
    double expected_X[8];
    for (int ii = 0; ii < 2; ii++)
    {
        for (int jj = 0; jj < 4; jj++)
        {
            expected_X[ii * 4 + jj] =
                C_ref[col_perm_B[ii] * 5 + expected_col_perm[jj]];
        }
    }
    mu_assert("values", cmp_double_array(out->X, expected_X, 8));

    free(A_d);
    free(B_d);
    free_matrix(out_m);
    free_matrix(B_m);
    free_CSR_matrix(A);
    return 0;
}

/* BTA(CSR_matrix A, PD B) where A is a leaf-variable Jacobian (identity-in-block).
   A is (4, 8): row k has a 1 at column 4+k (variable v of size 4 at var_id=4).
   Expected: col_perm_out = {4+row_perm_B[kk]} = {4+1, 4+3} = {5, 7}, and X_C =
   X_B^T. */
const char *test_BTA_pd_csr_leaf_variable(void)
{
    CSR_matrix *A = new_CSR_matrix(4, 8, 4);
    for (int k = 0; k < 4; k++)
    {
        A->p[k] = k;
        A->i[k] = 4 + k;
        A->x[k] = 1.0;
    }
    A->p[4] = 4;

    int row_perm_B[2] = {1, 3};
    int col_perm_B[2] = {0, 2};
    double XB[4] = {10.0, 20.0, 30.0, 40.0}; /* row-major (2, 2) */
    matrix *B_m = new_permuted_dense(4, 4, 2, 2, row_perm_B, col_perm_B, XB);
    permuted_dense *B = (permuted_dense *) B_m;

    matrix *out_m = BTA_pd_csr_alloc(B, A);
    permuted_dense *out = (permuted_dense *) out_m;

    int expected_col_perm[2] = {5, 7};
    mu_assert("m0", out->m0 == 2);
    mu_assert("n0", out->n0 == 2);
    mu_assert("row_perm", cmp_int_array(out->row_perm, col_perm_B, 2));
    mu_assert("col_perm", cmp_int_array(out->col_perm, expected_col_perm, 2));

    BTA_pd_csr_fill_values(B, A, out);

    /* X_C should be X_B^T = [[10, 30], [20, 40]] row-major. */
    double expected_X[4] = {10.0, 30.0, 20.0, 40.0};
    mu_assert("values", cmp_double_array(out->X, expected_X, 4));

    free_matrix(out_m);
    free_matrix(B_m);
    free_CSR_matrix(A);
    return 0;
}

/* BTA(CSR_matrix A, PD B) where A has no entries in any row of row_perm_B.
   Output dense block should have n0 = 0. */
const char *test_BTA_pd_csr_no_overlap(void)
{
    /* A: rows 0 and 2 have entries; rows 1 and 3 (row_perm_B) are empty. */
    CSR_matrix *A = new_CSR_matrix(4, 5, 3);
    A->p[0] = 0;
    A->p[1] = 2;
    A->p[2] = 2;
    A->p[3] = 3;
    A->p[4] = 3;
    int Ai[3] = {1, 4, 2};
    double Ax[3] = {1.0, 2.0, 3.0};
    memcpy(A->i, Ai, sizeof Ai);
    memcpy(A->x, Ax, sizeof Ax);

    int row_perm_B[2] = {1, 3}; /* rows that ARE empty in A */
    int col_perm_B[2] = {0, 2};
    double XB[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *B_m = new_permuted_dense(4, 4, 2, 2, row_perm_B, col_perm_B, XB);
    permuted_dense *B = (permuted_dense *) B_m;

    matrix *out_m = BTA_pd_csr_alloc(B, A);
    permuted_dense *out = (permuted_dense *) out_m;

    mu_assert("m0", out->m0 == 2);
    mu_assert("n0", out->n0 == 0);

    /* Fill should be a no-op (0-sized dense block). */
    BTA_pd_csr_fill_values(B, A, out);

    free_matrix(out_m);
    free_matrix(B_m);
    free_CSR_matrix(A);
    return 0;
}

/* Tests for the production CSR-pd kernel pair (B=CSR, A=PD). The BTA fill
   variant lives here in old-code because production only calls the BTDA
   path; the alloc is still in src/utils/permuted_dense.c. */

/* BTA(CSR_matrix B, PD A): basic correctness against a dense reference.
   A is (4, 5) PD with row_perm = [1, 3], col_perm = [0, 2], dense block (2, 2).
   B is (4, 4) CSR_matrix with arbitrary sparsity. */
const char *test_BTA_csr_pd_basic(void)
{
    /* PD A: m=4, n=5, row_perm = [1, 3], col_perm = [0, 2].
       X = [[1, 2], [3, 4]] (2 x 2 row-major). */
    int row_perm_A[2] = {1, 3};
    int col_perm_A[2] = {0, 2};
    double XA[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *A_m = new_permuted_dense(4, 5, 2, 2, row_perm_A, col_perm_A, XA);
    permuted_dense *A = (permuted_dense *) A_m;

    /* CSR_matrix B: m=4, n=4.
       row 0: cols {1, 3}
       row 1: cols {0, 2}
       row 2: cols {2}
       row 3: cols {0, 3} */
    CSR_matrix *B = new_CSR_matrix(4, 4, 7);
    B->p[0] = 0;
    B->p[1] = 2;
    B->p[2] = 4;
    B->p[3] = 5;
    B->p[4] = 7;
    int Bi[7] = {1, 3, 0, 2, 2, 0, 3};
    double Bx[7] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0};
    memcpy(B->i, Bi, sizeof Bi);
    memcpy(B->x, Bx, sizeof Bx);

    matrix *out_m = BTA_csr_pd_alloc(B, A);
    permuted_dense *out = (permuted_dense *) out_m;

    /* row_active = union of B's cols in rows 1 and 3
                  = {0, 2} ∪ {0, 3} = {0, 2, 3}, size 3. */
    int expected_row_perm[3] = {0, 2, 3};
    mu_assert("out m", out_m->m == 4); /* B.n */
    mu_assert("out n", out_m->n == 5); /* A.n */
    mu_assert("m0", out->m0 == 3);
    mu_assert("n0", out->n0 == 2);
    mu_assert("row_perm", cmp_int_array(out->row_perm, expected_row_perm, 3));
    mu_assert("col_perm", cmp_int_array(out->col_perm, col_perm_A, 2));

    BTA_csr_pd_fill_values(B, A, out);

    /* Reference: dense B^T A, extract block at (row_active × col_perm_A).
       Scatter inlined locally to avoid coupling to static helpers. */
    double *A_d = (double *) calloc(4 * 5, sizeof(double));
    double *B_d = (double *) calloc(4 * 4, sizeof(double));
    for (int kk = 0; kk < A->m0; kk++)
        for (int jj = 0; jj < A->n0; jj++)
            A_d[A->row_perm[kk] * 5 + A->col_perm[jj]] = A->X[kk * A->n0 + jj];
    for (int i = 0; i < B->m; i++)
        for (int e = B->p[i]; e < B->p[i + 1]; e++) B_d[i * 4 + B->i[e]] = B->x[e];

    double C_ref[4 * 5];
    memset(C_ref, 0, sizeof C_ref);
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            double s = 0.0;
            for (int k = 0; k < 4; k++)
            {
                s += B_d[k * 4 + i] * A_d[k * 5 + j];
            }
            C_ref[i * 5 + j] = s;
        }
    }
    double expected_X[6];
    for (int ii = 0; ii < 3; ii++)
    {
        for (int jj = 0; jj < 2; jj++)
        {
            expected_X[ii * 2 + jj] =
                C_ref[expected_row_perm[ii] * 5 + col_perm_A[jj]];
        }
    }
    mu_assert("values", cmp_double_array(out->X, expected_X, 6));

    free(A_d);
    free(B_d);
    free_matrix(out_m);
    free_CSR_matrix(B);
    free_matrix(A_m);
    return 0;
}

/* BTA(CSR_matrix B, PD A) where B is a leaf-variable Jacobian (identity-in-block).
   B is (4, 8): row k has a 1 at column 4+k (variable v of size 4 at var_id=4).
   Expected: row_perm_out = {4+row_perm_A[kk]} = {4+1, 4+3} = {5, 7}, X_C = X_A. */
const char *test_BTA_csr_pd_leaf_variable(void)
{
    int row_perm_A[2] = {1, 3};
    int col_perm_A[2] = {0, 2};
    double XA[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *A_m = new_permuted_dense(4, 5, 2, 2, row_perm_A, col_perm_A, XA);
    permuted_dense *A = (permuted_dense *) A_m;

    CSR_matrix *B = new_CSR_matrix(4, 8, 4);
    for (int k = 0; k < 4; k++)
    {
        B->p[k] = k;
        B->i[k] = 4 + k;
        B->x[k] = 1.0;
    }
    B->p[4] = 4;

    matrix *out_m = BTA_csr_pd_alloc(B, A);
    permuted_dense *out = (permuted_dense *) out_m;

    int expected_row_perm[2] = {5, 7};
    mu_assert("m0", out->m0 == 2);
    mu_assert("n0", out->n0 == 2);
    mu_assert("row_perm", cmp_int_array(out->row_perm, expected_row_perm, 2));
    mu_assert("col_perm", cmp_int_array(out->col_perm, col_perm_A, 2));

    BTA_csr_pd_fill_values(B, A, out);

    /* X_C should equal X_A. */
    mu_assert("values", cmp_double_array(out->X, XA, 4));

    free_matrix(out_m);
    free_CSR_matrix(B);
    free_matrix(A_m);
    return 0;
}

/* BTA(CSR_matrix B, PD A) where B has no entries in any row of row_perm_A.
   Output dense block should have m0 = 0. */
const char *test_BTA_csr_pd_no_overlap(void)
{
    int row_perm_A[2] = {1, 3};
    int col_perm_A[2] = {0, 2};
    double XA[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *A_m = new_permuted_dense(4, 5, 2, 2, row_perm_A, col_perm_A, XA);
    permuted_dense *A = (permuted_dense *) A_m;

    /* B: rows 0 and 2 have entries; rows 1 and 3 (row_perm_A) are empty. */
    CSR_matrix *B = new_CSR_matrix(4, 4, 3);
    B->p[0] = 0;
    B->p[1] = 2;
    B->p[2] = 2;
    B->p[3] = 3;
    B->p[4] = 3;
    int Bi[3] = {0, 1, 2};
    double Bx[3] = {1.0, 2.0, 3.0};
    memcpy(B->i, Bi, sizeof Bi);
    memcpy(B->x, Bx, sizeof Bx);

    matrix *out_m = BTA_csr_pd_alloc(B, A);
    permuted_dense *out = (permuted_dense *) out_m;

    mu_assert("m0", out->m0 == 0);
    mu_assert("n0", out->n0 == 2);

    /* Fill should be a no-op (0-sized dense block on the row axis). */
    BTA_csr_pd_fill_values(B, A, out);

    free_matrix(out_m);
    free_CSR_matrix(B);
    free_matrix(A_m);
    return 0;
}

/* BTA_pd_csc_alloc + BTDA_pd_csc_fill_values should match the legacy
   CSR-pd kernels in old-code on both alloc structure and BTDA values.
   Uses a d with negative + zero entries to exercise sign / drop paths. */
const char *test_BTA_pd_csc_matches_csr(void)
{
    /* Same A and B as test_BTA_pd_csr_basic. */
    CSR_matrix *A_csr = new_CSR_matrix(4, 5, 7);
    A_csr->p[0] = 0;
    A_csr->p[1] = 2;
    A_csr->p[2] = 4;
    A_csr->p[3] = 5;
    A_csr->p[4] = 7;
    int Ai[7] = {1, 4, 0, 2, 2, 1, 4};
    double Ax[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    memcpy(A_csr->i, Ai, sizeof Ai);
    memcpy(A_csr->x, Ax, sizeof Ax);

    int *iwork = (int *) malloc(MAX(A_csr->m, A_csr->n) * sizeof(int));
    CSC_matrix *A_csc = csr_to_csc_alloc(A_csr, iwork);
    csr_to_csc_fill_values(A_csr, A_csc, iwork);

    int row_perm_B[2] = {1, 3};
    int col_perm_B[2] = {0, 2};
    double XB[4] = {10.0, 20.0, 30.0, 40.0};
    matrix *B_m = new_permuted_dense(4, 4, 2, 2, row_perm_B, col_perm_B, XB);
    permuted_dense *B = (permuted_dense *) B_m;

    double d[4] = {1.5, -2.0, 0.0, 3.5};

    /* CSR variant (baseline, from old-code). */
    matrix *C_csr_m = BTA_pd_csr_alloc(B, A_csr);
    permuted_dense *C_csr = (permuted_dense *) C_csr_m;
    BTDA_pd_csr_fill_values(B, d, A_csr, C_csr);

    /* CSC variant (under test). */
    matrix *C_csc_m = BTA_pd_csc_alloc(B, A_csc);
    permuted_dense *C_csc = (permuted_dense *) C_csc_m;
    BTDA_pd_csc_fill_values(B, d, A_csc, C_csc);

    /* Structural equality. */
    mu_assert("m matches", C_csc_m->m == C_csr_m->m);
    mu_assert("n matches", C_csc_m->n == C_csr_m->n);
    mu_assert("m0 matches", C_csc->m0 == C_csr->m0);
    mu_assert("n0 matches", C_csc->n0 == C_csr->n0);
    mu_assert("row_perm matches",
              cmp_int_array(C_csc->row_perm, C_csr->row_perm, C_csr->m0));
    mu_assert("col_perm matches",
              cmp_int_array(C_csc->col_perm, C_csr->col_perm, C_csr->n0));

    /* Value equality (tolerance-based; dot ordering differs vs dgemm). */
    mu_assert("BTDA values match",
              cmp_double_array(C_csc->X, C_csr->X, C_csr->m0 * C_csr->n0));

    free_matrix(C_csr_m);
    free_matrix(C_csc_m);
    free_matrix(B_m);
    free_CSC_matrix(A_csc);
    free_CSR_matrix(A_csr);
    free(iwork);
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

#endif /* TEST_OLD_PERMUTED_DENSE_H */
