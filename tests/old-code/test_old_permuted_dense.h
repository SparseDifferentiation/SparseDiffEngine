#ifndef TEST_OLD_PERMUTED_DENSE_H
#define TEST_OLD_PERMUTED_DENSE_H

#include "minunit.h"
#include "old-code/old_permuted_dense.h"
#include "test_helpers.h"
#include "utils/CSR_matrix.h"
#include "utils/permuted_dense.h"
#include <stdlib.h>
#include <string.h>

/* Direct unit tests for the legacy CSR-pd BTA kernels in old-code. They no
   longer sit on a production path (matrix_BTA dispatcher hard-wires the
   CSC variants), but the kernels remain as reference implementations and
   as the CSR side of the cross-comparison test in test_permuted_dense.h. */

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

#endif /* TEST_OLD_PERMUTED_DENSE_H */
