#ifndef TEST_PERMUTED_DENSE_H
#define TEST_PERMUTED_DENSE_H

#include "minunit.h"
#include "old-code/old_permuted_dense.h"
#include "test_helpers.h"
#include "utils/CSC_matrix.h"
#include "utils/matrix_BTA.h"
#include "utils/permuted_dense.h"
#include "utils/sparse_matrix.h"
#include "utils/stacked_pd.h"
#include "utils/utils.h"
#include <stdlib.h>
#include <string.h>

/* 5x6 matrix with a 3x2 dense block at rows {1, 2, 4}, cols {0, 3}:

       global view:
       [0  0  0  0  0  0]
       [1  0  0  2  0  0]
       [3  0  0  4  0  0]
       [0  0  0  0  0  0]
       [5  0  0  6  0  0]                                                */
const char *test_permuted_dense_to_csr_basic(void)
{
    int row_perm[3] = {1, 2, 4};
    int col_perm[2] = {0, 3};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    matrix *M = new_permuted_dense(5, 6, 3, 2, row_perm, col_perm, X);

    CSR_matrix *C = M->to_csr(M);
    int Cp_expected[6] = {0, 0, 2, 4, 4, 6};
    int Ci_expected[6] = {0, 3, 0, 3, 0, 3};
    double Cx_expected[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    mu_assert("dim m", C->m == 5);
    mu_assert("dim n", C->n == 6);
    mu_assert("nnz", C->nnz == 6);
    mu_assert("p", cmp_int_array(C->p, Cp_expected, 6));
    mu_assert("i", cmp_int_array(C->i, Ci_expected, 6));
    mu_assert("x", cmp_double_array(C->x, Cx_expected, 6));

    free_matrix(M);
    return 0;
}

/* Empty dense block (m0 = n0 = 0): result is an m x n CSR_matrix with
   no nonzeros. */
const char *test_permuted_dense_to_csr_empty(void)
{
    matrix *M = new_permuted_dense(4, 5, 0, 0, NULL, NULL, NULL);

    CSR_matrix *C = M->to_csr(M);
    int Cp_expected[5] = {0, 0, 0, 0, 0};
    mu_assert("nnz", C->nnz == 0);
    mu_assert("p", cmp_int_array(C->p, Cp_expected, 5));

    free_matrix(M);
    return 0;
}

/* Full dense (row_perm = [0..m), col_perm = [0..n)): result is the dense
   matrix in CSR_matrix. */
const char *test_permuted_dense_to_csr_full(void)
{
    int row_perm[2] = {0, 1};
    int col_perm[3] = {0, 1, 2};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    matrix *M = new_permuted_dense(2, 3, 2, 3, row_perm, col_perm, X);

    CSR_matrix *C = M->to_csr(M);
    int Cp_expected[3] = {0, 3, 6};
    int Ci_expected[6] = {0, 1, 2, 0, 1, 2};
    double Cx_expected[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    mu_assert("p", cmp_int_array(C->p, Cp_expected, 3));
    mu_assert("i", cmp_int_array(C->i, Ci_expected, 6));
    mu_assert("x", cmp_double_array(C->x, Cx_expected, 6));

    free_matrix(M);
    return 0;
}

/* Single dense row, two dense cols. Tests rows with no entries before
   and after the active row. */
const char *test_permuted_dense_to_csr_single_row(void)
{
    int row_perm[1] = {2};
    int col_perm[2] = {1, 4};
    double X[2] = {7.0, 9.0};

    matrix *M = new_permuted_dense(4, 5, 1, 2, row_perm, col_perm, X);

    CSR_matrix *C = M->to_csr(M);
    int Cp_expected[5] = {0, 0, 0, 2, 2};
    int Ci_expected[2] = {1, 4};
    double Cx_expected[2] = {7.0, 9.0};

    mu_assert("p", cmp_int_array(C->p, Cp_expected, 5));
    mu_assert("i", cmp_int_array(C->i, Ci_expected, 2));
    mu_assert("x", cmp_double_array(C->x, Cx_expected, 2));

    free_matrix(M);
    return 0;
}

/* Single dense col across multiple rows. */
const char *test_permuted_dense_to_csr_single_col(void)
{
    int row_perm[3] = {0, 2, 3};
    int col_perm[1] = {2};
    double X[3] = {1.0, 2.0, 3.0};

    matrix *M = new_permuted_dense(4, 4, 3, 1, row_perm, col_perm, X);

    CSR_matrix *C = M->to_csr(M);
    int Cp_expected[5] = {0, 1, 1, 2, 3};
    int Ci_expected[3] = {2, 2, 2};
    double Cx_expected[3] = {1.0, 2.0, 3.0};

    mu_assert("p", cmp_int_array(C->p, Cp_expected, 5));
    mu_assert("i", cmp_int_array(C->i, Ci_expected, 3));
    mu_assert("x", cmp_double_array(C->x, Cx_expected, 3));

    free_matrix(M);
    return 0;
}

/* DA_fill_values: compare against CSR_matrix DA_fill_values on the equivalent
   CSR_matrix.

   PD is the 5x6 matrix from the basic to_csr test, with d a length-5
   global-row diagonal including a negative and zero entry. */
const char *test_DA_pd_fill_values(void)
{
    int row_perm[3] = {1, 2, 4};
    int col_perm[2] = {0, 3};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double d[5] = {7.0, -1.5, 0.0, 9.0, 2.5};

    matrix *M = new_permuted_dense(5, 6, 3, 2, row_perm, col_perm, X);
    matrix *M_out = new_permuted_dense(5, 6, 3, 2, row_perm, col_perm, NULL);
    permuted_dense *pd = (permuted_dense *) M;
    permuted_dense *pd_out = (permuted_dense *) M_out;

    DA_pd_fill_values(d, pd, pd_out);

    /* Ground truth: build CSR_matrix of self, run DA_fill_values, compare. */
    CSR_matrix *csr = M->to_csr(M);
    CSR_matrix *csr_expected = new_csr_copy_sparsity(csr);
    DA_fill_values(d, csr, csr_expected);

    CSR_matrix *csr_out = M_out->to_csr(M_out);
    mu_assert("x", cmp_double_array(csr_out->x, csr_expected->x, csr->nnz));

    free_CSR_matrix(csr_expected);
    free_matrix(M);
    free_matrix(M_out);
    return 0;
}

/* ATA_alloc: structure-only check. Output is 6x6 with a 2x2 dense block at
   perms {0, 3} (= self.col_perm on both sides). Values are uninitialized
   here; ATDA_fill_values is the value-producing op. */
const char *test_ATA_pd_alloc(void)
{
    int row_perm[3] = {1, 2, 4};
    int col_perm[2] = {0, 3};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    matrix *M = new_permuted_dense(5, 6, 3, 2, row_perm, col_perm, X);
    permuted_dense *pd = (permuted_dense *) M;

    matrix *M_ata = ATA_pd_alloc(pd);
    permuted_dense *pd_ata = (permuted_dense *) M_ata;

    int perm_expected[2] = {0, 3};
    mu_assert("m", M_ata->m == 6);
    mu_assert("n", M_ata->n == 6);
    mu_assert("m0", pd_ata->m0 == 2);
    mu_assert("n0", pd_ata->n0 == 2);
    mu_assert("row_perm", cmp_int_array(pd_ata->row_perm, perm_expected, 2));
    mu_assert("col_perm", cmp_int_array(pd_ata->col_perm, perm_expected, 2));

    free_matrix(M);
    free_matrix(M_ata);
    return 0;
}

/* ATDA: same 5x6 PD, d with negative + zero entries to catch sign bugs.
   Hand-computed: d_perm = [-1.5, 0, 2.5], Y = diag(d_perm) X gives
   [[-1.5,-3],[0,0],[12.5,15]], and X^T Y = [[61,72],[72,84]]. */
const char *test_ATDA_pd_fill_values(void)
{
    int row_perm[3] = {1, 2, 4};
    int col_perm[2] = {0, 3};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double d[5] = {7.0, -1.5, 0.0, 9.0, 2.5};

    matrix *M = new_permuted_dense(5, 6, 3, 2, row_perm, col_perm, X);
    permuted_dense *pd = (permuted_dense *) M;

    matrix *M_out = ATA_pd_alloc(pd);
    permuted_dense *pd_out = (permuted_dense *) M_out;
    ATDA_pd_fill_values(pd, d, pd_out);

    double X_expected[4] = {61.0, 72.0, 72.0, 84.0};
    mu_assert("X", cmp_double_array(pd_out->X, X_expected, 4));

    free_matrix(M);
    free_matrix(M_out);
    return 0;
}

/* PD x CSC_matrix: J is 6x4. col 0 empty; col 1 has rows {0,3} (vals 10, 20);
   col 2 has row {2} (val 30, but row 2 not in col_perm_self = {0,3} so col 2
   is INACTIVE); col 3 has row {3} (val 40). Active cols: {1, 3}.

   Expected: m0=3, n0=2, row_perm={1,2,4}, col_perm={1,3}.
   Values: out.X[:,0] = 10*[1,3,5] + 20*[2,4,6] = [50,110,170],
           out.X[:,1] = 40*[2,4,6] = [80,160,240]. */
const char *test_permuted_dense_times_csc(void)
{
    int row_perm[3] = {1, 2, 4};
    int col_perm[2] = {0, 3};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    matrix *M = new_permuted_dense(5, 6, 3, 2, row_perm, col_perm, X);
    permuted_dense *pd = (permuted_dense *) M;

    CSC_matrix *J = new_CSC_matrix(6, 4, 4);
    int Jp[5] = {0, 0, 2, 3, 4};
    int Ji[4] = {0, 3, 2, 3};
    double Jx[4] = {10.0, 20.0, 30.0, 40.0};
    memcpy(J->p, Jp, 5 * sizeof(int));
    memcpy(J->i, Ji, 4 * sizeof(int));
    memcpy(J->x, Jx, 4 * sizeof(double));

    matrix *M_out = BA_pd_csc_alloc(pd, J);
    permuted_dense *pd_out = (permuted_dense *) M_out;
    BA_pd_csc_fill_values(pd->X, pd->n0, pd->col_inv, J, pd_out);

    int row_perm_expected[3] = {1, 2, 4};
    int col_perm_expected[2] = {1, 3};
    double X_expected[6] = {50.0, 80.0, 110.0, 160.0, 170.0, 240.0};

    mu_assert("m", M_out->m == 5);
    mu_assert("n", M_out->n == 4);
    mu_assert("m0", pd_out->m0 == 3);
    mu_assert("n0", pd_out->n0 == 2);
    mu_assert("row_perm", cmp_int_array(pd_out->row_perm, row_perm_expected, 3));
    mu_assert("col_perm", cmp_int_array(pd_out->col_perm, col_perm_expected, 2));
    mu_assert("X", cmp_double_array(pd_out->X, X_expected, 6));

    free_matrix(M);
    free_matrix(M_out);
    free_CSC_matrix(J);
    return 0;
}

/* PD x CSC_matrix edge case: every column of J has its only nonzero outside
   col_perm_self, so col_perm_out is empty (n0 = 0). */
const char *test_permuted_dense_times_csc_no_active(void)
{
    int row_perm[3] = {1, 2, 4};
    int col_perm[2] = {0, 3};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    matrix *M = new_permuted_dense(5, 6, 3, 2, row_perm, col_perm, X);
    permuted_dense *pd = (permuted_dense *) M;

    /* J: col 0 has row {1}, col 1 has row {5}. Neither in col_perm_self. */
    CSC_matrix *J = new_CSC_matrix(6, 2, 2);
    int Jp[3] = {0, 1, 2};
    int Ji[2] = {1, 5};
    double Jx[2] = {100.0, 200.0};
    memcpy(J->p, Jp, 3 * sizeof(int));
    memcpy(J->i, Ji, 2 * sizeof(int));
    memcpy(J->x, Jx, 2 * sizeof(double));

    matrix *M_out = BA_pd_csc_alloc(pd, J);
    permuted_dense *pd_out = (permuted_dense *) M_out;
    BA_pd_csc_fill_values(pd->X, pd->n0, pd->col_inv, J, pd_out);

    mu_assert("m", M_out->m == 5);
    mu_assert("n", M_out->n == 2);
    mu_assert("m0", pd_out->m0 == 3);
    mu_assert("n0", pd_out->n0 == 0);

    free_matrix(M);
    free_matrix(M_out);
    free_CSC_matrix(J);
    return 0;
}

/* to_csr vtable method: lazy CSR_matrix view. First call allocates pd->csr_cache;
   subsequent calls refresh values to reflect the current pd->X. */
const char *test_permuted_dense_to_csr_lazy(void)
{
    int row_perm[3] = {1, 2, 4};
    int col_perm[2] = {0, 3};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    matrix *M = new_permuted_dense(5, 6, 3, 2, row_perm, col_perm, X);
    permuted_dense *pd = (permuted_dense *) M;

    mu_assert("csr_cache initially NULL", pd->csr_cache == NULL);

    CSR_matrix *csr = M->to_csr(M);
    mu_assert("csr_cache populated", pd->csr_cache != NULL);
    mu_assert("returns the cache", csr == pd->csr_cache);

    double expected[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    mu_assert("values match X", cmp_double_array(csr->x, expected, 6));

    /* Mutate X and re-call to_csr: values must reflect the change. */
    pd->X[0] = 99.0;
    csr = M->to_csr(M);
    mu_assert("refresh picks up new value", csr->x[0] == 99.0);

    free_matrix(M);
    return 0;
}

/* Sanity check: col_inv is built correctly. col_perm = {0, 3} on n = 6
   should give col_inv = {0, -1, -1, 1, -1, -1}. */
const char *test_permuted_dense_col_inv(void)
{
    int row_perm[1] = {0};
    int col_perm[2] = {0, 3};
    double X[2] = {0.0, 0.0};

    matrix *M = new_permuted_dense(1, 6, 1, 2, row_perm, col_perm, X);
    permuted_dense *pd = (permuted_dense *) M;

    int expected[6] = {0, -1, -1, 1, -1, -1};
    mu_assert("col_inv", cmp_int_array(pd->col_inv, expected, 6));

    free_matrix(M);
    return 0;
}

/* PD index_alloc / index_fill_values: select rows from a PD; output must be
   another PD with row_perm equal to the output positions where indices[i]
   hit the source row_perm. */
const char *test_permuted_dense_index(void)
{
    /* Source PD, shape (6, 4), dense block at rows {1, 3, 4} x cols {0, 2}. */
    int row_perm[3] = {1, 3, 4};
    int col_perm[2] = {0, 2};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    matrix *M = new_permuted_dense(6, 4, 3, 2, row_perm, col_perm, X);

    /* Index by [0, 3, 1, 5, 4]:
       - position 0 -> source row 0 (not in row_perm, zero)
       - position 1 -> source row 3 (in row_perm at ii=1, dense)
       - position 2 -> source row 1 (in row_perm at ii=0, dense)
       - position 3 -> source row 5 (not in row_perm, zero)
       - position 4 -> source row 4 (in row_perm at ii=2, dense) */
    int indices[5] = {0, 3, 1, 5, 4};
    matrix *out = M->index_alloc(M, indices, 5);
    permuted_dense *out_pd = (permuted_dense *) out;

    mu_assert("out m", out->m == 5);
    mu_assert("out n", out->n == 4);
    mu_assert("out nnz", out->nnz == 6); /* m0=3 * n0=2 */
    mu_assert("m0", out_pd->m0 == 3);
    mu_assert("n0", out_pd->n0 == 2);

    int expected_row_perm[3] = {1, 2, 4};
    mu_assert("row_perm", cmp_int_array(out_pd->row_perm, expected_row_perm, 3));
    int expected_col_perm[2] = {0, 2};
    mu_assert("col_perm", cmp_int_array(out_pd->col_perm, expected_col_perm, 2));

    M->index_fill_values(M, indices, 5, out);

    /* Row 0 of out (i=1) = source row 3 = X[1, :] = {3, 4}.
       Row 1 of out (i=2) = source row 1 = X[0, :] = {1, 2}.
       Row 2 of out (i=4) = source row 4 = X[2, :] = {5, 6}. */
    double expected_X[6] = {3.0, 4.0, 1.0, 2.0, 5.0, 6.0};
    mu_assert("values", cmp_double_array(out_pd->X, expected_X, 6));

    free_matrix(out);
    free_matrix(M);
    return 0;
}

/* PD promote_alloc / promote_fill_values: tile a 1-row PD into a
   `size`-row PD where every row is a copy of the source row. */
const char *test_permuted_dense_promote(void)
{
    /* Source PD, shape (1, 5), single dense row at row 0, cols {1, 3}. */
    int row_perm[1] = {0};
    int col_perm[2] = {1, 3};
    double X[2] = {7.0, 9.0};
    matrix *M = new_permuted_dense(1, 5, 1, 2, row_perm, col_perm, X);

    matrix *out = M->promote_alloc(M, 4);
    permuted_dense *out_pd = (permuted_dense *) out;

    mu_assert("out m", out->m == 4);
    mu_assert("out n", out->n == 5);
    mu_assert("out nnz", out->nnz == 8); /* m0=4 * n0=2 */
    mu_assert("m0", out_pd->m0 == 4);
    mu_assert("n0", out_pd->n0 == 2);

    int expected_row_perm[4] = {0, 1, 2, 3};
    mu_assert("row_perm", cmp_int_array(out_pd->row_perm, expected_row_perm, 4));
    int expected_col_perm[2] = {1, 3};
    mu_assert("col_perm", cmp_int_array(out_pd->col_perm, expected_col_perm, 2));

    M->promote_fill_values(M, out);

    double expected_X[8] = {7.0, 9.0, 7.0, 9.0, 7.0, 9.0, 7.0, 9.0};
    mu_assert("values", cmp_double_array(out_pd->X, expected_X, 8));

    free_matrix(out);
    free_matrix(M);
    return 0;
}

/* PD broadcast_alloc / broadcast_fill_values, SCALAR variant.
   (1, 5) PD with single dense row -> (d1*d2, 5) PD with that row tiled. */
const char *test_permuted_dense_broadcast_scalar(void)
{
    int row_perm[1] = {0};
    int col_perm[2] = {1, 3};
    double X[2] = {7.0, 9.0};
    matrix *M = new_permuted_dense(1, 5, 1, 2, row_perm, col_perm, X);

    int d1 = 2, d2 = 3; /* out shape (2, 3), m = 6 */
    matrix *out = M->broadcast_alloc(M, BROADCAST_SCALAR, d1, d2);
    permuted_dense *out_pd = (permuted_dense *) out;

    mu_assert("out m", out->m == 6);
    mu_assert("out n", out->n == 5);
    mu_assert("m0", out_pd->m0 == 6);
    mu_assert("n0", out_pd->n0 == 2);
    int expected_rp[6] = {0, 1, 2, 3, 4, 5};
    mu_assert("row_perm", cmp_int_array(out_pd->row_perm, expected_rp, 6));

    M->broadcast_fill_values(M, BROADCAST_SCALAR, d1, d2, out);
    double expected_X[12] = {7, 9, 7, 9, 7, 9, 7, 9, 7, 9, 7, 9};
    mu_assert("values", cmp_double_array(out_pd->X, expected_X, 12));

    free_matrix(out);
    free_matrix(M);
    return 0;
}

/* PD broadcast_alloc / broadcast_fill_values, ROW variant.
   (1, d2) input has Jacobian of shape (d2, n_vars). Source PD: m=d2=3,
   row_perm={0, 2} (rows 0 and 2 dense), col_perm={1, 4}, single dense row
   per m0. Output (d1, d2) = (2, 3): each child row replicated d1=2
   times. */
const char *test_permuted_dense_broadcast_row(void)
{
    int row_perm[2] = {0, 2};
    int col_perm[2] = {1, 4};
    double X[4] = {1.0, 2.0,  /* row corresponding to child row 0 */
                   3.0, 4.0}; /* row corresponding to child row 2 */
    matrix *M = new_permuted_dense(3, 6, 2, 2, row_perm, col_perm, X);

    int d1 = 2, d2 = 3; /* output (2, 3), out m = 6 */
    matrix *out = M->broadcast_alloc(M, BROADCAST_ROW, d1, d2);
    permuted_dense *out_pd = (permuted_dense *) out;

    mu_assert("out m", out->m == 6);
    mu_assert("m0", out_pd->m0 == 4); /* d1 * 2 */
    mu_assert("n0", out_pd->n0 == 2);
    /* row_perm = {child_row_perm[0]*d1, +1, child_row_perm[1]*d1, +1}
                = {0, 1, 4, 5} */
    int expected_rp[4] = {0, 1, 4, 5};
    mu_assert("row_perm", cmp_int_array(out_pd->row_perm, expected_rp, 4));

    M->broadcast_fill_values(M, BROADCAST_ROW, d1, d2, out);
    /* each child row replicated d1 times */
    double expected_X[8] = {1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0};
    mu_assert("values", cmp_double_array(out_pd->X, expected_X, 8));

    free_matrix(out);
    free_matrix(M);
    return 0;
}

/* PD broadcast_alloc / broadcast_fill_values, COL variant.
   (d1, 1) input has Jacobian of shape (d1, n_vars). Source PD: m=d1=3,
   row_perm={0, 2}, col_perm={1, 4}, two dense rows. Output (d1, d2) = (3, 2),
   out m = 6: each child row appears d2 times, shifted by j*d1. */
const char *test_permuted_dense_broadcast_col(void)
{
    int row_perm[2] = {0, 2};
    int col_perm[2] = {1, 4};
    double X[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *M = new_permuted_dense(3, 6, 2, 2, row_perm, col_perm, X);

    int d1 = 3, d2 = 2;
    matrix *out = M->broadcast_alloc(M, BROADCAST_COL, d1, d2);
    permuted_dense *out_pd = (permuted_dense *) out;

    mu_assert("out m", out->m == 6);
    mu_assert("m0", out_pd->m0 == 4); /* d2 * 2 */
    mu_assert("n0", out_pd->n0 == 2);
    /* row_perm = {0+0, 0+2, 3+0, 3+2} = {0, 2, 3, 5} */
    int expected_rp[4] = {0, 2, 3, 5};
    mu_assert("row_perm", cmp_int_array(out_pd->row_perm, expected_rp, 4));

    M->broadcast_fill_values(M, BROADCAST_COL, d1, d2, out);
    /* X = d2 copies of full source X block */
    double expected_X[8] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    mu_assert("values", cmp_double_array(out_pd->X, expected_X, 8));

    free_matrix(out);
    free_matrix(M);
    return 0;
}

/* PD diag_vec_alloc / diag_vec_fill_values.
   Source PD shape (3, 6) with m0=2 (rows 0 and 2) -> output PD shape
   (9, 6) with the same 2 dense rows mapped to positions {0, 8} = {0*4, 2*4}. */
const char *test_permuted_dense_diag_vec(void)
{
    int row_perm[2] = {0, 2};
    int col_perm[2] = {1, 4};
    double X[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *M = new_permuted_dense(3, 6, 2, 2, row_perm, col_perm, X);

    matrix *out = M->diag_vec_alloc(M);
    permuted_dense *out_pd = (permuted_dense *) out;

    mu_assert("out m", out->m == 9);
    mu_assert("out n", out->n == 6);
    mu_assert("m0", out_pd->m0 == 2);
    mu_assert("n0", out_pd->n0 == 2);
    /* row_perm = {0*(n+1), 2*(n+1)} = {0, 8} */
    int expected_rp[2] = {0, 8};
    mu_assert("row_perm", cmp_int_array(out_pd->row_perm, expected_rp, 2));
    int expected_cp[2] = {1, 4};
    mu_assert("col_perm", cmp_int_array(out_pd->col_perm, expected_cp, 2));

    M->diag_vec_fill_values(M, out);
    /* X is identical to the source X */
    double expected_X[4] = {1.0, 2.0, 3.0, 4.0};
    mu_assert("values", cmp_double_array(out_pd->X, expected_X, 4));

    free_matrix(out);
    free_matrix(M);
    return 0;
}

/* ---- Helpers for BTA / BTDA tests ---- */

/* Scatter a PD into a dense m x n_global buffer (row-major), zero-filled.
   Buffer is allocated by the caller. */
static void scatter_pd_to_dense(const permuted_dense *pd, int n_global,
                                double *dense)
{
    int m = pd->base.m;
    memset(dense, 0, (size_t) m * (size_t) n_global * sizeof(double));
    for (int ii = 0; ii < pd->m0; ii++)
    {
        int row = pd->row_perm[ii];
        for (int jj = 0; jj < pd->n0; jj++)
        {
            int col = pd->col_perm[jj];
            dense[row * n_global + col] = pd->X[ii * pd->n0 + jj];
        }
    }
}

/* BTA: A and B share row_perm = [1, 3]; both have m=4, distinct col_perms.
   C = B^T A is computed via the primitive and compared against a hand
   reference X_B^T X_A. */
const char *test_permuted_dense_BTA_matching_row_perm(void)
{
    int row_perm[2] = {1, 3};
    int col_perm_A[2] = {0, 2};
    int col_perm_B[2] = {1, 3};
    /* X_A is (2, 2), X_B is (2, 2), both row-major. */
    double XA[4] = {1.0, 2.0, 3.0, 4.0}; /* rows: [1,2], [3,4] */
    double XB[4] = {5.0, 6.0, 7.0, 8.0}; /* rows: [5,6], [7,8] */
    matrix *A_m = new_permuted_dense(4, 4, 2, 2, row_perm, col_perm_A, XA);
    matrix *B_m = new_permuted_dense(4, 4, 2, 2, row_perm, col_perm_B, XB);
    permuted_dense *A = (permuted_dense *) A_m;
    permuted_dense *B = (permuted_dense *) B_m;

    matrix *C_m = BTA_pd_pd_alloc(B, A);
    permuted_dense *C = (permuted_dense *) C_m;

    mu_assert("out m", C_m->m == 4); /* B.n */
    mu_assert("out n", C_m->n == 4); /* A.n */
    mu_assert("m0", C->m0 == 2);
    mu_assert("n0", C->n0 == 2);
    mu_assert("row_perm", cmp_int_array(C->row_perm, col_perm_B, 2));
    mu_assert("col_perm", cmp_int_array(C->col_perm, col_perm_A, 2));

    BTA_pd_pd_fill_values(B, A, C);

    /* Reference: X_B^T X_A. With X_B = [[5,6],[7,8]], X_A = [[1,2],[3,4]]:
       X_B^T = [[5,7],[6,8]]. X_B^T X_A = [[5*1+7*3, 5*2+7*4], [6*1+8*3, 6*2+8*4]]
                                        = [[26, 38], [30, 44]]. */
    double expected[4] = {26.0, 38.0, 30.0, 44.0};
    mu_assert("values", cmp_double_array(C->X, expected, 4));

    free_matrix(C_m);
    free_matrix(B_m);
    free_matrix(A_m);
    return 0;
}

/* BTA with empty row intersection: row_perm_A = [0, 2], row_perm_B = [1, 3].
   BTA_pd_pd_alloc should return an empty C (nnz = 0); the fill
   kernels should short-circuit without crashing. */
const char *test_permuted_dense_BTA_empty_overlap(void)
{
    int row_perm_A[2] = {0, 2};
    int row_perm_B[2] = {1, 3};
    int col_perm_A[2] = {0, 2};
    int col_perm_B[2] = {1, 3};
    double XA[4] = {1.0, 2.0, 3.0, 4.0};
    double XB[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *A_m = new_permuted_dense(4, 4, 2, 2, row_perm_A, col_perm_A, XA);
    matrix *B_m = new_permuted_dense(4, 4, 2, 2, row_perm_B, col_perm_B, XB);
    permuted_dense *A = (permuted_dense *) A_m;
    permuted_dense *B = (permuted_dense *) B_m;

    matrix *C_m = BTA_pd_pd_alloc(B, A);
    permuted_dense *C = (permuted_dense *) C_m;

    mu_assert("out m", C_m->m == 4); /* B.n */
    mu_assert("out n", C_m->n == 4); /* A.n */
    mu_assert("m0", C->m0 == 0);
    mu_assert("n0", C->n0 == 0);
    mu_assert("nnz", C_m->nnz == 0);

    /* fill kernels should be safe no-ops on empty C. */
    BTA_pd_pd_fill_values(B, A, C);
    double d[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_pd_pd_fill_values(B, d, A, C);

    free_matrix(C_m);
    free_matrix(B_m);
    free_matrix(A_m);
    return 0;
}

/* BTA with partial overlap: row_perm_A = [1, 3, 5], row_perm_B = [3, 5, 7].
   Intersection = {3, 5}. */
const char *test_permuted_dense_BTA_partial_overlap(void)
{
    int row_perm_A[3] = {1, 3, 5};
    int row_perm_B[3] = {3, 5, 7};
    int col_perm_A[2] = {0, 2};
    int col_perm_B[2] = {1, 3};
    /* X_A rows correspond to A row_perm order: row 0 -> source row 1, row 1 -> 3,
     * row 2 -> 5. */
    double XA[6] = {1.0, 2.0,  /* row 1 (NOT in B) */
                    3.0, 4.0,  /* row 3 (in B at pos 0) */
                    5.0, 6.0}; /* row 5 (in B at pos 1) */
    /* X_B rows: row 0 -> source row 3, row 1 -> 5, row 2 -> 7. */
    double XB[6] = {10.0, 20.0,  /* row 3 (in A at pos 1) */
                    30.0, 40.0,  /* row 5 (in A at pos 2) */
                    50.0, 60.0}; /* row 7 (NOT in A) */
    matrix *A_m = new_permuted_dense(8, 4, 3, 2, row_perm_A, col_perm_A, XA);
    matrix *B_m = new_permuted_dense(8, 4, 3, 2, row_perm_B, col_perm_B, XB);
    permuted_dense *A = (permuted_dense *) A_m;
    permuted_dense *B = (permuted_dense *) B_m;

    matrix *C_m = BTA_pd_pd_alloc(B, A);
    permuted_dense *C = (permuted_dense *) C_m;
    BTA_pd_pd_fill_values(B, A, C);

    /* Reference: scatter A, B to dense 8x4, compute B^T A, compare block at
       (col_perm_B, col_perm_A). */
    double *A_d = (double *) calloc((size_t) 8 * 4, sizeof(double));
    double *B_d = (double *) calloc((size_t) 8 * 4, sizeof(double));
    scatter_pd_to_dense(A, 4, A_d);
    scatter_pd_to_dense(B, 4, B_d);

    /* Reference C_ref is 4x4 = B_d^T (4x8) * A_d (8x4). */
    double C_ref[16];
    memset(C_ref, 0, sizeof C_ref);
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            double s = 0.0;
            for (int k = 0; k < 8; k++)
            {
                s += B_d[k * 4 + i] * A_d[k * 4 + j];
            }
            C_ref[i * 4 + j] = s;
        }
    }

    /* Extract reference block at (col_perm_B, col_perm_A) and compare to C->X. */
    double expected[4];
    for (int ii = 0; ii < 2; ii++)
    {
        for (int jj = 0; jj < 2; jj++)
        {
            expected[ii * 2 + jj] = C_ref[col_perm_B[ii] * 4 + col_perm_A[jj]];
        }
    }
    mu_assert("values", cmp_double_array(C->X, expected, 4));

    free(A_d);
    free(B_d);
    free_matrix(C_m);
    free_matrix(B_m);
    free_matrix(A_m);
    return 0;
}

/* Full BTDA decomposition: tmp = diag(w) A; C = B^T tmp. Compare against a
   dense triple product B_d^T diag(w) A_d. */
const char *test_permuted_dense_BTDA_decomposition(void)
{
    int row_perm[3] = {0, 1, 2};
    int col_perm_A[2] = {0, 2};
    int col_perm_B[2] = {1, 3};
    double XA[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double XB[6] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    double w[3] = {2.0, -1.0, 3.0};

    matrix *A_m = new_permuted_dense(3, 4, 3, 2, row_perm, col_perm_A, XA);
    matrix *B_m = new_permuted_dense(3, 4, 3, 2, row_perm, col_perm_B, XB);
    permuted_dense *A = (permuted_dense *) A_m;
    permuted_dense *B = (permuted_dense *) B_m;

    /* tmp has the same sparsity as A. */
    matrix *tmp_m = A_m->copy_sparsity(A_m);
    permuted_dense *tmp = (permuted_dense *) tmp_m;
    DA_pd_fill_values(w, A, tmp);

    matrix *C_m = BTA_pd_pd_alloc(B, tmp);
    permuted_dense *C = (permuted_dense *) C_m;
    BTA_pd_pd_fill_values(B, tmp, C);

    /* Reference: dense B_d^T diag(w) A_d, extract (col_perm_B, col_perm_A) block. */
    double *A_d = (double *) calloc((size_t) 3 * 4, sizeof(double));
    double *B_d = (double *) calloc((size_t) 3 * 4, sizeof(double));
    scatter_pd_to_dense(A, 4, A_d);
    scatter_pd_to_dense(B, 4, B_d);

    double C_ref[16];
    memset(C_ref, 0, sizeof C_ref);
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            double s = 0.0;
            for (int k = 0; k < 3; k++)
            {
                s += B_d[k * 4 + i] * w[k] * A_d[k * 4 + j];
            }
            C_ref[i * 4 + j] = s;
        }
    }
    double expected[4];
    for (int ii = 0; ii < 2; ii++)
    {
        for (int jj = 0; jj < 2; jj++)
        {
            expected[ii * 2 + jj] = C_ref[col_perm_B[ii] * 4 + col_perm_A[jj]];
        }
    }
    mu_assert("values", cmp_double_array(C->X, expected, 4));

    free(A_d);
    free(B_d);
    free_matrix(C_m);
    free_matrix(tmp_m);
    free_matrix(B_m);
    free_matrix(A_m);
    return 0;
}

/* BTA(CSR_matrix A, PD B): basic correctness against a dense reference.
   A is (4, 5) CSR_matrix with mixed sparsity; B is (4, 4) PD with row_perm = [1, 3],
   col_perm = [0, 2], dense block (2, 2). */
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

/* BA_pd_matrices: C = B @ A where B is full-block PD (the production
   shape gated by left_matmul.c) and A is PD with non-trivial perms.
   B (2x3) row_perm=[0,1], col_perm=[0,1,2], X_B=[[1,2,3],[4,5,6]].
   A (3x5) row_perm=[0,2], col_perm=[1,4], X_A=[[7,8],[9,10]].
   Hand-computed C (2x5) nonzero at cols {1,4}: X_C=[[34,38],[82,92]]. */
const char *test_BA_pd_matrices_pd_pd_full_block_B(void)
{
    int row_perm_B[2] = {0, 1};
    int col_perm_B[3] = {0, 1, 2};
    double XB[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    matrix *B_m = new_permuted_dense(2, 3, 2, 3, row_perm_B, col_perm_B, XB);

    int row_perm_A[2] = {0, 2};
    int col_perm_A[2] = {1, 4};
    double XA[4] = {7.0, 8.0, 9.0, 10.0};
    matrix *A_m = new_permuted_dense(3, 5, 2, 2, row_perm_A, col_perm_A, XA);

    matrix *C_m = BA_pd_matrices_alloc((permuted_dense *) B_m, A_m);
    BA_pd_matrices_fill_values((permuted_dense *) B_m, A_m, (permuted_dense *) C_m);

    permuted_dense *C = (permuted_dense *) C_m;
    mu_assert("dim m", C_m->m == 2);
    mu_assert("dim n", C_m->n == 5);
    mu_assert("m0", C->m0 == 2);
    mu_assert("n0", C->n0 == 2);
    int expected_row_perm[2] = {0, 1};
    int expected_col_perm[2] = {1, 4};
    mu_assert("row_perm", cmp_int_array(C->row_perm, expected_row_perm, 2));
    mu_assert("col_perm", cmp_int_array(C->col_perm, expected_col_perm, 2));
    double expected_X[4] = {34.0, 38.0, 82.0, 92.0};
    mu_assert("X", cmp_double_array(C->X, expected_X, 4));

    free_matrix(C_m);
    free_matrix(A_m);
    free_matrix(B_m);
    return 0;
}

/* BA_pd_matrices with general (non-full-block) B. B->col_perm and
   A->row_perm only partially overlap, exercising the
   sorted_intersect_indices gather path.
   B (2x5) row_perm=[0,1], col_perm=[1,3], X_B=[[1,2],[3,4]].
   A (5x4) row_perm=[1,2], col_perm=[0,3], X_A=[[5,6],[7,8]].
   Intersection K = {1,3} ∩ {1,2} = {1}, s=1.
   Hand-computed C (2x4) nonzero at cols {0,3}: X_C=[[5,6],[15,18]]. */
const char *test_BA_pd_matrices_pd_pd_general_B(void)
{
    int row_perm_B[2] = {0, 1};
    int col_perm_B[2] = {1, 3};
    double XB[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *B_m = new_permuted_dense(2, 5, 2, 2, row_perm_B, col_perm_B, XB);

    int row_perm_A[2] = {1, 2};
    int col_perm_A[2] = {0, 3};
    double XA[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *A_m = new_permuted_dense(5, 4, 2, 2, row_perm_A, col_perm_A, XA);

    matrix *C_m = BA_pd_matrices_alloc((permuted_dense *) B_m, A_m);
    BA_pd_matrices_fill_values((permuted_dense *) B_m, A_m, (permuted_dense *) C_m);

    permuted_dense *C = (permuted_dense *) C_m;
    mu_assert("dim m", C_m->m == 2);
    mu_assert("dim n", C_m->n == 4);
    mu_assert("m0", C->m0 == 2);
    mu_assert("n0", C->n0 == 2);
    int expected_row_perm[2] = {0, 1};
    int expected_col_perm[2] = {0, 3};
    mu_assert("row_perm", cmp_int_array(C->row_perm, expected_row_perm, 2));
    mu_assert("col_perm", cmp_int_array(C->col_perm, expected_col_perm, 2));
    double expected_X[4] = {5.0, 6.0, 15.0, 18.0};
    mu_assert("X", cmp_double_array(C->X, expected_X, 4));

    free_matrix(C_m);
    free_matrix(A_m);
    free_matrix(B_m);
    return 0;
}

/* BA_pd_matrices with sparse A. Same B and same global A content as the
   pd_pd_general_B test — the dispatcher routes through BA_pd_csc_*
   and should yield byte-identical output. */
const char *test_BA_pd_matrices_pd_csc(void)
{
    int row_perm_B[2] = {0, 1};
    int col_perm_B[2] = {1, 3};
    double XB[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *B_m = new_permuted_dense(2, 5, 2, 2, row_perm_B, col_perm_B, XB);

    /* A as 5x4 sparse_matrix, same nonzero values as the PD case:
       (1,0)=5, (1,3)=6, (2,0)=7, (2,3)=8. */
    CSR_matrix *csr = new_CSR_matrix(5, 4, 4);
    int Ap[6] = {0, 0, 2, 4, 4, 4};
    int Ai[4] = {0, 3, 0, 3};
    double Ax[4] = {5.0, 6.0, 7.0, 8.0};
    memcpy(csr->p, Ap, 6 * sizeof(int));
    memcpy(csr->i, Ai, 4 * sizeof(int));
    memcpy(csr->x, Ax, 4 * sizeof(double));
    matrix *A_m = new_sparse_matrix(csr);

    matrix *C_m = BA_pd_matrices_alloc((permuted_dense *) B_m, A_m);
    A_m->refresh_csc_values(A_m); /* values must be fresh before fill */
    BA_pd_matrices_fill_values((permuted_dense *) B_m, A_m, (permuted_dense *) C_m);

    permuted_dense *C = (permuted_dense *) C_m;
    mu_assert("dim m", C_m->m == 2);
    mu_assert("dim n", C_m->n == 4);
    mu_assert("m0", C->m0 == 2);
    mu_assert("n0", C->n0 == 2);
    int expected_row_perm[2] = {0, 1};
    int expected_col_perm[2] = {0, 3};
    mu_assert("row_perm", cmp_int_array(C->row_perm, expected_row_perm, 2));
    mu_assert("col_perm", cmp_int_array(C->col_perm, expected_col_perm, 2));
    double expected_X[4] = {5.0, 6.0, 15.0, 18.0};
    mu_assert("X", cmp_double_array(C->X, expected_X, 4));

    free_matrix(C_m);
    free_matrix(A_m);
    free_matrix(B_m);
    return 0;
}

/* BA_pd_matrices dispatcher routes B(PD) @ A(spd) to BA_pd_spd_*. Fixture
   mirrors test_BA_pd_spd_two_blocks_disjoint_cols; the assertion is that
   the dispatcher and the direct kernel produce identical output. */
const char *test_BA_pd_matrices_spd_A(void)
{
    int B_rp[3] = {0, 1, 2};
    int B_cp[3] = {0, 1, 3};
    double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *B = new_permuted_dense(3, 4, 3, 3, B_rp, B_cp, BX);

    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 4};
    double A0X[4] = {10, 11, 12, 13};
    matrix *A0 = new_permuted_dense(4, 6, 2, 2, A0_rp, A0_cp, A0X);

    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 4};
    double A1X[4] = {20, 21, 22, 23};
    matrix *A1 = new_permuted_dense(4, 6, 2, 2, A1_rp, A1_cp, A1X);

    permuted_dense *blocks[2] = {(permuted_dense *) A0, (permuted_dense *) A1};
    matrix *A = new_stacked_pd(4, 6, 2, blocks, NULL, NULL);

    matrix *C_m = BA_pd_matrices_alloc((permuted_dense *) B, A);
    BA_pd_matrices_fill_values((permuted_dense *) B, A, (permuted_dense *) C_m);
    permuted_dense *C = (permuted_dense *) C_m;

    mu_assert("C m", C_m->m == 3);
    mu_assert("C n", C_m->n == 6);
    mu_assert("C m0", C->m0 == 3);
    mu_assert("C n0", C->n0 == 3);
    int rp_exp[3] = {0, 1, 2};
    int cp_exp[3] = {0, 1, 4};
    mu_assert("C row_perm", cmp_int_array(C->row_perm, rp_exp, 3));
    mu_assert("C col_perm", cmp_int_array(C->col_perm, cp_exp, 3));
    double CX_exp[9] = {34, 66, 106, 100, 132, 247, 166, 198, 388};
    mu_assert("C X", cmp_double_array(C->X, CX_exp, 9));

    free_matrix(C_m);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* BA_pd_matrices fast path: B->col_perm == A->row_perm exactly, so the
   slow-path gather is skipped and one cblas_dgemm runs directly on
   B->X and A->X.
   B (2x4) row_perm=[0,1], col_perm=[1,3], X_B=[[1,2],[3,4]].
   A (4x3) row_perm=[1,3], col_perm=[0,2], X_A=[[5,6],[7,8]].
   Matching col_perm_B == row_perm_A == [1,3] triggers the fast path.
   Hand-computed C (2x3) nonzero at cols {0,2}: X_C=[[19,22],[43,50]]. */
const char *test_BA_pd_matrices_fast_path(void)
{
    int row_perm_B[2] = {0, 1};
    int col_perm_B[2] = {1, 3};
    double XB[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *B_m = new_permuted_dense(2, 4, 2, 2, row_perm_B, col_perm_B, XB);

    int row_perm_A[2] = {1, 3};
    int col_perm_A[2] = {0, 2};
    double XA[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *A_m = new_permuted_dense(4, 3, 2, 2, row_perm_A, col_perm_A, XA);

    matrix *C_m = BA_pd_matrices_alloc((permuted_dense *) B_m, A_m);
    BA_pd_matrices_fill_values((permuted_dense *) B_m, A_m, (permuted_dense *) C_m);

    permuted_dense *C = (permuted_dense *) C_m;
    mu_assert("dim m", C_m->m == 2);
    mu_assert("dim n", C_m->n == 3);
    mu_assert("m0", C->m0 == 2);
    mu_assert("n0", C->n0 == 2);
    int expected_row_perm[2] = {0, 1};
    int expected_col_perm[2] = {0, 2};
    mu_assert("row_perm", cmp_int_array(C->row_perm, expected_row_perm, 2));
    mu_assert("col_perm", cmp_int_array(C->col_perm, expected_col_perm, 2));
    double expected_X[4] = {19.0, 22.0, 43.0, 50.0};
    mu_assert("X", cmp_double_array(C->X, expected_X, 4));

    free_matrix(C_m);
    free_matrix(A_m);
    free_matrix(B_m);
    return 0;
}

#endif /* TEST_PERMUTED_DENSE_H */
