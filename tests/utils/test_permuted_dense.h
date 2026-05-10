#ifndef TEST_PERMUTED_DENSE_H
#define TEST_PERMUTED_DENSE_H

#include "minunit.h"
#include "test_helpers.h"
#include "utils/CSC_Matrix.h"
#include "utils/permuted_dense.h"
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

    Matrix *M = new_permuted_dense(5, 6, 3, 2, row_perm, col_perm, X);
    Permuted_Dense *pd = (Permuted_Dense *) M;

    CSR_Matrix *C = permuted_dense_to_csr_alloc(pd);
    permuted_dense_to_csr_fill_values(pd, C);

    int Cp_expected[6] = {0, 0, 2, 4, 4, 6};
    int Ci_expected[6] = {0, 3, 0, 3, 0, 3};
    double Cx_expected[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    mu_assert("dim m", C->m == 5);
    mu_assert("dim n", C->n == 6);
    mu_assert("nnz", C->nnz == 6);
    mu_assert("p", cmp_int_array(C->p, Cp_expected, 6));
    mu_assert("i", cmp_int_array(C->i, Ci_expected, 6));
    mu_assert("x", cmp_double_array(C->x, Cx_expected, 6));

    free_csr_matrix(C);
    free_matrix(M);
    return 0;
}

/* Empty dense block (dense_m = dense_n = 0): result is an m x n CSR with
   no nonzeros. */
const char *test_permuted_dense_to_csr_empty(void)
{
    Matrix *M = new_permuted_dense(4, 5, 0, 0, NULL, NULL, NULL);
    Permuted_Dense *pd = (Permuted_Dense *) M;

    CSR_Matrix *C = permuted_dense_to_csr_alloc(pd);
    permuted_dense_to_csr_fill_values(pd, C);

    int Cp_expected[5] = {0, 0, 0, 0, 0};
    mu_assert("nnz", C->nnz == 0);
    mu_assert("p", cmp_int_array(C->p, Cp_expected, 5));

    free_csr_matrix(C);
    free_matrix(M);
    return 0;
}

/* Full dense (row_perm = [0..m), col_perm = [0..n)): result is the dense
   matrix in CSR. */
const char *test_permuted_dense_to_csr_full(void)
{
    int row_perm[2] = {0, 1};
    int col_perm[3] = {0, 1, 2};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    Matrix *M = new_permuted_dense(2, 3, 2, 3, row_perm, col_perm, X);
    Permuted_Dense *pd = (Permuted_Dense *) M;

    CSR_Matrix *C = permuted_dense_to_csr_alloc(pd);
    permuted_dense_to_csr_fill_values(pd, C);

    int Cp_expected[3] = {0, 3, 6};
    int Ci_expected[6] = {0, 1, 2, 0, 1, 2};
    double Cx_expected[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    mu_assert("p", cmp_int_array(C->p, Cp_expected, 3));
    mu_assert("i", cmp_int_array(C->i, Ci_expected, 6));
    mu_assert("x", cmp_double_array(C->x, Cx_expected, 6));

    free_csr_matrix(C);
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

    Matrix *M = new_permuted_dense(4, 5, 1, 2, row_perm, col_perm, X);
    Permuted_Dense *pd = (Permuted_Dense *) M;

    CSR_Matrix *C = permuted_dense_to_csr_alloc(pd);
    permuted_dense_to_csr_fill_values(pd, C);

    int Cp_expected[5] = {0, 0, 0, 2, 2};
    int Ci_expected[2] = {1, 4};
    double Cx_expected[2] = {7.0, 9.0};

    mu_assert("p", cmp_int_array(C->p, Cp_expected, 5));
    mu_assert("i", cmp_int_array(C->i, Ci_expected, 2));
    mu_assert("x", cmp_double_array(C->x, Cx_expected, 2));

    free_csr_matrix(C);
    free_matrix(M);
    return 0;
}

/* Single dense col across multiple rows. */
const char *test_permuted_dense_to_csr_single_col(void)
{
    int row_perm[3] = {0, 2, 3};
    int col_perm[1] = {2};
    double X[3] = {1.0, 2.0, 3.0};

    Matrix *M = new_permuted_dense(4, 4, 3, 1, row_perm, col_perm, X);
    Permuted_Dense *pd = (Permuted_Dense *) M;

    CSR_Matrix *C = permuted_dense_to_csr_alloc(pd);
    permuted_dense_to_csr_fill_values(pd, C);

    int Cp_expected[5] = {0, 1, 1, 2, 3};
    int Ci_expected[3] = {2, 2, 2};
    double Cx_expected[3] = {1.0, 2.0, 3.0};

    mu_assert("p", cmp_int_array(C->p, Cp_expected, 5));
    mu_assert("i", cmp_int_array(C->i, Ci_expected, 3));
    mu_assert("x", cmp_double_array(C->x, Cx_expected, 3));

    free_csr_matrix(C);
    free_matrix(M);
    return 0;
}

/* DA_fill_values: compare against CSR DA_fill_values on the equivalent CSR.

   PD is the 5x6 matrix from the basic to_csr test, with d a length-5
   global-row diagonal including a negative and zero entry. */
const char *test_permuted_dense_DA_fill_values(void)
{
    int row_perm[3] = {1, 2, 4};
    int col_perm[2] = {0, 3};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double d[5] = {7.0, -1.5, 0.0, 9.0, 2.5};

    Matrix *M = new_permuted_dense(5, 6, 3, 2, row_perm, col_perm, X);
    Matrix *M_out = new_permuted_dense(5, 6, 3, 2, row_perm, col_perm, NULL);
    Permuted_Dense *pd = (Permuted_Dense *) M;
    Permuted_Dense *pd_out = (Permuted_Dense *) M_out;

    permuted_dense_DA_fill_values(d, pd, pd_out);

    /* Ground truth: build CSR of self, run DA_fill_values, compare. */
    CSR_Matrix *csr = permuted_dense_to_csr_alloc(pd);
    permuted_dense_to_csr_fill_values(pd, csr);
    CSR_Matrix *csr_expected = new_csr_copy_sparsity(csr);
    DA_fill_values(d, csr, csr_expected);

    CSR_Matrix *csr_out = permuted_dense_to_csr_alloc(pd_out);
    permuted_dense_to_csr_fill_values(pd_out, csr_out);

    mu_assert("x", cmp_double_array(csr_out->x, csr_expected->x, csr->nnz));

    free_csr_matrix(csr);
    free_csr_matrix(csr_expected);
    free_csr_matrix(csr_out);
    free_matrix(M);
    free_matrix(M_out);
    return 0;
}

/* ATA_alloc: structure-only check. Output is 6x6 with a 2x2 dense block at
   perms {0, 3} (= self.col_perm on both sides). Values are uninitialized
   here; ATDA_fill_values is the value-producing op. */
const char *test_permuted_dense_ATA_alloc(void)
{
    int row_perm[3] = {1, 2, 4};
    int col_perm[2] = {0, 3};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    Matrix *M = new_permuted_dense(5, 6, 3, 2, row_perm, col_perm, X);
    Permuted_Dense *pd = (Permuted_Dense *) M;

    Matrix *M_ata = permuted_dense_ATA_alloc(pd);
    Permuted_Dense *pd_ata = (Permuted_Dense *) M_ata;

    int perm_expected[2] = {0, 3};
    mu_assert("m", M_ata->m == 6);
    mu_assert("n", M_ata->n == 6);
    mu_assert("dense_m", pd_ata->dense_m == 2);
    mu_assert("dense_n", pd_ata->dense_n == 2);
    mu_assert("row_perm", cmp_int_array(pd_ata->row_perm, perm_expected, 2));
    mu_assert("col_perm", cmp_int_array(pd_ata->col_perm, perm_expected, 2));

    free_matrix(M);
    free_matrix(M_ata);
    return 0;
}

/* ATDA: same 5x6 PD, d with negative + zero entries to catch sign bugs.
   Hand-computed: d_perm = [-1.5, 0, 2.5], Y = diag(d_perm) X gives
   [[-1.5,-3],[0,0],[12.5,15]], and X^T Y = [[61,72],[72,84]]. */
const char *test_permuted_dense_ATDA_fill_values(void)
{
    int row_perm[3] = {1, 2, 4};
    int col_perm[2] = {0, 3};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double d[5] = {7.0, -1.5, 0.0, 9.0, 2.5};

    Matrix *M = new_permuted_dense(5, 6, 3, 2, row_perm, col_perm, X);
    Permuted_Dense *pd = (Permuted_Dense *) M;

    Matrix *M_out = permuted_dense_ATA_alloc(pd);
    Permuted_Dense *pd_out = (Permuted_Dense *) M_out;
    permuted_dense_ATDA_fill_values(pd, d, pd_out);

    double X_expected[4] = {61.0, 72.0, 72.0, 84.0};
    mu_assert("X", cmp_double_array(pd_out->X, X_expected, 4));

    free_matrix(M);
    free_matrix(M_out);
    return 0;
}

/* PD x CSC: J is 6x4. col 0 empty; col 1 has rows {0,3} (vals 10, 20);
   col 2 has row {2} (val 30, but row 2 not in col_perm_self = {0,3} so col 2
   is INACTIVE); col 3 has row {3} (val 40). Active cols: {1, 3}.

   Expected: dense_m=3, dense_n=2, row_perm={1,2,4}, col_perm={1,3}.
   Values: out.X[:,0] = 10*[1,3,5] + 20*[2,4,6] = [50,110,170],
           out.X[:,1] = 40*[2,4,6] = [80,160,240]. */
const char *test_permuted_dense_times_csc(void)
{
    int row_perm[3] = {1, 2, 4};
    int col_perm[2] = {0, 3};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Matrix *M = new_permuted_dense(5, 6, 3, 2, row_perm, col_perm, X);
    Permuted_Dense *pd = (Permuted_Dense *) M;

    CSC_Matrix *J = new_csc_matrix(6, 4, 4);
    int Jp[5] = {0, 0, 2, 3, 4};
    int Ji[4] = {0, 3, 2, 3};
    double Jx[4] = {10.0, 20.0, 30.0, 40.0};
    memcpy(J->p, Jp, 5 * sizeof(int));
    memcpy(J->i, Ji, 4 * sizeof(int));
    memcpy(J->x, Jx, 4 * sizeof(double));

    Matrix *M_out = permuted_dense_times_csc_alloc(pd, J);
    Permuted_Dense *pd_out = (Permuted_Dense *) M_out;
    permuted_dense_times_csc_fill_values(pd, J, pd_out);

    int row_perm_expected[3] = {1, 2, 4};
    int col_perm_expected[2] = {1, 3};
    double X_expected[6] = {50.0, 80.0, 110.0, 160.0, 170.0, 240.0};

    mu_assert("m", M_out->m == 5);
    mu_assert("n", M_out->n == 4);
    mu_assert("dense_m", pd_out->dense_m == 3);
    mu_assert("dense_n", pd_out->dense_n == 2);
    mu_assert("row_perm", cmp_int_array(pd_out->row_perm, row_perm_expected, 3));
    mu_assert("col_perm", cmp_int_array(pd_out->col_perm, col_perm_expected, 2));
    mu_assert("X", cmp_double_array(pd_out->X, X_expected, 6));

    free_matrix(M);
    free_matrix(M_out);
    free_csc_matrix(J);
    return 0;
}

/* PD x CSC edge case: every column of J has its only nonzero outside
   col_perm_self, so col_perm_out is empty (dense_n = 0). */
const char *test_permuted_dense_times_csc_no_active(void)
{
    int row_perm[3] = {1, 2, 4};
    int col_perm[2] = {0, 3};
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Matrix *M = new_permuted_dense(5, 6, 3, 2, row_perm, col_perm, X);
    Permuted_Dense *pd = (Permuted_Dense *) M;

    /* J: col 0 has row {1}, col 1 has row {5}. Neither in col_perm_self. */
    CSC_Matrix *J = new_csc_matrix(6, 2, 2);
    int Jp[3] = {0, 1, 2};
    int Ji[2] = {1, 5};
    double Jx[2] = {100.0, 200.0};
    memcpy(J->p, Jp, 3 * sizeof(int));
    memcpy(J->i, Ji, 2 * sizeof(int));
    memcpy(J->x, Jx, 2 * sizeof(double));

    Matrix *M_out = permuted_dense_times_csc_alloc(pd, J);
    Permuted_Dense *pd_out = (Permuted_Dense *) M_out;
    permuted_dense_times_csc_fill_values(pd, J, pd_out);

    mu_assert("m", M_out->m == 5);
    mu_assert("n", M_out->n == 2);
    mu_assert("dense_m", pd_out->dense_m == 3);
    mu_assert("dense_n", pd_out->dense_n == 0);

    free_matrix(M);
    free_matrix(M_out);
    free_csc_matrix(J);
    return 0;
}

/* Sanity check: col_inv is built correctly. col_perm = {0, 3} on n = 6
   should give col_inv = {0, -1, -1, 1, -1, -1}. */
const char *test_permuted_dense_col_inv(void)
{
    int row_perm[1] = {0};
    int col_perm[2] = {0, 3};
    double X[2] = {0.0, 0.0};

    Matrix *M = new_permuted_dense(1, 6, 1, 2, row_perm, col_perm, X);
    Permuted_Dense *pd = (Permuted_Dense *) M;

    int expected[6] = {0, -1, -1, 1, -1, -1};
    mu_assert("col_inv", cmp_int_array(pd->col_inv, expected, 6));

    free_matrix(M);
    return 0;
}

#endif /* TEST_PERMUTED_DENSE_H */
