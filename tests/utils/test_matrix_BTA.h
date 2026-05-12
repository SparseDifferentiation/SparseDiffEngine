#ifndef TEST_MATRIX_BTA_H
#define TEST_MATRIX_BTA_H

#include "minunit.h"
#include "old-code/old_permuted_dense.h"
#include "test_helpers.h"
#include "utils/CSR_matrix.h"
#include "utils/matrix_BTA.h"
#include "utils/permuted_dense.h"
#include "utils/sparse_matrix.h"
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

    /* Wrapper path. */
    matrix *C_m = BTA_matrices_alloc(A_m, B_m);
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
   BTDA_csr_pd_fill_values. */
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

    /* Wrapper path. */
    matrix *C_m = BTA_matrices_alloc(A_m, B_m);
    BTDA_matrices_fill_values(A_m, d, B_m, C_m);

    /* Direct primitive path. */
    matrix *A2_m = new_permuted_dense(4, 5, 2, 2, row_perm_A, col_perm_A, XA);
    permuted_dense *A2 = (permuted_dense *) A2_m;
    CSR_matrix *B2 = new_CSR_matrix(4, 4, 5);
    B2->p[0] = 0;
    B2->p[1] = 2;
    B2->p[2] = 3;
    B2->p[3] = 4;
    B2->p[4] = 5;
    memcpy(B2->i, Bi, sizeof Bi);
    memcpy(B2->x, Bx, sizeof Bx);
    matrix *C2 = BTA_csr_pd_alloc(B2, A2);
    BTDA_csr_pd_fill_values(B2, d, A2, (permuted_dense *) C2);

    mu_assert("values", cmp_double_array(C_m->x, C2->x, C_m->nnz));

    free_matrix(C_m);
    free_matrix(B_m);
    free_matrix(A_m);
    free_matrix(C2);
    free_CSR_matrix(B2);
    free_matrix(A2_m);
    return 0;
}

#endif /* TEST_MATRIX_BTA_H */
