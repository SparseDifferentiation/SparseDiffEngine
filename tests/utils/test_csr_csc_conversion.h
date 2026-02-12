#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "minunit.h"
#include "test_helpers.h"
#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"

/* Test CSR to CSC conversion with fill_sparsity and fill_values */
const char *test_csr_to_csc_split()
{
    /* Create a 4x5 CSR matrix A:
     * [1.0  0.0  0.0  0.0  1.0]
     * [0.0  0.0  3.0  0.0  0.0]
     * [0.0  2.0  0.0  0.0  0.0]
     * [0.0  0.0  0.0  4.0  0.0]
     */
    CSR_Matrix *A = new_csr_matrix(4, 5, 5);
    double Ax[5] = {1.0, 1.0, 3.0, 2.0, 4.0};
    int Ai[5] = {0, 4, 2, 1, 3};
    int Ap[5] = {0, 2, 3, 4, 5};
    memcpy(A->x, Ax, 5 * sizeof(double));
    memcpy(A->i, Ai, 5 * sizeof(int));
    memcpy(A->p, Ap, 5 * sizeof(int));

    /* Allocate workspace */
    int *iwork = (int *) malloc(A->n * sizeof(int));

    /* First, fill sparsity pattern */
    CSC_Matrix *C = csr_to_csc_fill_sparsity(A, iwork);

    /* Check sparsity pattern */
    int Cp_correct[6] = {0, 1, 2, 3, 4, 5};
    int Ci_correct[5] = {0, 2, 1, 3, 0};

    mu_assert("C col pointers incorrect", cmp_int_array(C->p, Cp_correct, 6));
    mu_assert("C row indices incorrect", cmp_int_array(C->i, Ci_correct, 5));

    /* Now fill values */
    csr_to_csc_fill_values(A, C, iwork);

    /* Check values */
    double Cx_correct[5] = {1.0, 2.0, 3.0, 4.0, 1.0};

    mu_assert("C vals incorrect", cmp_double_array(C->x, Cx_correct, 5));

    free(iwork);
    free_csr_matrix(A);
    free_csc_matrix(C);

    return 0;
}

/* Test CSC to CSR conversion with fill_sparsity */
const char *test_csc_to_csr_sparsity()
{
    /* Create a 4x5 CSC matrix A:
     * [1.0  0.0  0.0  0.0  2.0]
     * [0.0  0.0  3.0  0.0  0.0]
     * [0.0  4.0  0.0  0.0  0.0]
     * [0.0  0.0  0.0  5.0  0.0]
     */
    CSC_Matrix *A = new_csc_matrix(4, 5, 5);
    double Ax[5] = {1.0, 4.0, 3.0, 5.0, 2.0};
    int Ai[5] = {0, 2, 1, 3, 0};
    int Ap[6] = {0, 1, 2, 3, 4, 5};
    memcpy(A->x, Ax, 5 * sizeof(double));
    memcpy(A->i, Ai, 5 * sizeof(int));
    memcpy(A->p, Ap, 6 * sizeof(int));

    /* Allocate workspace */
    int *iwork = (int *) malloc(A->m * sizeof(int));

    /* Fill sparsity pattern */
    CSR_Matrix *C = csc_to_csr_fill_sparsity(A, iwork);

    /* Expected CSR format:
     * Row 0: [1.0 at col 0, 2.0 at col 4]
     * Row 1: [3.0 at col 2]
     * Row 2: [4.0 at col 1]
     * Row 3: [5.0 at col 3]
     */
    int Cp_correct[5] = {0, 2, 3, 4, 5};
    int Ci_correct[5] = {0, 4, 2, 1, 3};

    mu_assert("C row pointers incorrect", cmp_int_array(C->p, Cp_correct, 5));
    mu_assert("C col indices incorrect", cmp_int_array(C->i, Ci_correct, 5));
    mu_assert("C dimensions incorrect", C->m == 4 && C->n == 5);
    mu_assert("C nnz incorrect", C->nnz == 5);

    free(iwork);
    free_csc_matrix(A);
    free_csr_matrix(C);

    return 0;
}

/* Test CSC to CSR conversion with fill_values */
const char *test_csc_to_csr_values()
{
    /* Create a 4x5 CSC matrix A */
    CSC_Matrix *A = new_csc_matrix(4, 5, 5);
    double Ax[5] = {1.0, 4.0, 3.0, 5.0, 2.0};
    int Ai[5] = {0, 2, 1, 3, 0};
    int Ap[6] = {0, 1, 2, 3, 4, 5};
    memcpy(A->x, Ax, 5 * sizeof(double));
    memcpy(A->i, Ai, 5 * sizeof(int));
    memcpy(A->p, Ap, 6 * sizeof(int));

    /* Allocate workspace */
    int *iwork = (int *) malloc(A->m * sizeof(int));

    /* Fill sparsity pattern */
    CSR_Matrix *C = csc_to_csr_fill_sparsity(A, iwork);

    /* Fill values */
    csc_to_csr_fill_values(A, C, iwork);

    /* Check values */
    double Cx_correct[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

    mu_assert("C vals incorrect", cmp_double_array(C->x, Cx_correct, 5));

    free(iwork);
    free_csc_matrix(A);
    free_csr_matrix(C);

    return 0;
}

/* Test round-trip conversion: CSR -> CSC -> CSR */
const char *test_csr_csc_csr_roundtrip()
{
    /* Create a 3x4 CSR matrix A:
     * [1.0  2.0  0.0  3.0]
     * [0.0  4.0  5.0  0.0]
     * [6.0  0.0  7.0  8.0]
     */
    CSR_Matrix *A = new_csr_matrix(3, 4, 8);
    double Ax[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    int Ai[8] = {0, 1, 3, 1, 2, 0, 2, 3};
    int Ap[4] = {0, 3, 5, 8};
    memcpy(A->x, Ax, 8 * sizeof(double));
    memcpy(A->i, Ai, 8 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    /* Convert CSR to CSC */
    int *iwork_csc = (int *) malloc(A->n * sizeof(int));
    CSC_Matrix *B = csr_to_csc_fill_sparsity(A, iwork_csc);
    csr_to_csc_fill_values(A, B, iwork_csc);

    /* Convert CSC back to CSR */
    int *iwork_csr = (int *) malloc(B->m * sizeof(int));
    CSR_Matrix *C = csc_to_csr_fill_sparsity(B, iwork_csr);
    csc_to_csr_fill_values(B, C, iwork_csr);

    /* C should match A */
    mu_assert("Round-trip: vals incorrect", cmp_double_array(C->x, Ax, 8));
    mu_assert("Round-trip: col indices incorrect", cmp_int_array(C->i, Ai, 8));
    mu_assert("Round-trip: row pointers incorrect", cmp_int_array(C->p, Ap, 4));

    free(iwork_csc);
    free(iwork_csr);
    free_csr_matrix(A);
    free_csc_matrix(B);
    free_csr_matrix(C);

    return 0;
}
