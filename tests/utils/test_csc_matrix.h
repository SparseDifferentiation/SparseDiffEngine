#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "minunit.h"
#include "test_helpers.h"
#include "utils/CSC_Matrix.h"

/* Test ATA_alloc with a simple 3x3 example
 * A is 4x3 (4 rows, 3 columns):
 * [x  0  x]
 * [0  x  0]
 * [0  0  x]
 * [0  x  0]
 *
 * A^T A is 3x3:
 * [x  0  x]
 * [0  x  0]
 * [x  0  x]
 */
const char *test_ATA_alloc_simple()
{
    CSC_Matrix *A = new_csc_matrix(4, 3, 6);
    int Ap[4] = {0, 2, 3, 6};
    int Ai[5] = {0, 2, 1, 2, 1};
    memcpy(A->p, Ap, 4 * sizeof(int));
    memcpy(A->i, Ai, 5 * sizeof(int));

    /* Compute C = A^T A */
    CSR_Matrix *C = ATA_alloc(A);
    int expected_p[4] = {0, 2, 3, 5};
    int expected_i[5] = {0, 2, 1, 0, 2};

    mu_assert("p incorrect", cmp_int_array(C->p, expected_p, 4));
    mu_assert("i incorrect", cmp_int_array(C->i, expected_i, C->nnz));
    mu_assert("nnz incorrect", C->nnz == 5);

    free_csr_matrix(C);
    free_csc_matrix(A);

    return 0;
}

/* Test ATA_alloc with a sparse 3x4 matrix with no overlaps on some pairs
 * A is 3x4:
 * [1  0  0  2]
 * [0  1  0  0]
 * [0  0  1  0]
 *
 * A^T A is 4x4:
 * [1  0  0  2]
 * [0  1  0  0]
 * [0  0  1  0]
 * [2  0  0  4]
 *
 */
const char *test_ATA_alloc_diagonal_like()
{
    /* Create A in CSC format (3 rows, 4 cols, 4 nonzeros) */
    CSC_Matrix *A = new_csc_matrix(3, 4, 4);
    int Ap[5] = {0, 1, 2, 3, 4};
    int Ai[4] = {0, 1, 2, 0};
    memcpy(A->p, Ap, 5 * sizeof(int));
    memcpy(A->i, Ai, 4 * sizeof(int));
    CSR_Matrix *C = ATA_alloc(A);

    int expected_p[5] = {0, 2, 3, 4, 6};
    int expected_i[6] = {0, 3, 1, 2, 0, 3};

    mu_assert("p incorrect", cmp_int_array(C->p, expected_p, 5));
    mu_assert("i incorrect", cmp_int_array(C->i, expected_i, C->nnz));
    mu_assert("nnz incorrect", C->nnz == 6);

    free_csr_matrix(C);
    free_csc_matrix(A);

    return 0;
}

const char *test_ATA_alloc_random()
{
    /* Create A in CSC format  */
    CSC_Matrix *A = new_csc_matrix(10, 15, 15);
    int Ap[16] = {0, 1, 1, 1, 1, 4, 5, 6, 7, 8, 9, 11, 11, 11, 13, 15};
    int Ai[15] = {5, 0, 6, 9, 0, 5, 1, 3, 6, 0, 6, 3, 6, 6, 8};
    memcpy(A->p, Ap, 16 * sizeof(int));
    memcpy(A->i, Ai, 15 * sizeof(int));
    CSR_Matrix *C = ATA_alloc(A);

    int expected_p[16] = {0, 2, 2, 2, 2, 8, 11, 13, 14, 16, 21, 27, 27, 27, 33, 38};
    int expected_i[38] = {0,  6, 4,  5, 9,  10, 13, 14, 4, 5,  10, 0,  6,
                          7,  8, 13, 4, 9,  10, 13, 14, 4, 5,  9,  10, 13,
                          14, 4, 8,  9, 10, 13, 14, 4,  9, 10, 13, 14};

    mu_assert("p incorrect", cmp_int_array(C->p, expected_p, 16));
    mu_assert("i incorrect", cmp_int_array(C->i, expected_i, C->nnz));
    mu_assert("nnz incorrect", C->nnz == 38);

    free_csr_matrix(C);
    free_csc_matrix(A);

    return 0;
}
