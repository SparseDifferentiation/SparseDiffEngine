#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "minunit.h"
#include "test_helpers.h"
#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"
#include "utils/linalg.h"

/* Test block_left_multiply_fill_sparsity with simple case: single block */
const char *test_block_left_multiply_single_block()
{
    /* A is 2x3 CSR:
     * [1.0  0.0  0.0]
     * [0.0  1.0  1.0]
     */
    CSR_Matrix *A = new_csr_matrix(2, 3, 3);
    double Ax[3] = {1.0, 1.0, 1.0};
    int Ai[3] = {0, 1, 2};
    int Ap[4] = {0, 1, 3};
    memcpy(A->x, Ax, 3 * sizeof(double));
    memcpy(A->i, Ai, 3 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    /* J is 3x2 CSC (single block, so p=1):
     * [1.0  0.0]
     * [1.0  0.0]
     * [0.0  1.0]
     */
    CSC_Matrix *J = new_csc_matrix(3, 2, 3);
    double Jx[3] = {1.0, 1.0, 1.0};
    int Ji[3] = {0, 1, 2};
    int Jp[3] = {0, 2, 3};
    memcpy(J->x, Jx, 3 * sizeof(double));
    memcpy(J->i, Ji, 3 * sizeof(int));
    memcpy(J->p, Jp, 3 * sizeof(int));

    /* Compute C = A @ J1 (p=1 means just one block) */
    CSC_Matrix *C = block_left_multiply_fill_sparsity(A, J, 1);

    /* Expected C is 2x2:
     * C[0,0] = A[0,:] @ J[:,0] = 1.0 * 1.0 = 1.0 (row 0 has column 0, J col 0 has row 0)
     * C[1,0] = A[1,:] @ J[:,0] = 1.0*1.0 + 1.0*0.0 = 1.0 (row 1 has columns 1,2, J col 0 has row 1)
     * C[0,1] = A[0,:] @ J[:,1] = 0.0 (row 0 has column 0, J col 1 has row 2)
     * C[1,1] = A[1,:] @ J[:,1] = 1.0*1.0 = 1.0 (row 1 has columns 1,2, J col 1 has row 2)
     */
    mu_assert("C->m should be 2", C->m == 2);
    mu_assert("C->n should be 2", C->n == 2);
    mu_assert("C->nnz should be 3", C->nnz == 3);  /* nnz at (0,0), (1,0), (1,1) */

    /* Check column pointers */
    mu_assert("C->p[0] should be 0", C->p[0] == 0);
    mu_assert("C->p[1] should be 2", C->p[1] == 2);  /* column 0: rows 0,1 */
    mu_assert("C->p[2] should be 3", C->p[2] == 3);  /* column 1: row 1 */

    /* Check row indices */
    mu_assert("C->i[0] should be 0", C->i[0] == 0);  /* (0,0) */
    mu_assert("C->i[1] should be 1", C->i[1] == 1);  /* (1,0) */
    mu_assert("C->i[2] should be 1", C->i[2] == 1);  /* (1,1) */

    free_csc_matrix(C);
    free_csr_matrix(A);
    free_csc_matrix(J);
    return NULL;
}

/* Test block_left_multiply_fill_sparsity with two blocks */
const char *test_block_left_multiply_two_blocks()
{
    /* A is 2x2 CSR:
     * [1.0  0.0]
     * [0.0  1.0]
     */
    CSR_Matrix *A = new_csr_matrix(2, 2, 2);
    double Ax[2] = {1.0, 1.0};
    int Ai[2] = {0, 1};
    int Ap[3] = {0, 1, 2};
    memcpy(A->x, Ax, 2 * sizeof(double));
    memcpy(A->i, Ai, 2 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* J is 4x3 CSC (two blocks of 2 rows each):
     * Block 1 rows [0,1]:
     * [1.0  0.0  0.0]
     * [0.0  0.0  0.0]
     * Block 2 rows [2,3]:
     * [0.0  1.0  0.0]
     * [0.0  0.0  1.0]
     * So J is:
     * [1.0  0.0  0.0]
     * [0.0  0.0  0.0]
     * [0.0  1.0  0.0]
     * [0.0  0.0  1.0]
     */
    CSC_Matrix *J = new_csc_matrix(4, 3, 3);
    double Jx[3] = {1.0, 1.0, 1.0};
    int Ji[3] = {0, 2, 3};
    int Jp[4] = {0, 1, 2, 3};
    memcpy(J->x, Jx, 3 * sizeof(double));
    memcpy(J->i, Ji, 3 * sizeof(int));
    memcpy(J->p, Jp, 4 * sizeof(int));

    /* Compute C = [A @ J1; A @ J2] where:
     * J1 = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
     * J2 = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
     * 
     * C = [A @ J1; A @ J2] is 4x3:
     * A @ J1 = [[1, 0, 0], [0, 0, 0]] (row 0 of A matches col 0 of J1)
     * A @ J2 = [[0, 0, 0], [0, 1, 1]] (row 1 of A matches cols 1,2 of J2)
     * So C is:
     * [1.0  0.0  0.0]
     * [0.0  0.0  0.0]
     * [0.0  0.0  0.0]
     * [0.0  1.0  1.0]
     */
    CSC_Matrix *C = block_left_multiply_fill_sparsity(A, J, 2);

    mu_assert("C->m should be 4", C->m == 4);
    mu_assert("C->n should be 3", C->n == 3);
    mu_assert("C->nnz should be 3", C->nnz == 3);

    /* Check column pointers */
    mu_assert("C->p[0] should be 0", C->p[0] == 0);
    mu_assert("C->p[1] should be 1", C->p[1] == 1);
    mu_assert("C->p[2] should be 2", C->p[2] == 2);
    mu_assert("C->p[3] should be 3", C->p[3] == 3);

    /* Check row indices */
    mu_assert("C->i[0] should be 0", C->i[0] == 0);
    mu_assert("C->i[1] should be 2", C->i[1] == 2);
    mu_assert("C->i[2] should be 3", C->i[2] == 3);

    free_csc_matrix(C);
    free_csr_matrix(A);
    free_csc_matrix(J);
    return NULL;
}

/* Test block_left_multiply_fill_sparsity with all zero column in J */
const char *test_block_left_multiply_zero_column()
{
    /* A is 2x2 CSR (identity) */
    CSR_Matrix *A = new_csr_matrix(2, 2, 2);
    double Ax[2] = {1.0, 1.0};
    int Ai[2] = {0, 1};
    int Ap[3] = {0, 1, 2};
    memcpy(A->x, Ax, 2 * sizeof(double));
    memcpy(A->i, Ai, 2 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* J is 2x2 with an empty column:
     * [1.0  0.0]
     * [0.0  0.0]
     */
    CSC_Matrix *J = new_csc_matrix(2, 2, 1);
    double Jx[1] = {1.0};
    int Ji[1] = {0};
    int Jp[3] = {0, 1, 1};  /* Column 0 has one nonzero, column 1 is empty */
    memcpy(J->x, Jx, 1 * sizeof(double));
    memcpy(J->i, Ji, 1 * sizeof(int));
    memcpy(J->p, Jp, 3 * sizeof(int));

    CSC_Matrix *C = block_left_multiply_fill_sparsity(A, J, 1);

    mu_assert("C->m should be 2", C->m == 2);
    mu_assert("C->n should be 2", C->n == 2);
    mu_assert("C->nnz should be 1", C->nnz == 1);

    mu_assert("C->p[0] should be 0", C->p[0] == 0);
    mu_assert("C->p[1] should be 1", C->p[1] == 1);
    mu_assert("C->p[2] should be 1", C->p[2] == 1);

    mu_assert("C->i[0] should be 0", C->i[0] == 0);

    free_csc_matrix(C);
    free_csr_matrix(A);
    free_csc_matrix(J);
    return NULL;
}

/* Test csr_csc_matmul_alloc: C = A @ B where A is CSR and B is CSC */
const char *test_csr_csc_matmul_alloc_basic()
{
    /* A is 3x2 CSR:
     * [1.0  0.0]
     * [0.0  1.0]
     * [1.0  1.0]
     */
    CSR_Matrix *A = new_csr_matrix(3, 2, 4);
    double Ax[4] = {1.0, 1.0, 1.0, 1.0};
    int Ai[4] = {0, 1, 0, 1};
    int Ap[4] = {0, 1, 2, 4};
    memcpy(A->x, Ax, 4 * sizeof(double));
    memcpy(A->i, Ai, 4 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    /* B is 2x3 CSC:
     * [1.0  0.0  1.0]
     * [0.0  1.0  1.0]
     */
    CSC_Matrix *B = new_csc_matrix(2, 3, 4);
    double Bx[4] = {1.0, 1.0, 1.0, 1.0};
    int Bi[4] = {0, 1, 0, 1};
    int Bp[4] = {0, 1, 2, 4};
    memcpy(B->x, Bx, 4 * sizeof(double));
    memcpy(B->i, Bi, 4 * sizeof(int));
    memcpy(B->p, Bp, 4 * sizeof(int));

    /* C = A @ B is 3x3:
     * C = [[1, 0, 1],
     *      [0, 1, 1],
     *      [1, 1, 2]]
     */
    CSR_Matrix *C = csr_csc_matmul_alloc(A, B);

    mu_assert("C->m should be 3", C->m == 3);
    mu_assert("C->n should be 3", C->n == 3);
    mu_assert("C->nnz should be 7", C->nnz == 7);

    /* Check row pointers */
    mu_assert("C->p[0] should be 0", C->p[0] == 0);
    mu_assert("C->p[1] should be 2", C->p[1] == 2);
    mu_assert("C->p[2] should be 4", C->p[2] == 4);
    mu_assert("C->p[3] should be 7", C->p[3] == 7);

    /* Check column indices for row 0: columns 0, 2 */
    mu_assert("C->i[0] should be 0", C->i[0] == 0);
    mu_assert("C->i[1] should be 2", C->i[1] == 2);

    /* Check column indices for row 1: columns 1, 2 */
    mu_assert("C->i[2] should be 1", C->i[2] == 1);
    mu_assert("C->i[3] should be 2", C->i[3] == 2);

    /* Check column indices for row 2: columns 0, 1, 2 */
    mu_assert("C->i[4] should be 0", C->i[4] == 0);
    mu_assert("C->i[5] should be 1", C->i[5] == 1);
    mu_assert("C->i[6] should be 2", C->i[6] == 2);

    free_csr_matrix(C);
    free_csr_matrix(A);
    free_csc_matrix(B);
    return NULL;
}

/* Test csr_csc_matmul_alloc with sparse result */
const char *test_csr_csc_matmul_alloc_sparse()
{
    /* A is 2x3 CSR:
     * [1.0  0.0  0.0]
     * [0.0  0.0  1.0]
     */
    CSR_Matrix *A = new_csr_matrix(2, 3, 2);
    double Ax[2] = {1.0, 1.0};
    int Ai[2] = {0, 2};
    int Ap[3] = {0, 1, 2};
    memcpy(A->x, Ax, 2 * sizeof(double));
    memcpy(A->i, Ai, 2 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* B is 3x2 CSC:
     * [1.0  0.0]
     * [0.0  0.0]
     * [0.0  1.0]
     */
    CSC_Matrix *B = new_csc_matrix(3, 2, 2);
    double Bx[2] = {1.0, 1.0};
    int Bi[2] = {0, 2};
    int Bp[3] = {0, 1, 2};
    memcpy(B->x, Bx, 2 * sizeof(double));
    memcpy(B->i, Bi, 2 * sizeof(int));
    memcpy(B->p, Bp, 3 * sizeof(int));

    /* C = A @ B is 2x2:
     * C = [[1, 0],
     *      [0, 1]]
     */
    CSR_Matrix *C = csr_csc_matmul_alloc(A, B);

    mu_assert("C->m should be 2", C->m == 2);
    mu_assert("C->n should be 2", C->n == 2);
    mu_assert("C->nnz should be 2", C->nnz == 2);

    mu_assert("C->p[0] should be 0", C->p[0] == 0);
    mu_assert("C->p[1] should be 1", C->p[1] == 1);
    mu_assert("C->p[2] should be 2", C->p[2] == 2);

    mu_assert("C->i[0] should be 0", C->i[0] == 0);
    mu_assert("C->i[1] should be 1", C->i[1] == 1);

    free_csr_matrix(C);
    free_csr_matrix(A);
    free_csc_matrix(B);
    return NULL;
}
