#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "minunit.h"
#include "test_helpers.h"
#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"
#include "utils/linalg_sparse_matmuls.h"

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
    int Ap[3] = {0, 1, 3};
    memcpy(A->x, Ax, 3 * sizeof(double));
    memcpy(A->i, Ai, 3 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

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
     * C[0,0] = A[0,:] @ J[:,0] = 1.0 * 1.0 = 1.0 (row 0 has column 0, J col 0 has
     * row 0) C[1,0] = A[1,:] @ J[:,0] = 1.0*1.0 + 1.0*0.0 = 1.0 (row 1 has columns
     * 1,2, J col 0 has row 1) C[0,1] = A[0,:] @ J[:,1] = 0.0 (row 0 has column 0, J
     * col 1 has row 2) C[1,1] = A[1,:] @ J[:,1] = 1.0*1.0 = 1.0 (row 1 has columns
     * 1,2, J col 1 has row 2)
     */
    int expected_p[3] = {0, 2, 3};
    int expected_i[3] = {0, 1, 1};

    mu_assert("C dims incorrect", C->m == 2 && C->n == 2 && C->nnz == 3);
    mu_assert("C col pointers incorrect", cmp_int_array(C->p, expected_p, 3));
    mu_assert("C row indices incorrect", cmp_int_array(C->i, expected_i, 3));

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
    block_left_multiply_fill_values(A, J, C);

    int expected_p2[4] = {0, 1, 2, 3};
    int expected_i2[3] = {0, 2, 3};
    double expected_x2[3] = {1.0, 1.0, 1.0};

    mu_assert("C dims incorrect", C->m == 4 && C->n == 3 && C->nnz == 3);
    mu_assert("C col pointers incorrect", cmp_int_array(C->p, expected_p2, 4));
    mu_assert("C row indices incorrect", cmp_int_array(C->i, expected_i2, 3));
    mu_assert("C values incorrect", cmp_double_array(C->x, expected_x2, 3));

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
    int Jp[3] = {0, 1, 1}; /* Column 0 has one nonzero, column 1 is empty */
    memcpy(J->x, Jx, 1 * sizeof(double));
    memcpy(J->i, Ji, 1 * sizeof(int));
    memcpy(J->p, Jp, 3 * sizeof(int));

    CSC_Matrix *C = block_left_multiply_fill_sparsity(A, J, 1);

    int expected_p3[3] = {0, 1, 1};
    int expected_i3[1] = {0};

    mu_assert("C dims incorrect", C->m == 2 && C->n == 2 && C->nnz == 1);
    mu_assert("C col pointers incorrect", cmp_int_array(C->p, expected_p3, 3));
    mu_assert("C row indices incorrect", cmp_int_array(C->i, expected_i3, 1));

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

    int expected_p4[4] = {0, 2, 4, 7};
    int expected_i4[7] = {0, 2, 1, 2, 0, 1, 2};

    mu_assert("C dims incorrect", C->m == 3 && C->n == 3 && C->nnz == 7);
    mu_assert("C row pointers incorrect", cmp_int_array(C->p, expected_p4, 4));
    mu_assert("C col indices incorrect", cmp_int_array(C->i, expected_i4, 7));

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

    int expected_p5[3] = {0, 1, 2};
    int expected_i5[2] = {0, 1};

    mu_assert("C dims incorrect", C->m == 2 && C->n == 2 && C->nnz == 2);
    mu_assert("C row pointers incorrect", cmp_int_array(C->p, expected_p5, 3));
    mu_assert("C col indices incorrect", cmp_int_array(C->i, expected_i5, 2));

    free_csr_matrix(C);
    free_csr_matrix(A);
    free_csc_matrix(B);
    return NULL;
}

/* Test block_left_multiply_vec with single block: y = A @ x */
const char *test_block_left_multiply_vec_single_block()
{
    /* A is 2x3 CSR:
     * [1.0  0.0  2.0]
     * [0.0  3.0  0.0]
     */
    CSR_Matrix *A = new_csr_matrix(2, 3, 3);
    double Ax[3] = {1.0, 3.0, 2.0};
    int Ai[3] = {0, 1, 2};
    int Ap[3] = {0, 2, 3};
    memcpy(A->x, Ax, 3 * sizeof(double));
    memcpy(A->i, Ai, 3 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* x is (3*1)-length vector = [1.0, 2.0, 3.0] */
    double x[3] = {1.0, 2.0, 3.0};
    double y[2] = {0.0, 0.0};

    block_left_multiply_vec(A, x, y, 1);

    /* Expected: y = [1.0*1.0 + 0.0*2.0 + 2.0*3.0, 0.0*1.0 + 3.0*2.0 + 0.0*3.0]
     *             = [1.0 + 6.0, 6.0]
     *             = [7.0, 6.0]
     */
    double expected_y[2] = {7.0, 6.0};
    mu_assert("y values incorrect", cmp_double_array(y, expected_y, 2));

    free_csr_matrix(A);
    return NULL;
}

/* Test block_left_multiply_vec with two blocks: y = [A @ x1; A @ x2] */
const char *test_block_left_multiply_vec_two_blocks()
{
    /* A is 2x3 CSR:
     * [1.0  2.0  0.0]
     * [0.0  3.0  4.0]
     */
    CSR_Matrix *A = new_csr_matrix(2, 3, 4);
    double Ax[4] = {1.0, 2.0, 3.0, 4.0};
    int Ai[4] = {0, 1, 1, 2};
    int Ap[3] = {0, 2, 4};
    memcpy(A->x, Ax, 4 * sizeof(double));
    memcpy(A->i, Ai, 4 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* x is (3*2)-length vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
     * x1 = [1.0, 2.0, 3.0], x2 = [4.0, 5.0, 6.0]
     */
    double x[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double y[4] = {0.0, 0.0, 0.0, 0.0};

    block_left_multiply_vec(A, x, y, 2);

    /* Expected block 1: y[0:2] = A @ x1 = [1.0*1.0 + 2.0*2.0, 3.0*2.0 + 4.0*3.0] =
     * [5.0, 18.0] Expected block 2: y[2:4] = A @ x2 = [1.0*4.0 + 2.0*5.0, 3.0*5.0
     * + 4.0*6.0] = [14.0, 39.0]
     */
    double expected_y[4] = {5.0, 18.0, 14.0, 39.0};
    mu_assert("y values incorrect", cmp_double_array(y, expected_y, 4));

    free_csr_matrix(A);
    return NULL;
}

/* Test block_left_multiply_vec with sparse matrix and multiple blocks */
const char *test_block_left_multiply_vec_sparse()
{
    /* A is 3x4 CSR (very sparse):
     * [2.0  0.0  0.0  0.0]
     * [0.0  0.0  3.0  0.0]
     * [0.0  0.0  0.0  4.0]
     */
    CSR_Matrix *A = new_csr_matrix(3, 4, 3);
    double Ax[3] = {2.0, 3.0, 4.0};
    int Ai[3] = {0, 2, 3};
    int Ap[4] = {0, 1, 2, 3};
    memcpy(A->x, Ax, 3 * sizeof(double));
    memcpy(A->i, Ai, 3 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    /* x is (4*2)-length = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
     * x1 = [1.0, 2.0, 3.0, 4.0], x2 = [5.0, 6.0, 7.0, 8.0]
     */
    double x[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double y[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    block_left_multiply_vec(A, x, y, 2);

    /* Expected block 1: y[0:3] = A @ x1 = [2.0*1.0, 3.0*3.0, 4.0*4.0] =
     * [2.0, 9.0, 16.0] Expected block 2: y[3:6] = A @ x2 =
     * [2.0*5.0, 3.0*7.0, 4.0*8.0] = [10.0, 21.0, 32.0]
     */
    double expected_y[6] = {2.0, 9.0, 16.0, 10.0, 21.0, 32.0};
    mu_assert("y values incorrect", cmp_double_array(y, expected_y, 6));

    free_csr_matrix(A);
    return NULL;
}

/* Test block_left_multiply_vec with three blocks */
const char *test_block_left_multiply_vec_three_blocks()
{
    /* A is 2x2 CSR:
     * [1.0  2.0]
     * [3.0  4.0]
     */
    CSR_Matrix *A = new_csr_matrix(2, 2, 4);
    double Ax[4] = {1.0, 2.0, 3.0, 4.0};
    int Ai[4] = {0, 1, 0, 1};
    int Ap[3] = {0, 2, 4};
    memcpy(A->x, Ax, 4 * sizeof(double));
    memcpy(A->i, Ai, 4 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* x is (2*3)-length = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
     * x1 = [1.0, 2.0], x2 = [3.0, 4.0], x3 = [5.0, 6.0]
     */
    double x[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double y[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    block_left_multiply_vec(A, x, y, 3);

    /* Expected block 1: y[0:2] = A @ x1 = [1.0*1.0 + 2.0*2.0, 3.0*1.0 + 4.0*2.0] =
     * [5.0, 11.0] Expected block 2: y[2:4] = A @ x2 = [1.0*3.0 + 2.0*4.0, 3.0*3.0
     * + 4.0*4.0] = [11.0, 25.0] Expected block 3: y[4:6] = A @ x3 = [1.0*5.0
     * + 2.0*6.0, 3.0*5.0 + 4.0*6.0] = [17.0, 39.0]
     */
    double expected_y[6] = {5.0, 11.0, 11.0, 25.0, 17.0, 39.0};
    mu_assert("y values incorrect", cmp_double_array(y, expected_y, 6));

    free_csr_matrix(A);
    return NULL;
}
