#ifndef TEST_MATRIX_H
#define TEST_MATRIX_H

#include "minunit.h"
#include "test_helpers.h"
#include "utils/dense_matrix.h"
#include <stdlib.h>
#include <string.h>

/* Test dense block_left_mult_vec against known result.
   A = [1 2; 3 4] (2x2), x = [1; 2], p = 1
   y = A * x = [1*1+2*2; 3*1+4*2] = [5; 11] */
const char *test_dense_matrix_mult_vec(void)
{
    double data[] = {1.0, 2.0, 3.0, 4.0};
    Matrix *A = new_dense_matrix(2, 2, data);

    double x[] = {1.0, 2.0};
    double y[2] = {0.0, 0.0};

    A->block_left_mult_vec(A, x, y, 1);

    double y_expected[2] = {5.0, 11.0};
    mu_assert("y incorrect", cmp_double_array(y, y_expected, 2));

    free_matrix(A);
    return 0;
}

/* Test dense block_left_mult_vec with multiple blocks.
   A = [1 2; 3 4] (2x2), x = [1; 2; 3; 4], p = 2
   y = [A*[1;2]; A*[3;4]] = [5; 11; 11; 25] */
const char *test_dense_matrix_mult_vec_blocks(void)
{
    double data[] = {1.0, 2.0, 3.0, 4.0};
    Matrix *A = new_dense_matrix(2, 2, data);

    double x[] = {1.0, 2.0, 3.0, 4.0};
    double y[4] = {0};

    A->block_left_mult_vec(A, x, y, 2);

    double y_expected[4] = {5.0, 11.0, 11.0, 25.0};
    mu_assert("y incorrect", cmp_double_array(y, y_expected, 4));

    free_matrix(A);
    return 0;
}

/* Compare sparse vs dense block_left_mult_vec for a non-square matrix.
   A = [1 2 3; 4 5 6] (2x3), x = [1; 2; 3], p = 1 */
const char *test_sparse_vs_dense_mult_vec(void)
{
    /* Build CSR for A = [1 2 3; 4 5 6] */
    CSR_Matrix *csr = new_csr_matrix(2, 3, 6, NULL);
    int Ap[3] = {0, 3, 6};
    int Ai[6] = {0, 1, 2, 0, 1, 2};
    double Ax[6] = {1, 2, 3, 4, 5, 6};
    memcpy(csr->p, Ap, 3 * sizeof(int));
    memcpy(csr->i, Ai, 6 * sizeof(int));
    memcpy(csr->x, Ax, 6 * sizeof(double));

    double dense_data[] = {1, 2, 3, 4, 5, 6};

    Matrix *sparse = new_sparse_matrix(csr);
    Matrix *dense = new_dense_matrix(2, 3, dense_data);

    double x[] = {1.0, 2.0, 3.0};
    double y_sparse[2] = {0};
    double y_dense[2] = {0};

    sparse->block_left_mult_vec(sparse, x, y_sparse, 1);
    dense->block_left_mult_vec(dense, x, y_dense, 1);

    mu_assert("sparse vs dense mismatch", cmp_double_array(y_sparse, y_dense, 2));

    free_matrix(sparse);
    free_matrix(dense);
    free_csr_matrix(csr);
    return 0;
}

/* Test dense transpose */
const char *test_dense_matrix_trans(void)
{
    double data[] = {1, 2, 3, 4, 5, 6}; /* 2x3 */
    Matrix *A = new_dense_matrix(2, 3, data);
    Matrix *AT = dense_matrix_trans((const Dense_Matrix *) A);

    mu_assert("transpose m", AT->m == 3);
    mu_assert("transpose n", AT->n == 2);

    /* AT should be [1 4; 2 5; 3 6] stored row-major */
    Dense_Matrix *dm = (Dense_Matrix *) AT;
    double AT_expected[6] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    mu_assert("AT vals incorrect", cmp_double_array(dm->x, AT_expected, 6));

    free_matrix(A);
    free_matrix(AT);
    return 0;
}

/* Compare sparse vs dense block_left_mult_vec with p=2 blocks.
   A = [1 2; 3 4], x = [1; 2; 3; 4], p = 2 */
const char *test_sparse_vs_dense_mult_vec_blocks(void)
{
    CSR_Matrix *csr = new_csr_matrix(2, 2, 4, NULL);
    int Ap[3] = {0, 2, 4};
    int Ai[4] = {0, 1, 0, 1};
    double Ax[4] = {1, 2, 3, 4};
    memcpy(csr->p, Ap, 3 * sizeof(int));
    memcpy(csr->i, Ai, 4 * sizeof(int));
    memcpy(csr->x, Ax, 4 * sizeof(double));

    double dense_data[] = {1, 2, 3, 4};

    Matrix *sparse = new_sparse_matrix(csr);
    Matrix *dense = new_dense_matrix(2, 2, dense_data);

    double x[] = {1.0, 2.0, 3.0, 4.0};
    double y_sparse[4] = {0};
    double y_dense[4] = {0};

    sparse->block_left_mult_vec(sparse, x, y_sparse, 2);
    dense->block_left_mult_vec(dense, x, y_dense, 2);

    mu_assert("sparse vs dense blocks mismatch",
              cmp_double_array(y_sparse, y_dense, 4));

    free_matrix(sparse);
    free_matrix(dense);
    free_csr_matrix(csr);
    return 0;
}

#endif /* TEST_MATRIX_H */
