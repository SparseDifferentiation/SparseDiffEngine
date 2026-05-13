#ifndef TEST_MATRIX_H
#define TEST_MATRIX_H

#include "minunit.h"
#include "test_helpers.h"
#include "utils/dense_matrix.h"
#include "utils/permuted_dense.h"
#include "utils/sparse_matrix.h"
#include <stdlib.h>
#include <string.h>

/* Test dense block_left_mult_vec against known result.
   A = [1 2; 3 4] (2x2), x = [1; 2], p = 1
   y = A * x = [1*1+2*2; 3*1+4*2] = [5; 11] */
const char *test_dense_matrix_mult_vec(void)
{
    double data[] = {1.0, 2.0, 3.0, 4.0};
    matrix *A = new_dense_matrix(2, 2, data);

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
    matrix *A = new_dense_matrix(2, 2, data);

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
    /* Build CSR_matrix for A = [1 2 3; 4 5 6] */
    CSR_matrix *csr = new_CSR_matrix(2, 3, 6);
    int Ap[3] = {0, 3, 6};
    int Ai[6] = {0, 1, 2, 0, 1, 2};
    double Ax[6] = {1, 2, 3, 4, 5, 6};
    memcpy(csr->p, Ap, 3 * sizeof(int));
    memcpy(csr->i, Ai, 6 * sizeof(int));
    memcpy(csr->x, Ax, 6 * sizeof(double));

    double dense_data[] = {1, 2, 3, 4, 5, 6};

    matrix *sparse = new_sparse_matrix(csr);
    matrix *dense = new_dense_matrix(2, 3, dense_data);

    double x[] = {1.0, 2.0, 3.0};
    double y_sparse[2] = {0};
    double y_dense[2] = {0};

    sparse->block_left_mult_vec(sparse, x, y_sparse, 1);
    dense->block_left_mult_vec(dense, x, y_dense, 1);

    mu_assert("sparse vs dense mismatch", cmp_double_array(y_sparse, y_dense, 2));

    free_matrix(sparse);
    free_matrix(dense);
    return 0;
}

/* Test dense transpose */
const char *test_dense_matrix_trans(void)
{
    double data[] = {1, 2, 3, 4, 5, 6}; /* 2x3 */
    matrix *A = new_dense_matrix(2, 3, data);
    matrix *AT = dense_matrix_trans((const dense_matrix *) A);

    mu_assert("transpose m", AT->m == 3);
    mu_assert("transpose n", AT->n == 2);

    /* AT should be [1 4; 2 5; 3 6] stored row-major */
    dense_matrix *dm = (dense_matrix *) AT;
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
    CSR_matrix *csr = new_CSR_matrix(2, 2, 4);
    int Ap[3] = {0, 2, 4};
    int Ai[4] = {0, 1, 0, 1};
    double Ax[4] = {1, 2, 3, 4};
    memcpy(csr->p, Ap, 3 * sizeof(int));
    memcpy(csr->i, Ai, 4 * sizeof(int));
    memcpy(csr->x, Ax, 4 * sizeof(double));

    double dense_data[] = {1, 2, 3, 4};

    matrix *sparse = new_sparse_matrix(csr);
    matrix *dense = new_dense_matrix(2, 2, dense_data);

    double x[] = {1.0, 2.0, 3.0, 4.0};
    double y_sparse[4] = {0};
    double y_dense[4] = {0};

    sparse->block_left_mult_vec(sparse, x, y_sparse, 2);
    dense->block_left_mult_vec(dense, x, y_dense, 2);

    mu_assert("sparse vs dense blocks mismatch",
              cmp_double_array(y_sparse, y_dense, 4));

    free_matrix(sparse);
    free_matrix(dense);
    return 0;
}

/* Full-block permuted_dense acting as operator must be byte-equivalent to
   dense_matrix for all three block_left_mult_* slots. Mirrors the data of
   test_dense_matrix_mult_vec and exercises the new PD-as-operator path. */
const char *test_pd_operator_block_left_mult_vec(void)
{
    double data[] = {1.0, 2.0, 3.0, 4.0};
    int row_perm[2] = {0, 1};
    int col_perm[2] = {0, 1};
    matrix *A = new_permuted_dense(2, 2, 2, 2, row_perm, col_perm, data);

    double x[] = {1.0, 2.0};
    double y[2] = {0.0, 0.0};

    A->block_left_mult_vec(A, x, y, 1);

    double y_expected[2] = {5.0, 11.0};
    mu_assert("y incorrect", cmp_double_array(y, y_expected, 2));

    free_matrix(A);
    return 0;
}

/* Full-block PD operator vs dense_matrix: block_left_mult_sparsity and
   block_left_mult_values must produce byte-equivalent CSC outputs.
   J is a 6x2 CSC representing two identity-like columns into a single
   block (p=1), exercising both the single-nonzero fast path and the
   multi-nonzero densify path. */
const char *test_pd_operator_vs_dense_block_left_mult(void)
{
    /* A = [1 2 3; 4 5 6] (2x3). */
    double data[] = {1, 2, 3, 4, 5, 6};
    int row_perm[2] = {0, 1};
    int col_perm[3] = {0, 1, 2};
    matrix *A_pd = new_permuted_dense(2, 3, 2, 3, row_perm, col_perm, data);
    matrix *A_dm = new_dense_matrix(2, 3, data);

    /* J is 3x2 CSC: col 0 = [1.0 at row 0], col 1 = [2.0 at row 0, 3.0 at row 2].
       p = 1; output C is 2x2. */
    CSC_matrix *J = new_CSC_matrix(3, 2, 3);
    int Jp[3] = {0, 1, 3};
    int Ji[3] = {0, 0, 2};
    double Jx[3] = {1.0, 2.0, 3.0};
    memcpy(J->p, Jp, 3 * sizeof(int));
    memcpy(J->i, Ji, 3 * sizeof(int));
    memcpy(J->x, Jx, 3 * sizeof(double));

    CSC_matrix *C_pd = A_pd->block_left_mult_sparsity(A_pd, J, 1);
    CSC_matrix *C_dm = A_dm->block_left_mult_sparsity(A_dm, J, 1);

    mu_assert("nnz mismatch", C_pd->nnz == C_dm->nnz);
    mu_assert("p mismatch", cmp_int_array(C_pd->p, C_dm->p, 3));
    mu_assert("i mismatch", cmp_int_array(C_pd->i, C_dm->i, C_pd->nnz));

    A_pd->block_left_mult_values(A_pd, J, C_pd);
    A_dm->block_left_mult_values(A_dm, J, C_dm);

    mu_assert("x mismatch", cmp_double_array(C_pd->x, C_dm->x, C_pd->nnz));

    free_CSC_matrix(C_pd);
    free_CSC_matrix(C_dm);
    free_CSC_matrix(J);
    free_matrix(A_pd);
    free_matrix(A_dm);
    return 0;
}

#endif /* TEST_MATRIX_H */
