#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "minunit.h"
#include "test_helpers.h"
#include "utils/COO_Matrix.h"

const char *test_csr_to_coo()
{
    /* Create a 3x3 CSR matrix A:
     * [1.0  2.0  0.0]
     * [0.0  3.0  4.0]
     * [5.0  0.0  6.0]
     */
    CSR_Matrix *A = new_csr_matrix(3, 3, 6);
    double Ax[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int Ai[6] = {0, 1, 1, 2, 0, 2};
    int Ap[4] = {0, 2, 4, 6};
    memcpy(A->x, Ax, 6 * sizeof(double));
    memcpy(A->i, Ai, 6 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    COO_Matrix *coo = new_coo_matrix(A);

    mu_assert("m incorrect", coo->m == 3);
    mu_assert("n incorrect", coo->n == 3);
    mu_assert("nnz incorrect", coo->nnz == 6);

    int expected_rows[6] = {0, 0, 1, 1, 2, 2};
    int expected_cols[6] = {0, 1, 1, 2, 0, 2};
    double expected_x[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    mu_assert("rows incorrect",
              cmp_int_array(coo->rows, expected_rows, 6));
    mu_assert("cols incorrect",
              cmp_int_array(coo->cols, expected_cols, 6));
    mu_assert("vals incorrect",
              cmp_double_array(coo->x, expected_x, 6));

    free_coo_matrix(coo);
    free_csr_matrix(A);

    return 0;
}

const char *test_csr_to_coo_lower_triangular()
{
    /* Symmetric 3x3 matrix:
     * [1  2  3]
     * [2  5  6]
     * [3  6  9]
     */
    CSR_Matrix *A = new_csr_matrix(3, 3, 9);
    int Ap[4] = {0, 3, 6, 9};
    int Ai[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    double Ax[9] = {1, 2, 3, 2, 5, 6, 3, 6, 9};
    memcpy(A->p, Ap, 4 * sizeof(int));
    memcpy(A->i, Ai, 9 * sizeof(int));
    memcpy(A->x, Ax, 9 * sizeof(double));

    COO_Matrix *coo = new_coo_matrix_lower_triangular(A);

    mu_assert("ltri m incorrect", coo->m == 3);
    mu_assert("ltri n incorrect", coo->n == 3);
    mu_assert("ltri nnz incorrect", coo->nnz == 6);

    int expected_rows[6] = {0, 1, 1, 2, 2, 2};
    int expected_cols[6] = {0, 0, 1, 0, 1, 2};
    double expected_x[6] = {1, 2, 5, 3, 6, 9};
    int expected_map[6] = {0, 3, 4, 6, 7, 8};

    mu_assert("ltri rows incorrect",
              cmp_int_array(coo->rows, expected_rows, 6));
    mu_assert("ltri cols incorrect",
              cmp_int_array(coo->cols, expected_cols, 6));
    mu_assert("ltri vals incorrect",
              cmp_double_array(coo->x, expected_x, 6));
    mu_assert("ltri value_map incorrect",
              cmp_int_array(coo->value_map, expected_map, 6));

    free_coo_matrix(coo);
    free_csr_matrix(A);

    return 0;
}

const char *test_refresh_lower_triangular_coo()
{
    CSR_Matrix *A = new_csr_matrix(3, 3, 9);
    int Ap[4] = {0, 3, 6, 9};
    int Ai[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    double Ax[9] = {1, 2, 3, 2, 5, 6, 3, 6, 9};
    memcpy(A->p, Ap, 4 * sizeof(int));
    memcpy(A->i, Ai, 9 * sizeof(int));
    memcpy(A->x, Ax, 9 * sizeof(double));

    COO_Matrix *coo = new_coo_matrix_lower_triangular(A);

    double vals2[9] = {10, 20, 30, 20, 50, 60, 30, 60, 90};
    refresh_lower_triangular_coo(coo, vals2);

    double expected_x[6] = {10, 20, 50, 30, 60, 90};
    mu_assert("refresh vals incorrect",
              cmp_double_array(coo->x, expected_x, 6));

    free_coo_matrix(coo);
    free_csr_matrix(A);

    return 0;
}
