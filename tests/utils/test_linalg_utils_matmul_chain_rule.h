#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "minunit.h"
#include "test_helpers.h"
#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"
#include "utils/linalg_dense_sparse_matmuls.h"

/* Test YT_kron_I_alloc and YT_kron_I_fill_values
 *
 * C = (Y^T kron I_m) @ J
 * m=2, k=2, n=2, p=3
 *
 * Y (k x n, col-major [1,2,3,4]):
 *   [1  3]
 *   [2  4]
 *
 * J (mk=4 x p=3, CSC):
 *   [1  0  2]
 *   [0  1  0]
 *   [3  0  0]
 *   [0  0  1]
 *
 * C = (Y^T kron I_2) @ J:
 *   [ 7  0  2]
 *   [ 0  1  2]
 *   [15  0  6]
 *   [ 0  3  4]
 */
const char *test_YT_kron_I(void)
{
    int m = 2, k = 2, n = 2;

    /* J is 4x3 CSC */
    CSC_Matrix *J = new_csc_matrix(4, 3, 5);
    int Jp[4] = {0, 2, 3, 5};
    int Ji[5] = {0, 2, 1, 0, 3};
    double Jx[5] = {1.0, 3.0, 1.0, 2.0, 1.0};
    memcpy(J->p, Jp, 4 * sizeof(int));
    memcpy(J->i, Ji, 5 * sizeof(int));
    memcpy(J->x, Jx, 5 * sizeof(double));

    /* Y col-major: Y[0,0]=1, Y[1,0]=2, Y[0,1]=3, Y[1,1]=4 */
    double Y[4] = {1.0, 2.0, 3.0, 4.0};

    CSR_Matrix *C = YT_kron_I_alloc(m, k, n, J);

    /* Expected CSR (from scipy) */
    int exp_p[5] = {0, 2, 4, 6, 8};
    int exp_i[8] = {0, 2, 1, 2, 0, 2, 1, 2};
    double exp_x[8] = {7.0, 2.0, 1.0, 2.0, 15.0, 6.0, 3.0, 4.0};

    mu_assert("C dims", C->m == 4 && C->n == 3);
    mu_assert("C nnz", C->nnz == 8);
    mu_assert("C row ptrs", cmp_int_array(C->p, exp_p, 5));
    mu_assert("C col indices", cmp_int_array(C->i, exp_i, 8));

    YT_kron_I_fill_values(m, k, n, Y, J, C);
    mu_assert("C values", cmp_double_array(C->x, exp_x, 8));

    free_csr_matrix(C);
    free_csc_matrix(J);
    return NULL;
}

/* Test YT_kron_I with larger dimensions: m=3, k=2, n=3, p=4
 *
 * Y (k=2 x n=3, col-major [1,3,0.5,1,2,0.5]):
 *   [1.0  0.5  2.0]
 *   [3.0  1.0  0.5]
 *
 * J (mk=6 x p=4, CSC):
 *   [1  0  0  2]
 *   [0  0  1  0]
 *   [0  3  0  0]
 *   [2  0  0  1]
 *   [0  1  0  0]
 *   [0  0  4  0]
 *
 * C = (Y^T kron I_3) @ J  is 9 x 4
 */
const char *test_YT_kron_I_larger(void)
{
    int m = 3, k = 2, n = 3;

    /* J is 6x4 CSC */
    CSC_Matrix *J = new_csc_matrix(6, 4, 8);
    int Jp[5] = {0, 2, 4, 6, 8};
    int Ji[8] = {0, 3, 2, 4, 1, 5, 0, 3};
    double Jx[8] = {1.0, 2.0, 3.0, 1.0, 1.0, 4.0, 2.0, 1.0};
    memcpy(J->p, Jp, 5 * sizeof(int));
    memcpy(J->i, Ji, 8 * sizeof(int));
    memcpy(J->x, Jx, 8 * sizeof(double));

    /* Y col-major */
    double Y[6] = {1.0, 3.0, 0.5, 1.0, 2.0, 0.5};

    CSR_Matrix *C = YT_kron_I_alloc(m, k, n, J);

    /* Expected CSR (from scipy) */
    int exp_p[10] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
    int exp_i[18] = {0, 3, 1, 2, 1, 2, 0, 3, 1, 2, 1, 2, 0, 3, 1, 2, 1, 2};
    double exp_x[18] = {7.0, 5.0, 3.0, 1.0, 3.0, 12.0, 2.5, 2.0, 1.0,
                        0.5, 1.5, 4.0, 3.0, 4.5, 0.5,  2.0, 6.0, 2.0};

    mu_assert("C2 dims", C->m == 9 && C->n == 4);
    mu_assert("C2 nnz", C->nnz == 18);
    mu_assert("C2 row ptrs", cmp_int_array(C->p, exp_p, 10));
    mu_assert("C2 col indices", cmp_int_array(C->i, exp_i, 18));

    YT_kron_I_fill_values(m, k, n, Y, J, C);
    mu_assert("C2 values", cmp_double_array(C->x, exp_x, 18));

    free_csr_matrix(C);
    free_csc_matrix(J);
    return NULL;
}

/* Test I_kron_X_alloc and I_kron_X_fill_values
 *
 * C = (I_n kron X) @ J
 * m=2, k=2, n=2, p=3
 *
 * X (m x k, col-major [1,2,3,4]):
 *   [1  3]
 *   [2  4]
 *
 * J (kn=4 x p=3, CSC):
 *   [1  0  2]
 *   [0  1  0]
 *   [3  0  0]
 *   [0  0  1]
 *
 * C = (I_2 kron X) @ J:
 *   [1  3  2]
 *   [2  4  4]
 *   [3  0  3]
 *   [6  0  4]
 */
const char *test_I_kron_X(void)
{
    int m = 2, k = 2, n = 2;

    /* J is 4x3 CSC */
    CSC_Matrix *J = new_csc_matrix(4, 3, 5);
    int Jp[4] = {0, 2, 3, 5};
    int Ji[5] = {0, 2, 1, 0, 3};
    double Jx[5] = {1.0, 3.0, 1.0, 2.0, 1.0};
    memcpy(J->p, Jp, 4 * sizeof(int));
    memcpy(J->i, Ji, 5 * sizeof(int));
    memcpy(J->x, Jx, 5 * sizeof(double));

    /* X col-major */
    double X[4] = {1.0, 2.0, 3.0, 4.0};

    CSR_Matrix *C = I_kron_X_alloc(m, k, n, J);

    /* Expected CSR */
    int exp_p[5] = {0, 3, 6, 8, 10};
    int exp_i[10] = {0, 1, 2, 0, 1, 2, 0, 2, 0, 2};
    double exp_x[10] = {1.0, 3.0, 2.0, 2.0, 4.0, 4.0, 3.0, 3.0, 6.0, 4.0};

    mu_assert("C dims", C->m == 4 && C->n == 3);
    mu_assert("C nnz", C->nnz == 10);
    mu_assert("C row ptrs", cmp_int_array(C->p, exp_p, 5));
    mu_assert("C col indices", cmp_int_array(C->i, exp_i, 10));

    I_kron_X_fill_values(m, k, n, X, J, C);
    mu_assert("C values", cmp_double_array(C->x, exp_x, 10));

    free_csr_matrix(C);
    free_csc_matrix(J);
    return NULL;
}

/* Test I_kron_X with larger dimensions: m=3, k=2, n=2, p=4
 *
 * X (m=3 x k=2, col-major [1,2,3,0.5,1,0.5]):
 *   [1.0  0.5]
 *   [2.0  1.0]
 *   [3.0  0.5]
 *
 * J (kn=4 x p=4, CSC):
 *   [1  0  0  2]
 *   [0  3  1  0]
 *   [0  0  4  0]
 *   [2  0  0  1]
 *
 * C = (I_2 kron X) @ J  is 6 x 4
 */
const char *test_I_kron_X_larger(void)
{
    int m = 3, k = 2, n = 2;

    /* J is 4x4 CSC */
    CSC_Matrix *J = new_csc_matrix(4, 4, 7);
    int Jp[5] = {0, 2, 3, 5, 7};
    int Ji[7] = {0, 3, 1, 1, 2, 0, 3};
    double Jx[7] = {1.0, 2.0, 3.0, 1.0, 4.0, 2.0, 1.0};
    memcpy(J->p, Jp, 5 * sizeof(int));
    memcpy(J->i, Ji, 7 * sizeof(int));
    memcpy(J->x, Jx, 7 * sizeof(double));

    /* X col-major */
    double X[6] = {1.0, 2.0, 3.0, 0.5, 1.0, 0.5};

    CSR_Matrix *C = I_kron_X_alloc(m, k, n, J);

    /* Expected CSR */
    int exp_p[7] = {0, 4, 8, 12, 15, 18, 21};
    int exp_i[21] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 3, 0, 2, 3, 0, 2, 3};
    double exp_x[21] = {1.0, 1.5, 0.5, 2.0, 2.0, 3.0, 1.0, 4.0, 3.0,  1.5, 0.5,
                        6.0, 1.0, 4.0, 0.5, 2.0, 8.0, 1.0, 1.0, 12.0, 0.5};

    mu_assert("C2 dims", C->m == 6 && C->n == 4);
    mu_assert("C2 nnz", C->nnz == 21);
    mu_assert("C2 row ptrs", cmp_int_array(C->p, exp_p, 7));
    mu_assert("C2 col indices", cmp_int_array(C->i, exp_i, 21));

    I_kron_X_fill_values(m, k, n, X, J, C);
    mu_assert("C2 values", cmp_double_array(C->x, exp_x, 21));

    free_csr_matrix(C);
    free_csc_matrix(J);
    return NULL;
}
