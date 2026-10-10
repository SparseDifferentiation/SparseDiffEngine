#include <stdlib.h>
#include <string.h>

#include "minunit.h"
#include "sparse_linalg/CSR_matrix.h"
#include "test_helpers.h"

const char *test_transpose(void)
{
    CSR_matrix *A = new_CSR_matrix(4, 5, 5);
    double Ax[5] = {1.0, 1.0, 3.0, 2.0, 4.0};
    int Ai[5] = {0, 4, 1, 0, 1};
    int Ap[5] = {0, 2, 3, 4, 5};
    memcpy(A->x, Ax, 5 * sizeof(double));
    memcpy(A->i, Ai, 5 * sizeof(int));
    memcpy(A->p, Ap, 5 * sizeof(int));

    int iwork[5];
    CSR_matrix *AT = transpose(A, iwork);
    double ATx_correct[5] = {1.0, 2.0, 3.0, 4.0, 1.0};
    int ATi_correct[5] = {0, 2, 1, 3, 0};
    int ATp_correct[6] = {0, 2, 4, 4, 4, 5};
    mu_assert("AT vals incorrect", cmp_double_array(AT->x, ATx_correct, 5));
    mu_assert("AT cols incorrect", cmp_int_array(AT->i, ATi_correct, 5));
    mu_assert("AT rows incorrect", cmp_int_array(AT->p, ATp_correct, 6));

    free_CSR_matrix(A);
    free_CSR_matrix(AT);

    return 0;
}

const char *test_AT_alloc_and_fill(void)
{
    /* Create a 3x4 CSR_matrix matrix A:
     * [1.0  0.0  2.0  0.0]
     * [0.0  3.0  0.0  4.0]
     * [5.0  0.0  6.0  0.0]
     */
    CSR_matrix *A = new_CSR_matrix(3, 4, 6);
    double Ax[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int Ai[6] = {0, 2, 1, 3, 0, 2};
    int Ap[4] = {0, 2, 4, 6};
    memcpy(A->x, Ax, 6 * sizeof(double));
    memcpy(A->i, Ai, 6 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    /* Allocate A^T (should be 4x3) */
    int *iwork = (int *) malloc(A->n * sizeof(int));
    CSR_matrix *AT = AT_alloc(A, iwork);

    /* Fill values of A^T */
    AT_fill_values(A, AT, iwork);

    /* Expected A^T:
     * [1.0  0.0  5.0]
     * [0.0  3.0  0.0]
     * [2.0  0.0  6.0]
     * [0.0  4.0  0.0]
     */
    double ATx_correct[6] = {1.0, 5.0, 3.0, 2.0, 6.0, 4.0};
    int ATi_correct[6] = {0, 2, 1, 0, 2, 1};
    int ATp_correct[5] = {0, 2, 3, 5, 6};

    mu_assert("AT dimensions incorrect", AT->m == 4 && AT->n == 3);
    mu_assert("AT nnz incorrect", AT->nnz == 6);
    mu_assert("AT vals incorrect", cmp_double_array(AT->x, ATx_correct, 6));
    mu_assert("AT cols incorrect", cmp_int_array(AT->i, ATi_correct, 6));
    mu_assert("AT rows incorrect", cmp_int_array(AT->p, ATp_correct, 5));

    free_CSR_matrix(A);
    free_CSR_matrix(AT);
    free(iwork);

    return 0;
}
