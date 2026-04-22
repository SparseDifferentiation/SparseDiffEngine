#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "atoms/affine.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_kron_left_forward(void)
{
    /* Test: Z = kron(C, X) where
     * C is 2x2 sparse: [[1, 2], [0, 3]]
     * X is 2x2 variable (col-major): [[1, 3], [2, 4]]
     *
     * kron(C, X) = [[1*X, 2*X], [0*X, 3*X]]
     *            = [[1, 3, 2, 6],
     *               [2, 4, 4, 8],
     *               [0, 0, 3, 9],
     *               [0, 0, 6, 12]]
     */
    expr *X = new_variable(2, 2, 0, 4);

    CSR_Matrix *C = new_csr_matrix(2, 2, 3);
    int C_p[3] = {0, 2, 3};
    int C_i[3] = {0, 1, 1};
    double C_x[3] = {1.0, 2.0, 3.0};
    memcpy(C->p, C_p, 3 * sizeof(int));
    memcpy(C->i, C_i, 3 * sizeof(int));
    memcpy(C->x, C_x, 3 * sizeof(double));

    expr *Z = new_kron_left(NULL, X, C, 2, 2);

    /* X = [[1,3],[2,4]] in column-major */
    double u[4] = {1.0, 2.0, 3.0, 4.0};

    Z->forward(Z, u);

    /* (4x4) column-major */
    double expected[16] = {
        1.0, 2.0, 0.0, 0.0, /* col 0 */
        3.0, 4.0, 0.0, 0.0, /* col 1 */
        2.0, 4.0, 3.0, 6.0, /* col 2 */
        6.0, 8.0, 9.0, 12.0 /* col 3 */
    };

    mu_assert("kron_left d1 != 4", Z->d1 == 4);
    mu_assert("kron_left d2 != 4", Z->d2 == 4);
    mu_assert("kron_left size != 16", Z->size == 16);
    mu_assert("kron_left forward values", cmp_double_array(Z->value, expected, 16));

    free_csr_matrix(C);
    free_expr(Z);
    return 0;
}

const char *test_kron_left_forward_identity(void)
{
    /* Identity path: Z = kron(I_3, X) with X a (2 x 2) variable.
     * Result is block-diagonal: three copies of X stacked along the
     * diagonal. Verifies the sparse-driven path collapses to the
     * right block structure with no identity-detection code. */
    expr *X = new_variable(2, 2, 0, 4);

    CSR_Matrix *I3 = new_csr_matrix(3, 3, 3);
    int Ip[4] = {0, 1, 2, 3};
    int Ii[3] = {0, 1, 2};
    double Ix[3] = {1.0, 1.0, 1.0};
    memcpy(I3->p, Ip, 4 * sizeof(int));
    memcpy(I3->i, Ii, 3 * sizeof(int));
    memcpy(I3->x, Ix, 3 * sizeof(double));

    expr *Z = new_kron_left(NULL, X, I3, 2, 2);

    /* X = [[1,3],[2,4]] in column-major */
    double u[4] = {1.0, 2.0, 3.0, 4.0};
    Z->forward(Z, u);

    /* kron(I_3, X) is 6x6, column-major: each column holds one
     * column of X at a different row offset. */
    double expected[36] = {
        1.0, 2.0, 0.0, 0.0, 0.0, 0.0, /* col 0: X[:,0] at rows 0-1 */
        3.0, 4.0, 0.0, 0.0, 0.0, 0.0, /* col 1: X[:,1] at rows 0-1 */
        0.0, 0.0, 1.0, 2.0, 0.0, 0.0, /* col 2: X[:,0] at rows 2-3 */
        0.0, 0.0, 3.0, 4.0, 0.0, 0.0, /* col 3: X[:,1] at rows 2-3 */
        0.0, 0.0, 0.0, 0.0, 1.0, 2.0, /* col 4: X[:,0] at rows 4-5 */
        0.0, 0.0, 0.0, 0.0, 3.0, 4.0, /* col 5: X[:,1] at rows 4-5 */
    };

    mu_assert("kron_left identity d1 != 6", Z->d1 == 6);
    mu_assert("kron_left identity d2 != 6", Z->d2 == 6);
    mu_assert("kron_left identity values", cmp_double_array(Z->value, expected, 36));

    free_csr_matrix(I3);
    free_expr(Z);
    return 0;
}
