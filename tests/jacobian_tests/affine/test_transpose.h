
#ifndef TEST_TRANSPOSE_H
#define TEST_TRANSPOSE_H

#include "atoms/affine.h"
#include "minunit.h"
#include "test_helpers.h"
#include "utils/permuted_dense.h"
#include <math.h>
#include <stdio.h>

const char *test_jacobian_transpose(void)
{
    // A = [1 2; 3 4]
    CSR_matrix *A = new_CSR_matrix(2, 2, 4);
    int A_p[3] = {0, 2, 4};
    int A_i[4] = {0, 1, 0, 1};
    double A_x[4] = {1, 2, 3, 4};
    memcpy(A->p, A_p, 3 * sizeof(int));
    memcpy(A->i, A_i, 4 * sizeof(int));
    memcpy(A->x, A_x, 4 * sizeof(double));

    // X = [1 2; 3 4] (columnwise: x = [1 3 2 4])
    expr *X = new_variable(2, 2, 0, 4);
    expr *AX = new_left_matmul(NULL, X, A);
    expr *transpose_AX = new_transpose(AX);
    double u[4] = {1, 3, 2, 4};
    transpose_AX->forward(transpose_AX, u);
    jacobian_init(transpose_AX);
    transpose_AX->eval_jacobian(transpose_AX);

    // Jacobian of transpose_AX
    double expected_x[8] = {1, 2, 1, 2, 3, 4, 3, 4};
    int expected_p[5] = {0, 2, 4, 6, 8};
    int expected_i[8] = {0, 1, 2, 3, 0, 1, 2, 3};

    mu_assert("vals fail", cmp_values(transpose_AX->jacobian, expected_x, 8));
    mu_assert("sparsity fail",
              cmp_sparsity(transpose_AX->jacobian, expected_p, expected_i, 4, 8));
    free_expr(transpose_AX);
    free_CSR_matrix(A);
    return 0;
}

/* When the child of transpose has a PD Jacobian, the output should also be PD
   with the same col_perm and a permuted row_perm. Setup:
     u : 2x1 column variable, n_vars = 2.
     AU = left_matmul_dense(A, u) with A a 6x2 dense matrix => AU is 6x1.
          PD Jacobian: global (6, 2), m0=6, n0=2, row_perm=[0..5], col_perm=[0,1].
     R  = reshape(AU, 3, 2). copy_sparsity preserves PD.
     T  = transpose(R) with d1=2, d2=3. k(r) = (r/2) + (r%2)*3 = [0,3,1,4,2,5].
          All r are active, so output row_perm stays [0..5] and the dense
          block X is row-permuted: X_out[i, :] = X_c[k(i), :]. */
const char *test_jacobian_transpose_pd_preserved(void)
{
    double A_data[12] = {1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                         7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    expr *u = new_variable(2, 1, 0, 2);
    expr *AU = new_left_matmul_dense(NULL, u, 6, 2, A_data);
    expr *R = new_reshape(AU, 3, 2);
    expr *T = new_transpose(R);

    double u_vals[2] = {0.5, -1.5};
    T->forward(T, u_vals);
    jacobian_init(T);
    T->eval_jacobian(T);

    /* Structural: output Jacobian must be a PD. */
    mu_assert("transpose Jacobian should be PD", T->jacobian->is_permuted_dense);
    permuted_dense *pd_T = (permuted_dense *) T->jacobian;
    mu_assert("global m", T->jacobian->m == 6);
    mu_assert("global n", T->jacobian->n == 2);
    mu_assert("m0", pd_T->m0 == 6);
    mu_assert("n0", pd_T->n0 == 2);
    int expected_row_perm[6] = {0, 1, 2, 3, 4, 5};
    int expected_col_perm[2] = {0, 1};
    mu_assert("row_perm", cmp_int_array(pd_T->row_perm, expected_row_perm, 6));
    mu_assert("col_perm", cmp_int_array(pd_T->col_perm, expected_col_perm, 2));

    /* Numerical: X_out rows = A rows permuted by k(r) = [0,3,1,4,2,5]. */
    double expected_X[12] = {1.0,  2.0,   /* row 0 from A row 0 */
                             7.0,  8.0,   /* row 1 from A row 3 */
                             3.0,  4.0,   /* row 2 from A row 1 */
                             9.0,  10.0,  /* row 3 from A row 4 */
                             5.0,  6.0,   /* row 4 from A row 2 */
                             11.0, 12.0}; /* row 5 from A row 5 */
    mu_assert("X values", cmp_double_array(pd_T->X, expected_X, 12));

    free_expr(T);
    return 0;
}

#endif // TEST_TRANSPOSE_H
