
#ifndef TEST_TRANSPOSE_H
#define TEST_TRANSPOSE_H

#include "atoms/affine.h"
#include "minunit.h"
#include "test_helpers.h"
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

#endif // TEST_TRANSPOSE_H
