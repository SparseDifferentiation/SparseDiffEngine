
#ifndef TEST_TRANSPOSE_H
#define TEST_TRANSPOSE_H

#include "affine.h"
#include "bivariate.h"
#include "minunit.h"
#include "subexpr.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

const char *test_jacobian_transpose()
{
    /* A = [1 2; 3 4] as dense 2x2 CSR */
    CSR_Matrix *A = new_csr_matrix(2, 2, 4);
    int Ap[3] = {0, 2, 4};
    int Ai[4] = {0, 1, 0, 1};
    double Ax[4] = {1.0, 2.0, 3.0, 4.0};
    memcpy(A->p, Ap, 3 * sizeof(int));
    memcpy(A->i, Ai, 4 * sizeof(int));
    memcpy(A->x, Ax, 4 * sizeof(double));

    // X = [1 2; 3 4] (columnwise: x = [1 3 2 4])
    expr *X = new_variable(2, 2, 0, 4);
    expr *AX = new_left_matmul(NULL, X, A);
    free_csr_matrix(A);
    expr *transpose_AX = new_transpose(AX);
    double u[4] = {1, 3, 2, 4};
    transpose_AX->forward(transpose_AX, u);
    transpose_AX->jacobian_init(transpose_AX);
    transpose_AX->eval_jacobian(transpose_AX);

    // Jacobian of transpose_AX
    double expected_x[8] = {1, 2, 1, 2, 3, 4, 3, 4};
    int expected_p[5] = {0, 2, 4, 6, 8};
    int expected_i[8] = {0, 1, 2, 3, 0, 1, 2, 3};

    mu_assert("jacobian values fail",
              cmp_double_array(transpose_AX->jacobian->x, expected_x, 8));
    mu_assert("jacobian row ptr fail",
              cmp_int_array(transpose_AX->jacobian->p, expected_p, 5));
    mu_assert("jacobian col idx fail",
              cmp_int_array(transpose_AX->jacobian->i, expected_i, 8));
    free_expr(transpose_AX);
    return 0;
}

#endif // TEST_TRANSPOSE_H
