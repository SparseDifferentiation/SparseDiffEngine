#ifndef TEST_WSUM_HESS_TRANSPOSE_H
#define TEST_WSUM_HESS_TRANSPOSE_H

#include "atoms/affine.h"
#include "minunit.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>

const char *test_wsum_hess_transpose(void)
{

    expr *X = new_variable(2, 2, 0, 8);
    expr *Y = new_variable(2, 2, 4, 8);

    expr *XY = new_matmul(X, Y);
    expr *XYT = new_transpose(XY);

    double u[8] = {1, 3, 2, 4, 5, 7, 6, 8};
    XYT->forward(XYT, u);
    jacobian_init(XYT);
    wsum_hess_init(XYT);
    double w[4] = {1, 2, 3, 4};
    XYT->eval_wsum_hess(XYT, w);

    double expected_x[16] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 3, 1, 3, 2, 4, 2, 4};
    int expected_p[9] = {0, 2, 4, 6, 8, 10, 12, 14, 16};
    int expected_i[16] = {4, 6, 4, 6, 5, 7, 5, 7, 0, 1, 2, 3, 0, 1, 2, 3};

    mu_assert("vals fail", cmp_values(XYT->wsum_hess, expected_x, 16));
    mu_assert("sparsity fail",
              cmp_sparsity(XYT->wsum_hess, expected_p, expected_i, 8, 16));
    free_expr(XYT);

    return 0;
}

#endif // TEST_WSUM_HESS_TRANSPOSE_H
