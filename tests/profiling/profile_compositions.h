#include <stdio.h>
#include <stdlib.h>

#include "atoms/affine.h"
#include "atoms/bivariate_full_dom.h"
#include "atoms/elementwise_full_dom.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *profile_multiply_D_XA(void)
{
    int n = 100;
    int nn = n * n;

    srand(42);
    expr *X = new_variable(n, n, 0, nn);
    double *A_data = new_random_dense_data(nn);
    double *D_data = new_random_dense_data(nn);
    expr *D = new_constant(n, n, nn, D_data);

    expr *XA = new_left_matmul_dense(X, n, n, A_data);
    expr *obj = new_elementwise_mult(D, XA);

    double *u = new_random_dense_data(nn);
    profile_expr(obj, u, "multiply(D, X@A)  ");

    free(A_data);
    free(D_data);
    free(u);
    free_expr(obj);
    return 0;
}

const char *profile_exp_AX(void)
{
    int n = 75;
    int nn = n * n;

    srand(42);
    expr *X = new_variable(n, n, 0, nn);
    double *A_data = new_random_dense_data(nn);

    expr *AX = new_left_matmul_dense(X, n, n, A_data);
    expr *obj = new_exp(AX);

    double *u = new_random_dense_data(nn);
    profile_expr(obj, u, "exp(A@X)           ");

    free(A_data);
    free(u);
    free_expr(obj);
    return 0;
}

const char *profile_exp_XA(void)
{
    int n = 75;
    int nn = n * n;

    srand(42);
    expr *X = new_variable(n, n, 0, nn);
    double *A_data = new_random_dense_data(nn);

    expr *XA = new_right_matmul_dense(X, n, n, A_data);
    expr *obj = new_exp(XA);

    double *u = new_random_dense_data(nn);
    profile_expr(obj, u, "exp(X@A)           ");

    free(A_data);
    free(u);
    free_expr(obj);
    return 0;
}
