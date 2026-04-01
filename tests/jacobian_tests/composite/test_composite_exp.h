#include <math.h>
#include <stdio.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "test_helpers.h"

const char *test_jacobian_composite_exp(void)
{
    double u_vals[6] = {0, 0, 1, 2, 3, 0};

    CSR_Matrix *A = new_csr_matrix(2, 6, 6, NULL);
    double Ax[6] = {3, 2, 1, 2, 1, 1};
    int Ai[6] = {2, 3, 4, 2, 3, 4};
    int Ap[3] = {0, 3, 6};
    memcpy(A->x, Ax, 6 * sizeof(double));
    memcpy(A->i, Ai, 6 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    expr *u = new_variable(3, 1, 2, 6);
    expr *Au = new_linear(u, A, NULL);
    expr *exp_node = new_exp(Au);
    jacobian_init(exp_node);
    exp_node->forward(exp_node, u_vals);
    exp_node->eval_jacobian(exp_node);

    /* A*u = [10, 7], so exp(A*u) = [exp(10), exp(7)]
     * J = diag(exp(A*u)) * A */
    double e10 = exp(10.0);
    double e7 = exp(7.0);
    double vals[6] = {3 * e10, 2 * e10, 1 * e10, 2 * e7, 1 * e7, 1 * e7};
    int rows[3] = {0, 3, 6};
    int cols[6] = {2, 3, 4, 2, 3, 4};
    mu_assert("vals fail", cmp_double_array(exp_node->jacobian->x, vals, 6));
    mu_assert("rows fail", cmp_int_array(exp_node->jacobian->p, rows, 3));
    mu_assert("cols fail", cmp_int_array(exp_node->jacobian->i, cols, 6));
    free_expr(exp_node);
    free_csr_matrix(A);
    return 0;
}

/* f(u) = exp(A x) + exp(B y) */
const char *test_jacobian_composite_exp_add(void)
{
    double u_vals[7] = {0, 0, 1, 1, 1, 2, 2};

    CSR_Matrix *A = new_csr_matrix(3, 7, 9, NULL);
    double Ax[9] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
    int Ai[9] = {2, 3, 4, 2, 3, 4, 2, 3, 4};
    int Ap[4] = {0, 3, 6, 9};
    memcpy(A->x, Ax, 9 * sizeof(double));
    memcpy(A->i, Ai, 9 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    CSR_Matrix *B = new_csr_matrix(3, 7, 6, NULL);
    double Bx[6] = {1, 1, 2, 2, 3, 3};
    int Bi[6] = {5, 6, 5, 6, 5, 6};
    int Bp[4] = {0, 2, 4, 6};
    memcpy(B->x, Bx, 6 * sizeof(double));
    memcpy(B->i, Bi, 6 * sizeof(int));
    memcpy(B->p, Bp, 4 * sizeof(int));

    expr *x = new_variable(3, 1, 2, 7);
    expr *y = new_variable(2, 1, 5, 7);
    expr *Ax_expr = new_linear(x, A, NULL);
    expr *By_expr = new_linear(y, B, NULL);
    expr *exp_Ax = new_exp(Ax_expr);
    expr *exp_By = new_exp(By_expr);
    expr *sum = new_add(exp_Ax, exp_By);

    mu_assert("check_jacobian failed",
              check_jacobian(sum, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(sum);
    free_csr_matrix(A);
    free_csr_matrix(B);
    return 0;
}
