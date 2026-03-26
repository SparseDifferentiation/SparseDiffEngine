#include <string.h>

#include "affine.h"
#include "elementwise_full_dom.h"
#include "minunit.h"
#include "numerical_diff.h"

const char *test_check_jacobian_composite_exp(void)
{
    double u_vals[6] = {0, 0, 1, 2, 3, 0};

    CSR_Matrix *A = new_csr_matrix(2, 6, 6);
    double Ax[6] = {3, 2, 1, 2, 1, 1};
    int Ai[6] = {2, 3, 4, 2, 3, 4};
    int Ap[3] = {0, 3, 6};
    memcpy(A->x, Ax, 6 * sizeof(double));
    memcpy(A->i, Ai, 6 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    expr *u = new_variable(3, 1, 2, 6);
    expr *Au = new_linear(u, A, NULL);
    expr *exp_node = new_exp(Au);

    mu_assert("check_jacobian failed",
              check_jacobian(exp_node, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(exp_node);
    free_csr_matrix(A);
    return 0;
}

const char *test_check_wsum_hess_exp_composite(void)
{
    double u_vals[5] = {1, 2, 3, 4, 5};
    double w[3] = {-1, -2, -3};
    double Ax[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int Ai[] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    int Ap[] = {0, 5, 10, 15};
    CSR_Matrix *A_csr = new_csr_matrix(3, 5, 15);
    memcpy(A_csr->x, Ax, 15 * sizeof(double));
    memcpy(A_csr->i, Ai, 15 * sizeof(int));
    memcpy(A_csr->p, Ap, 4 * sizeof(int));

    expr *x = new_variable(5, 1, 0, 5);
    expr *Ax_node = new_linear(x, A_csr, NULL);
    expr *exp_node = new_exp(Ax_node);

    mu_assert("check_wsum_hess failed",
              check_wsum_hess(exp_node, u_vals, w, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(exp_node);
    free_csr_matrix(A_csr);
    return 0;
}
