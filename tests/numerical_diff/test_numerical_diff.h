#include <string.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "minunit.h"
#include "numerical_diff.h"

const char *test_check_jacobian_composite_log(void)
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
    expr *log_node = new_log(Au);

    mu_assert("check_jacobian failed",
              check_jacobian(log_node, u_vals, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(log_node);
    free_csr_matrix(A);
    return 0;
}
