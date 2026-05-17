#ifndef PROFILE_HESSIAN_EXP_AX_H
#define PROFILE_HESSIAN_EXP_AX_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "expr.h"
#include "minunit.h"
#include "utils/CSR_matrix.h"
#include "utils/Timer.h"
#include "utils/permuted_dense.h"
#include "utils/stacked_pd.h"
#include "utils/tracked_alloc.h"

/* Profile and validate two ways to compute the Hessian of
   w^T exp(A @ X) for n x n matrices A, X with n = 50.

   - Approach 1: build expr `exp(left_matmul(A, X))`, drive
     jacobian_init + wsum_hess_init (alloc) and forward + eval_jacobian
     + eval_wsum_hess (fill). Uses the existing CSR/CSC path.

   - Approach 2: manually construct the Jacobian
     J = ∂vec(A @ X)/∂vec(X) = I_n ⊗ A as a stacked_pd with n blocks
     (each block carries A's values at the corresponding row/col
     window). Call ATA_spd_alloc(J) (alloc) and ATDA_spd_fill_values(J,
     d, H) (fill), where d_k = w_k * exp((A@X)_k). The AX compute is
     done OUTSIDE the timer per user request.

   Both Hessians are densified to a 2500x2500 row-major buffer and
   compared via max-abs diff (asserted < 1e-9). */
const char *profile_hessian_exp_AX(void)
{
    const int n = 40;
    const int n_vars = n * n;
    const size_t N = (size_t) n_vars * (size_t) n_vars;
    srand(0);

    /* Random A (row-major), X (col-major vec), w. Values in [-0.5, 0.5]
       to keep exp() bounded and finite. */
    double *A_data = (double *) SP_MALLOC(n_vars * sizeof(double));
    double *X_vals = (double *) SP_MALLOC(n_vars * sizeof(double));
    double *w = (double *) SP_MALLOC(n_vars * sizeof(double));
    for (int k = 0; k < n_vars; k++)
    {
        A_data[k] = ((double) rand() / (double) RAND_MAX) - 0.5;
    }
    for (int k = 0; k < n_vars; k++)
    {
        X_vals[k] = ((double) rand() / (double) RAND_MAX) - 0.5;
    }
    for (int k = 0; k < n_vars; k++)
    {
        w[k] = ((double) rand() / (double) RAND_MAX) - 0.5;
    }

    /* ------------------------------------------------------------ */
    /* Approach 1: existing path via the expression tree.            */
    /* ------------------------------------------------------------ */
    expr *X_var = new_variable(n, n, 0, n_vars);
    expr *AX = new_left_matmul_dense(NULL, X_var, n, n, A_data);
    expr *node = new_exp(AX);

    Timer t1a;
    clock_gettime(CLOCK_MONOTONIC, &t1a.start);
    jacobian_init(node);
    wsum_hess_init(node);
    clock_gettime(CLOCK_MONOTONIC, &t1a.end);

    Timer t1f;
    clock_gettime(CLOCK_MONOTONIC, &t1f.start);
    node->forward(node, X_vals);
    node->eval_jacobian(node);
    node->eval_wsum_hess(node, w);
    clock_gettime(CLOCK_MONOTONIC, &t1f.end);

    matrix *H1 = node->wsum_hess;

    /* ------------------------------------------------------------ */
    /* Pre-compute AX and d (column-major) outside the approach-2    */
    /* timer.                                                        */
    /* AX_cm[j*n + i] = sum_k A_data[i*n + k] * X_vals[j*n + k]      */
    /*                = A[i, :] · X[:, j].                           */
    /* ------------------------------------------------------------ */
    double *AX_cm = (double *) SP_MALLOC(n_vars * sizeof(double));
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            double s = 0.0;
            for (int k = 0; k < n; k++)
            {
                s += A_data[i * n + k] * X_vals[j * n + k];
            }
            AX_cm[j * n + i] = s;
        }
    }
    double *d = (double *) SP_MALLOC(n_vars * sizeof(double));
    for (int k = 0; k < n_vars; k++)
    {
        d[k] = w[k] * exp(AX_cm[k]);
    }

    /* ------------------------------------------------------------ */
    /* Approach 2: manual stacked_pd Jacobian + ATA / ATDA.          */
    /* J has n_vars rows and n_vars cols, with n blocks. Block k     */
    /* covers row_perm = col_perm = {k*n, k*n+1, ..., k*n+n-1} and   */
    /* carries A's values (row-major within the n*n block).          */
    /* ------------------------------------------------------------ */
    Timer t2a;
    clock_gettime(CLOCK_MONOTONIC, &t2a.start);
    permuted_dense **j_blocks =
        (permuted_dense **) SP_MALLOC(n * sizeof(permuted_dense *));
    int *row_perm = (int *) SP_MALLOC(n * sizeof(int));
    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < n; i++)
        {
            row_perm[i] = k * n + i;
        }
        j_blocks[k] = (permuted_dense *) new_permuted_dense(
            n_vars, n_vars, n, n, row_perm, row_perm, A_data);
    }
    free(row_perm);
    matrix *J = new_stacked_pd(n_vars, n_vars, n, j_blocks, NULL, NULL);
    free(j_blocks);
    matrix *H2 = ATA_spd_alloc((stacked_pd *) J);
    clock_gettime(CLOCK_MONOTONIC, &t2a.end);

    Timer t2f;
    clock_gettime(CLOCK_MONOTONIC, &t2f.start);
    ATDA_spd_fill_values((stacked_pd *) J, d, (stacked_pd *) H2);
    clock_gettime(CLOCK_MONOTONIC, &t2f.end);

    /* ------------------------------------------------------------ */
    /* Densify both Hessians to a 2500 x 2500 row-major buffer and   */
    /* compute max-abs diff.                                         */
    /* ------------------------------------------------------------ */
    double *D1 = (double *) SP_CALLOC(N, sizeof(double));
    double *D2 = (double *) SP_CALLOC(N, sizeof(double));

    CSR_matrix *csr = H1->to_csr(H1);
    for (int i = 0; i < csr->m; i++)
    {
        for (int kk = csr->p[i]; kk < csr->p[i + 1]; kk++)
        {
            D1[(size_t) i * (size_t) n_vars + (size_t) csr->i[kk]] = csr->x[kk];
        }
    }

    stacked_pd *H2s = (stacked_pd *) H2;
    for (int k = 0; k < H2s->n_blocks; k++)
    {
        permuted_dense *blk = H2s->blocks[k];
        for (int i = 0; i < blk->m0; i++)
        {
            for (int j = 0; j < blk->n0; j++)
            {
                int gi = blk->row_perm[i];
                int gj = blk->col_perm[j];
                D2[(size_t) gi * (size_t) n_vars + (size_t) gj] =
                    blk->X[i * blk->n0 + j];
            }
        }
    }

    double max_diff = 0.0;
    for (size_t k = 0; k < N; k++)
    {
        double diff = fabs(D1[k] - D2[k]);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }

    printf("\n  Hessian of w^T exp(A@X), n=%d  (Hessian is %dx%d)\n", n, n_vars,
           n_vars);
    printf("  approach 1 alloc:  %8.3f ms\n", GET_ELAPSED_SECONDS(t1a) * 1000.0);
    printf("  approach 1 fill:   %8.3f ms\n", GET_ELAPSED_SECONDS(t1f) * 1000.0);
    printf("  approach 2 alloc:  %8.3f ms\n", GET_ELAPSED_SECONDS(t2a) * 1000.0);
    printf("  approach 2 fill:   %8.3f ms\n", GET_ELAPSED_SECONDS(t2f) * 1000.0);
    printf("  max-abs diff:      %.3e\n", max_diff);

    mu_assert("Hessians match", max_diff < 1e-9);

    free(D1);
    free(D2);
    free(d);
    free(AX_cm);
    free_matrix(H2);
    free_matrix(J);
    free_expr(node);
    free(A_data);
    free(X_vals);
    free(w);
    return 0;
}

#endif /* PROFILE_HESSIAN_EXP_AX_H */
