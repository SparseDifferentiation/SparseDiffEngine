#ifndef PROFILE_LOG_REG_H
#define PROFILE_LOG_REG_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "expr.h"
#include "minunit.h"
#include "utils/CSR_sum.h"
#include "utils/Timer.h"
#include "utils/permuted_dense.h"

/* Profile and validate Jacobian + Hessian of obj = sum(logistic(A x)).

   Path A: the engine's expression DAG (CSR_matrix/CSC_matrix chain rule).
   Path B: hardcoded chain rule using permuted_dense kernels for the dense
           linear algebra (DA and ATDA), plus the engine's CSR_matrix row-sum
           primitives for J_sum.

   Forward pass is excluded from timing. */
const char *profile_log_reg(void)
{
    int m = 2000;
    int n = 785;

    /* ---- Random A and initial x ---- */
    double *A_data = (double *) malloc((size_t) m * n * sizeof(double));
    double *u = (double *) malloc(n * sizeof(double));
    srand(42);
    for (int i = 0; i < m * n; i++)
    {
        A_data[i] = (double) rand() / RAND_MAX - 0.5;
    }
    for (int i = 0; i < n; i++)
    {
        u[i] = (double) rand() / RAND_MAX - 0.5;
    }

    /* ---- Build expression DAG (shared by both paths) ---- */
    expr *x = new_variable(n, 1, 0, n);
    expr *Ax = new_left_matmul_dense(NULL, x, m, n, A_data);
    expr *log_obj = new_logistic(Ax);
    expr *obj = new_sum(log_obj, -1);
    jacobian_init(obj);
    wsum_hess_init(obj);

    /* Forward (untimed). */
    obj->forward(obj, u);

    /* ---- Path A: time eval_jacobian and eval_wsum_hess separately ---- */
    Timer t_a_jac, t_a_hess;
    double w_one = 1.0;
    clock_gettime(CLOCK_MONOTONIC, &t_a_jac.start);
    obj->eval_jacobian(obj);
    clock_gettime(CLOCK_MONOTONIC, &t_a_jac.end);
    clock_gettime(CLOCK_MONOTONIC, &t_a_hess.start);
    obj->eval_wsum_hess(obj, &w_one);
    clock_gettime(CLOCK_MONOTONIC, &t_a_hess.end);
    double sec_a_jac = GET_ELAPSED_SECONDS(t_a_jac);
    double sec_a_hess = GET_ELAPSED_SECONDS(t_a_hess);

    /* ---- Path B setup (untimed) ---- */
    int *full_rows = (int *) malloc(m * sizeof(int));
    int *full_cols = (int *) malloc(n * sizeof(int));
    for (int i = 0; i < m; i++) full_rows[i] = i;
    for (int j = 0; j < n; j++) full_cols[j] = j;

    matrix *A_pd_M = new_permuted_dense(m, n, m, n, full_rows, full_cols, A_data);
    permuted_dense *A_pd = (permuted_dense *) A_pd_M;
    matrix *Jlog_M = new_permuted_dense(m, n, m, n, full_rows, full_cols, NULL);
    permuted_dense *Jlog_pd = (permuted_dense *) Jlog_M;
    matrix *H_pd_M = permuted_dense_ATA_alloc(A_pd);
    permuted_dense *H_pd = (permuted_dense *) H_pd_M;

    free(full_rows);
    free(full_cols);

    /* CSR_matrix scaffolding for the row-sum step (PD owns the cached CSR_matrix view). */
    CSR_matrix *Jlog_csr = Jlog_M->to_csr(Jlog_M);
    CSR_matrix *Jobj_csr = new_csr_matrix(1, n, n);
    int *iwork = (int *) malloc((size_t) m * n * sizeof(int));
    int *idx_map = (int *) malloc((size_t) m * n * sizeof(int));
    sum_all_rows_csr_alloc(Jlog_csr, Jobj_csr, iwork, idx_map);

    double *d2 = (double *) malloc(m * sizeof(double));
    double *w_ones = (double *) malloc(m * sizeof(double));
    for (int i = 0; i < m; i++) w_ones[i] = 1.0;

    /* ---- Path B: time the manual chain rule, Jacobian and Hessian separately ----
     */
    Timer t_b_jac, t_b_hess;
    /* dwork = sigmoid(z); used as the diagonal in DA below and (still in
       dwork) as sigmas read by local_wsum_hess. */
    clock_gettime(CLOCK_MONOTONIC, &t_b_jac.start);
    log_obj->local_jacobian(log_obj, log_obj->work->dwork);
    permuted_dense_DA_fill_values(log_obj->work->dwork, A_pd, Jlog_pd);
    memset(Jobj_csr->x, 0, Jobj_csr->nnz * sizeof(double));
    accumulator(Jlog_csr->x, Jlog_csr->nnz, idx_map, Jobj_csr->x);
    clock_gettime(CLOCK_MONOTONIC, &t_b_jac.end);
    clock_gettime(CLOCK_MONOTONIC, &t_b_hess.start);
    log_obj->local_wsum_hess(log_obj, d2, w_ones);
    permuted_dense_ATDA_fill_values(A_pd, d2, H_pd);
    clock_gettime(CLOCK_MONOTONIC, &t_b_hess.end);
    double sec_b_jac = GET_ELAPSED_SECONDS(t_b_jac);
    double sec_b_hess = GET_ELAPSED_SECONDS(t_b_hess);

    printf("\n");
    printf("                            Jacobian      Hessian        Total\n");
    printf("  Path A (engine CSR_matrix/CSC_matrix): %10.6fs  %10.6fs  %10.6fs\n", sec_a_jac,
           sec_a_hess, sec_a_jac + sec_a_hess);
    printf("  Path B (permuted_dense): %10.6fs  %10.6fs  %10.6fs\n", sec_b_jac,
           sec_b_hess, sec_b_jac + sec_b_hess);
    printf("  Speedup (A / B):         %10.2fx %10.2fx %10.2fx\n",
           sec_a_jac / sec_b_jac, sec_a_hess / sec_b_hess,
           (sec_a_jac + sec_a_hess) / (sec_b_jac + sec_b_hess));

    /* ---- Compare Jacobian (1 x n, both have full sparsity) ---- */
    CSR_matrix *J_a = obj->jacobian->to_csr(obj->jacobian);
    mu_assert("J n mismatch", J_a->n == Jobj_csr->n);
    mu_assert("J nnz mismatch", J_a->nnz == Jobj_csr->nnz);
    double max_J_diff = 0.0;
    for (int j = 0; j < J_a->nnz; j++)
    {
        double diff = fabs(J_a->x[j] - Jobj_csr->x[j]);
        if (diff > max_J_diff) max_J_diff = diff;
    }
    printf("  Jacobian max abs diff:   %10.3e\n", max_J_diff);
    mu_assert("Jacobian mismatch", max_J_diff < 1e-10);

    /* ---- Compare Hessian (n x n): scatter Path A's CSR_matrix into a dense
       n x n array, compare to H_pd->X (already dense row-major).
       Extract the CSR_matrix view ONCE: PD's to_csr does an O(m0 * n0)
       memcpy refresh per call, so calling it inside the inner loop is
       quadratically expensive. ---- */
    CSR_matrix *H_a = obj->wsum_hess->to_csr(obj->wsum_hess);
    double *H_a_dense = (double *) calloc((size_t) n * n, sizeof(double));
    for (int i = 0; i < n; i++)
    {
        for (int e = H_a->p[i]; e < H_a->p[i + 1]; e++)
        {
            int col = H_a->i[e];
            H_a_dense[i * n + col] = H_a->x[e];
        }
    }
    double max_H_diff = 0.0;
    for (size_t k = 0; k < (size_t) n * n; k++)
    {
        double diff = fabs(H_a_dense[k] - H_pd->X[k]);
        if (diff > max_H_diff) max_H_diff = diff;
    }
    printf("  Hessian max abs diff:    %10.3e\n", max_H_diff);
    mu_assert("Hessian mismatch", max_H_diff < 1e-10);

    /* ---- Cleanup ---- */
    free(H_a_dense);
    free(d2);
    free(w_ones);
    free(iwork);
    free(idx_map);
    free_csr_matrix(Jobj_csr);
    /* Jlog_csr is owned by Jlog_M's cache; released by free_matrix below. */
    free_matrix(H_pd_M);
    free_matrix(Jlog_M);
    free_matrix(A_pd_M);
    free_expr(obj);
    free(A_data);
    free(u);

    return 0;
}

#endif /* PROFILE_LOG_REG_H */
