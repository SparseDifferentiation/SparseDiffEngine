#ifndef PROFILE_BTA_PD_CSR_VS_CSC_H
#define PROFILE_BTA_PD_CSR_VS_CSC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "Timer.h"
#include "minunit.h"
#include "old-code/old_permuted_dense.h"
#include "sparse_linalg/CSR_matrix.h"
#include "sparse_linalg/matmul_dispatchers.h"
#include "sparse_linalg/permuted_dense.h"
#include "sparse_linalg/sparse_matrix.h"
#include "test_helpers.h"

/* Microbenchmark: compare the legacy CSR-pd BTA kernel (old-code) against the
   library's dispatcher path (BTA_matrices_alloc / BTDA_matrices_fill_values on
   a sparse_matrix, i.e. the BTA_pd_csc kernel) on trimmed_log_reg-shaped
   (m=2000, n0_B=785) inputs at two A densities. Output is one fill timing
   pair per row of the table. */
static void run_bench_one_density(int m, int n0_B, int n_A, int nnz_per_row,
                                  int N_ITERS, const char *label)
{
    /* B: PD with full m × n0_B dense block (row_perm = 0..m-1, col_perm =
       0..n0_B-1). Values arbitrary. */
    int *row_perm_B = (int *) malloc(m * sizeof(int));
    int *col_perm_B = (int *) malloc(n0_B * sizeof(int));
    double *XB = (double *) malloc(m * n0_B * sizeof(double));
    for (int i = 0; i < m; i++) row_perm_B[i] = i;
    for (int j = 0; j < n0_B; j++) col_perm_B[j] = j;
    for (int k = 0; k < m * n0_B; k++) XB[k] = (double) (k % 37) * 0.013 + 0.1;

    /* B's global shape: (m, n_B_global). Pick n_B_global = n0_B (no padding). */
    matrix *B_m = new_permuted_dense(m, n0_B, m, n0_B, row_perm_B, col_perm_B, XB);
    permuted_dense *B = (permuted_dense *) B_m;

    /* A: (m × n_A) CSR with `nnz_per_row` evenly-spaced nonzeros per row. */
    int total_nnz = m * nnz_per_row;
    CSR_matrix *A_csr = new_CSR_matrix(m, n_A, total_nnz);
    for (int row = 0; row <= m; row++) A_csr->p[row] = row * nnz_per_row;
    srand(42);
    for (int row = 0; row < m; row++)
    {
        /* Pick nnz_per_row distinct columns by sorted random sampling. */
        int *cols = (int *) malloc(nnz_per_row * sizeof(int));
        int picked = 0;
        while (picked < nnz_per_row)
        {
            int c = rand() % n_A;
            int dup = 0;
            for (int k = 0; k < picked; k++)
                if (cols[k] == c)
                {
                    dup = 1;
                    break;
                }
            if (!dup) cols[picked++] = c;
        }
        /* Insertion sort to keep CSR column-index invariant. */
        for (int a = 1; a < nnz_per_row; a++)
        {
            int v = cols[a];
            int b = a - 1;
            while (b >= 0 && cols[b] > v)
            {
                cols[b + 1] = cols[b];
                b--;
            }
            cols[b + 1] = v;
        }
        for (int k = 0; k < nnz_per_row; k++)
        {
            int e = A_csr->p[row] + k;
            A_csr->i[e] = cols[k];
            A_csr->x[e] = (double) ((row * 31 + cols[k]) % 53) * 0.027 + 0.05;
        }
        free(cols);
    }

    /* Dispatcher operand: a sparse_matrix around a copy of A (the CSR kernel
       keeps A_csr). The dispatcher builds the CSC cache during alloc; values
       are refreshed once since A never changes. */
    matrix *A_m = new_sparse_matrix(new_csr(A_csr));

    /* Allocate outputs once for each variant. */
    matrix *C_csr_m = BTA_pd_csr_alloc(B, A_csr);
    permuted_dense *C_csr = (permuted_dense *) C_csr_m;
    matrix *C_csc_m = BTA_matrices_alloc(A_m, B_m);
    A_m->refresh_csc_values(A_m);

    /* d for BTDA: all ones, so C = B^T diag(d) A = B^T A. */
    double *d_ones = (double *) malloc(m * sizeof(double));
    for (int i = 0; i < m; i++) d_ones[i] = 1.0;

    /* Warm-up + time CSR fill. */
    Timer t1;
    BTDA_pd_csr_fill_values(B, d_ones, A_csr, C_csr);
    clock_gettime(CLOCK_MONOTONIC, &t1.start);
    for (int it = 0; it < N_ITERS; it++)
        BTDA_pd_csr_fill_values(B, d_ones, A_csr, C_csr);
    clock_gettime(CLOCK_MONOTONIC, &t1.end);
    double t_csr_ms = GET_ELAPSED_SECONDS(t1) * 1000.0 / N_ITERS;

    /* Warm-up + time dispatcher (CSC kernel) fill. */
    Timer t2;
    BTDA_matrices_fill_values(A_m, d_ones, B_m, C_csc_m);
    clock_gettime(CLOCK_MONOTONIC, &t2.start);
    for (int it = 0; it < N_ITERS; it++)
        BTDA_matrices_fill_values(A_m, d_ones, B_m, C_csc_m);
    clock_gettime(CLOCK_MONOTONIC, &t2.end);
    double t_csc_ms = GET_ELAPSED_SECONDS(t2) * 1000.0 / N_ITERS;

    printf("  %-22s CSR = %7.3f ms   dispatcher(CSC) = %7.3f ms   ratio = %.2fx\n",
           label, t_csr_ms, t_csc_ms, t_csr_ms / t_csc_ms);

    free_matrix(C_csr_m);
    free_matrix(C_csc_m);
    free_matrix(B_m);
    free_matrix(A_m);
    free_CSR_matrix(A_csr);
    free(row_perm_B);
    free(col_perm_B);
    free(XB);
    free(d_ones);
}

const char *profile_BTA_pd_csr_vs_csc(void)
{
    int m = 2000;
    int n0_B = 785;
    int n_A = 2000;
    int N_ITERS = 50;

    printf("\nBTA pd × sparse fill benchmark (m=%d, n0_B=%d, n_A=%d, %d iters):\n",
           m, n0_B, n_A, N_ITERS);
    run_bench_one_density(m, n0_B, n_A, 1, N_ITERS, "leaf-var (1 nnz/row):");
    run_bench_one_density(m, n0_B, n_A, 50, N_ITERS, "dense-ish (50 nnz/row):");
    return 0;
}

#endif /* PROFILE_BTA_PD_CSR_VS_CSC_H */
