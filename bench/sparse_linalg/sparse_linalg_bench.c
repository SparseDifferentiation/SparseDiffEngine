/*
 * Copyright 2026 Daniel Cederberg and William Zhang
 *
 * This file is part of the SparseDiffEngine project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

/* sparse_linalg_bench: microbenchmarks for sparse_linalg kernels, built against
   the library alone.

   Usage: sparse_linalg_bench [m n_A n0_B iters [nnz_per_row]]

   Operands: A is an (m x n_A) sparse_matrix with nnz_per_row random distinct
   columns per row, B is an (m x n0_B) full-block permuted_dense, d is all ones.
   Times C = B^T diag(d) A through the polymorphic BTA_matrices_alloc /
   BTDA_matrices_fill_values pair (pd B, sparse A -> the BTDA_pd_csc kernel).
   Without nnz_per_row the two trimmed_log_reg-shaped densities {1, 50} run.
   Prints one key=value line per configuration; with -DSP_TRACK_MEMORY=ON the
   line also reports bytes allocated by the alloc and the transient peak during
   the timed fills (expected 0: fills must not allocate). */

#include <stdio.h>
#include <stdlib.h>

#include "sparse_linalg/CSR_matrix.h"
#include "sparse_linalg/Timer.h"
#include "sparse_linalg/matmul_dispatchers.h"
#include "sparse_linalg/matrix.h"
#include "sparse_linalg/permuted_dense.h"
#include "sparse_linalg/sparse_matrix.h"
#include "sparse_linalg/tracked_alloc.h"

#define BENCH_DEFAULT_M 2000
#define BENCH_DEFAULT_N_A 2000
#define BENCH_DEFAULT_N0_B 785
#define BENCH_DEFAULT_ITERS 50
#define BENCH_SEED 42u
#define BENCH_MAX_ARG 100000000L

static double rand_unit(void)
{
    return (double) rand() / ((double) RAND_MAX + 1.0);
}

/* m x n CSR with exactly nnz_per_row distinct ascending columns per row
   (same construction as tests/profiling/profile_BTA_pd_csr_vs_csc.h).
   Requires nnz_per_row <= n. */
static CSR_matrix *new_csr_fixed_nnz_per_row(int m, int n, int nnz_per_row)
{
    CSR_matrix *A = new_CSR_matrix(m, n, m * nnz_per_row);
    int *cols = (int *) malloc((size_t) nnz_per_row * sizeof(int));
    for (int i = 0; i <= m; i++) A->p[i] = i * nnz_per_row;
    for (int i = 0; i < m; i++)
    {
        int picked = 0;
        while (picked < nnz_per_row)
        {
            int c = rand() % n;
            int dup = 0;
            for (int k = 0; k < picked; k++)
            {
                if (cols[k] == c)
                {
                    dup = 1;
                    break;
                }
            }
            if (!dup) cols[picked++] = c;
        }
        /* Insertion sort keeps the CSR column-index invariant. */
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
            int ii = A->p[i] + k;
            A->i[ii] = cols[k];
            A->x[ii] = rand_unit() - 0.5;
        }
    }
    free(cols);
    return A;
}

static void bench_BTDA_pd_sparse(int m, int n_A, int n0_B, int nnz_per_row,
                                 int iters)
{
    srand(BENCH_SEED); /* identical operands for every configuration and run */

    matrix *A = new_sparse_matrix(new_csr_fixed_nnz_per_row(m, n_A, nnz_per_row));
    size_t XB_count = (size_t) m * (size_t) n0_B;
    double *XB = (double *) malloc(XB_count * sizeof(double));
    for (size_t k = 0; k < XB_count; k++) XB[k] = rand_unit() - 0.5;
    matrix *B = new_permuted_dense_full(m, n0_B, XB);
    double *d = (double *) malloc((size_t) m * sizeof(double));
    for (int i = 0; i < m; i++) d[i] = 1.0;

    Timer timer;
#ifdef SP_TRACK_MEMORY
    size_t live_before_alloc = g_allocated_bytes;
#endif
    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    matrix *C = BTA_matrices_alloc(A, B); /* also builds A's CSC cache structure */
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    double alloc_ms = GET_ELAPSED_SECONDS(timer) * 1e3;
#ifdef SP_TRACK_MEMORY
    size_t alloc_bytes = g_allocated_bytes - live_before_alloc;
#endif

    /* Dispatcher contract: the caller refreshes the sparse operand's CSC values
       before a fill. A never changes here, so once suffices and the loop times
       the kernel alone. */
    A->refresh_csc_values(A);
    BTDA_matrices_fill_values(A, d, B, C); /* warm-up */

#ifdef SP_TRACK_MEMORY
    size_t base = g_allocated_bytes;
    g_peak_bytes = base;
#endif
    double min_ms = 0.0;
    double sum_ms = 0.0;
    for (int it = 0; it < iters; it++)
    {
        clock_gettime(CLOCK_MONOTONIC, &timer.start);
        BTDA_matrices_fill_values(A, d, B, C);
        clock_gettime(CLOCK_MONOTONIC, &timer.end);
        double ms = GET_ELAPSED_SECONDS(timer) * 1e3;
        if (it == 0 || ms < min_ms) min_ms = ms;
        sum_ms += ms;
    }

    printf(
        "kernel=BTDA_matrices_fill_values(pd,sparse) m=%d n_A=%d n0_B=%d "
        "nnz_per_row=%d iters=%d alloc_ms=%.3f fill_min_ms=%.4f fill_mean_ms=%.4f",
        m, n_A, n0_B, nnz_per_row, iters, alloc_ms, min_ms, sum_ms / (double) iters);
#ifdef SP_TRACK_MEMORY
    printf(" alloc_bytes=%zu fill_peak_bytes=%zu\n", alloc_bytes,
           g_peak_bytes - base);
#else
    printf(" alloc_bytes=n/a fill_peak_bytes=n/a\n");
#endif

    free_matrix(C);
    free_matrix(B);
    free_matrix(A); /* also frees the wrapped CSR */
    free(XB);
    free(d);
}

static int parse_positive_int(const char *s, int *out)
{
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || v <= 0 || v > BENCH_MAX_ARG) return 0;
    *out = (int) v;
    return 1;
}

static int usage(const char *argv0)
{
    fprintf(stderr, "usage: %s [m n_A n0_B iters [nnz_per_row]]\n", argv0);
    return 2;
}

int main(int argc, char **argv)
{
    int m = BENCH_DEFAULT_M;
    int n_A = BENCH_DEFAULT_N_A;
    int n0_B = BENCH_DEFAULT_N0_B;
    int iters = BENCH_DEFAULT_ITERS;
    int nnz_list[2] = {1, 50};
    int n_configs = 2;

    if (argc != 1 && argc != 5 && argc != 6) return usage(argv[0]);
    if (argc >= 5 &&
        !(parse_positive_int(argv[1], &m) && parse_positive_int(argv[2], &n_A) &&
          parse_positive_int(argv[3], &n0_B) && parse_positive_int(argv[4], &iters)))
    {
        return usage(argv[0]);
    }
    if (argc == 6)
    {
        if (!parse_positive_int(argv[5], &nnz_list[0])) return usage(argv[0]);
        n_configs = 1;
    }

    for (int k = 0; k < n_configs; k++)
    {
        if (nnz_list[k] > n_A)
        {
            fprintf(stderr, "skipping nnz_per_row=%d > n_A=%d\n", nnz_list[k], n_A);
            continue;
        }
        bench_BTDA_pd_sparse(m, n_A, n0_B, nnz_list[k], iters);
    }
    return 0;
}
