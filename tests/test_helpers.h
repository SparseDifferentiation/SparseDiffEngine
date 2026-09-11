#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include "expr.h"
#include "utils/CSR_matrix.h"
#include "utils/matrix.h"

/* Compare two double arrays directly
 * Returns 1 if all values match, 0 otherwise */
int cmp_double_array(const double *actual, const double *expected, int size);

/* Compare two int arrays directly
 * Returns 1 if all values match, 0 otherwise */
int cmp_int_array(const int *actual, const int *expected, int size);

/* Verify M has shape (m, *) with exactly nnz entries, and that M's CSR_matrix
 * row pointers and column indices match exp_p (length m+1) and exp_i
 * (length nnz). Returns 1 on full match, 0 otherwise. */
int cmp_sparsity(matrix *M, const int *exp_p, const int *exp_i, int m, int nnz);

/* Verify M has nnz entries and that its value array matches exp_x of
 * length nnz. Returns 1 on full match, 0 otherwise. */
int cmp_values(const matrix *M, const double *exp_x, int nnz);

/* Check the CSR invariants: p[0] == 0, p nondecreasing, p[m] == nnz, and
 * all column indices in [0, n). Returns 1 if valid, 0 otherwise. Use on
 * hand-built fixtures before handing them to the code under test. */
int csr_is_valid(const CSR_matrix *A);

/* Create a random m x n CSR_matrix matrix with approximate nonzero density
 * in [0, 1]. Nonzero values are standard Gaussian (Box-Muller). */
CSR_matrix *new_csr_random(int m, int n, double density);

/* Only available with -DSP_TRACK_MEMORY=ON: reads the tracked allocator
 * counters, which do not exist in a default build. */
#ifdef SP_TRACK_MEMORY
#include "utils/tracked_alloc.h"

/* No-alloc-in-fill contract: after alloc and one warm-up fill, a second fill
 * must not touch the tracked allocator at all. Any transient sp_malloc inside
 * the fill raises g_peak_bytes above the baseline even if freed before
 * returning; a permanent one raises g_allocated_bytes. */
static inline int fill_is_alloc_free(void (*fill)(const void *ctx), const void *ctx)
{
    size_t base = g_allocated_bytes;
    g_peak_bytes = base;
    fill(ctx);
    return g_allocated_bytes == base && g_peak_bytes == base;
}
#endif

#endif /* TEST_HELPERS_H */
