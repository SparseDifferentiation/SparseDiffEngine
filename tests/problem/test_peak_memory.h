#ifndef TEST_PEAK_MEMORY_H
#define TEST_PEAK_MEMORY_H

#include <stdlib.h>

#include "atoms/affine.h"
#include "expr.h"
#include "minunit.h"
#include "problem.h"
#include "utils/tracked_alloc.h"

/* Peak-memory regression for the kron-path Jacobians (compact blocks, no
   inverse-permutation arrays).

   Row-sum + col-sum constraints on a matrix variable X (a x b) both take the
   dense left_matmul kron path: each Jacobian is a stacked_pd of p blocks
   (p = b and p = a) whose global column space is all of n_vars = a * b.
   Before the kron blocks were compact, every block eagerly carried col_inv
   (n_vars ints) plus row_inv, so init peaked at O((a + b) * a * b) ints of
   pure index metadata against O(a * b) true nnz: measured 1324 bytes/var
   (21.7 MB) at a = b = 128. With compact blocks the same init peaks at 292
   bytes/var (4.8 MB), dominated by inherent O(n_vars) structure (the
   children's identity Jacobians and their CSC caches). The 400 bytes/var
   bound sits between the two with wide margins on both sides. */
const char *test_peak_memory_kron_jacobian(void)
{
    const int a = 128;
    const int b = 128;
    const int n_vars = a * b;

    double *ones_a = (double *) malloc(a * sizeof(double));
    double *ones_b = (double *) malloc(b * sizeof(double));
    for (int i = 0; i < a; i++) ones_a[i] = 1.0;
    for (int j = 0; j < b; j++) ones_b[j] = 1.0;

    expr *x_obj = new_variable(a, b, 0, n_vars);
    expr *objective = new_sum(x_obj, -1);

    /* col sums: ones(1, a) @ X, shape (1, b) -> kron path with p = b blocks */
    expr *X1 = new_variable(a, b, 0, n_vars);
    expr *colsum = new_left_matmul_dense(NULL, X1, 1, a, ones_a);

    /* row sums: ones(1, b) @ X^T, shape (1, a) -> kron path with p = a blocks */
    expr *X2 = new_variable(a, b, 0, n_vars);
    expr *rowsum = new_left_matmul_dense(NULL, new_transpose(X2), 1, b, ones_b);

    expr *constraints[2] = {colsum, rowsum};
    problem *prob = new_problem(objective, constraints, 2, false);
    mu_assert("new_problem failed", prob != NULL);

    /* new_problem rebaselined g_peak_bytes; measure the growth that
       derivative initialization adds on top of the live bytes here. */
    size_t base = g_allocated_bytes;
    problem_init_derivatives(prob);
    size_t init_peak_growth = g_peak_bytes > base ? g_peak_bytes - base : 0;

    mu_assert("kron Jacobian init peak exceeds 400 bytes per variable",
              init_peak_growth < (size_t) 400 * (size_t) n_vars);

    free_problem(prob);
    free(ones_a);
    free(ones_b);
    return 0;
}

#endif /* TEST_PEAK_MEMORY_H */
