#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "minunit.h"
#include "test_helpers.h"
#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"
#include "utils/Timer.h"
#include "utils/utils.h"

const char *profile_ATA_alloc(void)
{
    int m = 600;
    int n = 600;
    double density = 0.05;

    CSR_Matrix *A_csr = new_csr_random(m, n, density);

    /* Convert to CSC for ATA_alloc */
    int *iwork = (int *) calloc(n, sizeof(int));
    CSC_Matrix *A_csc = csr_to_csc_alloc(A_csr, iwork);
    free(iwork);

    Timer timer;

    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    CSR_Matrix *C_old = ATA_alloc(A_csc);
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    printf("ATA_alloc      (%d x %d, density=%.2f): %8.3f seconds"
           "  (nnz = %d)\n",
           m, n, density, GET_ELAPSED_SECONDS(timer), C_old->nnz);

    free_csr_matrix(C_old);
    free_csc_matrix(A_csc);
    free_csr_matrix(A_csr);
    return 0;
}

const char *profile_ATDA_fill(void)
{
    int m = 300;
    int n = 300;
    double density = 0.05;
    int n_iters = 80;

    CSR_Matrix *A_csr = new_csr_random(m, n, density);

    int *iwork = (int *) calloc(n, sizeof(int));
    CSC_Matrix *A_csc = csr_to_csc_alloc(A_csr, iwork);
    free(iwork);

    CSR_Matrix *C = ATA_alloc(A_csc);

    /* Random diagonal */
    double *d = (double *) malloc(m * sizeof(double));
    for (int i = 0; i < m; i++)
    {
        d[i] = 1.0 + 0.01 * i;
    }

    Timer timer;

    /* Original ATDA_fill_values */
    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    for (int iter = 0; iter < n_iters; iter++)
    {
        ATDA_fill_values(A_csc, d, C);
    }
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    printf("ATDA_fill_values              (%d iters): %8.3f s\n", n_iters,
           GET_ELAPSED_SECONDS(timer));

    /* Save reference values for verification */
    double *ref = (double *) malloc(C->nnz * sizeof(double));
    memcpy(ref, C->x, C->nnz * sizeof(double));

    /* Precompute matching pairs */
    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    ATA_fill_matching_pairs(A_csc, C);
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    printf("ATA_fill_matching_pairs:                  %8.3f s\n",
           GET_ELAPSED_SECONDS(timer));

    /* Matching pairs version */
    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    for (int iter = 0; iter < n_iters; iter++)
    {
        ATDA_fill_values_matching_pairs(A_csc, d, C);
    }
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    printf("ATDA_fill_values_matching     (%d iters): %8.3f s\n", n_iters,
           GET_ELAPSED_SECONDS(timer));

    /* Verify all values match */
    mu_assert("ATDA matching pairs values mismatch",
              cmp_double_array(C->x, ref, C->nnz));
    free(ref);

    free(d);
    free_csr_matrix(C);
    free_csc_matrix(A_csc);
    free_csr_matrix(A_csr);
    return 0;
}
