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

    free_csc_matrix(A_csc);
    free_csr_matrix(A_csr);
    return 0;
}
