#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "minunit.h"
#include "test_helpers.h"
#include "utils/CSC_matrix.h"
#include "utils/CSR_matrix.h"
#include "utils/iVec.h"
#include "utils/linalg_sparse_matmuls.h"
#include "utils/tracked_alloc.h"
#include "utils/utils.h"

/* Test block_left_multiply_fill_sparsity with simple case: single block */
const char *test_block_left_multiply_single_block(void)
{
    /* A is 2x3 CSR_matrix:
     * [1.0  0.0  0.0]
     * [0.0  1.0  1.0]
     */
    CSR_matrix *A = new_CSR_matrix(2, 3, 3);
    double Ax[3] = {1.0, 1.0, 1.0};
    int Ai[3] = {0, 1, 2};
    int Ap[3] = {0, 1, 3};
    memcpy(A->x, Ax, 3 * sizeof(double));
    memcpy(A->i, Ai, 3 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* J is 3x2 CSC_matrix (single block, so p=1):
     * [1.0  0.0]
     * [1.0  0.0]
     * [0.0  1.0]
     */
    CSC_matrix *J = new_CSC_matrix(3, 2, 3);
    double Jx[3] = {1.0, 1.0, 1.0};
    int Ji[3] = {0, 1, 2};
    int Jp[3] = {0, 2, 3};
    memcpy(J->x, Jx, 3 * sizeof(double));
    memcpy(J->i, Ji, 3 * sizeof(int));
    memcpy(J->p, Jp, 3 * sizeof(int));

    /* Compute C = A @ J1 (p=1 means just one block) */
    CSC_matrix *C = block_left_multiply_fill_sparsity(A, J, 1);

    /* Expected C is 2x2:
     * C[0,0] = A[0,:] @ J[:,0] = 1.0 * 1.0 = 1.0 (row 0 has column 0, J col 0 has
     * row 0) C[1,0] = A[1,:] @ J[:,0] = 1.0*1.0 + 1.0*0.0 = 1.0 (row 1 has columns
     * 1,2, J col 0 has row 1) C[0,1] = A[0,:] @ J[:,1] = 0.0 (row 0 has column 0, J
     * col 1 has row 2) C[1,1] = A[1,:] @ J[:,1] = 1.0*1.0 = 1.0 (row 1 has columns
     * 1,2, J col 1 has row 2)
     */
    int expected_p[3] = {0, 2, 3};
    int expected_i[3] = {0, 1, 1};

    mu_assert("C dims incorrect", C->m == 2 && C->n == 2 && C->nnz == 3);
    mu_assert("C col pointers incorrect", cmp_int_array(C->p, expected_p, 3));
    mu_assert("C row indices incorrect", cmp_int_array(C->i, expected_i, 3));

    free_CSC_matrix(C);
    free_CSR_matrix(A);
    free_CSC_matrix(J);
    return NULL;
}

/* Test block_left_multiply_fill_sparsity with two blocks */
const char *test_block_left_multiply_two_blocks(void)
{
    /* A is 2x2 CSR_matrix:
     * [1.0  0.0]
     * [0.0  1.0]
     */
    CSR_matrix *A = new_CSR_matrix(2, 2, 2);
    double Ax[2] = {1.0, 1.0};
    int Ai[2] = {0, 1};
    int Ap[3] = {0, 1, 2};
    memcpy(A->x, Ax, 2 * sizeof(double));
    memcpy(A->i, Ai, 2 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* J is 4x3 CSC_matrix (two blocks of 2 rows each):
     * Block 1 rows [0,1]:
     * [1.0  0.0  0.0]
     * [0.0  0.0  0.0]
     * Block 2 rows [2,3]:
     * [0.0  1.0  0.0]
     * [0.0  0.0  1.0]
     * So J is:
     * [1.0  0.0  0.0]
     * [0.0  0.0  0.0]
     * [0.0  1.0  0.0]
     * [0.0  0.0  1.0]
     */
    CSC_matrix *J = new_CSC_matrix(4, 3, 3);
    double Jx[3] = {1.0, 1.0, 1.0};
    int Ji[3] = {0, 2, 3};
    int Jp[4] = {0, 1, 2, 3};
    memcpy(J->x, Jx, 3 * sizeof(double));
    memcpy(J->i, Ji, 3 * sizeof(int));
    memcpy(J->p, Jp, 4 * sizeof(int));

    /* Compute C = [A @ J1; A @ J2] where:
     * J1 = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
     * J2 = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
     *
     * C = [A @ J1; A @ J2] is 4x3:
     * A @ J1 = [[1, 0, 0], [0, 0, 0]] (row 0 of A matches col 0 of J1)
     * A @ J2 = [[0, 0, 0], [0, 1, 1]] (row 1 of A matches cols 1,2 of J2)
     * So C is:
     * [1.0  0.0  0.0]
     * [0.0  0.0  0.0]
     * [0.0  0.0  0.0]
     * [0.0  1.0  1.0]
     */
    CSC_matrix *C = block_left_multiply_fill_sparsity(A, J, 2);
    block_left_multiply_fill_values(A, J, C);

    int expected_p2[4] = {0, 1, 2, 3};
    int expected_i2[3] = {0, 2, 3};
    double expected_x2[3] = {1.0, 1.0, 1.0};

    mu_assert("C dims incorrect", C->m == 4 && C->n == 3 && C->nnz == 3);
    mu_assert("C col pointers incorrect", cmp_int_array(C->p, expected_p2, 4));
    mu_assert("C row indices incorrect", cmp_int_array(C->i, expected_i2, 3));
    mu_assert("C values incorrect", cmp_double_array(C->x, expected_x2, 3));

    free_CSC_matrix(C);
    free_CSR_matrix(A);
    free_CSC_matrix(J);
    return NULL;
}

/* Test block_left_multiply_fill_sparsity with all zero column in J */
const char *test_block_left_multiply_zero_column(void)
{
    /* A is 2x2 CSR_matrix (identity) */
    CSR_matrix *A = new_CSR_matrix(2, 2, 2);
    double Ax[2] = {1.0, 1.0};
    int Ai[2] = {0, 1};
    int Ap[3] = {0, 1, 2};
    memcpy(A->x, Ax, 2 * sizeof(double));
    memcpy(A->i, Ai, 2 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* J is 2x2 with an empty column:
     * [1.0  0.0]
     * [0.0  0.0]
     */
    CSC_matrix *J = new_CSC_matrix(2, 2, 1);
    double Jx[1] = {1.0};
    int Ji[1] = {0};
    int Jp[3] = {0, 1, 1}; /* Column 0 has one nonzero, column 1 is empty */
    memcpy(J->x, Jx, 1 * sizeof(double));
    memcpy(J->i, Ji, 1 * sizeof(int));
    memcpy(J->p, Jp, 3 * sizeof(int));

    CSC_matrix *C = block_left_multiply_fill_sparsity(A, J, 1);

    int expected_p3[3] = {0, 1, 1};
    int expected_i3[1] = {0};

    mu_assert("C dims incorrect", C->m == 2 && C->n == 2 && C->nnz == 1);
    mu_assert("C col pointers incorrect", cmp_int_array(C->p, expected_p3, 3));
    mu_assert("C row indices incorrect", cmp_int_array(C->i, expected_i3, 1));

    free_CSC_matrix(C);
    free_CSR_matrix(A);
    free_CSC_matrix(J);
    return NULL;
}

/* Test csr_csc_matmul_alloc: C = A @ B where A is CSR_matrix and B is CSC_matrix */
const char *test_csr_csc_matmul_alloc_basic(void)
{
    /* A is 3x2 CSR_matrix:
     * [1.0  0.0]
     * [0.0  1.0]
     * [1.0  1.0]
     */
    CSR_matrix *A = new_CSR_matrix(3, 2, 4);
    double Ax[4] = {1.0, 1.0, 1.0, 1.0};
    int Ai[4] = {0, 1, 0, 1};
    int Ap[4] = {0, 1, 2, 4};
    memcpy(A->x, Ax, 4 * sizeof(double));
    memcpy(A->i, Ai, 4 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    /* B is 2x3 CSC_matrix:
     * [1.0  0.0  1.0]
     * [0.0  1.0  1.0]
     */
    CSC_matrix *B = new_CSC_matrix(2, 3, 4);
    double Bx[4] = {1.0, 1.0, 1.0, 1.0};
    int Bi[4] = {0, 1, 0, 1};
    int Bp[4] = {0, 1, 2, 4};
    memcpy(B->x, Bx, 4 * sizeof(double));
    memcpy(B->i, Bi, 4 * sizeof(int));
    memcpy(B->p, Bp, 4 * sizeof(int));

    /* C = A @ B is 3x3:
     * C = [[1, 0, 1],
     *      [0, 1, 1],
     *      [1, 1, 2]]
     */
    CSR_matrix *C = csr_csc_matmul_alloc(A, B);

    int expected_p4[4] = {0, 2, 4, 7};
    int expected_i4[7] = {0, 2, 1, 2, 0, 1, 2};

    mu_assert("C dims incorrect", C->m == 3 && C->n == 3 && C->nnz == 7);
    mu_assert("C row pointers incorrect", cmp_int_array(C->p, expected_p4, 4));
    mu_assert("C col indices incorrect", cmp_int_array(C->i, expected_i4, 7));

    free_CSR_matrix(C);
    free_CSR_matrix(A);
    free_CSC_matrix(B);
    return NULL;
}

/* Test csr_csc_matmul_alloc with sparse result */
const char *test_csr_csc_matmul_alloc_sparse(void)
{
    /* A is 2x3 CSR_matrix:
     * [1.0  0.0  0.0]
     * [0.0  0.0  1.0]
     */
    CSR_matrix *A = new_CSR_matrix(2, 3, 2);
    double Ax[2] = {1.0, 1.0};
    int Ai[2] = {0, 2};
    int Ap[3] = {0, 1, 2};
    memcpy(A->x, Ax, 2 * sizeof(double));
    memcpy(A->i, Ai, 2 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* B is 3x2 CSC_matrix:
     * [1.0  0.0]
     * [0.0  0.0]
     * [0.0  1.0]
     */
    CSC_matrix *B = new_CSC_matrix(3, 2, 2);
    double Bx[2] = {1.0, 1.0};
    int Bi[2] = {0, 2};
    int Bp[3] = {0, 1, 2};
    memcpy(B->x, Bx, 2 * sizeof(double));
    memcpy(B->i, Bi, 2 * sizeof(int));
    memcpy(B->p, Bp, 3 * sizeof(int));

    /* C = A @ B is 2x2:
     * C = [[1, 0],
     *      [0, 1]]
     */
    CSR_matrix *C = csr_csc_matmul_alloc(A, B);

    int expected_p5[3] = {0, 1, 2};
    int expected_i5[2] = {0, 1};

    mu_assert("C dims incorrect", C->m == 2 && C->n == 2 && C->nnz == 2);
    mu_assert("C row pointers incorrect", cmp_int_array(C->p, expected_p5, 3));
    mu_assert("C col indices incorrect", cmp_int_array(C->i, expected_i5, 2));

    free_CSR_matrix(C);
    free_CSR_matrix(A);
    free_CSC_matrix(B);
    return NULL;
}

/* Dedup + ordering: one block whose child entries hit the same A row through
 * several columns (dedup), with CSC column lists that interleave so a missing
 * sort would emit rows out of ascending order. */
const char *test_block_left_multiply_dedup_order(void)
{
    /* A is 4x3 CSR_matrix:
     * [1.0  0.0  2.0]     CSC col 0: rows {0, 2}
     * [0.0  3.0  0.0]     CSC col 1: rows {1, 3}
     * [4.0  0.0  0.0]     CSC col 2: rows {0, 3}
     * [0.0  5.0  6.0]
     */
    CSR_matrix *A = new_CSR_matrix(4, 3, 6);
    double Ax[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int Ai[6] = {0, 2, 1, 0, 1, 2};
    int Ap[5] = {0, 2, 3, 4, 6};
    memcpy(A->x, Ax, 6 * sizeof(double));
    memcpy(A->i, Ai, 6 * sizeof(int));
    memcpy(A->p, Ap, 5 * sizeof(int));

    /* J is 3x1 dense column (p=1): child entries 0, 1, 2. Gathering CSC columns
     * in child order visits rows 0,2 then 1,3 then 0,3 — rows 0 and 3 twice
     * (dedup) and out of order (sort). Expected: all four rows, ascending. */
    CSC_matrix *J = new_CSC_matrix(3, 1, 3);
    double Jx[3] = {1.0, 1.0, 1.0};
    int Ji[3] = {0, 1, 2};
    int Jp[2] = {0, 3};
    memcpy(J->x, Jx, 3 * sizeof(double));
    memcpy(J->i, Ji, 3 * sizeof(int));
    memcpy(J->p, Jp, 2 * sizeof(int));

    CSC_matrix *C = block_left_multiply_fill_sparsity(A, J, 1);

    int expected_p[2] = {0, 4};
    int expected_i[4] = {0, 1, 2, 3};

    mu_assert("C dims incorrect", C->m == 4 && C->n == 1 && C->nnz == 4);
    mu_assert("C col pointers incorrect", cmp_int_array(C->p, expected_p, 2));
    mu_assert("C row indices incorrect", cmp_int_array(C->i, expected_i, 4));

    free_CSC_matrix(C);
    free_CSR_matrix(A);
    free_CSC_matrix(J);
    return NULL;
}

/* Reference implementation of the block sparsity (the original per-row
 * has_overlap scan) used to cross-check the gather rewrite on random input. */
static CSC_matrix *block_left_multiply_fill_sparsity_ref(const CSR_matrix *A,
                                                         const CSC_matrix *J, int p)
{
    int m = A->m;
    int n = A->n;

    int *Cp = (int *) sp_malloc((J->n + 1) * sizeof(int));
    iVec *Ci = iVec_new(J->nnz > 0 ? J->nnz : 1);
    Cp[0] = 0;

    for (int j = 0; j < J->n; j++)
    {
        if (J->p[j] == J->p[j + 1])
        {
            Cp[j + 1] = Cp[j];
            continue;
        }
        int jj = J->p[j];
        for (int block = 0; block < p; block++)
        {
            int block_start = block * n;
            int block_end = block_start + n;
            while (jj < J->p[j + 1] && J->i[jj] < block_start) jj++;
            int block_jj_start = jj;
            while (jj < J->p[j + 1] && J->i[jj] < block_end) jj++;
            int nnz_in_block = jj - block_jj_start;
            if (nnz_in_block == 0) continue;
            for (int i = 0; i < m; i++)
            {
                int a_len = A->p[i + 1] - A->p[i];
                if (has_overlap(A->i + A->p[i], a_len, J->i + block_jj_start,
                                nnz_in_block, block_start))
                    iVec_append(Ci, block * m + i);
            }
        }
        Cp[j + 1] = Ci->len;
    }

    CSC_matrix *C = new_CSC_matrix(m * p, J->n, Ci->len);
    memcpy(C->p, Cp, (J->n + 1) * sizeof(int));
    memcpy(C->i, Ci->data, Ci->len * sizeof(int));
    sp_free(Cp);
    iVec_free(Ci);
    return C;
}

/* Random cross-check: the gather rewrite must be byte-identical to the
 * original has_overlap scan. */
const char *test_block_left_multiply_matches_reference_random(void)
{
    srand(42);
    int n = 15, p = 3, k = 8;
    int iwork_size;

    for (int trial = 0; trial < 5; trial++)
    {
        CSR_matrix *A = new_csr_random(20, n, 0.2);
        CSR_matrix *G = new_csr_random(n * p, k, 0.15);

        iwork_size = G->n;
        int *iwork =
            (int *) sp_malloc((iwork_size > 0 ? iwork_size : 1) * sizeof(int));
        CSC_matrix *J = csr_to_csc_alloc(G, iwork);
        sp_free(iwork);

        CSC_matrix *C = block_left_multiply_fill_sparsity(A, J, p);
        CSC_matrix *C_ref = block_left_multiply_fill_sparsity_ref(A, J, p);

        mu_assert("random block sparsity nnz mismatch", C->nnz == C_ref->nnz);
        mu_assert("random block sparsity col pointers mismatch",
                  cmp_int_array(C->p, C_ref->p, J->n + 1));
        mu_assert("random block sparsity row indices mismatch",
                  cmp_int_array(C->i, C_ref->i, C_ref->nnz));

        free_CSC_matrix(C);
        free_CSC_matrix(C_ref);
        free_CSC_matrix(J);
        free_CSR_matrix(G);
        free_CSR_matrix(A);
    }
    return NULL;
}

/* Dedup + ordering for csr_csc_matmul_alloc: A's row hits the same B column
 * through several B rows (dedup), with B's CSR row lists interleaved so a
 * missing sort would emit columns out of ascending order. */
const char *test_csr_csc_matmul_alloc_dedup_order(void)
{
    /* A is 1x2 CSR_matrix: [1.0  1.0] */
    CSR_matrix *A = new_CSR_matrix(1, 2, 2);
    double Ax[2] = {1.0, 1.0};
    int Ai[2] = {0, 1};
    int Ap[2] = {0, 2};
    memcpy(A->x, Ax, 2 * sizeof(double));
    memcpy(A->i, Ai, 2 * sizeof(int));
    memcpy(A->p, Ap, 2 * sizeof(int));

    /* B is 2x3 CSC_matrix:
     * [1.0  0.0  1.0]     CSR row 0: cols {0, 2}
     * [0.0  1.0  1.0]     CSR row 1: cols {1, 2}
     * Gathering rows 0 then 1 visits cols 0,2 then 1,2 — col 2 twice (dedup)
     * and out of order (sort). Expected: cols {0, 1, 2} ascending. */
    CSC_matrix *B = new_CSC_matrix(2, 3, 4);
    double Bx[4] = {1.0, 1.0, 1.0, 1.0};
    int Bi[4] = {0, 1, 0, 1};
    int Bp[4] = {0, 1, 2, 4};
    memcpy(B->x, Bx, 4 * sizeof(double));
    memcpy(B->i, Bi, 4 * sizeof(int));
    memcpy(B->p, Bp, 4 * sizeof(int));

    CSR_matrix *C = csr_csc_matmul_alloc(A, B);

    int expected_p[2] = {0, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("C dims incorrect", C->m == 1 && C->n == 3 && C->nnz == 3);
    mu_assert("C row pointers incorrect", cmp_int_array(C->p, expected_p, 2));
    mu_assert("C col indices incorrect", cmp_int_array(C->i, expected_i, 3));

    free_CSR_matrix(C);
    free_CSR_matrix(A);
    free_CSC_matrix(B);
    return NULL;
}

/* Reference implementation of csr_csc_matmul_alloc (the original full
 * row-times-column has_overlap loop) for the random cross-check. */
static CSR_matrix *csr_csc_matmul_alloc_ref(const CSR_matrix *A, const CSC_matrix *B)
{
    int m = A->m;
    int p = B->n;

    int *Cp = (int *) sp_malloc((m + 1) * sizeof(int));
    iVec *Ci = iVec_new(m);
    Cp[0] = 0;

    int nnz = 0;
    for (int i = 0; i < m; i++)
    {
        int len_a = A->p[i + 1] - A->p[i];
        for (int j = 0; j < p; j++)
        {
            int len_b = B->p[j + 1] - B->p[j];
            if (has_overlap(A->i + A->p[i], len_a, B->i + B->p[j], len_b, 0))
            {
                iVec_append(Ci, j);
                nnz++;
            }
        }
        Cp[i + 1] = nnz;
    }

    CSR_matrix *C = new_CSR_matrix(m, p, nnz);
    memcpy(C->p, Cp, (m + 1) * sizeof(int));
    memcpy(C->i, Ci->data, nnz * sizeof(int));
    sp_free(Cp);
    iVec_free(Ci);
    return C;
}

/* Random cross-check: the gather rewrite must be byte-identical to the
 * original has_overlap pair loop. */
const char *test_csr_csc_matmul_alloc_matches_reference_random(void)
{
    srand(7);

    for (int trial = 0; trial < 5; trial++)
    {
        CSR_matrix *A = new_csr_random(20, 15, 0.2);
        CSR_matrix *G = new_csr_random(15, 12, 0.2);

        int *iwork = (int *) sp_malloc(G->n * sizeof(int));
        CSC_matrix *B = csr_to_csc_alloc(G, iwork);
        sp_free(iwork);

        CSR_matrix *C = csr_csc_matmul_alloc(A, B);
        CSR_matrix *C_ref = csr_csc_matmul_alloc_ref(A, B);

        mu_assert("random matmul sparsity nnz mismatch", C->nnz == C_ref->nnz);
        mu_assert("random matmul sparsity row pointers mismatch",
                  cmp_int_array(C->p, C_ref->p, A->m + 1));
        mu_assert("random matmul sparsity col indices mismatch",
                  cmp_int_array(C->i, C_ref->i, C_ref->nnz));

        free_CSR_matrix(C);
        free_CSR_matrix(C_ref);
        free_CSC_matrix(B);
        free_CSR_matrix(G);
        free_CSR_matrix(A);
    }
    return NULL;
}

/* Test block_left_multiply_vec with single block: y = A @ x */
const char *test_block_left_multiply_vec_single_block(void)
{
    /* A is 2x3 CSR_matrix:
     * [1.0  0.0  2.0]
     * [0.0  3.0  0.0]
     */
    CSR_matrix *A = new_CSR_matrix(2, 3, 3);
    double Ax[3] = {1.0, 3.0, 2.0};
    int Ai[3] = {0, 1, 2};
    int Ap[3] = {0, 2, 3};
    memcpy(A->x, Ax, 3 * sizeof(double));
    memcpy(A->i, Ai, 3 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* x is (3*1)-length vector = [1.0, 2.0, 3.0] */
    double x[3] = {1.0, 2.0, 3.0};
    double y[2] = {0.0, 0.0};

    block_left_multiply_vec(A, x, y, 1);

    /* Expected: y = [1.0*1.0 + 0.0*2.0 + 2.0*3.0, 0.0*1.0 + 3.0*2.0 + 0.0*3.0]
     *             = [1.0 + 6.0, 6.0]
     *             = [7.0, 6.0]
     */
    double expected_y[2] = {7.0, 6.0};
    mu_assert("y values incorrect", cmp_double_array(y, expected_y, 2));

    free_CSR_matrix(A);
    return NULL;
}

/* Test block_left_multiply_vec with two blocks: y = [A @ x1; A @ x2] */
const char *test_block_left_multiply_vec_two_blocks(void)
{
    /* A is 2x3 CSR_matrix:
     * [1.0  2.0  0.0]
     * [0.0  3.0  4.0]
     */
    CSR_matrix *A = new_CSR_matrix(2, 3, 4);
    double Ax[4] = {1.0, 2.0, 3.0, 4.0};
    int Ai[4] = {0, 1, 1, 2};
    int Ap[3] = {0, 2, 4};
    memcpy(A->x, Ax, 4 * sizeof(double));
    memcpy(A->i, Ai, 4 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* x is (3*2)-length vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
     * x1 = [1.0, 2.0, 3.0], x2 = [4.0, 5.0, 6.0]
     */
    double x[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double y[4] = {0.0, 0.0, 0.0, 0.0};

    block_left_multiply_vec(A, x, y, 2);

    /* Expected block 1: y[0:2] = A @ x1 = [1.0*1.0 + 2.0*2.0, 3.0*2.0 + 4.0*3.0] =
     * [5.0, 18.0] Expected block 2: y[2:4] = A @ x2 = [1.0*4.0 + 2.0*5.0, 3.0*5.0
     * + 4.0*6.0] = [14.0, 39.0]
     */
    double expected_y[4] = {5.0, 18.0, 14.0, 39.0};
    mu_assert("y values incorrect", cmp_double_array(y, expected_y, 4));

    free_CSR_matrix(A);
    return NULL;
}

/* Test block_left_multiply_vec with sparse matrix and multiple blocks */
const char *test_block_left_multiply_vec_sparse(void)
{
    /* A is 3x4 CSR_matrix (very sparse):
     * [2.0  0.0  0.0  0.0]
     * [0.0  0.0  3.0  0.0]
     * [0.0  0.0  0.0  4.0]
     */
    CSR_matrix *A = new_CSR_matrix(3, 4, 3);
    double Ax[3] = {2.0, 3.0, 4.0};
    int Ai[3] = {0, 2, 3};
    int Ap[4] = {0, 1, 2, 3};
    memcpy(A->x, Ax, 3 * sizeof(double));
    memcpy(A->i, Ai, 3 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    /* x is (4*2)-length = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
     * x1 = [1.0, 2.0, 3.0, 4.0], x2 = [5.0, 6.0, 7.0, 8.0]
     */
    double x[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double y[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    block_left_multiply_vec(A, x, y, 2);

    /* Expected block 1: y[0:3] = A @ x1 = [2.0*1.0, 3.0*3.0, 4.0*4.0] =
     * [2.0, 9.0, 16.0] Expected block 2: y[3:6] = A @ x2 =
     * [2.0*5.0, 3.0*7.0, 4.0*8.0] = [10.0, 21.0, 32.0]
     */
    double expected_y[6] = {2.0, 9.0, 16.0, 10.0, 21.0, 32.0};
    mu_assert("y values incorrect", cmp_double_array(y, expected_y, 6));

    free_CSR_matrix(A);
    return NULL;
}

/* Test block_left_multiply_vec with three blocks */
const char *test_block_left_multiply_vec_three_blocks(void)
{
    /* A is 2x2 CSR_matrix:
     * [1.0  2.0]
     * [3.0  4.0]
     */
    CSR_matrix *A = new_CSR_matrix(2, 2, 4);
    double Ax[4] = {1.0, 2.0, 3.0, 4.0};
    int Ai[4] = {0, 1, 0, 1};
    int Ap[3] = {0, 2, 4};
    memcpy(A->x, Ax, 4 * sizeof(double));
    memcpy(A->i, Ai, 4 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* x is (2*3)-length = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
     * x1 = [1.0, 2.0], x2 = [3.0, 4.0], x3 = [5.0, 6.0]
     */
    double x[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double y[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    block_left_multiply_vec(A, x, y, 3);

    /* Expected block 1: y[0:2] = A @ x1 = [1.0*1.0 + 2.0*2.0, 3.0*1.0 + 4.0*2.0] =
     * [5.0, 11.0] Expected block 2: y[2:4] = A @ x2 = [1.0*3.0 + 2.0*4.0, 3.0*3.0
     * + 4.0*4.0] = [11.0, 25.0] Expected block 3: y[4:6] = A @ x3 = [1.0*5.0
     * + 2.0*6.0, 3.0*5.0 + 4.0*6.0] = [17.0, 39.0]
     */
    double expected_y[6] = {5.0, 11.0, 11.0, 25.0, 17.0, 39.0};
    mu_assert("y values incorrect", cmp_double_array(y, expected_y, 6));

    free_CSR_matrix(A);
    return NULL;
}
