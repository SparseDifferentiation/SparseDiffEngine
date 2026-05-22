#ifndef TEST_OLD_STACKED_PD_H
#define TEST_OLD_STACKED_PD_H

#include "minunit.h"
#include "old-code/old_matrix_BTA.h"
#include "old-code/old_stacked_pd_linalg.h"
#include "test_helpers.h"
#include "utils/CSC_matrix.h"
#include "utils/CSR_matrix.h"
#include "utils/permuted_dense.h"
#include "utils/sparse_matrix.h"
#include "utils/stacked_pd.h"
#include <stdlib.h>
#include <string.h>

/* B is 4x4 spd with two blocks; A is 4x3 CSC chosen so every block of B
   produces a nonempty result. Result spd should keep both source blocks. */
const char *test_BA_spd_csc_two_blocks_both_kept(void)
{
    /* block 0: rows {0, 1}, cols {0, 2}, X = [[1, 2], [3, 4]]
       block 1: rows {3},    cols {1, 3}, X = [5, 6]                    */
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 2};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[1] = {3};
    int col_perm_1[2] = {1, 3};
    double X1[2] = {5.0, 6.0};
    matrix *blk1 = new_permuted_dense(4, 4, 1, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *B_m = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);

    /* A: 4x3 CSC.
       col 0: (row 0, 1), (row 2, 2)
       col 1: (row 1, 3), (row 3, 4)
       col 2: (row 2, 5)
       For block 0 (col_perm {0, 2}): cols 0 and 2 of A hit -> C0 cols {0, 2}.
       For block 1 (col_perm {1, 3}): only col 1 of A hits -> C1 cols {1}. */
    CSC_matrix *A = new_CSC_matrix(4, 3, 5);
    int Ap[4] = {0, 2, 4, 5};
    int Ai[5] = {0, 2, 1, 3, 2};
    double Ax[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    memcpy(A->p, Ap, sizeof(Ap));
    memcpy(A->i, Ai, sizeof(Ai));
    memcpy(A->x, Ax, sizeof(Ax));

    matrix *C_m = BA_spd_csc_alloc((stacked_pd *) B_m, A);
    BA_spd_csc_fill_values((stacked_pd *) B_m, A, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("C m", C_m->m == 4);
    mu_assert("C n", C_m->n == 3);
    mu_assert("n_blocks", C->n_blocks == 2);
    int expected_src_p[3] = {0, 1, 2};
    int expected_src[2] = {0, 1};
    mu_assert("src_block_idx_p",
              cmp_int_array(C->src_block_idx_p, expected_src_p, 3));
    mu_assert("src_block_idx", cmp_int_array(C->src_block_idx, expected_src, 2));

    /* C0: 2x2 dense at rows {0, 1}, cols {0, 2}.
         C0[0,0] = 1*1 + 2*2 =  5
         C0[0,2] = 1*0 + 2*5 = 10
         C0[1,0] = 3*1 + 4*2 = 11
         C0[1,2] = 3*0 + 4*5 = 20                                       */
    permuted_dense *C0 = C->blocks[0];
    int C0_col_expected[2] = {0, 2};
    double C0_X_expected[4] = {5.0, 10.0, 11.0, 20.0};
    mu_assert("C0 m0", C0->m0 == 2);
    mu_assert("C0 n0", C0->n0 == 2);
    mu_assert("C0 row_perm", cmp_int_array(C0->row_perm, row_perm_0, 2));
    mu_assert("C0 col_perm", cmp_int_array(C0->col_perm, C0_col_expected, 2));
    mu_assert("C0 X", cmp_double_array(C0->X, C0_X_expected, 4));

    /* C1: 1x1 dense at row {3}, col {1}.
         C1[3,1] = 5*3 + 6*4 = 39                                       */
    permuted_dense *C1 = C->blocks[1];
    int C1_col_expected[1] = {1};
    double C1_X_expected[1] = {39.0};
    mu_assert("C1 m0", C1->m0 == 1);
    mu_assert("C1 n0", C1->n0 == 1);
    mu_assert("C1 row_perm", cmp_int_array(C1->row_perm, row_perm_1, 1));
    mu_assert("C1 col_perm", cmp_int_array(C1->col_perm, C1_col_expected, 1));
    mu_assert("C1 X", cmp_double_array(C1->X, C1_X_expected, 1));

    free_matrix(C_m);
    free_matrix(B_m);
    free_CSC_matrix(A);
    return 0;
}

/* B has 2 blocks; A intersects only block 0's col_perm. Result drops
   block 1 and the src_block_idx records the surviving source index. */
const char *test_BA_spd_csc_one_block_dropped(void)
{
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 2};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[1] = {3};
    int col_perm_1[2] = {1, 3};
    double X1[2] = {5.0, 6.0};
    matrix *blk1 = new_permuted_dense(4, 4, 1, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *B_m = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);

    /* A: 4x2 CSC, both cols hit only rows {0, 2}. Block 1 (col_perm
       {1, 3}) gets no intersection and is dropped.
       col 0: (0, 1), (2, 2)
       col 1: (0, 3), (2, 4)                                            */
    CSC_matrix *A = new_CSC_matrix(4, 2, 4);
    int Ap[3] = {0, 2, 4};
    int Ai[4] = {0, 2, 0, 2};
    double Ax[4] = {1.0, 2.0, 3.0, 4.0};
    memcpy(A->p, Ap, sizeof(Ap));
    memcpy(A->i, Ai, sizeof(Ai));
    memcpy(A->x, Ax, sizeof(Ax));

    matrix *C_m = BA_spd_csc_alloc((stacked_pd *) B_m, A);
    BA_spd_csc_fill_values((stacked_pd *) B_m, A, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("n_blocks", C->n_blocks == 1);
    int expected_src_p[2] = {0, 1};
    int expected_src[1] = {0};
    mu_assert("src_block_idx_p",
              cmp_int_array(C->src_block_idx_p, expected_src_p, 2));
    mu_assert("src_block_idx", cmp_int_array(C->src_block_idx, expected_src, 1));

    /* C0: 2x2 dense at rows {0, 1}, cols {0, 1}.
         C0[0,0] = 1*1 + 2*2 =  5
         C0[0,1] = 1*3 + 2*4 = 11
         C0[1,0] = 3*1 + 4*2 = 11
         C0[1,1] = 3*3 + 4*4 = 25                                       */
    permuted_dense *C0 = C->blocks[0];
    int C0_col_expected[2] = {0, 1};
    double C0_X_expected[4] = {5.0, 11.0, 11.0, 25.0};
    mu_assert("C0 col_perm", cmp_int_array(C0->col_perm, C0_col_expected, 2));
    mu_assert("C0 X", cmp_double_array(C0->X, C0_X_expected, 4));

    free_matrix(C_m);
    free_matrix(B_m);
    free_CSC_matrix(A);
    return 0;
}

/* A has no nonzeros in any block's col_perm. Result spd has zero blocks
   and is safe to free. */
const char *test_BA_spd_csc_all_blocks_dropped(void)
{
    int row_perm_0[1] = {0};
    int col_perm_0[1] = {0};
    double X0[1] = {1.0};
    matrix *blk0 = new_permuted_dense(4, 4, 1, 1, row_perm_0, col_perm_0, X0);

    int row_perm_1[1] = {2};
    int col_perm_1[1] = {2};
    double X1[1] = {3.0};
    matrix *blk1 = new_permuted_dense(4, 4, 1, 1, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *B_m = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);

    /* A: 4x1 CSC with a single nonzero at row 1; rows {1} disjoint from
       both block col_perms {0} and {2}. */
    CSC_matrix *A = new_CSC_matrix(4, 1, 1);
    A->p[0] = 0;
    A->p[1] = 1;
    A->i[0] = 1;
    A->x[0] = 7.0;

    matrix *C_m = BA_spd_csc_alloc((stacked_pd *) B_m, A);
    BA_spd_csc_fill_values((stacked_pd *) B_m, A, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("C m", C_m->m == 4);
    mu_assert("C n", C_m->n == 1);
    mu_assert("n_blocks", C->n_blocks == 0);
    mu_assert("nnz", C_m->nnz == 0);

    free_matrix(C_m);
    free_matrix(B_m);
    free_CSC_matrix(A);
    return 0;
}

/* BA_spd_pd: C = B(spd) @ A(PD). Per-block thin loop over B's blocks
   delegating to BA_pd_pd_*. Both blocks contribute -> 2 output blocks. */
const char *test_BA_spd_pd_two_blocks_both_kept(void)
{
    /* B: 4x3 spd, two blocks.
       B_0: rows {0,1}, cols {0,2}, X = [[7,8],[9,10]].
       B_1: rows {2,3}, cols {1,2}, X = [[11,12],[13,14]].                  */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {7, 8, 9, 10};
    matrix *B0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);

    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {11, 12, 13, 14};
    matrix *B1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);

    permuted_dense *blocks[2] = {(permuted_dense *) B0, (permuted_dense *) B1};
    matrix *B = new_stacked_pd(4, 3, 2, blocks, NULL, NULL);

    /* A: 3x4 PD with row_perm = {0,1,2} (full), col_perm = {0,2},
       X = [[1,2],[3,4],[5,6]].                                             */
    int A_rp[3] = {0, 1, 2};
    int A_cp[2] = {0, 2};
    double AX[6] = {1, 2, 3, 4, 5, 6};
    matrix *A = new_permuted_dense(3, 4, 3, 2, A_rp, A_cp, AX);

    matrix *C_m = BA_spd_pd_alloc((stacked_pd *) B, (permuted_dense *) A);
    BA_spd_pd_fill_values((stacked_pd *) B, (permuted_dense *) A,
                          (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("C m", C_m->m == 4);
    mu_assert("C n", C_m->n == 4);
    mu_assert("C n_blocks", C->n_blocks == 2);

    /* C_0 = B_0 @ A: intersection cols/rows = {0,2}.
       dgemm of B_0 = [[7,8],[9,10]] and A rows at {0,2} = [[1,2],[5,6]]
       = [[7+40, 14+48],[9+50, 18+60]] = [[47,62],[59,78]].                */
    permuted_dense *C0 = C->blocks[0];
    int C0_rp_exp[2] = {0, 1};
    int C0_cp_exp[2] = {0, 2};
    double C0_X_exp[4] = {47, 62, 59, 78};
    mu_assert("C0 m0", C0->m0 == 2);
    mu_assert("C0 n0", C0->n0 == 2);
    mu_assert("C0 row_perm", cmp_int_array(C0->row_perm, C0_rp_exp, 2));
    mu_assert("C0 col_perm", cmp_int_array(C0->col_perm, C0_cp_exp, 2));
    mu_assert("C0 X", cmp_double_array(C0->X, C0_X_exp, 4));

    /* C_1 = B_1 @ A: intersection cols/rows = {1,2}.
       dgemm of B_1 = [[11,12],[13,14]] and A rows at {1,2} = [[3,4],[5,6]]
       = [[33+60, 44+72],[39+70, 52+84]] = [[93,116],[109,136]].           */
    permuted_dense *C1 = C->blocks[1];
    int C1_rp_exp[2] = {2, 3};
    int C1_cp_exp[2] = {0, 2};
    double C1_X_exp[4] = {93, 116, 109, 136};
    mu_assert("C1 m0", C1->m0 == 2);
    mu_assert("C1 n0", C1->n0 == 2);
    mu_assert("C1 row_perm", cmp_int_array(C1->row_perm, C1_rp_exp, 2));
    mu_assert("C1 col_perm", cmp_int_array(C1->col_perm, C1_cp_exp, 2));
    mu_assert("C1 X", cmp_double_array(C1->X, C1_X_exp, 4));

    int src_exp[2] = {0, 1};
    mu_assert("src_block_idx", cmp_int_array(C->src_block_idx, src_exp, 2));

    free_matrix(C_m);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* BA_spd_pd: B_1's col_perm has empty intersection with A's row_perm,
   so its contribution is dropped. Output has 1 block, src = {0}. */
const char *test_BA_spd_pd_one_block_dropped(void)
{
    /* B: 4x6 spd, two blocks.
       B_0: rows {0,1}, cols {0,2}, X = [[7,8],[9,10]].
       B_1: rows {2,3}, cols {4,5}, X = [[1,1],[1,1]].                      */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {7, 8, 9, 10};
    matrix *B0 = new_permuted_dense(4, 6, 2, 2, B0_rp, B0_cp, B0X);

    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {4, 5};
    double B1X[4] = {1, 1, 1, 1};
    matrix *B1 = new_permuted_dense(4, 6, 2, 2, B1_rp, B1_cp, B1X);

    permuted_dense *blocks[2] = {(permuted_dense *) B0, (permuted_dense *) B1};
    matrix *B = new_stacked_pd(4, 6, 2, blocks, NULL, NULL);

    /* A: 6x4 PD with row_perm = {0,2} (so B_1's cols {4,5} miss it),
       col_perm = {0,2}, X = [[1,2],[5,6]].                                 */
    int A_rp[2] = {0, 2};
    int A_cp[2] = {0, 2};
    double AX[4] = {1, 2, 5, 6};
    matrix *A = new_permuted_dense(6, 4, 2, 2, A_rp, A_cp, AX);

    matrix *C_m = BA_spd_pd_alloc((stacked_pd *) B, (permuted_dense *) A);
    BA_spd_pd_fill_values((stacked_pd *) B, (permuted_dense *) A,
                          (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("C n_blocks", C->n_blocks == 1);
    mu_assert("C m", C_m->m == 4);
    mu_assert("C n", C_m->n == 4);
    int src_exp[1] = {0};
    mu_assert("src_block_idx", cmp_int_array(C->src_block_idx, src_exp, 1));

    free_matrix(C_m);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* BA_spd_spd: both B-blocks contribute, each from exactly one A-block. */
const char *test_BA_spd_spd_two_blocks_both_kept(void)
{
    /* B: 4x6 spd.
       B_0: rows {0,1}, cols {0,1}, X = [[1,2],[3,4]].
       B_1: rows {2,3}, cols {3,4}, X = [[5,6],[7,8]].                  */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 1};
    double B0X[4] = {1, 2, 3, 4};
    matrix *B0 = new_permuted_dense(4, 6, 2, 2, B0_rp, B0_cp, B0X);

    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {3, 4};
    double B1X[4] = {5, 6, 7, 8};
    matrix *B1 = new_permuted_dense(4, 6, 2, 2, B1_rp, B1_cp, B1X);

    permuted_dense *B_blocks[2] = {(permuted_dense *) B0, (permuted_dense *) B1};
    matrix *B = new_stacked_pd(4, 6, 2, B_blocks, NULL, NULL);

    /* A: 6x4 spd.
       A_0: rows {0,1,2}, cols {0,1}, X = [[10,11],[12,13],[14,15]].
       A_1: rows {3,4,5}, cols {2,3}, X = [[20,21],[22,23],[24,25]].     */
    int A0_rp[3] = {0, 1, 2};
    int A0_cp[2] = {0, 1};
    double A0X[6] = {10, 11, 12, 13, 14, 15};
    matrix *A0 = new_permuted_dense(6, 4, 3, 2, A0_rp, A0_cp, A0X);

    int A1_rp[3] = {3, 4, 5};
    int A1_cp[2] = {2, 3};
    double A1X[6] = {20, 21, 22, 23, 24, 25};
    matrix *A1 = new_permuted_dense(6, 4, 3, 2, A1_rp, A1_cp, A1X);

    permuted_dense *A_blocks[2] = {(permuted_dense *) A0, (permuted_dense *) A1};
    matrix *A = new_stacked_pd(6, 4, 2, A_blocks, NULL, NULL);

    matrix *C_m = BA_spd_spd_alloc((stacked_pd *) B, (stacked_pd *) A);
    BA_spd_spd_fill_values((stacked_pd *) B, (stacked_pd *) A, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("C m", C_m->m == 4);
    mu_assert("C n", C_m->n == 4);
    mu_assert("n_blocks", C->n_blocks == 2);
    int sp_exp[3] = {0, 1, 2};
    int s_exp[2] = {0, 1};
    mu_assert("src_block_idx_p", cmp_int_array(C->src_block_idx_p, sp_exp, 3));
    mu_assert("src_block_idx", cmp_int_array(C->src_block_idx, s_exp, 2));

    /* Block 0: row {0,1}, col {0,1}.
       temp = [[1,2],[3,4]] @ [[10,11],[12,13]] = [[34,37],[78,85]].     */
    permuted_dense *C0 = C->blocks[0];
    int C0_cp_exp[2] = {0, 1};
    double C0_X_exp[4] = {34, 37, 78, 85};
    mu_assert("C0 row_perm", cmp_int_array(C0->row_perm, B0_rp, 2));
    mu_assert("C0 col_perm", cmp_int_array(C0->col_perm, C0_cp_exp, 2));
    mu_assert("C0 X", cmp_double_array(C0->X, C0_X_exp, 4));

    /* Block 1: row {2,3}, col {2,3}.
       temp = [[5,6],[7,8]] @ [[20,21],[22,23]] = [[232,243],[316,331]]. */
    permuted_dense *C1 = C->blocks[1];
    int C1_cp_exp[2] = {2, 3};
    double C1_X_exp[4] = {232, 243, 316, 331};
    mu_assert("C1 row_perm", cmp_int_array(C1->row_perm, B1_rp, 2));
    mu_assert("C1 col_perm", cmp_int_array(C1->col_perm, C1_cp_exp, 2));
    mu_assert("C1 X", cmp_double_array(C1->X, C1_X_exp, 4));

    free_matrix(C_m);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* B has 2 blocks; only B_0's cols overlap an A-block. B_1 is dropped. */
const char *test_BA_spd_spd_one_block_dropped(void)
{
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 1};
    double B0X[4] = {1, 2, 3, 4};
    matrix *B0 = new_permuted_dense(4, 6, 2, 2, B0_rp, B0_cp, B0X);

    /* B_1 cols {4,5} won't intersect any A-block row_perm below. */
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {4, 5};
    double B1X[4] = {5, 6, 7, 8};
    matrix *B1 = new_permuted_dense(4, 6, 2, 2, B1_rp, B1_cp, B1X);

    permuted_dense *B_blocks[2] = {(permuted_dense *) B0, (permuted_dense *) B1};
    matrix *B = new_stacked_pd(4, 6, 2, B_blocks, NULL, NULL);

    /* A: A_0 rows {0,1} hits B_0; A_1 rows {2,3} disjoint from B_1 cols {4,5}. */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 1};
    double A0X[4] = {10, 11, 12, 13};
    matrix *A0 = new_permuted_dense(6, 4, 2, 2, A0_rp, A0_cp, A0X);

    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {2, 3};
    double A1X[4] = {20, 21, 22, 23};
    matrix *A1 = new_permuted_dense(6, 4, 2, 2, A1_rp, A1_cp, A1X);

    permuted_dense *A_blocks[2] = {(permuted_dense *) A0, (permuted_dense *) A1};
    matrix *A = new_stacked_pd(6, 4, 2, A_blocks, NULL, NULL);

    matrix *C_m = BA_spd_spd_alloc((stacked_pd *) B, (stacked_pd *) A);
    BA_spd_spd_fill_values((stacked_pd *) B, (stacked_pd *) A, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("n_blocks", C->n_blocks == 1);
    int sp_exp[2] = {0, 1};
    int s_exp[1] = {0};
    mu_assert("src_block_idx_p", cmp_int_array(C->src_block_idx_p, sp_exp, 2));
    mu_assert("src_block_idx", cmp_int_array(C->src_block_idx, s_exp, 1));

    permuted_dense *C0 = C->blocks[0];
    double C0_X_exp[4] = {34, 37, 78, 85};
    mu_assert("C0 X", cmp_double_array(C0->X, C0_X_exp, 4));

    free_matrix(C_m);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* No B-block's col_perm overlaps any A-block row_perm: all dropped. */
const char *test_BA_spd_spd_all_blocks_dropped(void)
{
    int B0_rp[1] = {0};
    int B0_cp[1] = {0};
    double B0X[1] = {1};
    matrix *B0 = new_permuted_dense(2, 4, 1, 1, B0_rp, B0_cp, B0X);

    int B1_rp[1] = {1};
    int B1_cp[1] = {1};
    double B1X[1] = {2};
    matrix *B1 = new_permuted_dense(2, 4, 1, 1, B1_rp, B1_cp, B1X);

    permuted_dense *B_blocks[2] = {(permuted_dense *) B0, (permuted_dense *) B1};
    matrix *B = new_stacked_pd(2, 4, 2, B_blocks, NULL, NULL);

    int A0_rp[1] = {2};
    int A0_cp[1] = {0};
    double A0X[1] = {10};
    matrix *A0 = new_permuted_dense(4, 2, 1, 1, A0_rp, A0_cp, A0X);

    int A1_rp[1] = {3};
    int A1_cp[1] = {1};
    double A1X[1] = {20};
    matrix *A1 = new_permuted_dense(4, 2, 1, 1, A1_rp, A1_cp, A1X);

    permuted_dense *A_blocks[2] = {(permuted_dense *) A0, (permuted_dense *) A1};
    matrix *A = new_stacked_pd(4, 2, 2, A_blocks, NULL, NULL);

    matrix *C_m = BA_spd_spd_alloc((stacked_pd *) B, (stacked_pd *) A);
    BA_spd_spd_fill_values((stacked_pd *) B, (stacked_pd *) A, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("C m", C_m->m == 2);
    mu_assert("C n", C_m->n == 2);
    mu_assert("n_blocks", C->n_blocks == 0);
    mu_assert("nnz", C_m->nnz == 0);

    free_matrix(C_m);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* Empty A: every B-block drops; output empty. */
const char *test_BA_spd_spd_empty_A(void)
{
    int B0_rp[1] = {0};
    int B0_cp[1] = {0};
    double B0X[1] = {1};
    matrix *B0 = new_permuted_dense(2, 3, 1, 1, B0_rp, B0_cp, B0X);

    int B1_rp[1] = {1};
    int B1_cp[1] = {1};
    double B1X[1] = {2};
    matrix *B1 = new_permuted_dense(2, 3, 1, 1, B1_rp, B1_cp, B1X);

    permuted_dense *B_blocks[2] = {(permuted_dense *) B0, (permuted_dense *) B1};
    matrix *B = new_stacked_pd(2, 3, 2, B_blocks, NULL, NULL);

    matrix *A = new_stacked_pd(3, 4, 0, NULL, NULL, NULL);

    matrix *C_m = BA_spd_spd_alloc((stacked_pd *) B, (stacked_pd *) A);
    BA_spd_spd_fill_values((stacked_pd *) B, (stacked_pd *) A, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("n_blocks", C->n_blocks == 0);
    mu_assert("nnz", C_m->nnz == 0);

    free_matrix(C_m);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* Empty B: loop body doesn't execute; output empty. */
const char *test_BA_spd_spd_empty_B(void)
{
    matrix *B = new_stacked_pd(2, 3, 0, NULL, NULL, NULL);

    int A0_rp[1] = {0};
    int A0_cp[1] = {0};
    double A0X[1] = {10};
    matrix *A0 = new_permuted_dense(3, 4, 1, 1, A0_rp, A0_cp, A0X);

    permuted_dense *A_blocks[1] = {(permuted_dense *) A0};
    matrix *A = new_stacked_pd(3, 4, 1, A_blocks, NULL, NULL);

    matrix *C_m = BA_spd_spd_alloc((stacked_pd *) B, (stacked_pd *) A);
    BA_spd_spd_fill_values((stacked_pd *) B, (stacked_pd *) A, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("C m", C_m->m == 2);
    mu_assert("C n", C_m->n == 4);
    mu_assert("n_blocks", C->n_blocks == 0);

    free_matrix(C_m);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* Two-phase: alloc once on test-1 inputs, mutate B_0->X, refill, verify. */
const char *test_BA_spd_spd_alloc_then_fill_values(void)
{
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 1};
    double B0X[4] = {1, 2, 3, 4};
    matrix *B0 = new_permuted_dense(4, 6, 2, 2, B0_rp, B0_cp, B0X);

    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {3, 4};
    double B1X[4] = {5, 6, 7, 8};
    matrix *B1 = new_permuted_dense(4, 6, 2, 2, B1_rp, B1_cp, B1X);

    permuted_dense *B_blocks[2] = {(permuted_dense *) B0, (permuted_dense *) B1};
    matrix *B = new_stacked_pd(4, 6, 2, B_blocks, NULL, NULL);

    int A0_rp[3] = {0, 1, 2};
    int A0_cp[2] = {0, 1};
    double A0X[6] = {10, 11, 12, 13, 14, 15};
    matrix *A0 = new_permuted_dense(6, 4, 3, 2, A0_rp, A0_cp, A0X);

    int A1_rp[3] = {3, 4, 5};
    int A1_cp[2] = {2, 3};
    double A1X[6] = {20, 21, 22, 23, 24, 25};
    matrix *A1 = new_permuted_dense(6, 4, 3, 2, A1_rp, A1_cp, A1X);

    permuted_dense *A_blocks[2] = {(permuted_dense *) A0, (permuted_dense *) A1};
    matrix *A = new_stacked_pd(6, 4, 2, A_blocks, NULL, NULL);

    matrix *C_m = BA_spd_spd_alloc((stacked_pd *) B, (stacked_pd *) A);
    BA_spd_spd_fill_values((stacked_pd *) B, (stacked_pd *) A, (stacked_pd *) C_m);

    /* Mutate B_0->X to [[10, 20],[30, 40]]. */
    permuted_dense *B0_pd = (permuted_dense *) B0;
    B0_pd->X[0] = 10;
    B0_pd->X[1] = 20;
    B0_pd->X[2] = 30;
    B0_pd->X[3] = 40;

    BA_spd_spd_fill_values((stacked_pd *) B, (stacked_pd *) A, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    /* New block 0 X:
         [[10,20],[30,40]] @ [[10,11],[12,13]]
         = [[340, 370],[780, 850]].                                      */
    double C0_X_exp[4] = {340, 370, 780, 850};
    mu_assert("C0 X refilled", cmp_double_array(C->blocks[0]->X, C0_X_exp, 4));

    /* Block 1 unchanged. */
    double C1_X_exp[4] = {232, 243, 316, 331};
    mu_assert("C1 X unchanged", cmp_double_array(C->blocks[1]->X, C1_X_exp, 4));

    free_matrix(C_m);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* BA_spd_matrices dispatcher with A = sparse_matrix. Same B + numerical
   inputs as test_BA_spd_csc_two_blocks_both_kept, but A is built as a
   sparse_matrix wrapping CSR and we call through the dispatcher; the
   result must match the direct-call expected values. */
const char *test_BA_spd_matrices_sparse_A(void)
{
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 2};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[1] = {3};
    int col_perm_1[2] = {1, 3};
    double X1[2] = {5.0, 6.0};
    matrix *blk1 = new_permuted_dense(4, 4, 1, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *B_m = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);

    /* A: 4x3, same values as the direct-call test but expressed as CSR.
       col 0: (0,1), (2,2);  col 1: (1,3), (3,4);  col 2: (2,5).
       Row-major CSR layout:
         row 0: col 0 = 1
         row 1: col 1 = 3
         row 2: col 0 = 2, col 2 = 5
         row 3: col 1 = 4
       p = [0, 1, 2, 4, 5], i = [0, 1, 0, 2, 1], x = [1, 3, 2, 5, 4]. */
    CSR_matrix *csr = new_CSR_matrix(4, 3, 5);
    int Ap[5] = {0, 1, 2, 4, 5};
    int Ai[5] = {0, 1, 0, 2, 1};
    double Ax[5] = {1.0, 3.0, 2.0, 5.0, 4.0};
    memcpy(csr->p, Ap, sizeof(Ap));
    memcpy(csr->i, Ai, sizeof(Ai));
    memcpy(csr->x, Ax, sizeof(Ax));
    matrix *A_m = new_sparse_matrix(csr);

    matrix *C_m = BA_spd_matrices_alloc((stacked_pd *) B_m, A_m);
    A_m->refresh_csc_values(A_m); /* per dispatcher contract */
    BA_spd_matrices_fill_values((stacked_pd *) B_m, A_m, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    /* Expected values match test_BA_spd_csc_two_blocks_both_kept. */
    mu_assert("n_blocks", C->n_blocks == 2);

    permuted_dense *C0 = C->blocks[0];
    double C0_X_exp[4] = {5.0, 10.0, 11.0, 20.0};
    mu_assert("C0 X", cmp_double_array(C0->X, C0_X_exp, 4));

    permuted_dense *C1 = C->blocks[1];
    double C1_X_exp[1] = {39.0};
    mu_assert("C1 X", cmp_double_array(C1->X, C1_X_exp, 1));

    free_matrix(C_m);
    free_matrix(B_m);
    free_matrix(A_m);
    return 0;
}

/* BA_spd_matrices dispatcher with A = stacked_pd. Same inputs as
   test_BA_spd_spd_two_blocks_both_kept; calling through the dispatcher
   must yield identical per-block values. */
const char *test_BA_spd_matrices_spd_A(void)
{
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 1};
    double B0X[4] = {1, 2, 3, 4};
    matrix *B0 = new_permuted_dense(4, 6, 2, 2, B0_rp, B0_cp, B0X);

    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {3, 4};
    double B1X[4] = {5, 6, 7, 8};
    matrix *B1 = new_permuted_dense(4, 6, 2, 2, B1_rp, B1_cp, B1X);

    permuted_dense *B_blocks[2] = {(permuted_dense *) B0, (permuted_dense *) B1};
    matrix *B_m = new_stacked_pd(4, 6, 2, B_blocks, NULL, NULL);

    int A0_rp[3] = {0, 1, 2};
    int A0_cp[2] = {0, 1};
    double A0X[6] = {10, 11, 12, 13, 14, 15};
    matrix *A0 = new_permuted_dense(6, 4, 3, 2, A0_rp, A0_cp, A0X);

    int A1_rp[3] = {3, 4, 5};
    int A1_cp[2] = {2, 3};
    double A1X[6] = {20, 21, 22, 23, 24, 25};
    matrix *A1 = new_permuted_dense(6, 4, 3, 2, A1_rp, A1_cp, A1X);

    permuted_dense *A_blocks[2] = {(permuted_dense *) A0, (permuted_dense *) A1};
    matrix *A_m = new_stacked_pd(6, 4, 2, A_blocks, NULL, NULL);

    matrix *C_m = BA_spd_matrices_alloc((stacked_pd *) B_m, A_m);
    BA_spd_matrices_fill_values((stacked_pd *) B_m, A_m, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("n_blocks", C->n_blocks == 2);

    double C0_X_exp[4] = {34, 37, 78, 85};
    mu_assert("C0 X", cmp_double_array(C->blocks[0]->X, C0_X_exp, 4));

    double C1_X_exp[4] = {232, 243, 316, 331};
    mu_assert("C1 X", cmp_double_array(C->blocks[1]->X, C1_X_exp, 4));

    free_matrix(C_m);
    free_matrix(B_m);
    free_matrix(A_m);
    return 0;
}

#endif /* TEST_OLD_STACKED_PD_H */
