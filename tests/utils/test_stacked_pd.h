#ifndef TEST_STACKED_PD_H
#define TEST_STACKED_PD_H

#include "minunit.h"
#include "test_helpers.h"
#include "utils/CSC_matrix.h"
#include "utils/permuted_dense.h"
#include "utils/stacked_pd.h"
#include <stdlib.h>
#include <string.h>

/* Construct a 2-block spd, check identity src_block_idx_*, and free. */
const char *test_stacked_pd_construct_and_free(void)
{
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 2};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(5, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[1] = {3};
    int col_perm_1[3] = {0, 1, 3};
    double X1[3] = {5.0, 6.0, 7.0};
    matrix *blk1 = new_permuted_dense(5, 4, 1, 3, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *M = new_stacked_pd(5, 4, 2, blocks, NULL, NULL);
    stacked_pd *s = (stacked_pd *) M;

    mu_assert("m", M->m == 5);
    mu_assert("n", M->n == 4);
    mu_assert("nnz", M->nnz == 4 + 3);
    mu_assert("n_blocks", s->n_blocks == 2);
    int expected_src_p[3] = {0, 1, 2};
    int expected_src[2] = {0, 1};
    mu_assert("src_block_idx_p",
              cmp_int_array(s->src_block_idx_p, expected_src_p, 3));
    mu_assert("src_block_idx", cmp_int_array(s->src_block_idx, expected_src, 2));

    free_matrix(M);
    return 0;
}

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

/* Coalesce input with disjoint row perms (and disjoint cols): output
   reproduces the input structure (one PD per source block, in min-row
   order). */
const char *test_coalesce_no_overlap(void)
{
    /* block 0: rows {0, 1}, cols {0, 1}, X = [[1, 2], [3, 4]]
       block 1: rows {2, 3}, cols {2, 3}, X = [[5, 6], [7, 8]]          */
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 1};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[2] = {2, 3};
    int col_perm_1[2] = {2, 3};
    double X1[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *blk1 = new_permuted_dense(4, 4, 2, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *src_m = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);

    matrix *out_m = coalesce_spd_alloc((stacked_pd *) src_m);
    coalesce_spd_fill_values((stacked_pd *) src_m, (stacked_pd *) out_m);
    stacked_pd *out = (stacked_pd *) out_m;

    mu_assert("n_blocks", out->n_blocks == 2);
    int expected_src_p[3] = {0, 1, 2};
    int expected_src[2] = {0, 1};
    mu_assert("src_block_idx_p",
              cmp_int_array(out->src_block_idx_p, expected_src_p, 3));
    mu_assert("src_block_idx", cmp_int_array(out->src_block_idx, expected_src, 2));

    permuted_dense *O0 = out->blocks[0];
    mu_assert("O0 row_perm", cmp_int_array(O0->row_perm, row_perm_0, 2));
    mu_assert("O0 col_perm", cmp_int_array(O0->col_perm, col_perm_0, 2));
    mu_assert("O0 X", cmp_double_array(O0->X, X0, 4));

    permuted_dense *O1 = out->blocks[1];
    mu_assert("O1 row_perm", cmp_int_array(O1->row_perm, row_perm_1, 2));
    mu_assert("O1 col_perm", cmp_int_array(O1->col_perm, col_perm_1, 2));
    mu_assert("O1 X", cmp_double_array(O1->X, X1, 4));

    free_matrix(out_m);
    free_matrix(src_m);
    return 0;
}

/* Coalesce input with one shared row (the transpose-style example from
   the design conversation). Three unique signatures produce three
   output PDs, ordered by min row. */
const char *test_coalesce_three_signatures(void)
{
    /* block 0 (id 0): rows {2, 4}, cols {0, 1}, X = [[1, 2], [3, 4]]
       block 1 (id 1): rows {0, 2}, cols {3},    X = [10, 11]
       row 2 is shared. col_perms {0,1} and {3} are disjoint -> ok.    */
    int row_perm_0[2] = {2, 4};
    int col_perm_0[2] = {0, 1};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(5, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[2] = {0, 2};
    int col_perm_1[1] = {3};
    double X1[2] = {10.0, 11.0};
    matrix *blk1 = new_permuted_dense(5, 4, 2, 1, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *src_m = new_stacked_pd_unchecked(5, 4, 2, blocks, NULL, NULL);

    matrix *out_m = coalesce_spd_alloc((stacked_pd *) src_m);
    coalesce_spd_fill_values((stacked_pd *) src_m, (stacked_pd *) out_m);
    stacked_pd *out = (stacked_pd *) out_m;

    /* Expected (ordered by min row):
         sig {1}:    row {0},  cols {3},       X = [10]
         sig {0, 1}: row {2},  cols {0, 1, 3}, X = [1, 2, 11]
         sig {0}:    row {4},  cols {0, 1},    X = [3, 4]              */
    mu_assert("n_blocks", out->n_blocks == 3);

    int expected_src_p[4] = {0, 1, 3, 4};
    int expected_src[4] = {1, 0, 1, 0};
    mu_assert("src_block_idx_p",
              cmp_int_array(out->src_block_idx_p, expected_src_p, 4));
    mu_assert("src_block_idx", cmp_int_array(out->src_block_idx, expected_src, 4));

    permuted_dense *O0 = out->blocks[0];
    int O0_row[1] = {0};
    int O0_col[1] = {3};
    double O0_X[1] = {10.0};
    mu_assert("O0 row", cmp_int_array(O0->row_perm, O0_row, 1));
    mu_assert("O0 col", cmp_int_array(O0->col_perm, O0_col, 1));
    mu_assert("O0 X", cmp_double_array(O0->X, O0_X, 1));

    permuted_dense *O1 = out->blocks[1];
    int O1_row[1] = {2};
    int O1_col[3] = {0, 1, 3};
    double O1_X[3] = {1.0, 2.0, 11.0};
    mu_assert("O1 row", cmp_int_array(O1->row_perm, O1_row, 1));
    mu_assert("O1 col", cmp_int_array(O1->col_perm, O1_col, 3));
    mu_assert("O1 X", cmp_double_array(O1->X, O1_X, 3));

    permuted_dense *O2 = out->blocks[2];
    int O2_row[1] = {4};
    int O2_col[2] = {0, 1};
    double O2_X[2] = {3.0, 4.0};
    mu_assert("O2 row", cmp_int_array(O2->row_perm, O2_row, 1));
    mu_assert("O2 col", cmp_int_array(O2->col_perm, O2_col, 2));
    mu_assert("O2 X", cmp_double_array(O2->X, O2_X, 2));

    free_matrix(out_m);
    free_matrix(src_m);
    return 0;
}

/* Two rows share the same signature -> they land in the SAME output
   PD, demonstrating the group-by-signature reduction. */
const char *test_coalesce_shared_signature_merges_rows(void)
{
    /* block A (id 0): rows {0, 1, 2}, cols {0}, X = [a0, a1, a2]
       block B (id 1): rows {1, 2, 3}, cols {1}, X = [b1, b2, b3]
       rows 1 and 2 both have signature {A, B}; cols disjoint -> ok.   */
    int row_perm_A[3] = {0, 1, 2};
    int col_perm_A[1] = {0};
    double XA[3] = {1.0, 2.0, 3.0};
    matrix *blkA = new_permuted_dense(4, 2, 3, 1, row_perm_A, col_perm_A, XA);

    int row_perm_B[3] = {1, 2, 3};
    int col_perm_B[1] = {1};
    double XB[3] = {10.0, 20.0, 30.0};
    matrix *blkB = new_permuted_dense(4, 2, 3, 1, row_perm_B, col_perm_B, XB);

    permuted_dense *blocks[2] = {(permuted_dense *) blkA, (permuted_dense *) blkB};
    matrix *src_m = new_stacked_pd_unchecked(4, 2, 2, blocks, NULL, NULL);

    matrix *out_m = coalesce_spd_alloc((stacked_pd *) src_m);
    coalesce_spd_fill_values((stacked_pd *) src_m, (stacked_pd *) out_m);
    stacked_pd *out = (stacked_pd *) out_m;

    /* Expected (ordered by min row):
         sig {0}:    row {0},    col {0},    X = [1]
         sig {0, 1}: rows {1,2}, cols {0,1}, X = [2, 10, 3, 20]
         sig {1}:    row {3},    col {1},    X = [30]                  */
    mu_assert("n_blocks", out->n_blocks == 3);

    int expected_src_p[4] = {0, 1, 3, 4};
    int expected_src[4] = {0, 0, 1, 1};
    mu_assert("src_block_idx_p",
              cmp_int_array(out->src_block_idx_p, expected_src_p, 4));
    mu_assert("src_block_idx", cmp_int_array(out->src_block_idx, expected_src, 4));

    permuted_dense *O0 = out->blocks[0];
    int O0_row[1] = {0};
    int O0_col[1] = {0};
    double O0_X[1] = {1.0};
    mu_assert("O0 row", cmp_int_array(O0->row_perm, O0_row, 1));
    mu_assert("O0 col", cmp_int_array(O0->col_perm, O0_col, 1));
    mu_assert("O0 X", cmp_double_array(O0->X, O0_X, 1));

    permuted_dense *O1 = out->blocks[1];
    int O1_row[2] = {1, 2};
    int O1_col[2] = {0, 1};
    double O1_X[4] = {2.0, 10.0, 3.0, 20.0};
    mu_assert("O1 m0", O1->m0 == 2);
    mu_assert("O1 n0", O1->n0 == 2);
    mu_assert("O1 row", cmp_int_array(O1->row_perm, O1_row, 2));
    mu_assert("O1 col", cmp_int_array(O1->col_perm, O1_col, 2));
    mu_assert("O1 X", cmp_double_array(O1->X, O1_X, 4));

    permuted_dense *O2 = out->blocks[2];
    int O2_row[1] = {3};
    int O2_col[1] = {1};
    double O2_X[1] = {30.0};
    mu_assert("O2 row", cmp_int_array(O2->row_perm, O2_row, 1));
    mu_assert("O2 col", cmp_int_array(O2->col_perm, O2_col, 1));
    mu_assert("O2 X", cmp_double_array(O2->X, O2_X, 1));

    free_matrix(out_m);
    free_matrix(src_m);
    return 0;
}

/* Two-phase test: alloc once, mutate source X buffers, refill, re-check. */
const char *test_coalesce_alloc_then_fill_values(void)
{
    int row_perm_0[2] = {2, 4};
    int col_perm_0[2] = {0, 1};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(5, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[2] = {0, 2};
    int col_perm_1[1] = {3};
    double X1[2] = {10.0, 11.0};
    matrix *blk1 = new_permuted_dense(5, 4, 2, 1, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *src_m = new_stacked_pd_unchecked(5, 4, 2, blocks, NULL, NULL);
    stacked_pd *src = (stacked_pd *) src_m;

    matrix *out_m = coalesce_spd_alloc(src);
    coalesce_spd_fill_values(src, (stacked_pd *) out_m);

    /* Mutate source values then refill. New values:
         blk0: X = [100, 200, 300, 400]
         blk1: X = [50, 60]                                             */
    src->blocks[0]->X[0] = 100.0;
    src->blocks[0]->X[1] = 200.0;
    src->blocks[0]->X[2] = 300.0;
    src->blocks[0]->X[3] = 400.0;
    src->blocks[1]->X[0] = 50.0;
    src->blocks[1]->X[1] = 60.0;

    coalesce_spd_fill_values(src, (stacked_pd *) out_m);
    stacked_pd *out = (stacked_pd *) out_m;

    /* Block order (by min row): {1} -> {0,1} -> {0}.
         O0: row {0},  col {3},        X = [50]
         O1: row {2},  cols {0, 1, 3}, X = [100, 200, 60]
         O2: row {4},  cols {0, 1},    X = [300, 400]                  */
    double O0_X[1] = {50.0};
    mu_assert("O0 X refilled", cmp_double_array(out->blocks[0]->X, O0_X, 1));

    double O1_X[3] = {100.0, 200.0, 60.0};
    mu_assert("O1 X refilled", cmp_double_array(out->blocks[1]->X, O1_X, 3));

    double O2_X[2] = {300.0, 400.0};
    mu_assert("O2 X refilled", cmp_double_array(out->blocks[2]->X, O2_X, 2));

    free_matrix(out_m);
    free_matrix(src_m);
    return 0;
}

/* Empty input (n_blocks == 0): output is also empty and safe to free. */
const char *test_coalesce_empty_input(void)
{
    matrix *src_m = new_stacked_pd(4, 4, 0, NULL, NULL, NULL);
    matrix *out_m = coalesce_spd_alloc((stacked_pd *) src_m);
    coalesce_spd_fill_values((stacked_pd *) src_m, (stacked_pd *) out_m);
    stacked_pd *out = (stacked_pd *) out_m;

    mu_assert("m", out_m->m == 4);
    mu_assert("n", out_m->n == 4);
    mu_assert("n_blocks", out->n_blocks == 0);
    mu_assert("nnz", out_m->nnz == 0);

    free_matrix(out_m);
    free_matrix(src_m);
    return 0;
}

/* Transpose where source blocks have disjoint rows AND disjoint cols.
   Transposed blocks also have disjoint rows; coalesce leaves them
   unchanged. */
const char *test_transpose_spd_no_overlap(void)
{
    /* block 0: rows {0, 1}, cols {0, 1}, X = [[1, 2], [3, 4]]
       block 1: rows {2, 3}, cols {2, 3}, X = [[5, 6], [7, 8]]          */
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 1};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[2] = {2, 3};
    int col_perm_1[2] = {2, 3};
    double X1[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *blk1 = new_permuted_dense(4, 4, 2, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *src_m = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);

    matrix *out_m = transpose_spd_alloc((stacked_pd *) src_m);
    transpose_spd_fill_values((stacked_pd *) src_m, (stacked_pd *) out_m);
    stacked_pd *out = (stacked_pd *) out_m;

    mu_assert("m", out_m->m == 4);
    mu_assert("n", out_m->n == 4);
    mu_assert("n_blocks", out->n_blocks == 2);
    int expected_src_p[3] = {0, 1, 2};
    int expected_src[2] = {0, 1};
    mu_assert("src_block_idx_p",
              cmp_int_array(out->src_block_idx_p, expected_src_p, 3));
    mu_assert("src_block_idx", cmp_int_array(out->src_block_idx, expected_src, 2));

    /* Transpose of [[1,2],[3,4]] is [[1,3],[2,4]] (row-major flatten:
       [1, 3, 2, 4]); similarly for the second block. */
    permuted_dense *O0 = out->blocks[0];
    double O0_X_expected[4] = {1.0, 3.0, 2.0, 4.0};
    mu_assert("O0 row_perm", cmp_int_array(O0->row_perm, row_perm_0, 2));
    mu_assert("O0 col_perm", cmp_int_array(O0->col_perm, col_perm_0, 2));
    mu_assert("O0 X", cmp_double_array(O0->X, O0_X_expected, 4));

    permuted_dense *O1 = out->blocks[1];
    double O1_X_expected[4] = {5.0, 7.0, 6.0, 8.0};
    mu_assert("O1 row_perm", cmp_int_array(O1->row_perm, row_perm_1, 2));
    mu_assert("O1 col_perm", cmp_int_array(O1->col_perm, col_perm_1, 2));
    mu_assert("O1 X", cmp_double_array(O1->X, O1_X_expected, 4));

    free_matrix(out_m);
    free_matrix(src_m);
    return 0;
}

/* Transpose where source blocks share a column. After per-block
   transpose two raw blocks share a row, so coalesce merges and produces
   three output PDs (the worked example from the design conversation). */
const char *test_transpose_spd_overlap_coalesces(void)
{
    /* Source (4x5):
         block 0: rows {0, 1}, cols {2, 4}, X = [[1, 2], [3, 4]]
         block 1: rows {3},    cols {0, 2}, X = [10, 11]
       (Disjoint rows; cells {(0,2),(0,4),(1,2),(1,4)} and {(3,0),(3,2)}
       are disjoint.)                                                    */
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {2, 4};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 5, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[1] = {3};
    int col_perm_1[2] = {0, 2};
    double X1[2] = {10.0, 11.0};
    matrix *blk1 = new_permuted_dense(4, 5, 1, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *src_m = new_stacked_pd(4, 5, 2, blocks, NULL, NULL);

    matrix *out_m = transpose_spd_alloc((stacked_pd *) src_m);
    transpose_spd_fill_values((stacked_pd *) src_m, (stacked_pd *) out_m);
    stacked_pd *out = (stacked_pd *) out_m;

    /* Transposed (5x4):
         raw block 0^T: rows {2, 4}, cols {0, 1}, X = [[1, 3], [2, 4]]
         raw block 1^T: rows {0, 2}, cols {3},    X = [10, 11]
       Row 2 shared. Coalesce produces (ordered by min row):
         sig {1}:    row {0},  col {3},        X = [10]
         sig {0, 1}: row {2},  cols {0, 1, 3}, X = [1, 3, 11]
         sig {0}:    row {4},  cols {0, 1},    X = [2, 4]               */
    mu_assert("m", out_m->m == 5);
    mu_assert("n", out_m->n == 4);
    mu_assert("n_blocks", out->n_blocks == 3);

    int expected_src_p[4] = {0, 1, 3, 4};
    int expected_src[4] = {1, 0, 1, 0};
    mu_assert("src_block_idx_p",
              cmp_int_array(out->src_block_idx_p, expected_src_p, 4));
    mu_assert("src_block_idx", cmp_int_array(out->src_block_idx, expected_src, 4));

    permuted_dense *O0 = out->blocks[0];
    int O0_row[1] = {0};
    int O0_col[1] = {3};
    double O0_X[1] = {10.0};
    mu_assert("O0 row", cmp_int_array(O0->row_perm, O0_row, 1));
    mu_assert("O0 col", cmp_int_array(O0->col_perm, O0_col, 1));
    mu_assert("O0 X", cmp_double_array(O0->X, O0_X, 1));

    permuted_dense *O1 = out->blocks[1];
    int O1_row[1] = {2};
    int O1_col[3] = {0, 1, 3};
    double O1_X[3] = {1.0, 3.0, 11.0};
    mu_assert("O1 row", cmp_int_array(O1->row_perm, O1_row, 1));
    mu_assert("O1 col", cmp_int_array(O1->col_perm, O1_col, 3));
    mu_assert("O1 X", cmp_double_array(O1->X, O1_X, 3));

    permuted_dense *O2 = out->blocks[2];
    int O2_row[1] = {4};
    int O2_col[2] = {0, 1};
    double O2_X[2] = {2.0, 4.0};
    mu_assert("O2 row", cmp_int_array(O2->row_perm, O2_row, 1));
    mu_assert("O2 col", cmp_int_array(O2->col_perm, O2_col, 2));
    mu_assert("O2 X", cmp_double_array(O2->X, O2_X, 2));

    free_matrix(out_m);
    free_matrix(src_m);
    return 0;
}

/* Two-phase: alloc once, mutate src values, refill, re-check. */
const char *test_transpose_spd_alloc_then_fill_values(void)
{
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {2, 4};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 5, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[1] = {3};
    int col_perm_1[2] = {0, 2};
    double X1[2] = {10.0, 11.0};
    matrix *blk1 = new_permuted_dense(4, 5, 1, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *src_m = new_stacked_pd(4, 5, 2, blocks, NULL, NULL);
    stacked_pd *src = (stacked_pd *) src_m;

    matrix *out_m = transpose_spd_alloc(src);
    transpose_spd_fill_values(src, (stacked_pd *) out_m);

    /* Mutate src values then refill:
         blk0 X -> [100, 200, 300, 400]
         blk1 X -> [50, 60]                                             */
    src->blocks[0]->X[0] = 100.0;
    src->blocks[0]->X[1] = 200.0;
    src->blocks[0]->X[2] = 300.0;
    src->blocks[0]->X[3] = 400.0;
    src->blocks[1]->X[0] = 50.0;
    src->blocks[1]->X[1] = 60.0;

    transpose_spd_fill_values(src, (stacked_pd *) out_m);
    stacked_pd *out = (stacked_pd *) out_m;

    /* Expected after refill:
         O0: X = [50]
         O1: X = [100, 300, 60]
         O2: X = [200, 400]                                             */
    double O0_X[1] = {50.0};
    mu_assert("O0 X refilled", cmp_double_array(out->blocks[0]->X, O0_X, 1));
    double O1_X[3] = {100.0, 300.0, 60.0};
    mu_assert("O1 X refilled", cmp_double_array(out->blocks[1]->X, O1_X, 3));
    double O2_X[2] = {200.0, 400.0};
    mu_assert("O2 X refilled", cmp_double_array(out->blocks[2]->X, O2_X, 2));

    free_matrix(out_m);
    free_matrix(src_m);
    return 0;
}

/* Empty source: output also empty; work is an empty spd; free safe. */
const char *test_transpose_spd_empty(void)
{
    matrix *src_m = new_stacked_pd(4, 5, 0, NULL, NULL, NULL);
    matrix *out_m = transpose_spd_alloc((stacked_pd *) src_m);
    transpose_spd_fill_values((stacked_pd *) src_m, (stacked_pd *) out_m);
    stacked_pd *out = (stacked_pd *) out_m;

    mu_assert("m", out_m->m == 5);
    mu_assert("n", out_m->n == 4);
    mu_assert("n_blocks", out->n_blocks == 0);
    mu_assert("nnz", out_m->nnz == 0);
    mu_assert("work not NULL", out->work != NULL);
    mu_assert("work n_blocks", out->work->n_blocks == 0);

    free_matrix(out_m);
    free_matrix(src_m);
    return 0;
}

/* Single full-dense block: trivial case, just a dense transpose. */
const char *test_transpose_spd_single_block_full(void)
{
    /* 2x3 dense block, X = [[1, 2, 3], [4, 5, 6]] */
    double data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    matrix *blk = new_permuted_dense_full(2, 3, data);

    permuted_dense *blocks[1] = {(permuted_dense *) blk};
    matrix *src_m = new_stacked_pd(2, 3, 1, blocks, NULL, NULL);

    matrix *out_m = transpose_spd_alloc((stacked_pd *) src_m);
    transpose_spd_fill_values((stacked_pd *) src_m, (stacked_pd *) out_m);
    stacked_pd *out = (stacked_pd *) out_m;

    mu_assert("m", out_m->m == 3);
    mu_assert("n", out_m->n == 2);
    mu_assert("n_blocks", out->n_blocks == 1);

    /* Transposed 3x2 dense: [[1, 4], [2, 5], [3, 6]] -> [1, 4, 2, 5, 3, 6]. */
    permuted_dense *O = out->blocks[0];
    int O_row[3] = {0, 1, 2};
    int O_col[2] = {0, 1};
    double O_X[6] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    mu_assert("O m0", O->m0 == 3);
    mu_assert("O n0", O->n0 == 2);
    mu_assert("O row", cmp_int_array(O->row_perm, O_row, 3));
    mu_assert("O col", cmp_int_array(O->col_perm, O_col, 2));
    mu_assert("O X", cmp_double_array(O->X, O_X, 6));

    free_matrix(out_m);
    free_matrix(src_m);
    return 0;
}

#endif /* TEST_STACKED_PD_H */
