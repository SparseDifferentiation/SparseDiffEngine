#ifndef TEST_STACKED_PD_H
#define TEST_STACKED_PD_H

#include "minunit.h"
#include "test_helpers.h"
#include "utils/permuted_dense.h"
#include "utils/stacked_pd.h"
#include "utils/stacked_pd_linalg.h"
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

/* Empty source: output also empty; pre_coalesce is an empty spd; free safe. */
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
    mu_assert("pre_coalesce not NULL", out->pre_coalesce != NULL);
    mu_assert("pre_coalesce n_blocks", out->pre_coalesce->n_blocks == 0);

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

/* copy_sparsity_spd_alloc preserves block structure and produces a fresh
   spd with identity src_block_idx_*, NULL pre_coalesce, and uninitialized X
   buffers. */
const char *test_copy_sparsity_spd_alloc(void)
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
    matrix *src_m = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);
    stacked_pd *src = (stacked_pd *) src_m;

    matrix *out_m = copy_sparsity_spd_alloc(src);
    stacked_pd *out = (stacked_pd *) out_m;

    mu_assert("m", out_m->m == 4);
    mu_assert("n", out_m->n == 4);
    mu_assert("nnz", out_m->nnz == src_m->nnz);
    mu_assert("n_blocks", out->n_blocks == 2);
    int expected_src_p[3] = {0, 1, 2};
    int expected_src[2] = {0, 1};
    mu_assert("src_block_idx_p",
              cmp_int_array(out->src_block_idx_p, expected_src_p, 3));
    mu_assert("src_block_idx", cmp_int_array(out->src_block_idx, expected_src, 2));
    mu_assert("pre_coalesce NULL", out->pre_coalesce == NULL);

    permuted_dense *O0 = out->blocks[0];
    mu_assert("O0 m0", O0->m0 == 2);
    mu_assert("O0 n0", O0->n0 == 2);
    mu_assert("O0 row_perm", cmp_int_array(O0->row_perm, row_perm_0, 2));
    mu_assert("O0 col_perm", cmp_int_array(O0->col_perm, col_perm_0, 2));

    permuted_dense *O1 = out->blocks[1];
    mu_assert("O1 m0", O1->m0 == 1);
    mu_assert("O1 n0", O1->n0 == 2);
    mu_assert("O1 row_perm", cmp_int_array(O1->row_perm, row_perm_1, 1));
    mu_assert("O1 col_perm", cmp_int_array(O1->col_perm, col_perm_1, 2));

    free_matrix(out_m);
    free_matrix(src_m);
    return 0;
}

/* DA on a 2-block spd: per-row scaling by d[global_row]. */
const char *test_DA_spd_two_blocks(void)
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
    matrix *A_m = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);
    stacked_pd *A = (stacked_pd *) A_m;

    double d[4] = {10.0, 100.0, 1000.0, 10000.0};

    matrix *C_m = copy_sparsity_spd_alloc(A);
    DA_spd_fill_values(d, A, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    /* C0 row 0 scaled by d[0]=10; row 1 by d[1]=100. */
    double C0_X_expected[4] = {10.0, 20.0, 300.0, 400.0};
    mu_assert("C0 X", cmp_double_array(C->blocks[0]->X, C0_X_expected, 4));

    /* C1 row 0 (= global row 2) scaled by d[2]=1000; row 1 (= row 3) by
       d[3]=10000. */
    double C1_X_expected[4] = {5000.0, 6000.0, 70000.0, 80000.0};
    mu_assert("C1 X", cmp_double_array(C->blocks[1]->X, C1_X_expected, 4));

    free_matrix(C_m);
    free_matrix(A_m);
    return 0;
}

/* DA on an empty spd: no-op, no crash. */
const char *test_DA_spd_empty(void)
{
    matrix *A_m = new_stacked_pd(4, 4, 0, NULL, NULL, NULL);
    matrix *C_m = copy_sparsity_spd_alloc((stacked_pd *) A_m);
    double d[4] = {1.0, 2.0, 3.0, 4.0};
    DA_spd_fill_values(d, (stacked_pd *) A_m, (stacked_pd *) C_m);

    mu_assert("n_blocks", ((stacked_pd *) C_m)->n_blocks == 0);
    mu_assert("nnz", C_m->nnz == 0);

    free_matrix(C_m);
    free_matrix(A_m);
    return 0;
}

/* DA on a single full-dense block. */
const char *test_DA_spd_single_block_full(void)
{
    /* 2x3 dense, X = [[1, 2, 3], [4, 5, 6]] */
    double data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    matrix *blk = new_permuted_dense_full(2, 3, data);

    permuted_dense *blocks[1] = {(permuted_dense *) blk};
    matrix *A_m = new_stacked_pd(2, 3, 1, blocks, NULL, NULL);

    double d[2] = {10.0, 100.0};

    matrix *C_m = copy_sparsity_spd_alloc((stacked_pd *) A_m);
    DA_spd_fill_values(d, (stacked_pd *) A_m, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    /* row 0 scaled by 10, row 1 by 100. */
    double C0_X_expected[6] = {10.0, 20.0, 30.0, 400.0, 500.0, 600.0};
    mu_assert("C X", cmp_double_array(C->blocks[0]->X, C0_X_expected, 6));

    free_matrix(C_m);
    free_matrix(A_m);
    return 0;
}

/* ATA_spd_alloc with disjoint source col_perms: no merging; one output
   PD per source block. */
const char *test_ATA_spd_alloc_disjoint_cols(void)
{
    /* block 0: rows {0,1}, cols {0,1}, X = [[1,2],[3,4]]
       block 1: rows {2,3}, cols {2,3}, X = [[5,6],[7,8]]              */
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 1};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[2] = {2, 3};
    int col_perm_1[2] = {2, 3};
    double X1[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *blk1 = new_permuted_dense(4, 4, 2, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A_m = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);

    matrix *C_m = ATA_spd_alloc((stacked_pd *) A_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("m", C_m->m == 4);
    mu_assert("n", C_m->n == 4);
    mu_assert("n_blocks", C->n_blocks == 2);
    int expected_src_p[3] = {0, 1, 2};
    int expected_src[2] = {0, 1};
    mu_assert("src_block_idx_p",
              cmp_int_array(C->src_block_idx_p, expected_src_p, 3));
    mu_assert("src_block_idx", cmp_int_array(C->src_block_idx, expected_src, 2));

    permuted_dense *C0 = C->blocks[0];
    mu_assert("C0 row_perm", cmp_int_array(C0->row_perm, col_perm_0, 2));
    mu_assert("C0 col_perm", cmp_int_array(C0->col_perm, col_perm_0, 2));

    permuted_dense *C1 = C->blocks[1];
    mu_assert("C1 row_perm", cmp_int_array(C1->row_perm, col_perm_1, 2));
    mu_assert("C1 col_perm", cmp_int_array(C1->col_perm, col_perm_1, 2));

    free_matrix(C_m);
    free_matrix(A_m);
    return 0;
}

/* ATA_spd_alloc with overlapping source col_perms: three sigs ({0},
   {0,1}, {1}) produce three output PDs ordered by min row. */
const char *test_ATA_spd_alloc_overlapping_cols(void)
{
    /* block 0: rows {0,1}, cols {0,1}, X arbitrary
       block 1: rows {2,3}, cols {1,2}, X arbitrary                    */
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 1};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[2] = {2, 3};
    int col_perm_1[2] = {1, 2};
    double X1[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A_m = new_stacked_pd(4, 3, 2, blocks, NULL, NULL);

    matrix *C_m = ATA_spd_alloc((stacked_pd *) A_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("m", C_m->m == 3);
    mu_assert("n", C_m->n == 3);
    mu_assert("n_blocks", C->n_blocks == 3);

    /* Ordering by min row: sig {0} (col 0) -> sig {0,1} (col 1) -> sig {1} (col 2).
     */
    int expected_src_p[4] = {0, 1, 3, 4};
    int expected_src[4] = {0, 0, 1, 1};
    mu_assert("src_block_idx_p",
              cmp_int_array(C->src_block_idx_p, expected_src_p, 4));
    mu_assert("src_block_idx", cmp_int_array(C->src_block_idx, expected_src, 4));

    permuted_dense *C0 = C->blocks[0];
    int C0_row[1] = {0};
    int C0_col[2] = {0, 1};
    mu_assert("C0 row_perm", cmp_int_array(C0->row_perm, C0_row, 1));
    mu_assert("C0 col_perm", cmp_int_array(C0->col_perm, C0_col, 2));

    permuted_dense *C1 = C->blocks[1];
    int C1_row[1] = {1};
    int C1_col[3] = {0, 1, 2};
    mu_assert("C1 row_perm", cmp_int_array(C1->row_perm, C1_row, 1));
    mu_assert("C1 col_perm", cmp_int_array(C1->col_perm, C1_col, 3));

    permuted_dense *C2 = C->blocks[2];
    int C2_row[1] = {2};
    int C2_col[2] = {1, 2};
    mu_assert("C2 row_perm", cmp_int_array(C2->row_perm, C2_row, 1));
    mu_assert("C2 col_perm", cmp_int_array(C2->col_perm, C2_col, 2));

    free_matrix(C_m);
    free_matrix(A_m);
    return 0;
}

/* ATDA on the disjoint-cols input: values per block match hand-computed
   B_k^T diag(d[R_k]) B_k. */
const char *test_ATDA_spd_disjoint_cols(void)
{
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 1};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[2] = {2, 3};
    int col_perm_1[2] = {2, 3};
    double X1[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *blk1 = new_permuted_dense(4, 4, 2, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A_m = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);

    double d[4] = {1.0, 2.0, 3.0, 4.0};

    matrix *C_m = ATA_spd_alloc((stacked_pd *) A_m);
    ATDA_spd_fill_values((stacked_pd *) A_m, d, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    /* B_0^T diag(1, 2) B_0 = [[19, 26], [26, 36]]. */
    double C0_X_expected[4] = {19.0, 26.0, 26.0, 36.0};
    mu_assert("C0 X", cmp_double_array(C->blocks[0]->X, C0_X_expected, 4));

    /* B_1^T diag(3, 4) B_1 = [[271, 314], [314, 364]]. */
    double C1_X_expected[4] = {271.0, 314.0, 314.0, 364.0};
    mu_assert("C1 X", cmp_double_array(C->blocks[1]->X, C1_X_expected, 4));

    free_matrix(C_m);
    free_matrix(A_m);
    return 0;
}

/* ATDA on overlapping cols: middle PD accumulates from both sources. */
const char *test_ATDA_spd_overlapping_cols(void)
{
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 1};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[2] = {2, 3};
    int col_perm_1[2] = {1, 2};
    double X1[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A_m = new_stacked_pd(4, 3, 2, blocks, NULL, NULL);

    double d[4] = {1.0, 2.0, 3.0, 4.0};

    matrix *C_m = ATA_spd_alloc((stacked_pd *) A_m);
    ATDA_spd_fill_values((stacked_pd *) A_m, d, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    /* C0: row {0}, cols {0, 1}, from B_0 only.
       (A^T d A)[0,0] = 1*1 + 2*9 = 19; [0,1] = 1*2 + 2*12 = 26. */
    double C0_X_expected[2] = {19.0, 26.0};
    mu_assert("C0 X", cmp_double_array(C->blocks[0]->X, C0_X_expected, 2));

    /* C1: row {1}, cols {0, 1, 2}, accumulated from B_0 and B_1.
       [1,0] = B_0 contribution = 1*2 + 2*12 = 26.
       [1,1] = 1*4 + 2*16 (B_0) + 3*25 + 4*49 (B_1) = 36 + 271 = 307.
       [1,2] = B_1 contribution = 3*30 + 4*56 = 90 + 224 = 314. */
    double C1_X_expected[3] = {26.0, 307.0, 314.0};
    mu_assert("C1 X", cmp_double_array(C->blocks[1]->X, C1_X_expected, 3));

    /* C2: row {2}, cols {1, 2}, from B_1 only.
       [2,1] = 3*5*6 + 4*7*8 = 90 + 224 = 314; [2,2] = 3*36 + 4*64 = 364. */
    double C2_X_expected[2] = {314.0, 364.0};
    mu_assert("C2 X", cmp_double_array(C->blocks[2]->X, C2_X_expected, 2));

    free_matrix(C_m);
    free_matrix(A_m);
    return 0;
}

/* Two-phase: alloc once, mutate A values, refill. */
const char *test_ATDA_spd_alloc_then_fill_values(void)
{
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 1};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[2] = {2, 3};
    int col_perm_1[2] = {2, 3};
    double X1[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *blk1 = new_permuted_dense(4, 4, 2, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A_m = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);
    stacked_pd *A = (stacked_pd *) A_m;

    double d[4] = {1.0, 2.0, 3.0, 4.0};

    matrix *C_m = ATA_spd_alloc(A);
    ATDA_spd_fill_values(A, d, (stacked_pd *) C_m);

    /* Mutate A's block 0 X to [2, 0, 0, 2] (= 2 * identity at rows {0,1},
       cols {0,1}). Then A^T d A on block 0 = diag(d[0], d[1]) * 4 =
       [[4, 0], [0, 8]]. */
    A->blocks[0]->X[0] = 2.0;
    A->blocks[0]->X[1] = 0.0;
    A->blocks[0]->X[2] = 0.0;
    A->blocks[0]->X[3] = 2.0;

    ATDA_spd_fill_values(A, d, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    double C0_X_expected[4] = {4.0, 0.0, 0.0, 8.0};
    mu_assert("C0 X refilled", cmp_double_array(C->blocks[0]->X, C0_X_expected, 4));

    /* Block 1 wasn't mutated; matches the disjoint-cols expected value. */
    double C1_X_expected[4] = {271.0, 314.0, 314.0, 364.0};
    mu_assert("C1 X unchanged", cmp_double_array(C->blocks[1]->X, C1_X_expected, 4));

    free_matrix(C_m);
    free_matrix(A_m);
    return 0;
}

/* Empty source: output also empty; free safely. */
const char *test_ATDA_spd_empty(void)
{
    matrix *A_m = new_stacked_pd(4, 4, 0, NULL, NULL, NULL);
    matrix *C_m = ATA_spd_alloc((stacked_pd *) A_m);
    double d[4] = {1.0, 2.0, 3.0, 4.0};
    ATDA_spd_fill_values((stacked_pd *) A_m, d, (stacked_pd *) C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("n_blocks", C->n_blocks == 0);
    mu_assert("nnz", C_m->nnz == 0);
    mu_assert("pre_coalesce not NULL", C->pre_coalesce != NULL);
    mu_assert("pre_coalesce n_blocks", C->pre_coalesce->n_blocks == 0);

    free_matrix(C_m);
    free_matrix(A_m);
    return 0;
}

/* BA_pd_spd worked example: two contributing A-blocks. */
const char *test_BA_pd_spd_two_blocks_disjoint_cols(void)
{
    /* B: 3x4 PD. row_perm = {0,1,2}, col_perm = {0,1,3},
       X = [[1,2,3],[4,5,6],[7,8,9]].                                    */
    int B_rp[3] = {0, 1, 2};
    int B_cp[3] = {0, 1, 3};
    double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *B = new_permuted_dense(3, 4, 3, 3, B_rp, B_cp, BX);

    /* A: 4x6 spd, two blocks.
       A_0: rows {0,1}, cols {0,4}, X = [[10,11],[12,13]].
       A_1: rows {2,3}, cols {1,4}, X = [[20,21],[22,23]].               */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 4};
    double A0X[4] = {10, 11, 12, 13};
    matrix *A0 = new_permuted_dense(4, 6, 2, 2, A0_rp, A0_cp, A0X);

    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 4};
    double A1X[4] = {20, 21, 22, 23};
    matrix *A1 = new_permuted_dense(4, 6, 2, 2, A1_rp, A1_cp, A1X);

    permuted_dense *blocks[2] = {(permuted_dense *) A0, (permuted_dense *) A1};
    matrix *A = new_stacked_pd(4, 6, 2, blocks, NULL, NULL);

    matrix *C_m = BA_pd_spd_alloc((permuted_dense *) B, (stacked_pd *) A);
    BA_pd_spd_fill_values((permuted_dense *) B, (stacked_pd *) A,
                          (permuted_dense *) C_m);
    permuted_dense *C = (permuted_dense *) C_m;

    /* C is 3x6 with row_perm {0,1,2}, col_perm {0,1,4}. */
    mu_assert("C m", C_m->m == 3);
    mu_assert("C n", C_m->n == 6);
    mu_assert("C m0", C->m0 == 3);
    mu_assert("C n0", C->n0 == 3);
    int rp_exp[3] = {0, 1, 2};
    int cp_exp[3] = {0, 1, 4};
    mu_assert("C row_perm", cmp_int_array(C->row_perm, rp_exp, 3));
    mu_assert("C col_perm", cmp_int_array(C->col_perm, cp_exp, 3));

    /* C[0,0]= 1*10+2*12 = 34;  C[0,1]= 3*22 = 66;
       C[0,4]= 1*11+2*13+3*23 = 106;
       C[1,0]= 4*10+5*12 = 100; C[1,1]= 6*22 = 132;
       C[1,4]= 4*11+5*13+6*23 = 247;
       C[2,0]= 7*10+8*12 = 166; C[2,1]= 9*22 = 198;
       C[2,4]= 7*11+8*13+9*23 = 388.                                     */
    double CX_exp[9] = {34, 66, 106, 100, 132, 247, 166, 198, 388};
    mu_assert("C X", cmp_double_array(C->X, CX_exp, 9));

    free_matrix(C_m);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* Only one A-block intersects B's col_perm. */
const char *test_BA_pd_spd_only_one_block_contributes(void)
{
    /* B has col_perm {0,1}; A_0 rows {0,1} hits, A_1 rows {3,4} misses. */
    int B_rp[2] = {0, 1};
    int B_cp[2] = {0, 1};
    double BX[4] = {1, 2, 3, 4};
    matrix *B = new_permuted_dense(2, 5, 2, 2, B_rp, B_cp, BX);

    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {5, 6, 7, 8};
    matrix *A0 = new_permuted_dense(5, 3, 2, 2, A0_rp, A0_cp, A0X);

    int A1_rp[2] = {3, 4};
    int A1_cp[1] = {1};
    double A1X[2] = {99, 100};
    matrix *A1 = new_permuted_dense(5, 3, 2, 1, A1_rp, A1_cp, A1X);

    permuted_dense *blocks[2] = {(permuted_dense *) A0, (permuted_dense *) A1};
    matrix *A = new_stacked_pd(5, 3, 2, blocks, NULL, NULL);

    matrix *C_m = BA_pd_spd_alloc((permuted_dense *) B, (stacked_pd *) A);
    BA_pd_spd_fill_values((permuted_dense *) B, (stacked_pd *) A,
                          (permuted_dense *) C_m);
    permuted_dense *C = (permuted_dense *) C_m;

    /* Only A_0 contributes -> col_perm = {0, 2}.
       C[0,0]= 1*5+2*7 = 19; C[0,2]= 1*6+2*8 = 22;
       C[1,0]= 3*5+4*7 = 43; C[1,2]= 3*6+4*8 = 50.                       */
    int cp_exp[2] = {0, 2};
    double CX_exp[4] = {19, 22, 43, 50};
    mu_assert("C n0", C->n0 == 2);
    mu_assert("C col_perm", cmp_int_array(C->col_perm, cp_exp, 2));
    mu_assert("C X", cmp_double_array(C->X, CX_exp, 4));

    free_matrix(C_m);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* B's col_perm disjoint from every A-block's row_perm: empty output. */
const char *test_BA_pd_spd_no_blocks_contribute(void)
{
    int B_rp[2] = {0, 1};
    int B_cp[1] = {0};
    double BX[2] = {1, 2};
    matrix *B = new_permuted_dense(2, 4, 2, 1, B_rp, B_cp, BX);

    int A0_rp[1] = {1};
    int A0_cp[1] = {0};
    double A0X[1] = {7};
    matrix *A0 = new_permuted_dense(4, 4, 1, 1, A0_rp, A0_cp, A0X);

    int A1_rp[1] = {3};
    int A1_cp[1] = {1};
    double A1X[1] = {9};
    matrix *A1 = new_permuted_dense(4, 4, 1, 1, A1_rp, A1_cp, A1X);

    permuted_dense *blocks[2] = {(permuted_dense *) A0, (permuted_dense *) A1};
    matrix *A = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);

    matrix *C_m = BA_pd_spd_alloc((permuted_dense *) B, (stacked_pd *) A);
    BA_pd_spd_fill_values((permuted_dense *) B, (stacked_pd *) A,
                          (permuted_dense *) C_m);
    permuted_dense *C = (permuted_dense *) C_m;

    mu_assert("C n0", C->n0 == 0);
    mu_assert("C nnz", C_m->nnz == 0);

    free_matrix(C_m);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* Empty A (n_blocks = 0): output has n0 = 0. */
const char *test_BA_pd_spd_empty_A(void)
{
    int B_rp[1] = {0};
    int B_cp[1] = {0};
    double BX[1] = {5};
    matrix *B = new_permuted_dense(1, 3, 1, 1, B_rp, B_cp, BX);

    matrix *A = new_stacked_pd(3, 4, 0, NULL, NULL, NULL);

    matrix *C_m = BA_pd_spd_alloc((permuted_dense *) B, (stacked_pd *) A);
    BA_pd_spd_fill_values((permuted_dense *) B, (stacked_pd *) A,
                          (permuted_dense *) C_m);

    mu_assert("C n0", ((permuted_dense *) C_m)->n0 == 0);
    mu_assert("C nnz", C_m->nnz == 0);

    free_matrix(C_m);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* Both A-blocks contribute AND have overlapping col_perms: cols accumulate. */
const char *test_BA_pd_spd_overlapping_col_perms(void)
{
    /* B: 2x3, row_perm {0,1}, col_perm {0,1,2}, X = [[1,2,3],[4,5,6]].   */
    int B_rp[2] = {0, 1};
    int B_cp[3] = {0, 1, 2};
    double BX[6] = {1, 2, 3, 4, 5, 6};
    matrix *B = new_permuted_dense(2, 3, 2, 3, B_rp, B_cp, BX);

    /* A_0: rows {0,1}, cols {0,1}, X = [[10,11],[12,13]].
       A_1: rows {2},   cols {1,2}, X = [20,21]. col 1 overlaps A_0.    */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 1};
    double A0X[4] = {10, 11, 12, 13};
    matrix *A0 = new_permuted_dense(3, 3, 2, 2, A0_rp, A0_cp, A0X);

    int A1_rp[1] = {2};
    int A1_cp[2] = {1, 2};
    double A1X[2] = {20, 21};
    matrix *A1 = new_permuted_dense(3, 3, 1, 2, A1_rp, A1_cp, A1X);

    permuted_dense *blocks[2] = {(permuted_dense *) A0, (permuted_dense *) A1};
    matrix *A = new_stacked_pd(3, 3, 2, blocks, NULL, NULL);

    matrix *C_m = BA_pd_spd_alloc((permuted_dense *) B, (stacked_pd *) A);
    BA_pd_spd_fill_values((permuted_dense *) B, (stacked_pd *) A,
                          (permuted_dense *) C_m);
    permuted_dense *C = (permuted_dense *) C_m;

    /* Output col_perm = {0, 1, 2}.
       Global A entries:
         A[0,0]=10, A[0,1]=11, A[1,0]=12, A[1,1]=13, A[2,1]=20, A[2,2]=21.
       C[0,0]= 1*10+2*12 = 34;
       C[0,1]= 1*11+2*13+3*20 = 11+26+60 = 97;
       C[0,2]= 3*21 = 63;
       C[1,0]= 4*10+5*12 = 100;
       C[1,1]= 4*11+5*13+6*20 = 44+65+120 = 229;
       C[1,2]= 6*21 = 126.                                              */
    int cp_exp[3] = {0, 1, 2};
    double CX_exp[6] = {34, 97, 63, 100, 229, 126};
    mu_assert("C n0", C->n0 == 3);
    mu_assert("C col_perm", cmp_int_array(C->col_perm, cp_exp, 3));
    mu_assert("C X", cmp_double_array(C->X, CX_exp, 6));

    free_matrix(C_m);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* Two-phase: alloc once, mutate inputs, refill, verify. */
const char *test_BA_pd_spd_alloc_then_fill_values(void)
{
    int B_rp[2] = {0, 1};
    int B_cp[2] = {0, 1};
    double BX[4] = {1, 2, 3, 4};
    matrix *B = new_permuted_dense(2, 3, 2, 2, B_rp, B_cp, BX);

    int A0_rp[2] = {0, 1};
    int A0_cp[1] = {0};
    double A0X[2] = {10, 20};
    matrix *A0 = new_permuted_dense(3, 2, 2, 1, A0_rp, A0_cp, A0X);

    permuted_dense *blocks[1] = {(permuted_dense *) A0};
    matrix *A = new_stacked_pd(3, 2, 1, blocks, NULL, NULL);

    matrix *C_m = BA_pd_spd_alloc((permuted_dense *) B, (stacked_pd *) A);
    BA_pd_spd_fill_values((permuted_dense *) B, (stacked_pd *) A,
                          (permuted_dense *) C_m);

    /* Mutate B and A_0 values. */
    permuted_dense *B_pd = (permuted_dense *) B;
    B_pd->X[0] = 5;
    B_pd->X[1] = 6;
    B_pd->X[2] = 7;
    B_pd->X[3] = 8;
    permuted_dense *A0_pd = (permuted_dense *) A0;
    A0_pd->X[0] = 100;
    A0_pd->X[1] = 200;

    BA_pd_spd_fill_values((permuted_dense *) B, (stacked_pd *) A,
                          (permuted_dense *) C_m);
    permuted_dense *C = (permuted_dense *) C_m;

    /* C col_perm = {0}.
       C[0,0]= 5*100+6*200 = 1700;
       C[1,0]= 7*100+8*200 = 2300.                                       */
    double CX_exp[2] = {1700, 2300};
    mu_assert("C X refilled", cmp_double_array(C->X, CX_exp, 2));

    free_matrix(C_m);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* Vtable: copy_sparsity through M->copy_sparsity(M). */
const char *test_spd_vtable_copy_sparsity(void)
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
    matrix *M = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);

    matrix *out_m = M->copy_sparsity(M);
    stacked_pd *out = (stacked_pd *) out_m;

    mu_assert("n_blocks", out->n_blocks == 2);
    mu_assert("O0 row_perm", cmp_int_array(out->blocks[0]->row_perm, row_perm_0, 2));
    mu_assert("O0 col_perm", cmp_int_array(out->blocks[0]->col_perm, col_perm_0, 2));
    mu_assert("O1 row_perm", cmp_int_array(out->blocks[1]->row_perm, row_perm_1, 1));
    mu_assert("O1 col_perm", cmp_int_array(out->blocks[1]->col_perm, col_perm_1, 2));

    free_matrix(out_m);
    free_matrix(M);
    return 0;
}

/* Vtable: DA_fill_values through A->DA_fill_values(d, A, C). */
const char *test_spd_vtable_DA_fill_values(void)
{
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 1};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[2] = {2, 3};
    int col_perm_1[2] = {2, 3};
    double X1[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *blk1 = new_permuted_dense(4, 4, 2, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A_m = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);

    double d[4] = {10.0, 100.0, 1000.0, 10000.0};

    matrix *C_m = A_m->copy_sparsity(A_m);
    A_m->DA_fill_values(d, A_m, C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    double C0_X_exp[4] = {10.0, 20.0, 300.0, 400.0};
    mu_assert("C0 X", cmp_double_array(C->blocks[0]->X, C0_X_exp, 4));

    double C1_X_exp[4] = {5000.0, 6000.0, 70000.0, 80000.0};
    mu_assert("C1 X", cmp_double_array(C->blocks[1]->X, C1_X_exp, 4));

    free_matrix(C_m);
    free_matrix(A_m);
    return 0;
}

/* Vtable: ATA_alloc through A->ATA_alloc(A). */
const char *test_spd_vtable_ATA_alloc(void)
{
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 1};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[2] = {2, 3};
    int col_perm_1[2] = {2, 3};
    double X1[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *blk1 = new_permuted_dense(4, 4, 2, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A_m = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);

    matrix *C_m = A_m->ATA_alloc(A_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("n_blocks", C->n_blocks == 2);
    mu_assert("C0 row_perm", cmp_int_array(C->blocks[0]->row_perm, col_perm_0, 2));
    mu_assert("C0 col_perm", cmp_int_array(C->blocks[0]->col_perm, col_perm_0, 2));

    free_matrix(C_m);
    free_matrix(A_m);
    return 0;
}

/* Vtable: ATDA_fill_values through A->ATDA_fill_values(A, d, C). */
const char *test_spd_vtable_ATDA_fill_values(void)
{
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 1};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[2] = {2, 3};
    int col_perm_1[2] = {2, 3};
    double X1[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *blk1 = new_permuted_dense(4, 4, 2, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A_m = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);

    double d[4] = {1.0, 2.0, 3.0, 4.0};

    matrix *C_m = A_m->ATA_alloc(A_m);
    A_m->ATDA_fill_values(A_m, d, C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    /* Same expected values as test_ATDA_spd_disjoint_cols. */
    double C0_X_exp[4] = {19.0, 26.0, 26.0, 36.0};
    mu_assert("C0 X", cmp_double_array(C->blocks[0]->X, C0_X_exp, 4));

    double C1_X_exp[4] = {271.0, 314.0, 314.0, 364.0};
    mu_assert("C1 X", cmp_double_array(C->blocks[1]->X, C1_X_exp, 4));

    free_matrix(C_m);
    free_matrix(A_m);
    return 0;
}

/* Vtable: transpose_alloc + transpose_fill_values. */
const char *test_spd_vtable_transpose(void)
{
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

    matrix *out_m = src_m->transpose_alloc(src_m);
    src_m->transpose_fill_values(src_m, out_m);
    stacked_pd *out = (stacked_pd *) out_m;

    /* Same expected values as test_transpose_spd_no_overlap. */
    mu_assert("n_blocks", out->n_blocks == 2);
    double O0_X_exp[4] = {1.0, 3.0, 2.0, 4.0};
    mu_assert("O0 X", cmp_double_array(out->blocks[0]->X, O0_X_exp, 4));
    double O1_X_exp[4] = {5.0, 7.0, 6.0, 8.0};
    mu_assert("O1 X", cmp_double_array(out->blocks[1]->X, O1_X_exp, 4));

    free_matrix(out_m);
    free_matrix(src_m);
    return 0;
}

/* Vtable: refresh_csc_values is a no-op; verify no crash. */
const char *test_spd_vtable_refresh_csc_values_noop(void)
{
    int row_perm[1] = {0};
    int col_perm[1] = {0};
    double X[1] = {1.0};
    matrix *blk = new_permuted_dense(1, 1, 1, 1, row_perm, col_perm, X);

    permuted_dense *blocks[1] = {(permuted_dense *) blk};
    matrix *M = new_stacked_pd(1, 1, 1, blocks, NULL, NULL);

    M->refresh_csc_values(M);

    free_matrix(M);
    return 0;
}

/* index_* on spd: rows are routed to the source block (if any) that
   carries them; per-block index_* is reused; empty blocks are dropped. */
const char *test_spd_vtable_index(void)
{
    /* 6x4 spd:
       block 0: rows {0,1}, cols {0,2}, X = [[1,2],[3,4]]
       block 1: rows {2,3}, cols {1,3}, X = [[5,6],[7,8]]                  */
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 2};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(6, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[2] = {2, 3};
    int col_perm_1[2] = {1, 3};
    double X1[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *blk1 = new_permuted_dense(6, 4, 2, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *M = new_stacked_pd(6, 4, 2, blocks, NULL, NULL);

    /* indices = [3, 0, 5, 1]: row 3 → blk1, row 0/1 → blk0, row 5 → none.
       Expected output (4 rows, dropping output position 2):
         output row 0 = src row 3 = blk1 row [7, 8] in cols {1,3}
         output row 1 = src row 0 = blk0 row [1, 2] in cols {0,2}
         output row 3 = src row 1 = blk0 row [3, 4] in cols {0,2}        */
    int indices[4] = {3, 0, 5, 1};
    matrix *C_m = M->index_alloc(M, indices, 4);
    M->index_fill_values(M, indices, 4, C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("n_blocks", C->n_blocks == 2);
    mu_assert("base.m", C_m->m == 4);
    mu_assert("base.n", C_m->n == 4);

    /* Per the per-block alloc, blk0 yields output positions where indices
       hits {0,1}: that's i=1 (indices[1]=0) and i=3 (indices[3]=1). So
       output block 0 carries row_perm={1,3}, col_perm={0,2}, X=[1,2,3,4]. */
    permuted_dense *out0 = C->blocks[0];
    int expected_row_perm_0[2] = {1, 3};
    int expected_col_perm_0[2] = {0, 2};
    double expected_X0[4] = {1.0, 2.0, 3.0, 4.0};
    mu_assert("out0 m0", out0->m0 == 2);
    mu_assert("out0 n0", out0->n0 == 2);
    mu_assert("out0 row_perm",
              cmp_int_array(out0->row_perm, expected_row_perm_0, 2));
    mu_assert("out0 col_perm",
              cmp_int_array(out0->col_perm, expected_col_perm_0, 2));
    mu_assert("out0 X", cmp_double_array(out0->X, expected_X0, 4));

    /* blk1 hit only at i=0 (indices[0]=3 → blk1 row 1). Output block 1:
       row_perm={0}, col_perm={1,3}, X=[7,8].                              */
    permuted_dense *out1 = C->blocks[1];
    int expected_row_perm_1[1] = {0};
    int expected_col_perm_1[2] = {1, 3};
    double expected_X1[2] = {7.0, 8.0};
    mu_assert("out1 m0", out1->m0 == 1);
    mu_assert("out1 n0", out1->n0 == 2);
    mu_assert("out1 row_perm",
              cmp_int_array(out1->row_perm, expected_row_perm_1, 1));
    mu_assert("out1 col_perm",
              cmp_int_array(out1->col_perm, expected_col_perm_1, 2));
    mu_assert("out1 X", cmp_double_array(out1->X, expected_X1, 2));

    /* src_block_idx: both source blocks survived, identity mapping. */
    mu_assert("src_block_idx[0]", C->src_block_idx[0] == 0);
    mu_assert("src_block_idx[1]", C->src_block_idx[1] == 1);

    free_matrix(C_m);
    free_matrix(M);
    return 0;
}

/* promote_* on spd: replicate the single row across `size` rows; per-block
   delegation drops empty blocks so output has at most one block. */
const char *test_spd_vtable_promote(void)
{
    /* 1x4 spd, single block carries row 0 at cols {0, 2} with values
       [9, 11]. Promote to size=3.                                         */
    int row_perm[1] = {0};
    int col_perm[2] = {0, 2};
    double X[2] = {9.0, 11.0};
    matrix *blk = new_permuted_dense(1, 4, 1, 2, row_perm, col_perm, X);

    permuted_dense *blocks[1] = {(permuted_dense *) blk};
    matrix *M = new_stacked_pd(1, 4, 1, blocks, NULL, NULL);

    matrix *C_m = M->promote_alloc(M, 3);
    M->promote_fill_values(M, C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("n_blocks", C->n_blocks == 1);
    mu_assert("base.m", C_m->m == 3);
    mu_assert("base.n", C_m->n == 4);

    permuted_dense *out0 = C->blocks[0];
    int expected_row_perm[3] = {0, 1, 2};
    int expected_col_perm[2] = {0, 2};
    double expected_X[6] = {9.0, 11.0, 9.0, 11.0, 9.0, 11.0};
    mu_assert("out0 m0", out0->m0 == 3);
    mu_assert("out0 n0", out0->n0 == 2);
    mu_assert("out0 row_perm", cmp_int_array(out0->row_perm, expected_row_perm, 3));
    mu_assert("out0 col_perm", cmp_int_array(out0->col_perm, expected_col_perm, 2));
    mu_assert("out0 X", cmp_double_array(out0->X, expected_X, 6));

    free_matrix(C_m);
    free_matrix(M);
    return 0;
}

/* diag_vec_* on spd: per-block row_perm entries are rescaled r -> r*(n+1);
   X buffers are unchanged; structure is preserved (same n_blocks). */
const char *test_spd_vtable_diag_vec(void)
{
    /* 4x4 spd. Input represents the Jacobian of a length-4 vector w.r.t.
       4 variables. Output represents the Jacobian of diag(v) (16x4) with
       only diagonal rows {0, 5, 10, 15} populated.
       Block 0: rows {0, 2}, cols {0, 1}, X = [[1, 2], [3, 4]]
       Block 1: rows {1, 3}, cols {2, 3}, X = [[5, 6], [7, 8]]              */
    int row_perm_0[2] = {0, 2};
    int col_perm_0[2] = {0, 1};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[2] = {1, 3};
    int col_perm_1[2] = {2, 3};
    double X1[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *blk1 = new_permuted_dense(4, 4, 2, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *M = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);

    matrix *C_m = M->diag_vec_alloc(M);
    M->diag_vec_fill_values(M, C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("n_blocks", C->n_blocks == 2);
    mu_assert("base.m", C_m->m == 16);
    mu_assert("base.n", C_m->n == 4);

    /* Block 0: row_perm = {0*5, 2*5} = {0, 10}; col_perm and X unchanged. */
    permuted_dense *out0 = C->blocks[0];
    int expected_row_perm_0[2] = {0, 10};
    mu_assert("out0 m0", out0->m0 == 2);
    mu_assert("out0 n0", out0->n0 == 2);
    mu_assert("out0 row_perm",
              cmp_int_array(out0->row_perm, expected_row_perm_0, 2));
    mu_assert("out0 col_perm", cmp_int_array(out0->col_perm, col_perm_0, 2));
    mu_assert("out0 X", cmp_double_array(out0->X, X0, 4));

    /* Block 1: row_perm = {1*5, 3*5} = {5, 15}; col_perm and X unchanged. */
    permuted_dense *out1 = C->blocks[1];
    int expected_row_perm_1[2] = {5, 15};
    mu_assert("out1 m0", out1->m0 == 2);
    mu_assert("out1 n0", out1->n0 == 2);
    mu_assert("out1 row_perm",
              cmp_int_array(out1->row_perm, expected_row_perm_1, 2));
    mu_assert("out1 col_perm", cmp_int_array(out1->col_perm, col_perm_1, 2));
    mu_assert("out1 X", cmp_double_array(out1->X, X1, 4));

    free_matrix(C_m);
    free_matrix(M);
    return 0;
}

/* broadcast_* on spd (BROADCAST_ROW): input Jac for a (1, 4) matrix has
   m=4; broadcasting to (2, 4) gives output Jac with m=8. Per-block PD
   rescales row_perm entries r -> {r*d1, r*d1+1, ..., r*d1+d1-1}. */
const char *test_spd_vtable_broadcast_row(void)
{
    /* Input: 4x4 spd (Jac of a (1, 4) matrix-valued node), two blocks.
       Block 0: rows {0, 1}, cols {0, 1}, X = [[1, 2], [3, 4]]
       Block 1: rows {2, 3}, cols {2, 3}, X = [[5, 6], [7, 8]]              */
    int row_perm_0[2] = {0, 1};
    int col_perm_0[2] = {0, 1};
    double X0[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *blk0 = new_permuted_dense(4, 4, 2, 2, row_perm_0, col_perm_0, X0);

    int row_perm_1[2] = {2, 3};
    int col_perm_1[2] = {2, 3};
    double X1[4] = {5.0, 6.0, 7.0, 8.0};
    matrix *blk1 = new_permuted_dense(4, 4, 2, 2, row_perm_1, col_perm_1, X1);

    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *M = new_stacked_pd(4, 4, 2, blocks, NULL, NULL);

    /* d1=2, d2=4 -> output Jac is 8x4 (matrix value (2, 4) vectorized). */
    matrix *C_m = M->broadcast_alloc(M, BROADCAST_ROW, 2, 4);
    M->broadcast_fill_values(M, BROADCAST_ROW, 2, 4, C_m);
    stacked_pd *C = (stacked_pd *) C_m;

    mu_assert("n_blocks", C->n_blocks == 2);
    mu_assert("base.m", C_m->m == 8);
    mu_assert("base.n", C_m->n == 4);

    /* Block 0: row_perm = {0*2, 0*2+1, 1*2, 1*2+1} = {0, 1, 2, 3}.
       col_perm unchanged. X: each input row replicated d1=2 times.        */
    permuted_dense *out0 = C->blocks[0];
    int expected_row_perm_0[4] = {0, 1, 2, 3};
    double expected_X0[8] = {1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0};
    mu_assert("out0 m0", out0->m0 == 4);
    mu_assert("out0 n0", out0->n0 == 2);
    mu_assert("out0 row_perm",
              cmp_int_array(out0->row_perm, expected_row_perm_0, 4));
    mu_assert("out0 col_perm", cmp_int_array(out0->col_perm, col_perm_0, 2));
    mu_assert("out0 X", cmp_double_array(out0->X, expected_X0, 8));

    /* Block 1: row_perm = {2*2, 2*2+1, 3*2, 3*2+1} = {4, 5, 6, 7}.        */
    permuted_dense *out1 = C->blocks[1];
    int expected_row_perm_1[4] = {4, 5, 6, 7};
    double expected_X1[8] = {5.0, 6.0, 5.0, 6.0, 7.0, 8.0, 7.0, 8.0};
    mu_assert("out1 m0", out1->m0 == 4);
    mu_assert("out1 n0", out1->n0 == 2);
    mu_assert("out1 row_perm",
              cmp_int_array(out1->row_perm, expected_row_perm_1, 4));
    mu_assert("out1 col_perm", cmp_int_array(out1->col_perm, col_perm_1, 2));
    mu_assert("out1 X", cmp_double_array(out1->X, expected_X1, 8));

    free_matrix(C_m);
    free_matrix(M);
    return 0;
}

#endif /* TEST_STACKED_PD_H */
