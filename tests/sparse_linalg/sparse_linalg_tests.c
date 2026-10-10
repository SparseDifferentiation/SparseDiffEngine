#include <stdio.h>

#include "minunit.h"

/* Library-only suite: this executable links sparse_linalg alone, so every
   header below may include only sparse_linalg/<name>.h, minunit.h,
   test_helpers.h and sibling test_<name>.h files (checked at configure time
   in CMakeLists.txt). Tests that need old-code or engine kernels live under
   tests/old-code and run from all_tests. */
#include "test_COO_matrix.h"
#include "test_alloc_overflow.h"
#include "test_cblas.h"
#include "test_csc_matrix.h"
#include "test_csr_csc_conversion.h"
#include "test_csr_matrix.h"
#include "test_linalg_sparse_matmuls.h"
#include "test_linalg_utils_matmul_chain_rule.h"
#include "test_matmul_dispatchers.h"
#include "test_matrix.h"
#include "test_permuted_dense.h"
#include "test_row_gather.h"
#include "test_row_reduce.h"
#include "test_stacked_pd.h"

int main(void)
{
    printf("=== Running sparse_linalg Tests ===\n\n");

    int tests_run = 0;

    printf("--- sparse_linalg Tests ---\n");
    mu_run_test(test_sat_mul_int_clamps_on_overflow, tests_run);
    mu_run_test(test_row_gather_alloc_no_int_overflow, tests_run);
    mu_run_test(test_cblas_ddot, tests_run);
    mu_run_test(test_transpose, tests_run);
    mu_run_test(test_AT_alloc_and_fill, tests_run);
    mu_run_test(test_csr_to_csc_split, tests_run);
    mu_run_test(test_csc_to_csr_sparsity, tests_run);
    mu_run_test(test_csc_to_csr_values, tests_run);
    mu_run_test(test_csr_csc_csr_roundtrip, tests_run);
    mu_run_test(test_block_left_multiply_single_block, tests_run);
    mu_run_test(test_block_left_multiply_two_blocks, tests_run);
    mu_run_test(test_block_left_multiply_zero_column, tests_run);
    mu_run_test(test_block_left_multiply_dedup_order, tests_run);
    mu_run_test(test_block_left_multiply_matches_reference_random, tests_run);
    mu_run_test(test_csr_csc_matmul_alloc_basic, tests_run);
    mu_run_test(test_csr_csc_matmul_alloc_sparse, tests_run);
    mu_run_test(test_csr_csc_matmul_alloc_dedup_order, tests_run);
    mu_run_test(test_csr_csc_matmul_alloc_matches_reference_random, tests_run);
    mu_run_test(test_block_left_multiply_vec_single_block, tests_run);
    mu_run_test(test_block_left_multiply_vec_two_blocks, tests_run);
    mu_run_test(test_block_left_multiply_vec_sparse, tests_run);
    mu_run_test(test_block_left_multiply_vec_three_blocks, tests_run);
    mu_run_test(test_ATA_alloc_simple, tests_run);
    mu_run_test(test_ATA_alloc_diagonal_like, tests_run);
    mu_run_test(test_ATA_alloc_random, tests_run);
    mu_run_test(test_ATA_alloc_random2, tests_run);
    mu_run_test(test_BTA_alloc_and_BTDA_fill, tests_run);
    mu_run_test(test_csr_to_coo, tests_run);
    mu_run_test(test_csr_to_coo_lower_triangular, tests_run);
    mu_run_test(test_refresh_lower_triangular_coo, tests_run);
    mu_run_test(test_pd_mult_vec_basic, tests_run);
    mu_run_test(test_pd_mult_vec_blocks, tests_run);
    mu_run_test(test_sparse_vs_pd_mult_vec, tests_run);
    mu_run_test(test_pd_trans_full_block, tests_run);
    mu_run_test(test_sparse_vs_pd_mult_vec_blocks, tests_run);
    mu_run_test(test_pd_operator_block_left_mult_vec, tests_run);
    mu_run_test(test_permuted_dense_to_csr_basic, tests_run);
    mu_run_test(test_permuted_dense_to_csr_empty, tests_run);
    mu_run_test(test_permuted_dense_to_csr_full, tests_run);
    mu_run_test(test_permuted_dense_to_csr_single_row, tests_run);
    mu_run_test(test_permuted_dense_to_csr_single_col, tests_run);
    mu_run_test(test_DA_pd_fill_values, tests_run);
    mu_run_test(test_ATA_pd_alloc, tests_run);
    mu_run_test(test_ATDA_pd_fill_values, tests_run);
    mu_run_test(test_permuted_dense_times_csc, tests_run);
    mu_run_test(test_permuted_dense_times_csc_no_active, tests_run);
    mu_run_test(test_permuted_dense_to_csr_lazy, tests_run);
    mu_run_test(test_permuted_dense_col_inv, tests_run);
    mu_run_test(test_permuted_dense_row_gather, tests_run);
    mu_run_test(test_row_gather_sparse, tests_run);
    mu_run_test(test_row_gather_pd_vs_sparse_twin, tests_run);
    mu_run_test(test_row_gather_spd_vs_sparse_twin, tests_run);
#ifdef SP_TRACK_MEMORY
    mu_run_test(test_row_gather_spd_fill_no_transient_alloc, tests_run);
#endif
    mu_run_test(test_row_reduce_sparse, tests_run);
    mu_run_test(test_row_reduce_pd, tests_run);
    mu_run_test(test_row_reduce_spd_cross_block_accumulate, tests_run);
    mu_run_test(test_row_reduce_spd_within_block, tests_run);
    mu_run_test(test_row_reduce_spd_all_to_one, tests_run);
#ifdef SP_TRACK_MEMORY
    mu_run_test(test_row_reduce_spd_fill_no_transient_alloc, tests_run);
#endif
    mu_run_test(test_permuted_dense_diag_vec, tests_run);
    mu_run_test(test_permuted_dense_BTA_matching_row_perm, tests_run);
    mu_run_test(test_permuted_dense_BTA_empty_overlap, tests_run);
    mu_run_test(test_permuted_dense_BTA_partial_overlap, tests_run);
    mu_run_test(test_permuted_dense_BTDA_decomposition, tests_run);
    mu_run_test(test_permuted_dense_BTDA_matching_row_perm, tests_run);
    mu_run_test(test_permuted_dense_BTDA_partial_overlap, tests_run);
    mu_run_test(test_BA_pd_matrices_pd_pd_full_block_B, tests_run);
    mu_run_test(test_BA_pd_matrices_pd_pd_general_B, tests_run);
    mu_run_test(test_BA_pd_matrices_pd_csc, tests_run);
    mu_run_test(test_BA_pd_matrices_spd_A, tests_run);
    mu_run_test(test_BA_pd_matrices_fast_path, tests_run);
    mu_run_test(test_BA_spd_csc_two_blocks_both_kept, tests_run);
    mu_run_test(test_BA_spd_csc_one_block_dropped, tests_run);
    mu_run_test(test_BA_spd_csc_all_blocks_dropped, tests_run);
    mu_run_test(test_BA_spd_pd_two_blocks_both_kept, tests_run);
    mu_run_test(test_BA_spd_pd_one_block_dropped, tests_run);
    mu_run_test(test_BA_spd_spd_two_blocks_both_kept, tests_run);
    mu_run_test(test_BA_spd_spd_one_block_dropped, tests_run);
    mu_run_test(test_BA_spd_spd_all_blocks_dropped, tests_run);
    mu_run_test(test_BA_spd_spd_empty_A, tests_run);
    mu_run_test(test_BA_spd_spd_empty_B, tests_run);
    mu_run_test(test_BA_spd_spd_alloc_then_fill_values, tests_run);
    mu_run_test(test_BTDA_matrices_pd_pd, tests_run);
    mu_run_test(test_BTDA_matrices_pd_csr, tests_run);
    mu_run_test(test_BTA_csc_pd_basic, tests_run);
    mu_run_test(test_BTA_csc_pd_partial_and_excluded_cols, tests_run);
    mu_run_test(test_BTA_csc_pd_empty, tests_run);
    mu_run_test(test_BTDA_matrices_spd_pd, tests_run);
    mu_run_test(test_BTDA_matrices_pd_spd, tests_run);
    mu_run_test(test_BTA_pd_csc_basic, tests_run);
    mu_run_test(test_BTA_pd_csc_partial_and_excluded_cols, tests_run);
    mu_run_test(test_BTA_pd_csc_empty, tests_run);
    mu_run_test(test_BTDA_matrices_spd_spd, tests_run);
    mu_run_test(test_BTA_pd_spd_two_blocks_both_kept, tests_run);
    mu_run_test(test_BTDA_pd_spd_two_blocks_both_kept, tests_run);
    mu_run_test(test_BTDA_spd_pd_overlapping_cp, tests_run);
#ifdef SP_TRACK_MEMORY
    mu_run_test(test_BTDA_fill_no_transient_alloc, tests_run);
#endif
    mu_run_test(test_BTA_spd_pd_overlapping_cp, tests_run);
    mu_run_test(test_BTDA_spd_csc_overlapping_cp, tests_run);
    mu_run_test(test_BTA_spd_csc_overlapping, tests_run);
    mu_run_test(test_BTA_spd_csc_block_no_overlap, tests_run);
    mu_run_test(test_BTDA_spd_spd_overlapping, tests_run);
    mu_run_test(test_BTA_spd_spd_overlapping, tests_run);
    mu_run_test(test_BTA_spd_spd_multi_A_per_block, tests_run);
    mu_run_test(test_BTA_spd_spd_nonoverlapping_block, tests_run);
    mu_run_test(test_BTA_spd_matrices_pd_A, tests_run);
    mu_run_test(test_BTA_spd_matrices_csc_A, tests_run);
    mu_run_test(test_BTA_spd_matrices_spd_A, tests_run);
    mu_run_test(test_BTA_pd_matrices_pd_A, tests_run);
    mu_run_test(test_BTA_pd_matrices_csc_A, tests_run);
    mu_run_test(test_BTA_pd_matrices_spd_A, tests_run);
    mu_run_test(test_BTDA_csc_spd_overlapping, tests_run);
    mu_run_test(test_BTA_csc_spd_overlapping, tests_run);
    mu_run_test(test_BTA_csc_spd_block_no_overlap, tests_run);
    mu_run_test(test_BTA_sparse_matrices_pd_A, tests_run);
    mu_run_test(test_BTA_sparse_matrices_csc_A, tests_run);
    mu_run_test(test_BTA_sparse_matrices_spd_A, tests_run);
    mu_run_test(test_BTA_matrices_fill_pd_spd, tests_run);
    mu_run_test(test_BTA_matrices_fill_spd_csc, tests_run);
    mu_run_test(test_BTA_matrices_fill_csc_pd, tests_run);
    mu_run_test(test_BTA_matrices_fill_csc_csc, tests_run);
    mu_run_test(test_BA_pd_kron_spd_no_cache_staleness, tests_run);
    mu_run_test(test_BA_pd_spd_transpose_cache_refresh, tests_run);
    mu_run_test(test_stacked_pd_construct_and_free, tests_run);
    mu_run_test(test_coalesce_no_overlap, tests_run);
    mu_run_test(test_coalesce_three_signatures, tests_run);
    mu_run_test(test_coalesce_shared_signature_merges_rows, tests_run);
    mu_run_test(test_coalesce_alloc_then_fill_values, tests_run);
    mu_run_test(test_coalesce_empty_input, tests_run);
    mu_run_test(test_transpose_spd_no_overlap, tests_run);
    mu_run_test(test_transpose_spd_overlap_coalesces, tests_run);
    mu_run_test(test_transpose_spd_alloc_then_fill_values, tests_run);
    mu_run_test(test_transpose_spd_empty, tests_run);
    mu_run_test(test_transpose_spd_single_block_full, tests_run);
    mu_run_test(test_copy_sparsity_spd_alloc, tests_run);
    mu_run_test(test_DA_spd_two_blocks, tests_run);
    mu_run_test(test_DA_spd_empty, tests_run);
    mu_run_test(test_DA_spd_single_block_full, tests_run);
    mu_run_test(test_ATA_spd_alloc_disjoint_cols, tests_run);
    mu_run_test(test_ATA_spd_alloc_overlapping_cols, tests_run);
    mu_run_test(test_ATDA_spd_disjoint_cols, tests_run);
    mu_run_test(test_ATDA_spd_overlapping_cols, tests_run);
    mu_run_test(test_ATDA_spd_alloc_then_fill_values, tests_run);
    mu_run_test(test_ATDA_spd_empty, tests_run);
    mu_run_test(test_BA_pd_spd_two_blocks_disjoint_cols, tests_run);
    mu_run_test(test_BA_pd_spd_only_one_block_contributes, tests_run);
    mu_run_test(test_BA_pd_spd_no_blocks_contribute, tests_run);
    mu_run_test(test_BA_pd_spd_empty_A, tests_run);
    mu_run_test(test_BA_pd_spd_overlapping_col_perms, tests_run);
    mu_run_test(test_BA_pd_spd_alloc_then_fill_values, tests_run);
    mu_run_test(test_spd_vtable_copy_sparsity, tests_run);
    mu_run_test(test_spd_vtable_DA_fill_values, tests_run);
    mu_run_test(test_spd_vtable_ATA_alloc, tests_run);
    mu_run_test(test_spd_vtable_ATDA_fill_values, tests_run);
    mu_run_test(test_spd_vtable_transpose, tests_run);
    mu_run_test(test_spd_vtable_refresh_csc_values_noop, tests_run);
    mu_run_test(test_spd_vtable_row_gather, tests_run);
    mu_run_test(test_spd_vtable_diag_vec, tests_run);
    mu_run_test(test_YT_kron_I, tests_run);
    mu_run_test(test_YT_kron_I_larger, tests_run);
    mu_run_test(test_I_kron_X, tests_run);
    mu_run_test(test_I_kron_X_larger, tests_run);

    printf("\n=== All %d sparse_linalg tests passed ===\n", tests_run);
    return 0;
}
