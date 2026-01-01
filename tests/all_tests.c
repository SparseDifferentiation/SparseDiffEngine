#include <stdio.h>

#include "minunit.h"

/* Include all test headers */
#include "forward_pass/affine/test_add.h"
#include "forward_pass/affine/test_hstack.h"
#include "forward_pass/affine/test_linear_op.h"
#include "forward_pass/affine/test_sum.h"
#include "forward_pass/affine/test_variable_constant.h"
#include "forward_pass/composite/test_composite.h"
#include "forward_pass/elementwise/test_exp.h"
#include "forward_pass/elementwise/test_log.h"
#include "jacobian_tests/test_composite.h"
#include "jacobian_tests/test_elementwise_mult.h"
#include "jacobian_tests/test_hstack.h"
#include "jacobian_tests/test_log.h"
#include "jacobian_tests/test_quad_form.h"
#include "jacobian_tests/test_quad_over_lin.h"
#include "jacobian_tests/test_rel_entr.h"
#include "jacobian_tests/test_sum.h"
#include "utils/test_csc_matrix.h"
#include "utils/test_csr_matrix.h"

int main(void)
{
    printf("=== Running All Tests ===\n\n");

    int tests_run = 0;

    printf("--- Forward Pass Tests ---\n");
    mu_run_test(test_variable, tests_run);
    mu_run_test(test_constant, tests_run);
    mu_run_test(test_addition, tests_run);
    mu_run_test(test_linear_op, tests_run);
    mu_run_test(test_exp, tests_run);
    mu_run_test(test_log, tests_run);
    mu_run_test(test_composite, tests_run);
    mu_run_test(test_sum_axis_neg1, tests_run);
    mu_run_test(test_sum_axis_0, tests_run);
    mu_run_test(test_sum_axis_1, tests_run);
    mu_run_test(test_hstack_forward_vectors, tests_run);
    mu_run_test(test_hstack_forward_matrix, tests_run);

    printf("\n--- Jacobian Tests ---\n");
    mu_run_test(test_jacobian_log, tests_run);
    mu_run_test(test_jacobian_composite_log, tests_run);
    mu_run_test(test_jacobian_composite_log_add, tests_run);
    mu_run_test(test_jacobian_rel_entr_vector_args_1, tests_run);
    mu_run_test(test_jacobian_rel_entr_vector_args_2, tests_run);
    mu_run_test(test_jacobian_elementwise_mult_1, tests_run);
    mu_run_test(test_jacobian_elementwise_mult_2, tests_run);
    mu_run_test(test_jacobian_elementwise_mult_3, tests_run);
    mu_run_test(test_jacobian_elementwise_mult_4, tests_run);
    mu_run_test(test_quad_over_lin1, tests_run);
    mu_run_test(test_quad_over_lin2, tests_run);
    mu_run_test(test_quad_over_lin3, tests_run);
    mu_run_test(test_quad_over_lin4, tests_run);
    mu_run_test(test_quad_over_lin5, tests_run);
    mu_run_test(test_quad_form, tests_run);
    mu_run_test(test_quad_form2, tests_run);
    mu_run_test(test_jacobian_sum_log, tests_run);
    mu_run_test(test_jacobian_sum_mult, tests_run);
    mu_run_test(test_jacobian_sum_log_axis_0, tests_run);
    mu_run_test(test_jacobian_sum_add_log_axis_0, tests_run);
    mu_run_test(test_jacobian_sum_log_axis_1, tests_run);
    mu_run_test(test_jacobian_hstack_vectors, tests_run);
    mu_run_test(test_jacobian_hstack_matrix, tests_run);

    printf("\n--- Utility Tests ---\n");
    mu_run_test(test_diag_csr_mult, tests_run);
    mu_run_test(test_csr_sum, tests_run);
    mu_run_test(test_csr_sum2, tests_run);
    mu_run_test(test_transpose, tests_run);
    mu_run_test(test_csr_to_csc1, tests_run);
    mu_run_test(test_csr_to_csc2, tests_run);
    mu_run_test(test_csr_vecmat_values_sparse, tests_run);
    mu_run_test(test_sum_all_rows_csr, tests_run);
    mu_run_test(test_sum_block_of_rows_csr, tests_run);
    mu_run_test(test_sum_evenly_spaced_rows_csr, tests_run);
    mu_run_test(test_ATA_alloc_simple, tests_run);
    mu_run_test(test_ATA_alloc_diagonal_like, tests_run);
    mu_run_test(test_ATA_alloc_random, tests_run);
    mu_run_test(test_ATA_alloc_random2, tests_run);

    printf("\n=== All %d tests passed ===\n", tests_run);

    return 0;
}
