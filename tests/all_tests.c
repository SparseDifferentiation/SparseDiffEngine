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
#include "wsum_hess/elementwise/test_entr.h"
#include "wsum_hess/elementwise/test_exp.h"
#include "wsum_hess/elementwise/test_hyperbolic.h"
#include "wsum_hess/elementwise/test_log.h"
#include "wsum_hess/elementwise/test_logistic.h"
#include "wsum_hess/elementwise/test_power.h"
#include "wsum_hess/elementwise/test_trig.h"
#include "wsum_hess/elementwise/test_xexp.h"
#include "wsum_hess/test_hstack.h"
#include "wsum_hess/test_multiply.h"
#include "wsum_hess/test_rel_entr.h"
#include "wsum_hess/test_sum.h"

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
    mu_run_test(test_jacobian_log_matrix, tests_run);
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
    mu_run_test(test_wsum_hess_multiply_1, tests_run);
    mu_run_test(test_wsum_hess_multiply_2, tests_run);

    printf("\n--- Weighted Sum of Hessian Tests ---\n");
    mu_run_test(test_wsum_hess_log, tests_run);
    mu_run_test(test_wsum_hess_log_composite, tests_run);
    mu_run_test(test_wsum_hess_exp, tests_run);
    mu_run_test(test_wsum_hess_entr, tests_run);
    mu_run_test(test_wsum_hess_logistic, tests_run);
    mu_run_test(test_wsum_hess_power, tests_run);
    mu_run_test(test_wsum_hess_xexp, tests_run);
    mu_run_test(test_wsum_hess_sin, tests_run);
    mu_run_test(test_wsum_hess_cos, tests_run);
    mu_run_test(test_wsum_hess_tan, tests_run);
    mu_run_test(test_wsum_hess_sinh, tests_run);
    mu_run_test(test_wsum_hess_tanh, tests_run);
    mu_run_test(test_wsum_hess_asinh, tests_run);
    mu_run_test(test_wsum_hess_atanh, tests_run);
    mu_run_test(test_wsum_hess_sum_log_linear, tests_run);
    mu_run_test(test_wsum_hess_sum_log_axis0, tests_run);
    mu_run_test(test_wsum_hess_sum_log_axis1, tests_run);
    mu_run_test(test_wsum_hess_rel_entr_1, tests_run);
    mu_run_test(test_wsum_hess_rel_entr_2, tests_run);
    mu_run_test(test_wsum_hess_hstack, tests_run);
    mu_run_test(test_wsum_hess_hstack_matrix, tests_run);
    mu_run_test(test_wsum_hess_multiply_linear_ops, tests_run);

    printf("\n--- Utility Tests ---\n");
    mu_run_test(test_diag_csr_mult, tests_run);
    mu_run_test(test_csr_sum, tests_run);
    mu_run_test(test_csr_sum2, tests_run);
    mu_run_test(test_transpose, tests_run);
    mu_run_test(test_AT_alloc_and_fill, tests_run);
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
    mu_run_test(test_BTA_alloc_and_BTDA_fill, tests_run);

    printf("\n=== All %d tests passed ===\n", tests_run);

    return 0;
}
