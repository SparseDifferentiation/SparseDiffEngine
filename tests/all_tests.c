#include <stdio.h>

#include "minunit.h"

/* Include all test headers */
#include "forward_pass/affine/test_add.h"
#include "forward_pass/affine/test_linear_op.h"
#include "forward_pass/affine/test_variable_constant.h"
#include "forward_pass/composite/test_composite.h"
#include "forward_pass/elementwise/test_exp.h"
#include "forward_pass/elementwise/test_log.h"
#include "jacobian_tests/test_composite.h"
#include "jacobian_tests/test_log.h"
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

    printf("\n--- Jacobian Tests ---\n");
    mu_run_test(test_jacobian_log, tests_run);
    mu_run_test(test_jacobian_composite_log, tests_run);
    mu_run_test(test_jacobian_composite_log_add, tests_run);

    printf("\n--- Utility Tests ---\n");
    mu_run_test(test_diag_csr_mult, tests_run);
    mu_run_test(test_csr_sum, tests_run);
    mu_run_test(test_csr_sum2, tests_run);

    printf("\n=== All %d tests passed ===\n", tests_run);

    return 0;
}
