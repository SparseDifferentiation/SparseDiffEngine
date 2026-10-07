#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "atoms/affine.h"
#include "atoms/bivariate_restricted_dom.h"
#include "atoms/elementwise_full_dom.h"
#include "atoms/elementwise_restricted_dom.h"
#include "expr.h"
#include "minunit.h"
#include "test_helpers.h"

const char *test_log(void)
{
    double u[2] = {1.0, 2.718281828};
    expr *var = new_variable(2, 1, 0, 2);
    expr *log_node = new_log(var);
    log_node->forward(log_node, u);
    double correct[2] = {log(1.0), log(2.718281828)};
    mu_assert("fail", cmp_double_array(log_node->value, correct, 2));
    free_expr(log_node);
    return 0;
}

/* Restricted-domain atoms are leaf-only: a non-variable child must be
   rejected at construction (NULL), and the caller keeps ownership of it. */
const char *test_restricted_rejects_non_variable_child(void)
{
    CSR_matrix *A = new_csr_random(3, 4, 1.0);
    expr *x = new_variable(4, 1, 0, 4);
    expr *Ax = new_left_matmul(NULL, x, A);

    mu_assert("log of affine child should be NULL", new_log(Ax) == NULL);
    mu_assert("rel_entr of affine child should be NULL",
              new_rel_entr_vector_args(Ax, x) == NULL);
    mu_assert("quad_over_lin with non-scalar affine denominator should be NULL",
              new_quad_over_lin(x, Ax) == NULL);

    free_expr(Ax);
    free_CSR_matrix(A);
    return 0;
}
