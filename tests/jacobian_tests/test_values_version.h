#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "atoms/affine.h"
#include "atoms/elementwise_full_dom.h"
#include "expr.h"
#include "minunit.h"
#include "numerical_diff.h"
#include "test_helpers.h"
#include "utils/sparse_matrix.h"

/* Non-affine node: every eval_jacobian call bumps the jacobian's
 * values_version. */
const char *test_values_version_non_affine(void)
{
    double u[3] = {0.1, 0.2, 0.3};
    expr *x = new_variable(3, 1, 0, 3);
    expr *e = new_exp(x);

    jacobian_init(e);
    e->forward(e, u);

    uint64_t v0 = e->jacobian->values_version;
    eval_jacobian(e);
    mu_assert("first eval must bump", e->jacobian->values_version == v0 + 1);
    eval_jacobian(e);
    mu_assert("second eval must bump", e->jacobian->values_version == v0 + 2);

    free_expr(e);
    return 0;
}

/* Affine node: the bump is skipped once the node has been evaluated this
 * parameter epoch (values provably identical), and re-armed by
 * expr_set_needs_refresh. */
const char *test_values_version_affine(void)
{
    double u[3] = {0.1, 0.2, 0.3};
    expr *x = new_variable(3, 1, 0, 3);
    expr *m = new_neg(x);

    jacobian_init(m);
    m->forward(m, u);

    uint64_t v0 = m->jacobian->values_version;
    eval_jacobian(m);
    mu_assert("first eval must bump", m->jacobian->values_version == v0 + 1);
    eval_jacobian(m);
    mu_assert("re-eval of affine node must not bump",
              m->jacobian->values_version == v0 + 1);

    expr_set_needs_refresh(m);
    eval_jacobian(m);
    mu_assert("eval after parameter refresh must bump",
              m->jacobian->values_version == v0 + 2);

    free_expr(m);
    return 0;
}

/* Affine node: once evaluated this parameter epoch, eval_jacobian skips the
 * impl entirely — proven by poisoning the jacobian values and observing that
 * a re-eval does not rewrite them. expr_set_needs_refresh re-arms the eval,
 * which then restores the true values. */
const char *test_impl_skip_affine(void)
{
    double u[3] = {0.1, 0.2, 0.3};
    expr *x = new_variable(3, 1, 0, 3);
    expr *m = new_neg(x);

    jacobian_init(m);
    m->forward(m, u);
    eval_jacobian(m);

    uint64_t v1 = m->jacobian->values_version;
    int nnz = m->jacobian->nnz;
    mu_assert("neg jacobian must have entries", nnz == 3);
    for (int ii = 0; ii < nnz; ii++)
    {
        mu_assert("neg jacobian value must be -1", m->jacobian->x[ii] == -1.0);
        m->jacobian->x[ii] = 42.0; /* poison */
    }

    eval_jacobian(m);
    mu_assert("re-eval of affine node must not bump",
              m->jacobian->values_version == v1);
    for (int ii = 0; ii < nnz; ii++)
    {
        mu_assert("impl must not have run (poison must survive)",
                  m->jacobian->x[ii] == 42.0);
    }

    expr_set_needs_refresh(m);
    eval_jacobian(m);
    mu_assert("eval after refresh must bump", m->jacobian->values_version == v1 + 1);
    for (int ii = 0; ii < nnz; ii++)
    {
        mu_assert("eval after refresh must restore true values",
                  m->jacobian->x[ii] == -1.0);
    }

    free_expr(m);
    return 0;
}

/* sparse_matrix CSC mirror: refresh_csc_values fills once per write and
 * dedupes repeated calls with no write in between. */
const char *test_values_version_csc_mirror_dedup(void)
{
    double u[3] = {0.1, 0.2, 0.3};
    expr *x = new_variable(3, 1, 0, 3);
    expr *e = new_exp(x);

    jacobian_init(e);
    e->forward(e, u);
    eval_jacobian(e);

    /* jacobian of exp over a leaf variable is a diagonal sparse_matrix, so
       the CSC mirror holds the same values in the same order */
    double expected[3] = {exp(u[0]), exp(u[1]), exp(u[2])};
    sparse_matrix *sm = (sparse_matrix *) e->jacobian;

    e->jacobian->refresh_csc_values(e->jacobian);
    mu_assert("csc_seen must catch up to values_version",
              sm->csc_seen == sm->base.values_version);
    mu_assert("csc values after first refresh",
              cmp_double_array(sm->csc_cache->x, expected, 3));

    /* no write in between: second refresh must be a no-op and leave the
       values correct */
    e->jacobian->refresh_csc_values(e->jacobian);
    mu_assert("csc values after deduped refresh",
              cmp_double_array(sm->csc_cache->x, expected, 3));

    free_expr(e);
    return 0;
}

/* stacked_pd CSR view: to_csr returns a stable pointer, skips the value
 * copy when nothing was written, and refreshes after a write + bump. */
const char *test_values_version_stacked_pd_to_csr(void)
{
    /* exp(A @ X) with X a 3x2 matrix variable: left_matmul_dense has
       n_blocks = 2, so the child jacobian is stacked_pd and the (non-affine)
       exp node's jacobian mirrors that type. */
    double A[6] = {1.0, -0.5, 2.0, 0.5, 1.5, -1.0};
    double u1[6] = {0.1, 0.2, 0.3, -0.1, -0.2, -0.3};
    double u2[6] = {0.4, -0.3, 0.2, 0.1, 0.3, -0.4};

    expr *x = new_variable(3, 2, 0, 6);
    expr *L = new_left_matmul_dense(NULL, x, 2, 3, A);
    expr *e = new_exp(L);

    jacobian_init(e);
    e->forward(e, u1);
    eval_jacobian(e);
    mu_assert("jacobian must be stacked_pd", e->jacobian->is_stacked_pd);

    CSR_matrix *view = e->jacobian->to_csr(e->jacobian);
    double *vals1 = (double *) malloc(view->nnz * sizeof(double));
    memcpy(vals1, view->x, view->nnz * sizeof(double));

    /* no write in between: same view pointer, same values */
    mu_assert("view pointer must be stable",
              e->jacobian->to_csr(e->jacobian) == view);
    mu_assert("values unchanged without a write",
              cmp_double_array(view->x, vals1, view->nnz));

    /* write + bump: the view must reflect the new values */
    e->forward(e, u2);
    eval_jacobian(e);
    mu_assert("view pointer must be stable after re-eval",
              e->jacobian->to_csr(e->jacobian) == view);
    mu_assert("values must change after a write",
              memcmp(view->x, vals1, view->nnz * sizeof(double)) != 0);
    mu_assert("jacobian values fresh after write+bump",
              check_jacobian_num(e, u2, NUMERICAL_DIFF_DEFAULT_H));

    free(vals1);
    free_expr(e);
    return 0;
}

/* Regression test for n-ary parameter-refresh propagation:
 * expr_set_needs_refresh must reach hstack args[] children. A
 * parameter-dependent affine child with a stacked_pd Jacobian would
 * otherwise keep its bump-skip armed across a parameter update, and
 * hstack's to_csr read of the child would serve the previous epoch's
 * values. */
const char *test_values_version_param_under_hstack(void)
{
    int n_vars = 4;
    double u[4] = {1.0, 2.0, 3.0, 4.0};
    expr *X = new_variable(2, 2, 0, n_vars);

    /* A @ X with dense 2x2 A: n_blocks = 2, so the jacobian is stacked_pd */
    double A[4] = {1.0, 2.0, 3.0, 4.0};
    expr *AX = new_left_matmul_dense(NULL, X, 2, 2, A);

    /* p * (A @ X): affine, parameter-dependent */
    double p0 = 2.0;
    expr *p = new_parameter(1, 1, 0, n_vars, &p0);
    expr *pAX = new_scalar_mult(p, AX);

    expr *args[2] = {pAX, X};
    expr *h = new_hstack(args, 2, n_vars);

    jacobian_init(h);
    h->forward(h, u);
    eval_jacobian(h);

    /* the first hstack block holds pAX's jacobian, which is linear in p */
    int nnz1 = pAX->jacobian->nnz;
    double *expected = (double *) malloc(nnz1 * sizeof(double));
    memcpy(expected, h->jacobian->x, nnz1 * sizeof(double));

    /* problem_update_params equivalent: p = 2.0 -> 3.0 */
    double p1 = 3.0;
    memcpy(p->value, &p1, sizeof(double));
    expr_set_needs_refresh(h);

    h->forward(h, u);
    eval_jacobian(h);

    for (int k = 0; k < nnz1; k++)
    {
        expected[k] *= p1 / p0;
    }
    mu_assert("hstack jacobian must pick up the new parameter value",
              cmp_double_array(h->jacobian->x, expected, nnz1));

    free(expected);
    free_expr(h);
    return 0;
}

/* Regression test for stale spd Hessian work matrices: two chained
 * non-affine elementwise atoms over a dense left_matmul with n_blocks > 1
 * make hess_term1/hess_term2 stacked_pd. They are written outside the eval
 * wrappers and read via to_csr in sum_matrices_fill_values, so a missing
 * values_version bump would leave the second Hessian eval reading stale
 * CSR caches. */
const char *test_values_version_spd_hess_terms(void)
{
    double A[6] = {0.8, -0.4, 0.6, 0.3, 0.9, -0.7};
    double u1[6] = {0.1, 0.2, 0.3, -0.1, -0.2, -0.3};
    double u2[6] = {0.3, -0.2, 0.1, 0.2, -0.3, 0.4};
    double w1[4] = {1.0, -2.0, 0.5, 1.5};
    double w2[4] = {-0.5, 1.0, 2.0, -1.5};

    expr *x = new_variable(3, 2, 0, 6);
    expr *L = new_left_matmul_dense(NULL, x, 2, 3, A);
    expr *inner = new_exp(L);
    expr *outer = new_exp(inner);

    /* make sure this test exercises the intended path: both Hessian work
       matrices of the outer node must be stacked_pd */
    jacobian_init(outer);
    wsum_hess_init(outer);
    mu_assert("hess_term1 must be stacked_pd",
              outer->work->hess_term1->is_stacked_pd);
    mu_assert("hess_term2 must be stacked_pd",
              outer->work->hess_term2->is_stacked_pd);

    mu_assert("first hessian eval",
              check_wsum_hess(outer, u1, w1, NUMERICAL_DIFF_DEFAULT_H));
    mu_assert("second hessian eval with different input",
              check_wsum_hess(outer, u2, w2, NUMERICAL_DIFF_DEFAULT_H));

    free_expr(outer);
    return 0;
}
