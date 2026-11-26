/*
 * Copyright 2026 Daniel Cederberg and William Zhang
 *
 * This file is part of the SparseDiffEngine project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "expr.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include <stdlib.h>
#include <string.h>

/* Affinity is structural and the atoms' rules recurse into their children:
   memoize, or the recursion visits a shared node once per path. */
static bool expr_is_affine(const expr *node)
{
    Expr_Work *wk = node->work;
    if (wk->affine_state == AFFINITY_UNKNOWN)
    {
        wk->affine_state = node->is_affine_impl(node) ? AFFINE : NON_AFFINE;
    }
    return wk->affine_state == AFFINE;
}

/* Thread-local storage in C99 (compiler extensions; C11 has _Thread_local) */
#if defined(_MSC_VER)
#define EXPR_THREAD_LOCAL __declspec(thread)
#else
#define EXPR_THREAD_LOCAL __thread
#endif

/* Identifies the current forward pass; bumped by every top-level forward.
   Thread-local, so threads may run passes on DAGs that share no nodes. */
static EXPR_THREAD_LOCAL uint64_t fwd_epoch = 0;

void expr_forward(expr *node, const double *u)
{
    if (node->work->fwd_epoch == fwd_epoch) return;
    node->work->fwd_epoch = fwd_epoch;
    if (!node->is_affine(node)) node->work->jacobian_evaluated = false;
    node->forward_impl(node, u);
}

/* node->forward of every node: starts a new pass */
static void expr_forward_root(expr *node, const double *u)
{
    fwd_epoch++;
    expr_forward(node, u);
}

void init_expr(expr *node, int d1, int d2, int n_vars, forward_fn forward,
               jacobian_init_fn jacobian_init, eval_jacobian_fn eval_jacobian,
               is_affine_fn is_affine, wsum_hess_init_fn wsum_hess_init,
               wsum_hess_fn eval_wsum_hess, free_type_data_fn free_type_data)
{
    node->d1 = d1;
    node->d2 = d2;
    node->size = d1 * d2;
    node->n_vars = n_vars;
    node->refcount = 0;
    node->value = (double *) sp_calloc(d1 * d2, sizeof(double));
    node->var_id = NOT_A_VARIABLE;
    node->forward = expr_forward_root;
    node->forward_impl = forward;
    node->jacobian_init_impl = jacobian_init;
    node->eval_jacobian_impl = eval_jacobian;
    node->is_affine = expr_is_affine;
    node->is_affine_impl = is_affine;
    node->wsum_hess_init_impl = wsum_hess_init;
    node->eval_wsum_hess_impl = eval_wsum_hess;
    node->free_type_data = free_type_data;
    node->work = (Expr_Work *) sp_calloc(1, sizeof(Expr_Work));
    node->has_params = true; /* assume dirty until the first walk refines it */
}

void free_expr(expr *node)
{
    if (node == NULL) return;

    node->refcount--;
    if (node->refcount > 0) return; /* Still referenced elsewhere */

    /* recursively free children */
    free_expr(node->left);
    free_expr(node->right);
    node->left = NULL;
    node->right = NULL;

    /* free type-specific data */
    if (node->free_type_data)
    {
        node->free_type_data(node);
    }

    /* free value array and derivative matrices */
    sp_free(node->value);
    free_matrix(node->jacobian);
    free_matrix(node->wsum_hess);

    /* free workspace */
    if (node->work)
    {
        sp_free(node->work->dwork);
        sp_free(node->work->iwork);
        sp_free(node->work->local_jac_diag);
        sp_free(node->work->hess_w);
        free_matrix(node->work->hess_term1);
        free_matrix(node->work->hess_term2);
        sp_free(node->work);
    }

    /* free the node itself */
    sp_free(node);
}

void jacobian_init(expr *node)
{
    if (node == NULL || node->jacobian != NULL) return;
    node->jacobian_init_impl(node);
}

void wsum_hess_init(expr *node)
{
    if (node == NULL) return;

    /* one more in-edge; a shared node needs a buffer to sum their weights */
    if (++node->work->hess_n_in == 2)
    {
        node->work->hess_w = (double *) sp_malloc(node->size * sizeof(double));
    }

    if (node->wsum_hess != NULL) return;
    node->wsum_hess_init_impl(node);

    /* eval_wsum_hess never touches an affine node, so its zeros are set here */
    if (node->is_affine(node) && node->wsum_hess->nnz > 0)
    {
        memset(node->wsum_hess->x, 0, node->wsum_hess->nnz * sizeof(double));
    }
}

/* Runs the atom's eval and bumps the Jacobian's values_version, unless the
   values are still valid: the node (and with it the whole subtree walk) is
   then skipped and consumers correctly see "unchanged". A forward pass
   (non-affine nodes) and expr_set_needs_refresh re-arm the eval. */
void eval_jacobian(expr *node)
{
    if (node == NULL || node->work->jacobian_evaluated) return;
    node->eval_jacobian_impl(node);
    matrix_values_changed(node->jacobian);
    node->work->jacobian_evaluated = true;
}

/* Runs the atom's eval and bumps the Hessian's values_version. */
void eval_wsum_hess(expr *node, const double *w)
{
    if (node == NULL || node->is_affine(node)) return;
    Expr_Work *wk = node->work;

    /* A shared node is reached once per in-edge. Sum the weights; all callers
       but the last read zeros, and the last one triggers the evaluation. */
    if (wk->hess_n_in > 1)
    {
        if (wk->hess_visits == 0)
        {
            memcpy(wk->hess_w, w, node->size * sizeof(double));
            memset(node->wsum_hess->x, 0, node->wsum_hess->nnz * sizeof(double));
            matrix_values_changed(node->wsum_hess);
        }
        else
        {
            for (int i = 0; i < node->size; i++)
            {
                wk->hess_w[i] += w[i];
            }
        }
        if (++wk->hess_visits < wk->hess_n_in) return;
        wk->hess_visits = 0;
        w = wk->hess_w;
    }

    node->eval_wsum_hess_impl(node, w);
    matrix_values_changed(node->wsum_hess);
}

bool expr_set_needs_refresh(expr *node)
{
    if (node == NULL) return false;

    /* Known parameter-free: nothing below can have changed, so leave the
       node's jacobian_evaluated latch set and skip the whole subtree. */
    if (!node->has_params) return false;

    node->needs_parameter_refresh = true;

    /* Re-arm the eval_jacobian wrapper's values_version bump: the next eval
       after a parameter update may change even an affine node's values. */
    node->work->jacobian_evaluated = false;

    bool child_has_params = expr_set_needs_refresh(node->left);
    child_has_params |= expr_set_needs_refresh(node->right);
    if (node->set_needs_refresh_children != NULL)
    {
        child_has_params |= node->set_needs_refresh_children(node);
    }

    /* The hook reports nodes the left/right walk cannot see: a parameter
       leaf reports itself, hstack its args[], the coefficient atoms their
       param_source. So this assignment is the whole answer. */
    node->has_params = child_has_params;
    return node->has_params;
}

void expr_retain(expr *node)
{
    if (node == NULL) return;
    node->refcount++;
}
