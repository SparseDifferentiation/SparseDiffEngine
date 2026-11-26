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
#ifndef EXPR_H
#define EXPR_H

#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/matrix.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define JAC_IDXS_NOT_SET -1
#define NOT_A_VARIABLE -1

/* Function pointer types */
struct expr;
typedef void (*forward_fn)(struct expr *node, const double *u);
typedef void (*jacobian_init_fn)(struct expr *node);
typedef void (*wsum_hess_init_fn)(struct expr *node);
typedef void (*eval_jacobian_fn)(struct expr *node);
typedef void (*wsum_hess_fn)(struct expr *node, const double *w);
typedef void (*local_jacobian_fn)(struct expr *node, double *out);
typedef void (*local_wsum_hess_fn)(struct expr *node, double *out, const double *w);
typedef bool (*is_affine_fn)(const struct expr *node);
typedef void (*free_type_data_fn)(struct expr *node);
typedef bool (*set_needs_refresh_children_fn)(struct expr *node);

/* AFFINITY_UNKNOWN must be 0 (the workspace is calloc'ed) */
typedef enum
{
    AFFINITY_UNKNOWN = 0,
    AFFINE,
    NON_AFFINE
} affinity;

/* Workspace for derivative computation */
typedef struct
{
    double *dwork;
    int *iwork;

    affinity affine_state; /* memoized node->is_affine(node) */

    /* The Jacobian values are valid. Cleared by expr_set_needs_refresh and,
       for a non-affine node, by a new forward pass. */
    bool jacobian_evaluated;

    /* forward pass that last evaluated this node (see expr_forward) */
    uint64_t fwd_epoch;

    int hess_n_in;          /* number of wsum_hess_init calls (in-edges) */
    int hess_visits;        /* eval_wsum_hess calls so far in this pass */
    double *hess_w;         /* weights summed over them (iff hess_n_in > 1) */
    double *local_jac_diag; /* cached f'(g(x)) diagonal */
    matrix *hess_term1;     /* Jg^T D Jg workspace */
    matrix *hess_term2;     /* child wsum_hess workspace */
} Expr_Work;

/* Base expression node structure */
typedef struct expr
{
    // ------------------------------------------------------------------------
    //                         general quantities
    // ------------------------------------------------------------------------
    int d1, d2, size, n_vars, refcount, var_id;
    struct expr *left;
    struct expr *right;

    // ------------------------------------------------------------------------
    //                     oracle related quantities
    // ------------------------------------------------------------------------
    double *value;
    matrix *jacobian;
    matrix *wsum_hess;
    forward_fn forward;      /* starts a forward pass */
    forward_fn forward_impl; /* the atom's rule; recurses with expr_forward */
    jacobian_init_fn jacobian_init_impl;
    wsum_hess_init_fn wsum_hess_init_impl;
    eval_jacobian_fn eval_jacobian_impl;
    wsum_hess_fn eval_wsum_hess_impl;

    // ------------------------------------------------------------------------
    //                      other things
    // ------------------------------------------------------------------------
    is_affine_fn is_affine;             /* memoizing wrapper */
    is_affine_fn is_affine_impl;        /* the atom's (recursive) rule */
    local_jacobian_fn local_jacobian;   /* used by elementwise univariate atoms*/
    local_wsum_hess_fn local_wsum_hess; /* used by elementwise univariate atoms*/
    free_type_data_fn free_type_data;   /* Cleanup for type-specific fields */
    /* Recursion hook for expr_set_needs_refresh, which on its own only
       descends left/right. Nodes that hold parameter-bearing subtrees
       elsewhere install this so the walk reaches them: the parameter leaf
       (reports itself), hstack (args[]) and the coefficient atoms
       scalar_mult, vector_mult, kron, convolve, left_matmul and quad_form
       (param_source). NULL everywhere else.

       Contract: the hook MUST return whether the nodes it reaches contain
       an updatable parameter (param_id >= 0). The walk ORs the result into
       has_params; a hook that returns false lets the subtree be memoized
       parameter-free and pruned from the second update on. */
    set_needs_refresh_children_fn set_needs_refresh_children;
    Expr_Work *work; /* derivative workspace */
    /* Does this subtree contain an updatable parameter? Starts true so
       nothing is pruned before the first walk; expr_set_needs_refresh
       refines and memoizes it from the left/right walk and the
       set_needs_refresh_children hook. */
    bool has_params;
    /* Set to true by problem_update_params() via expr_set_needs_refresh()
       on every node whose subtree contains an updatable parameter;
       parameter-free subtrees are skipped. Atoms that cache parameter data
       (e.g. left_matmul_dense) check this flag before their forward
       pass: if true, they refresh their cached matrices from
       param_source->value and clear the flag to false. */
    bool needs_parameter_refresh;

    // name of node just for debugging - should be removed later
    char name[32];

} expr;

void init_expr(expr *node, int d1, int d2, int n_vars, forward_fn forward,
               jacobian_init_fn jacobian_init, eval_jacobian_fn eval_jacobian,
               is_affine_fn is_affine, wsum_hess_init_fn wsum_hess_init,
               wsum_hess_fn eval_wsum_hess, free_type_data_fn free_type_data);

void free_expr(expr *node);

/* Forward pass on a DAG. node->forward(node, u) starts a pass and evaluates
 * the node. Atoms recurse into their children with expr_forward, which
 * evaluates a node at most once per pass. A new value invalidates a non-affine
 * node's Jacobian. */
void expr_forward(expr *node, const double *u);

/* Guarded init: the impl runs once per node, however many parents it has.
 *
 * Every wsum_hess_init(node) call counts as one in-edge of the node and must
 * be matched by exactly one eval_wsum_hess(node, w) call per Hessian pass
 * (an atom does this for each child; problem does it for its roots). */
void jacobian_init(expr *node);
void wsum_hess_init(expr *node);

/* Eval wrappers: run the atom's eval_*_impl and bump the output matrix's
 * values_version so version-guarded caches (CSC mirrors, spd CSR views)
 * refresh. Always call these instead of the impl slots.
 *
 * eval_jacobian skips a node whose Jacobian is still valid.
 *
 * eval_wsum_hess evaluates a node with several in-edges once per pass, at the
 * sum of their weights. An atom must therefore only add a child's Hessian to
 * its own, never rescale it (a scaling goes into the child's weights). */
void eval_jacobian(expr *node);
void eval_wsum_hess(expr *node, const double *w);

/* Mark the subtree dirty and report whether it contains an updatable
 * parameter. A subtree known to be parameter-free is skipped entirely: its
 * values and derivatives cannot have changed, so re-arming it would only
 * force a recompute of a result we already hold. */
bool expr_set_needs_refresh(expr *node);

/* Reference counting helpers */
void expr_retain(expr *node);

#endif /* EXPR_H */
