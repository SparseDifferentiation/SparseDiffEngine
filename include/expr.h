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

#include "utils/CSC_matrix.h"
#include "utils/CSR_matrix.h"
#include "utils/matrix.h"
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
typedef void (*set_needs_refresh_children_fn)(struct expr *node);

/* Workspace for derivative computation */
typedef struct
{
    double *dwork;
    int *iwork;
    CSC_matrix *jacobian_csc;
    int *csc_work; /* for CSR_matrix-CSC_matrix conversion */

    /* jacobian->values_version that the jacobian_csc mirror reflects;
       expr_refresh_jacobian_csc refills iff it differs. */
    uint64_t jacobian_csc_seen;

    /* node->is_affine(node), computed once by jacobian_init (affinity is
       structural, and the recursive is_affine is too costly per eval). */
    bool is_affine_cached;

    /* True once eval_jacobian has run this parameter epoch; cleared by
       expr_set_needs_refresh. Only consulted for affine nodes, where it
       lets the eval_jacobian wrapper skip the values_version bump (same
       role the old jacobian_csc_filled latch played). */
    bool jacobian_evaluated;
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
    forward_fn forward;
    jacobian_init_fn jacobian_init_impl;
    wsum_hess_init_fn wsum_hess_init_impl;
    eval_jacobian_fn eval_jacobian_impl;
    wsum_hess_fn eval_wsum_hess_impl;

    // ------------------------------------------------------------------------
    //                      other things
    // ------------------------------------------------------------------------
    is_affine_fn is_affine;
    local_jacobian_fn local_jacobian;   /* used by elementwise univariate atoms*/
    local_wsum_hess_fn local_wsum_hess; /* used by elementwise univariate atoms*/
    free_type_data_fn free_type_data;   /* Cleanup for type-specific fields */
    /* Recursion hook for expr_set_needs_refresh: atoms holding children
       outside left/right (hstack's args[]) set this so the parameter-refresh
       walk reaches them. NULL for binary/unary atoms. */
    set_needs_refresh_children_fn set_needs_refresh_children;
    Expr_Work *work; /* derivative workspace */
    /* Set to true on all nodes by problem_update_params() via
       expr_set_needs_refresh(). Atoms that cache parameter data
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

/* Guarded init: skips if already initialized (safe for DAGs
 * where a node may be visited through multiple parents). */
void jacobian_init(expr *node);
void wsum_hess_init(expr *node);

/* Eval wrappers: run the atom's eval_*_impl and bump the output matrix's
 * values_version so version-guarded caches (CSC mirrors, spd CSR views)
 * refresh. Always call these instead of the impl slots. */
void eval_jacobian(expr *node);
void eval_wsum_hess(expr *node, const double *w);

/* Refresh work->jacobian_csc from node->jacobian iff its values changed. */
void expr_refresh_jacobian_csc(expr *node);

/* Initialize CSC_matrix form of the Jacobian from the CSR_matrix Jacobian.
 * Must be called after jacobian_init. */
void jacobian_csc_init(expr *node);

/* Recursively set needs_parameter_refresh on node and all children */
void expr_set_needs_refresh(expr *node);

/* Reference counting helpers */
void expr_retain(expr *node);

#endif /* EXPR_H */
