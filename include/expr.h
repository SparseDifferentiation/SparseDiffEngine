/*
 * Copyright 2026 Daniel Cederberg and William Zhang
 *
 * This file is part of the DNLP-differentiation-engine project.
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

#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"
#include <stdbool.h>
#include <stddef.h> /* size_t */
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

/* Workspace for derivative computation */
typedef struct
{
    double *dwork;
    int *iwork;
    CSC_Matrix *jacobian_csc;
    int *csc_work; /* for CSR-CSC conversion */

    /* jacobian_csc_filled is only used for affine functions to avoid redundant
       conversions. Could become relevant for non-affine functions if we start
       supporting common subexpressions on the Python side. */
    bool jacobian_csc_filled;
    double *local_jac_diag; /* cached f'(g(x)) diagonal */
    CSR_Matrix *hess_term1; /* Jg^T D Jg workspace */
    CSR_Matrix *hess_term2; /* child wsum_hess workspace */
} Expr_Work;

/* Base expression node structure */
typedef struct expr
{
    // ------------------------------------------------------------------------
    //                         general quantities
    // ------------------------------------------------------------------------
    int d1, d2, size, n_vars, refcount, var_id;
    size_t memory_bytes;
    bool visited;
    struct expr *left;
    struct expr *right;

    // ------------------------------------------------------------------------
    //                     oracle related quantities
    // ------------------------------------------------------------------------
    double *value;
    CSR_Matrix *jacobian;
    CSR_Matrix *wsum_hess;
    forward_fn forward;
    jacobian_init_fn jacobian_init_impl;
    wsum_hess_init_fn wsum_hess_init_impl;
    eval_jacobian_fn eval_jacobian;
    wsum_hess_fn eval_wsum_hess;

    // ------------------------------------------------------------------------
    //                      other things
    // ------------------------------------------------------------------------
    is_affine_fn is_affine;
    local_jacobian_fn local_jacobian;   /* used by elementwise univariate atoms*/
    local_wsum_hess_fn local_wsum_hess; /* used by elementwise univariate atoms*/
    free_type_data_fn free_type_data;   /* Cleanup for type-specific fields */
    Expr_Work *work;                    /* derivative workspace */

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

/* Initialize CSC form of the Jacobian from the CSR Jacobian.
 * Must be called after jacobian_init. */
void jacobian_csc_init(expr *node);

/* Reference counting helpers */
void expr_retain(expr *node);

#endif /* EXPR_H */
