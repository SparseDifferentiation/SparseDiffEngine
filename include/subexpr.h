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
#ifndef SUBEXPR_H
#define SUBEXPR_H

#include "expr.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/matrix.h"

/* Parameter ID for fixed constants (not updatable) */
#define PARAM_FIXED -1

/* Type-specific expression structures that "inherit" from expr */

/* Unified constant/parameter node. Constants use param_id == PARAM_FIXED.
 * Updatable parameters use param_id >= 0 (offset into global theta). */
typedef struct parameter_expr
{
    expr base;
    int param_id;
} parameter_expr;

/* Power: y = x^p */
typedef struct power_expr
{
    expr base;
    double p;
} power_expr;

/* Quadratic form: y = x'*Q*x. Q is a polymorphic matrix: a sparse (CSR) backend
   on the sparse path, or a dense (permuted_dense) backend on the dense path. */
typedef struct quad_over_lin_expr
{
    expr base;
    /* Composition case x = f(u): J = (2 f(u)/y)^T J_f + (-value/y) e_y^T, formed
       as the dispatcher's B^T A with d = 2 f(u)/y (n x 1 permuted_dense) as the
       transposed operand, plus a one-entry row at y's column, summed. */
    matrix *d;     /* n x 1 permuted_dense holding 2 f(u)/y */
    matrix *xpart; /* 1 x n_vars, d^T J_f (permuted_dense from BTA) */
    matrix *ypart; /* 1 x n_vars sparse with one entry at y->var_id */
} quad_over_lin_expr;

typedef struct quad_form_expr
{
    expr base;
    matrix *Q;
    /* Q * J_f for the composition chain-rule hessian, formed through the matrix
       dispatchers on both paths: a permuted_dense when Q is dense, J_f's kind
       when Q is sparse (Q symmetric, so it is Q^T J_f with Q as the transposed
       operand). */
    matrix *QJf;
    /* Q f(x) as an n x 1 permuted_dense: the transposed operand that forms the
       composition Jacobian 2 (Q f(x))^T J_f through the matrix dispatchers. */
    matrix *Qx;
    int n; /* quadratic dimension = left->size */

    /* parametric dense path: param_source feeds Q each solve (NULL otherwise) */
    expr *param_source;
} quad_form_expr;

/* Sum reduction along an axis */
typedef struct sum_expr
{
    expr base;
    int axis;
} sum_expr;

/* trace */
typedef struct trace_expr
{
    expr base;
    matrix *diag_rows; /* the d1 rows of J_X at X's diagonal (row_gather) */
} trace_expr;

/* Product of all entries */
typedef struct prod_expr
{
    expr base;
    int num_of_zeros;
    int zero_index;      /* index of zero element when num_of_zeros == 1 */
    double prod_nonzero; /* product of non-zero elements */
} prod_expr;

/* Product of entries along axis=0 (columnwise products) or axis = 1 (rowwise
 * products) */
typedef struct prod_axis
{
    expr base;
    int *num_of_zeros; /* num of zeros for each column / row depending on the axis*/
    int *zero_index;   /* stores idx of zero element per column / row */
    double *prod_nonzero; /* product of non-zero elements per column / row */
} prod_axis;

/* Horizontal stack (concatenate) */
typedef struct hstack_expr
{
    expr base;
    expr **args;
    int n_args;
    matrix **child_jac;  /* args' Jacobians: the operands of the vstack */
    matrix **child_hess; /* args' Hessians: the operands of the sum */
} hstack_expr;

/* Elementwise multiplication */
typedef struct elementwise_mult_expr
{
    expr base;
    matrix *C;  /* C  = Jg2^T diag(w) Jg1 (Sparse or PD) */
    matrix *CT; /* CT = C^T; same concrete type as C */
} elementwise_mult_expr;

/* Left matrix multiplication: y = A * f(x) where f(x) is an expression. A has
local column indices (f(x)'s entries), not global variable indices. */
typedef struct left_matmul_expr
{
    expr base;
    matrix *A;
    matrix *AT;
    int n_blocks;
    expr *param_source;
    void (*refresh_param_values)(struct left_matmul_expr *);
} left_matmul_expr;

/* Scalar multiplication: y = a * child where a comes from param_source */
typedef struct scalar_mult_expr
{
    expr base;
    expr *param_source;
} scalar_mult_expr;

/* Vector elementwise multiplication: y = a \circ child where a comes from
 * param_source */
typedef struct vector_mult_expr
{
    expr base;
    expr *param_source;
} vector_mult_expr;

/* 1D convolution: y = conv(a, child) where a is a length-m kernel held by
 * param_source. Output has size (m + n - 1) where n is the child length.
 * Forward and wsum_hess backprop are computed as direct loops; for Jacobian
 * we materialize T(a) as a CSR_matrix once at jacobian_init and reuse the engine's
 * block-left-mult machinery for composite children. */
typedef struct convolve_expr
{
    expr base;
    expr *param_source; /* length-m kernel */
    int m;              /* kernel length */
    int n;              /* input length */
    matrix *T;          /* (m+n-1) x n convolution matrix (sparse) */
} convolve_expr;

/* Kronecker product Z = kron(A, B) where one operand is variable-free (held by
 * param_source) and the other (child = node->left) carries the variables. Each
 * output entry gathers a single child entry scaled by an entry of the constant
 * operand; rows not covered by the constant's active blocks are inactive
 * (child_row == -1) and stay structurally zero. */
typedef struct kron_expr
{
    expr base;
    expr *param_source; /* the constant/parameter operand */
    int *child_row;     /* per output row: child entry gathered, -1 if inactive */
    int *coeff_idx;     /* per output row: index into param_source->value */
} kron_expr;

/* Bivariate matrix multiplication: Z = f(u) @ g(u) where both children
 * may be composite expressions. */
typedef struct matmul_expr
{
    expr base;
    /* Jacobian workspace (chain rule): J_h = (Y^T kron I_m) J_f + (I_n kron X) J_g.
       YT and X are full permuted_dense holding the children's current values. */
    matrix *YT;    /* n x k, Y^T */
    matrix *X;     /* m x k */
    matrix *term1; /* (Y^T kron I_m) @ J_f */
    matrix *term2; /* (I_n kron X) @ J_g */

    /* Hessian workspace (chain rule): H = C + C^T + H_f + H_g. */
    matrix *B;   /* cross-Hessian B(w), mk x kn */
    matrix *BJg; /* B @ J_g */
    matrix *C;   /* J_f^T @ B @ J_g */
    matrix *CT;  /* C^T */
} matmul_expr;

/* Index/slicing: y = child[indices] where indices is a list of flat positions */
typedef struct index_expr
{
    expr base;
    int *indices;        /* Flattened indices to select (owned, copied) */
    int n_idxs;          /* Number of selected elements */
    bool has_duplicates; /* True if indices have duplicates (affects Hessian path) */
} index_expr;

/* Broadcast shape used by the broadcast atom. */
typedef enum
{
    BROADCAST_ROW,   /* (1, n) -> (m, n) */
    BROADCAST_COL,   /* (m, 1) -> (m, n) */
    BROADCAST_SCALAR /* (1, 1) -> (m, n) */
} broadcast_type;

typedef struct broadcast_expr
{
    expr base;
    broadcast_type type;
} broadcast_expr;

#endif /* SUBEXPR_H */
