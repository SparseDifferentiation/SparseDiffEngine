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
#ifndef SUBEXPR_H
#define SUBEXPR_H

#include "expr.h"
#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"
#include "utils/matrix.h"

/* Forward declaration */
struct int_double_pair;

/* Parameter ID for fixed constants (not updatable) */
#define PARAM_FIXED -1

/* Type-specific expression structures that "inherit" from expr */

/* Unified constant/parameter node. Constants use param_id == PARAM_FIXED.
 * Updatable parameters use param_id >= 0 (offset into global theta). */
typedef struct parameter_expr
{
    expr base;
    int param_id;
    bool has_been_refreshed;
} parameter_expr;

/* Linear operator: y = A * x + b
 * The matrix A is stored as node->jacobian (CSR). */
typedef struct linear_op_expr
{
    expr base;
    double *b; /* constant offset vector (NULL if no offset) */
} linear_op_expr;

/* Power: y = x^p */
typedef struct power_expr
{
    expr base;
    double p;
} power_expr;

/* Quadratic form: y = x'*Q*x */
typedef struct quad_form_expr
{
    expr base;
    CSR_Matrix *Q;
    CSC_Matrix *QJf; /* Q * J_f in CSC (for chain rule hessian) */
} quad_form_expr;

/* Sum reduction along an axis */
typedef struct sum_expr
{
    expr base;
    int axis;
    int *idx_map; /* maps child nnz to summed-row positions */
} sum_expr;

/* trace */
typedef struct trace_expr
{
    expr base;
    int *idx_map; /* maps child nnz to summed-row positions */
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
    CSR_Matrix *CSR_work; /* for summing Hessians of children */
} hstack_expr;

/* Elementwise multiplication */
typedef struct elementwise_mult_expr
{
    expr base;
    CSR_Matrix *CSR_work1; /* C  = Jg2^T diag(w) Jg1 */
    CSR_Matrix *CSR_work2; /* CT = C^T */
    int *idx_map_C;        /* C[j]  -> wsum_hess pos */
    int *idx_map_CT;       /* CT[j] -> wsum_hess pos */
    int *idx_map_Hx;       /* x->wsum_hess[j] -> pos */
    int *idx_map_Hy;       /* y->wsum_hess[j] -> pos */
} elementwise_mult_expr;

/* Left matrix multiplication: y = A * f(x) where f(x) is an expression. Note that
here A does not have global column indices but it is a local matrix. This is an
important distinction compared to linear_op_expr. */
typedef struct left_matmul_expr
{
    expr base;
    Matrix *A;
    Matrix *AT;
    int n_blocks;
    CSC_Matrix *Jchild_CSC;
    CSC_Matrix *J_CSC;
    int *csc_to_csr_work;
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

/* Bivariate matrix multiplication: Z = f(u) @ g(u) where both children
 * may be composite expressions. */
typedef struct matmul_expr
{
    expr base;
    /* Jacobian workspace */
    CSR_Matrix *term1_CSR; /* (Y^T x I_m) @ J_f */
    CSR_Matrix *term2_CSR; /* (I_n x X) @ J_g */

    /* Hessian workspace (composite only) */
    CSR_Matrix *B;       /* cross-Hessian B(w), mk x kn */
    CSR_Matrix *BJg;     /* B @ J_g */
    CSC_Matrix *BJg_CSC; /* BJg in CSC */
    int *BJg_csc_work;   /* CSR-to-CSC workspace */
    CSR_Matrix *C;       /* J_f^T @ B @ J_g */
    CSR_Matrix *CT;      /* C^T */
    int *idx_map_C;
    int *idx_map_CT;
    int *idx_map_Hf;
    int *idx_map_Hg;
} matmul_expr;


/* Index/slicing: y = child[indices] where indices is a list of flat positions */
typedef struct index_expr
{
    expr base;
    int *indices;        /* Flattened indices to select (owned, copied) */
    int n_idxs;          /* Number of selected elements */
    bool has_duplicates; /* True if indices have duplicates (affects Hessian path) */
} index_expr;

/* Broadcast types */
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
