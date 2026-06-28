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
#include "utils/CSC_matrix.h"
#include "utils/CSR_matrix.h"
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
} parameter_expr;

/* Linear operator: y = A * x + b
 * The matrix A is stored as node->jacobian (CSR_matrix). */
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

/* Quadratic form: y = x'*Q*x. Q is a polymorphic matrix: a sparse (CSR) backend
   on the sparse path, or a dense (permuted_dense) backend on the dense path. */
typedef struct quad_form_expr
{
    expr base;
    matrix *Q;
    /* Q * J_f for the composition chain-rule hessian; exactly one is used per
       node. Sparse path: CSC (raw symmetric products, no matrix-vtable form).
       Dense path: permuted_dense via the matrix dispatchers. */
    CSC_matrix *QJf;
    matrix *QJf_dense;
    double *diag_w; /* length-n diagonal (= 2w) fed to BTDA on the dense path */
    int n;          /* quadratic dimension = left->size */

    /* parametric dense path: param_source feeds Q each solve (NULL otherwise) */
    expr *param_source;
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
    CSR_matrix *CSR_work; /* for summing Hessians of children */
} hstack_expr;

/* Elementwise multiplication */
typedef struct elementwise_mult_expr
{
    expr base;
    matrix *C;       /* C  = Jg2^T diag(w) Jg1 (Sparse or PD) */
    matrix *CT;      /* CT = C^T; same concrete type as C */
    int *idx_map_C;  /* C[j]  -> wsum_hess pos */
    int *idx_map_CT; /* CT[j] -> wsum_hess pos */
    int *idx_map_Hx; /* x->wsum_hess[j] -> pos */
    int *idx_map_Hy; /* y->wsum_hess[j] -> pos */
} elementwise_mult_expr;

/* Left matrix multiplication: y = A * f(x) where f(x) is an expression. Note that
here A does not have global column indices but it is a local matrix. This is an
important distinction compared to linear_op_expr. */
typedef struct left_matmul_expr
{
    expr base;
    matrix *A;
    matrix *AT;
    int n_blocks;
    CSC_matrix *Jchild_CSC;
    CSC_matrix *J_CSC;
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
    CSR_matrix *T;      /* (m+n-1) x n convolution matrix */
    CSC_matrix *Jchild_CSC;
} convolve_expr;

/* Kronecker product Z = kron(A, B) where exactly one operand is variable-free
 * (held by param_source) and the other (child = node->left) carries the
 * variables. Every output entry depends on a single child entry, so the output
 * Jacobian is the child Jacobian's rows gathered (with repetition) and scaled --
 * no coefficient matrix or matmul. child_row[OUT] and coeff_idx[OUT] depend only
 * on the operand shapes and are precomputed once at construction. */
typedef struct kron_expr
{
    expr base;
    expr *param_source; /* the constant/parameter operand */
    int p, q, r, s;     /* A is p x q, B is r x s */
    int const_is_left;  /* 1: A=param, B=child; 0: A=child, B=param */
    int *child_row;     /* size_out: child entry each output row gathers */
    int *coeff_idx;     /* size_out: index into param_source->value (the scale) */
} kron_expr;

/* Bivariate matrix multiplication: Z = f(u) @ g(u) where both children
 * may be composite expressions. */
typedef struct matmul_expr
{
    expr base;
    /* Jacobian workspace */
    CSR_matrix *term1_CSR; /* (Y^T x I_m) @ J_f */
    CSR_matrix *term2_CSR; /* (I_n x X) @ J_g */

    /* Hessian workspace (composite only) */
    CSR_matrix *B;       /* cross-Hessian B(w), mk x kn */
    CSR_matrix *BJg;     /* B @ J_g */
    CSC_matrix *BJg_CSC; /* BJg in CSC_matrix */
    int *BJg_csc_work;   /* CSR_matrix-to-CSC_matrix workspace */
    CSR_matrix *C;       /* J_f^T @ B @ J_g */
    CSR_matrix *CT;      /* C^T */
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

typedef struct broadcast_expr
{
    expr base;
    broadcast_type type;
} broadcast_expr;

#endif /* SUBEXPR_H */
