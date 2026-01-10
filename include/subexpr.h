#ifndef SUBEXPR_H
#define SUBEXPR_H

#include "expr.h"
#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"

/* Forward declaration */
struct int_double_pair;

/* Type-specific expression structures that "inherit" from expr */

/* Linear operator: y = A * x */
typedef struct linear_op_expr
{
    expr base;
    CSC_Matrix *A_csc;
    CSR_Matrix *A_csr;
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
} quad_form_expr;

/* Sum reduction along an axis */
typedef struct sum_expr
{
    expr base;
    int axis;
    int *idx_map; /* maps child nnz to summed-row positions */
} sum_expr;

/* Trace-like reduction: sums entries spaced by child->d1 */
typedef struct trace_expr
{
    expr base;
    struct int_double_pair *int_double_pairs; /* for sorting jacobian entries */
} trace_expr;

/* Product of all entries */
typedef struct prod_expr
{
    expr base;
    int num_of_zeros;
    int zero_index;      /* index of zero element when num_of_zeros == 1 */
    double prod_nonzero; /* product of non-zero elements */
} prod_expr;

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
    CSR_Matrix *CSR_work1;
    CSR_Matrix *CSR_work2;
} elementwise_mult_expr;

/* Left matrix multiplication: y = A * f(x) where f(x) is an expression. Note that
here A does not have global column indices but it is a local matrix. This is an
important distinction compared to linear_op_expr. */
typedef struct left_matmul_expr
{
    expr base;
    CSR_Matrix *A;
    CSR_Matrix *AT;
    CSC_Matrix *CSC_work;
} left_matmul_expr;

#endif /* SUBEXPR_H */
