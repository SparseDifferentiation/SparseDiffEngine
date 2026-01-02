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
    int p;
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
    struct int_double_pair *int_double_pairs; /* for sorting jacobian entries */
} sum_expr;

/* Horizontal stack (concatenate) */
typedef struct hstack_expr
{
    expr base;
    expr **args;
    int n_args;
} hstack_expr;

#endif /* SUBEXPR_H */
