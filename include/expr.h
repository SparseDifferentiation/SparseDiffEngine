#ifndef EXPR_H
#define EXPR_H

#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"
#include <stdbool.h>
#include <stddef.h>

#define JAC_IDXS_NOT_SET -1

/* Function pointer types */
struct expr;
typedef void (*forward_fn)(struct expr *node, const double *u);
typedef void (*jacobian_init_fn)(struct expr *node);
typedef void (*wsum_hess_init_fn)(struct expr *node);
typedef void (*eval_jacobian_fn)(struct expr *node);
typedef void (*wsum_hess_fn)(struct expr *node, double *w);
typedef void (*local_jacobian_fn)(struct expr *node, double *out);
typedef void (*local_wsum_hess_fn)(struct expr *node, double *out, double *w);
typedef bool (*is_affine_fn)(struct expr *node);
typedef void (*free_type_data_fn)(struct expr *node);

/* Base expression node structure - contains only common fields */
typedef struct expr
{
    // ------------------------------------------------------------------------
    //                         general quantities
    // ------------------------------------------------------------------------
    int d1, d2, size, n_vars, refcount, var_id;
    struct expr *left;
    struct expr *right;
    double *dwork;
    int *iwork;

    // ------------------------------------------------------------------------
    //                     oracle related quantities
    // ------------------------------------------------------------------------
    double *value;
    CSR_Matrix *jacobian;
    CSR_Matrix *wsum_hess;
    forward_fn forward;
    jacobian_init_fn jacobian_init;
    wsum_hess_init_fn wsum_hess_init;
    eval_jacobian_fn eval_jacobian;
    wsum_hess_fn eval_wsum_hess;

    // ------------------------------------------------------------------------
    //                      other things
    // ------------------------------------------------------------------------
    is_affine_fn is_affine;
    local_jacobian_fn local_jacobian;   /* used by elementwise univariate atoms*/
    local_wsum_hess_fn local_wsum_hess; /* used by elementwise univariate atoms*/
    free_type_data_fn free_type_data;   /* Cleanup for type-specific fields */

} expr;

expr *new_expr(int d1, int d2, int n_vars);
void free_expr(expr *node);

/* Reference counting helpers */
void expr_retain(expr *node);

#endif /* EXPR_H */
