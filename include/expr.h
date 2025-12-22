#ifndef EXPR_H
#define EXPR_H

#include "utils/CSR_Matrix.h"
#include <stdbool.h>
#include <stddef.h>

#define JAC_IDXS_NOT_SET -1

/* Forward declarations */
struct expr;

/* Function pointer types */
typedef void (*forward_fn)(struct expr *node, const double *u);
typedef void (*jacobian_init_fn)(struct expr *node);
typedef void (*eval_jacobian_fn)(struct expr *node);
typedef void (*eval_local_jacobian_fn)(struct expr *node, double *out);
typedef bool (*is_affine_fn)(struct expr *node);

/* Expression node structure */
typedef struct expr
{
    // ------------------------------------------------------------------------
    //                         general quantities
    // ------------------------------------------------------------------------
    int m;
    int n_vars;
    int var_id;
    struct expr *left;
    struct expr *right;
    double *dwork;

    // ------------------------------------------------------------------------
    //                     forward pass related quantities
    // ------------------------------------------------------------------------
    double *value;
    forward_fn forward;

    // ------------------------------------------------------------------------
    //                      jacobian related quantities
    // ------------------------------------------------------------------------
    CSR_Matrix *jacobian;
    jacobian_init_fn jacobian_init;
    eval_jacobian_fn eval_jacobian;
    eval_local_jacobian_fn eval_local_jacobian;
    is_affine_fn is_affine;

} expr;

expr *new_expr(int m, int n_vars);
void free_expr(expr *node);

#endif /* EXPR_H */
