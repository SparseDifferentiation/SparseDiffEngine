#ifndef BIVARIATE_H
#define BIVARIATE_H

#include "expr.h"

expr *new_elementwise_mult(expr *left, expr *right);
expr *new_rel_entr_vector_args(expr *left, expr *right);
expr *new_quad_over_lin(expr *left, expr *right);

expr *new_rel_entr_first_arg_scalar(expr *left, expr *right);
expr *new_rel_entr_second_arg_scalar(expr *left, expr *right);

/* Left matrix multiplication: A @ f(x) where A is a constant matrix */
expr *new_left_matmul(expr *u, const CSR_Matrix *A);

#endif /* BIVARIATE_H */
