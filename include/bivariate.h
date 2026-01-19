#ifndef BIVARIATE_H
#define BIVARIATE_H

#include "expr.h"

expr *new_elementwise_mult(expr *left, expr *right);
expr *new_rel_entr_vector_args(expr *left, expr *right);
expr *new_quad_over_lin(expr *left, expr *right);

expr *new_rel_entr_first_arg_scalar(expr *left, expr *right);
expr *new_rel_entr_second_arg_scalar(expr *left, expr *right);

/* Matrix multiplication: Z = X @ Y */
expr *new_matmul(expr *x, expr *y);

/* Left matrix multiplication: A @ f(x) where A is a constant matrix */
expr *new_left_matmul(expr *u, const CSR_Matrix *A);

/* Right matrix multiplication: f(x) @ A where A is a constant matrix */
expr *new_right_matmul(expr *u, const CSR_Matrix *A);

/* Constant scalar multiplication: a * f(x) where a is a constant double */
expr *new_const_scalar_mult(double a, expr *child);

/* Constant vector elementwise multiplication: a ∘ f(x) where a is constant */
expr *new_const_vector_mult(const double *a, expr *child);

#endif /* BIVARIATE_H */
