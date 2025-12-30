#ifndef AFFINE_H
#define AFFINE_H

#include "expr.h"
#include "utils/CSR_Matrix.h"

expr *new_linear(expr *u, const CSR_Matrix *A);

expr *new_add(expr *left, expr *right);
expr *new_sum(expr *child, int axis);

expr *new_constant(int d1, int d2, const double *values);
expr *new_variable(int d1, int d2, int var_id, int n_vars);

#endif /* AFFINE_H */
