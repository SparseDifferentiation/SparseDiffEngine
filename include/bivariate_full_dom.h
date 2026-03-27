#ifndef BIVARIATE_FULL_DOM_H
#define BIVARIATE_FULL_DOM_H

#include "expr.h"

expr *new_elementwise_mult(expr *left, expr *right);

/* Matrix multiplication: Z = X @ Y */
expr *new_matmul(expr *x, expr *y);

#endif /* BIVARIATE_FULL_DOM_H */
