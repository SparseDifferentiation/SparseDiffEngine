#ifndef OLD_AFFINE_H
#define OLD_AFFINE_H

#include "expr.h"
#include "utils/CSR_Matrix.h"

expr *new_linear(expr *u, const CSR_Matrix *A, const double *b);

#endif /* OLD_AFFINE_H */
