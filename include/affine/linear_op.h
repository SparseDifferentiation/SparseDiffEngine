#ifndef LINEAR_OP_H
#define LINEAR_OP_H

#include "expr.h"
#include "utils/CSR_Matrix.h"

/* linear operator f(u) = Au */
expr *new_linear(expr *u, const CSR_Matrix *A);

#endif /* LINEAR_OP_H */
