#ifndef OTHER_H
#define OTHER_H

#include "expr.h"
#include "subexpr.h"
#include "utils/CSR_Matrix.h"

expr *new_quad_form(expr *child, CSR_Matrix *Q);

/* product of all entries, without axis argument */
expr *new_prod(expr *child);

#endif /* OTHER_H */
