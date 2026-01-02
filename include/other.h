#ifndef OTHER_H
#define OTHER_H

#include "expr.h"
#include "subexpr.h"
#include "utils/CSR_Matrix.h"

/* Helper function to initialize a quad_form expr (can be used with derived types) */
void init_quad_form(expr *node, expr *child);

expr *new_quad_form(expr *child, CSR_Matrix *Q);

#endif /* OTHER_H */
