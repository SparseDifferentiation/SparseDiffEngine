#ifndef ELEMENTWISE_H
#define ELEMENTWISE_H

#include "expr.h"

expr *new_exp(expr *child);
expr *new_log(expr *child);
expr *new_entr(expr *child);
expr *new_sin(expr *child);
expr *new_cos(expr *child);
expr *new_tan(expr *child);

/* the jacobian for elementwise atoms are always initialized in the
   same way and implement the chain rule in the same way */
void jacobian_init_elementwise(expr *node);
void eval_jacobian_elementwise(expr *node);

/* no elementwise atoms are affine according to our convention,
   so we can have a common implementation */
bool is_affine_elementwise(expr *node);

#endif /* ELEMENTWISE_H */
