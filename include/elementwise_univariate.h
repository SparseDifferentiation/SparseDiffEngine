#ifndef ELEMENTWISE_H
#define ELEMENTWISE_H

#include "expr.h"

expr *new_exp(expr *child);
expr *new_log(expr *child);
expr *new_entr(expr *child);
expr *new_sin(expr *child);
expr *new_cos(expr *child);
expr *new_tan(expr *child);
expr *new_sinh(expr *child);
expr *new_tanh(expr *child);
expr *new_asinh(expr *child);
expr *new_atanh(expr *child);
expr *new_logistic(expr *child);
expr *new_power(expr *child, int p);
expr *new_xexp(expr *child);

/* the jacobian and wsum_hess for elementwise univariate atoms are always
   initialized in the same way and implement the chain rule in the same way */
void jacobian_init_elementwise(expr *node);
void eval_jacobian_elementwise(expr *node);
void wsum_hess_init_elementwise(expr *node);
void eval_wsum_hess_elementwise(expr *node, double *w);
expr *new_elementwise(expr *child);

/* no elementwise atoms are affine according to our convention,
   so we can have a common implementation */
bool is_affine_elementwise(expr *node);

#endif /* ELEMENTWISE_UNIVARIATE_H */
