#ifndef ELEMENTWISE_FULL_DOM_H
#define ELEMENTWISE_FULL_DOM_H

#include "expr.h"

/* Helper function to initialize an elementwise expr
 * (can be used with derived types) */
void init_elementwise(expr *node, expr *child);

expr *new_exp(expr *child);
expr *new_sin(expr *child);
expr *new_cos(expr *child);
expr *new_sinh(expr *child);
expr *new_tanh(expr *child);
expr *new_asinh(expr *child);
expr *new_logistic(expr *child);
expr *new_power(expr *child, double p);
expr *new_xexp(expr *child);
expr *new_normal_cdf(expr *child);

/* the jacobian and wsum_hess for elementwise full domain
   atoms are always initialized in the same way and
   implement the chain rule in the same way */
void jacobian_init_elementwise(expr *node);
void eval_jacobian_elementwise(expr *node);
void wsum_hess_init_elementwise(expr *node);
void eval_wsum_hess_elementwise(expr *node, const double *w);
expr *new_elementwise(expr *child);

/* no elementwise atoms are affine according to our
   convention, so we can have a common implementation */
bool is_affine_elementwise(const expr *node);

#endif /* ELEMENTWISE_FULL_DOM_H */
