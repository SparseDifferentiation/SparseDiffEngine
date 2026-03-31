#ifndef ELEMENTWISE_RESTRICTED_DOM_H
#define ELEMENTWISE_RESTRICTED_DOM_H

#include "expr.h"

/* Shared init functions for restricted domain atoms
 * (variable-child only, no linear operator support) */
void jacobian_init_restricted(expr *node);
void wsum_hess_init_restricted(expr *node);
bool is_affine_restricted(const expr *node);
expr *new_restricted(expr *child);

expr *new_log(expr *child);
expr *new_entr(expr *child);
expr *new_atanh(expr *child);
expr *new_tan(expr *child);

#endif /* ELEMENTWISE_RESTRICTED_DOM_H */
