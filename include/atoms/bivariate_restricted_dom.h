#ifndef BIVARIATE_RESTRICTED_DOM_H
#define BIVARIATE_RESTRICTED_DOM_H

#include "expr.h"

expr *new_quad_over_lin(expr *left, expr *right);
expr *new_rel_entr_vector_args(expr *left, expr *right);
expr *new_rel_entr_first_arg_scalar(expr *left, expr *right);
expr *new_rel_entr_second_arg_scalar(expr *left, expr *right);

#endif /* BIVARIATE_RESTRICTED_DOM_H */
