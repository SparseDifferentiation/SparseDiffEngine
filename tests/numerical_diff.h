#ifndef NUMERICAL_DIFF_H
#define NUMERICAL_DIFF_H

#include "expr.h"

#define NUMERICAL_DIFF_DEFAULT_H 1e-7

/* Compute dense numerical Jacobian via central differences.
 * Returns malloc'd row-major array (node->size x node->n_vars).
 * Caller must free(). */
double *numerical_jacobian(expr *node, const double *u, double h);

/* Evaluate analytical Jacobian, compute numerical Jacobian,
 * and compare. Returns 1 on match, 0 on mismatch.
 * Prints diagnostic on first failing entry. */
int check_jacobian(expr *node, const double *u, double h);

#endif /* NUMERICAL_DIFF_H */
