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

/* Compute dense numerical weighted-sum Hessian via central
 * differences on the gradient g(u) = J(u)^T w.
 * Returns malloc'd row-major array (n_vars x n_vars).
 * Caller must free(). */
double *numerical_wsum_hess(expr *node, const double *u, const double *w, double h);

/* Evaluate analytical wsum_hess, compute numerical wsum_hess,
 * and compare. Returns 1 on match, 0 on mismatch.
 * Prints diagnostic on first failing entry. */
int check_wsum_hess(expr *node, const double *u, const double *w, double h);

#endif /* NUMERICAL_DIFF_H */
