#ifndef PROBLEM_H
#define PROBLEM_H

#include "expr.h"
#include "utils/CSR_Matrix.h"
#include "utils/Timer.h"

typedef struct
{
    double time_init_derivatives;
    double time_eval_jacobian;
    double time_eval_gradient;
    double time_eval_hessian;
    double time_forward_obj;
    double time_forward_constraints;
} stats;

typedef struct problem
{
    expr *objective;
    expr **constraints;
    int n_constraints;
    int n_vars;
    int total_constraint_size;

    /* Allocated by new_problem */
    double *constraint_values;
    double *gradient_values;

    /* Allocated by problem_init_derivatives */
    CSR_Matrix *jacobian;
    CSR_Matrix *lagrange_hessian;
    int *hess_idx_map; /* Maps all wsum_hess nnz to lagrange_hessian (obj +
                          constraints) */

    /* Statistics for performance measurement */
    stats stats;
} problem;

/* Retains objective and constraints (shared ownership with caller) */
problem *new_problem(expr *objective, expr **constraints, int n_constraints);
void problem_init_derivatives(problem *prob);
void free_problem(problem *prob);

double problem_objective_forward(problem *prob, const double *u);
void problem_constraint_forward(problem *prob, const double *u);
void problem_gradient(problem *prob);
void problem_jacobian(problem *prob);
void problem_hessian(problem *prob, double obj_w, const double *w);

#endif
