#ifndef PROBLEM_H
#define PROBLEM_H

#include "expr.h"
#include "utils/CSR_Matrix.h"

typedef struct problem
{
    expr *objective;
    expr **constraints;
    int n_constraints;
    int n_vars;
    int total_constraint_size;

    /* Allocated by problem_allocate */
    double *constraint_values;
    double *gradient_values;
    CSR_Matrix *stacked_jac;
} problem;

/* Retains objective and constraints (shared ownership with caller) */
problem *new_problem(expr *objective, expr **constraints, int n_constraints);
void problem_allocate(problem *prob, const double *u);
void free_problem(problem *prob);

double problem_objective_forward(problem *prob, const double *u);
double *problem_constraint_forward(problem *prob, const double *u);
double *problem_gradient(problem *prob, const double *u);
CSR_Matrix *problem_jacobian(problem *prob, const double *u);

#endif
