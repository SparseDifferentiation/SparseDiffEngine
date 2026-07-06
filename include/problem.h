/*
 * Copyright 2026 Daniel Cederberg and William Zhang
 *
 * This file is part of the SparseDiffEngine project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef PROBLEM_H
#define PROBLEM_H

#include "expr.h"
#include "utils/COO_matrix.h"
#include "utils/CSR_matrix.h"
#include "utils/Timer.h"
#include <stdbool.h>

typedef struct
{
    double time_init_derivatives;
    double time_eval_jacobian;
    double time_eval_gradient;
    double time_eval_hessian;
    double time_forward_obj;
    double time_forward_constraints;

    int nnz_affine;
    int nnz_nonlinear; /* jacobian of nonlinear constraints */
    int nnz_hessian;
    int n_vars;
    int total_constraint_size;
    size_t memory_bytes;
} Diff_engine_stats;

typedef struct problem
{
    expr *objective;
    expr **constraints;
    int n_constraints;
    int n_vars;
    int total_constraint_size;

    /* parameter support */
    expr **param_nodes;
    int n_param_nodes;
    int total_parameter_size;

    /* allocated by new_problem */
    double *constraint_values;
    double *gradient_values;

    /* allocated by problem_init_derivatives */
    CSR_matrix *jacobian;
    CSR_matrix *lagrange_hessian;
    int *hess_idx_map; /* maps all wsum_hess nnz to lagrange_hessian */
    COO_matrix *jacobian_coo;
    COO_matrix *lagrange_hessian_coo; /* lower triangular part stored in COO */

    /* for the affine shortcut we keep track of the first time the jacobian and
     * hessian are called */
    bool jacobian_called;

    /* statistics for performance measurement */
    Diff_engine_stats stats;
    bool verbose;
} problem;

/* Retains objective and constraints (shared ownership with caller) */
problem *new_problem(expr *objective, expr **constraints, int n_constraints,
                     bool verbose);
void problem_init_jacobian(problem *prob);
void problem_init_hessian(problem *prob);
void problem_init_derivatives(problem *prob);
void problem_init_jacobian_coo(problem *prob);
/* Like problem_init_jacobian_coo, but adopts a caller-provided COO pattern
 * (as previously returned for a structurally identical problem) instead of
 * expanding it from the CSR. The expression-level Jacobian structures are
 * still initialized; only the COO view construction is skipped. Returns 0 on
 * success, -1 if nnz does not match this problem's Jacobian (the pattern is
 * then ignored and no COO view is created). */
int problem_init_jacobian_coo_from(problem *prob, const int *rows,
                                   const int *cols, int nnz);
void problem_init_hessian_coo_lower_triangular(problem *prob);
/* Hessian analog of problem_init_jacobian_coo_from: adopts a caller-provided
 * lower-triangular COO pattern. The value map is still derived from the CSR
 * (evaluation needs it); same 0 / -1 contract. */
int problem_init_hessian_coo_lower_triangular_from(problem *prob,
                                                   const int *rows,
                                                   const int *cols, int nnz);
void free_problem(problem *prob);

void problem_register_params(problem *prob, expr **param_nodes, int n_param_nodes);
void problem_update_params(problem *prob, const double *theta);

double problem_objective_forward(problem *prob, const double *u);
void problem_constraint_forward(problem *prob, const double *u);
void problem_gradient(problem *prob);
void problem_jacobian(problem *prob);
void problem_hessian(problem *prob, double obj_w, const double *w);

#endif
