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
#include "problem.h"
#include "subexpr.h"
#include "utils/CSR_sum.h"
#include "utils/stacked_pd.h"
#include "utils/tracked_alloc.h"
#include "utils/utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* forward declaration */
static void problem_lagrange_hess_fill_sparsity(problem *prob, int *iwork);

problem *new_problem(expr *objective, expr **constraints, int n_constraints,
                     bool verbose)
{
    /* we don't reset g_peak_bytes or g_allocated_bytes since allocations
       using sp_malloc/sp_calloc might have happened before new_problem in eg.,
       left_matmul, and their frees will subtract from this counter. */
    g_peak_bytes = g_allocated_bytes;
    problem *prob = (problem *) sp_calloc(1, sizeof(problem));
    if (!prob) return NULL;

    /* objective */
    prob->objective = objective;
    expr_retain(objective);
    prob->n_vars = objective->n_vars;
    prob->jacobian_called = false;

    /* constraints array */
    prob->total_constraint_size = 0;
    prob->n_constraints = n_constraints;
    if (n_constraints > 0)
    {
        prob->constraints = (expr **) sp_malloc(n_constraints * sizeof(expr *));
        for (int i = 0; i < n_constraints; i++)
        {
            prob->constraints[i] = constraints[i];
            prob->total_constraint_size += constraints[i]->size;
            expr_retain(constraints[i]);
        }
    }

    /* allocation */
    prob->constraint_values =
        (double *) sp_calloc(prob->total_constraint_size, sizeof(double));
    prob->gradient_values = (double *) sp_calloc(prob->n_vars, sizeof(double));

    /* Initialize statistics */
    prob->stats.time_init_derivatives = 0.0;
    prob->stats.time_eval_jacobian = 0.0;
    prob->stats.time_eval_gradient = 0.0;
    prob->stats.time_eval_hessian = 0.0;
    prob->stats.time_forward_obj = 0.0;
    prob->stats.time_forward_constraints = 0.0;
    prob->stats.nnz_affine = 0;
    prob->stats.nnz_nonlinear = 0;
    prob->stats.nnz_hessian = 0;
    prob->stats.n_vars = prob->n_vars;
    prob->stats.total_constraint_size = prob->total_constraint_size;

    prob->verbose = verbose;

    return prob;
}

static void problem_lagrange_hess_fill_sparsity(problem *prob, int *iwork)
{
    expr **constrs = prob->constraints;
    int *cols = iwork;
    int *col_to_pos = iwork; /* reused after qsort */
    int nnz = 0;
    CSR_matrix *H_obj =
        prob->objective->wsum_hess->to_csr(prob->objective->wsum_hess);
    CSR_matrix *H_c;
    CSR_matrix *H = prob->lagrange_hessian;
    H->p[0] = 0;

    // ----------------------------------------------------------------------
    //                      Fill sparsity pattern
    // ----------------------------------------------------------------------
    for (int row = 0; row < H->m; row++)
    {
        /* gather columns from objective hessian */
        int count = H_obj->p[row + 1] - H_obj->p[row];
        memcpy(cols, H_obj->i + H_obj->p[row], count * sizeof(int));

        /* gather columns from constraint hessians */
        for (int c_idx = 0; c_idx < prob->n_constraints; c_idx++)
        {
            H_c = constrs[c_idx]->wsum_hess->to_csr(constrs[c_idx]->wsum_hess);
            int c_len = H_c->p[row + 1] - H_c->p[row];
            memcpy(cols + count, H_c->i + H_c->p[row], c_len * sizeof(int));
            count += c_len;
        }

        /* find unique columns */
        sort_int_array(cols, count);
        int prev_col = -1;
        for (int j = 0; j < count; j++)
        {
            if (cols[j] != prev_col)
            {
                H->i[nnz] = cols[j];
                nnz++;
                prev_col = cols[j];
            }
        }

        H->p[row + 1] = nnz;
    }

    H->nnz = nnz;

    // ----------------------------------------------------------------------
    //                           Build idx map
    // ----------------------------------------------------------------------
    int idx_offset = 0;

    /* map objective hessian entries */
    int obj_start = idx_offset;
    for (int row = 0; row < H->m; row++)
    {
        for (int idx = H->p[row]; idx < H->p[row + 1]; idx++)
        {
            col_to_pos[H->i[idx]] = idx;
        }

        for (int j = H_obj->p[row]; j < H_obj->p[row + 1]; j++)
        {
            prob->hess_idx_map[idx_offset++] = col_to_pos[H_obj->i[j]];
        }
    }
    if (prob->objective->wsum_hess->is_stacked_pd)
    {
        compose_csr_idx_map_for_spd((const stacked_pd *) prob->objective->wsum_hess,
                                    H_obj, prob->hess_idx_map + obj_start);
    }

    /* map constraint hessian entries */
    for (int c_idx = 0; c_idx < prob->n_constraints; c_idx++)
    {
        H_c = constrs[c_idx]->wsum_hess->to_csr(constrs[c_idx]->wsum_hess);
        int c_start = idx_offset;
        for (int row = 0; row < H->m; row++)
        {
            for (int idx = H->p[row]; idx < H->p[row + 1]; idx++)
            {
                col_to_pos[H->i[idx]] = idx;
            }

            for (int j = H_c->p[row]; j < H_c->p[row + 1]; j++)
            {
                prob->hess_idx_map[idx_offset++] = col_to_pos[H_c->i[j]];
            }
        }
        if (constrs[c_idx]->wsum_hess->is_stacked_pd)
        {
            compose_csr_idx_map_for_spd(
                (const stacked_pd *) constrs[c_idx]->wsum_hess, H_c,
                prob->hess_idx_map + c_start);
        }
    }
}

void problem_init_jacobian(problem *prob)
{
    if (prob->jacobian != NULL) return;

    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    // -------------------------------------------------------------------------------
    //                           Jacobian structure
    // -------------------------------------------------------------------------------
    jacobian_init(prob->objective);
    int nnz = 0;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];
        jacobian_init(c);
        CSR_matrix *Jc = c->jacobian->to_csr(c->jacobian);
        nnz += Jc->nnz;

        if (c->is_affine(c))
        {
            prob->stats.nnz_affine += Jc->nnz;
        }
        else
        {
            prob->stats.nnz_nonlinear += Jc->nnz;
        }
    }

    prob->jacobian = new_CSR_matrix(prob->total_constraint_size, prob->n_vars, nnz);

    /* set sparsity pattern of jacobian */
    CSR_matrix *H = prob->jacobian;
    H->p[0] = 0;
    int row_offset = 0;
    int nnz_offset = 0;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];
        CSR_matrix *Jc = c->jacobian->to_csr(c->jacobian);

        for (int r = 1; r <= Jc->m; r++)
        {
            H->p[row_offset + r] = nnz_offset + Jc->p[r];
        }

        memcpy(H->i + nnz_offset, Jc->i, Jc->nnz * sizeof(int));
        row_offset += Jc->m;
        nnz_offset += Jc->nnz;
    }
    assert(nnz_offset == nnz);

    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_init_derivatives += GET_ELAPSED_SECONDS(timer);
}

void problem_init_hessian(problem *prob)
{
    if (prob->lagrange_hessian != NULL) return;

    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    // -------------------------------------------------------------------------------
    //                        Lagrange Hessian structure
    // -------------------------------------------------------------------------------
    wsum_hess_init(prob->objective);
    int nnz = prob->objective->wsum_hess->nnz;

    for (int i = 0; i < prob->n_constraints; i++)
    {
        wsum_hess_init(prob->constraints[i]);
        nnz += prob->constraints[i]->wsum_hess->nnz;
    }

    int hess_nnz_ub = MIN(nnz, sat_mul_int(prob->n_vars, prob->n_vars));
    prob->lagrange_hessian = new_CSR_matrix(prob->n_vars, prob->n_vars, hess_nnz_ub);

    /* affine shortcut */
    memset(prob->lagrange_hessian->x, 0, hess_nnz_ub * sizeof(double));

    prob->hess_idx_map = (int *) sp_malloc(nnz * sizeof(int));
    int *iwork = (int *) sp_malloc(MAX(nnz, prob->n_vars) * sizeof(int));
    problem_lagrange_hess_fill_sparsity(prob, iwork);
    prob->stats.nnz_hessian = prob->lagrange_hessian->nnz;
    sp_free(iwork);

    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_init_derivatives += GET_ELAPSED_SECONDS(timer);
}

void problem_init_jacobian_coo(problem *prob)
{
    problem_init_jacobian(prob);
    if (prob->jacobian_coo != NULL) return;

    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    prob->jacobian_coo = new_COO_matrix(prob->jacobian);
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_init_derivatives += GET_ELAPSED_SECONDS(timer);
}

int problem_init_jacobian_coo_from(problem *prob, const int *rows,
                                   const int *cols, int nnz)
{
    problem_init_jacobian(prob);
    if (prob->jacobian_coo != NULL) return 0;

    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    prob->jacobian_coo =
        new_COO_matrix_from_pattern(prob->jacobian, rows, cols, nnz);
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_init_derivatives += GET_ELAPSED_SECONDS(timer);

    return prob->jacobian_coo != NULL ? 0 : -1;
}

void problem_init_hessian_coo_lower_triangular(problem *prob)
{
    problem_init_hessian(prob);
    if (prob->lagrange_hessian_coo != NULL) return;

    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    prob->lagrange_hessian_coo =
        new_COO_matrix_lower_triangular(prob->lagrange_hessian);
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_init_derivatives += GET_ELAPSED_SECONDS(timer);
}

int problem_init_hessian_coo_lower_triangular_from(problem *prob,
                                                   const int *rows,
                                                   const int *cols, int nnz)
{
    problem_init_hessian(prob);
    if (prob->lagrange_hessian_coo != NULL) return 0;

    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    prob->lagrange_hessian_coo = new_COO_matrix_lower_triangular_from_pattern(
        prob->lagrange_hessian, rows, cols, nnz);
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_init_derivatives += GET_ELAPSED_SECONDS(timer);

    return prob->lagrange_hessian_coo != NULL ? 0 : -1;
}

void problem_init_derivatives(problem *prob)
{
    problem_init_jacobian(prob);
    problem_init_hessian(prob);
}

static inline void format_memory(size_t bytes, char *buf, size_t buf_size)
{
    if (bytes < 1024)
    {
        snprintf(buf, buf_size, "%zu B", bytes);
    }
    else if (bytes < 1024 * 1024)
    {
        snprintf(buf, buf_size, "%.2f KB", (double) bytes / 1024.0);
    }
    else
    {
        snprintf(buf, buf_size, "%.2f MB", (double) bytes / (1024.0 * 1024.0));
    }
}

static inline void print_end_message(const Diff_engine_stats *stats)
{
    printf("\n"
           "============================================================\n"
           "                SparseDifferentiation v%s\n"
           "  (c) D. Cederberg and W. Zhang, Stanford University, 2026\n"
           "============================================================\n",
           DIFF_ENGINE_VERSION);

    printf("\nProblem statistics:\n");
    printf("  Number of variables:                    %d\n", stats->n_vars);
    printf("  Number of constraints:                  %d\n",
           stats->total_constraint_size);
    printf("  Affine constraints (nnz):               %d\n", stats->nnz_affine);
    printf("  Jacobian nonlinear constraints (nnz):   %d\n", stats->nnz_nonlinear);
    printf("  Lagrange Hessian (nnz):                 %d\n", stats->nnz_hessian);
    char mem_buf[64];
    format_memory(stats->memory_bytes, mem_buf, sizeof(mem_buf));
    printf("  Peak memory:                            %s\n", mem_buf);

    printf("\nTiming (seconds):\n");
    printf("  Derivative structure (sparsity):     %8.3f\n",
           stats->time_init_derivatives);
    printf("  Jacobian evaluation:                 %8.3f\n",
           stats->time_eval_jacobian);
    printf("  Gradient evaluation:                 %8.3f\n",
           stats->time_eval_gradient);
    printf("  Hessian evaluation:                  %8.3f\n",
           stats->time_eval_hessian);
    printf("  Objective evaluation:                %8.3f\n",
           stats->time_forward_obj);
    printf("  Constraints evaluation:              %8.3f\n",
           stats->time_forward_constraints);

    double total_time = stats->time_init_derivatives + stats->time_eval_jacobian +
                        stats->time_eval_gradient + stats->time_eval_hessian +
                        stats->time_forward_obj + stats->time_forward_constraints;

    printf("  ----------------------------------------------\n");
    printf("  Total differentiation time:          %8.3f\n", total_time);
}

void free_problem(problem *prob)
{
    if (prob == NULL) return;

    prob->stats.memory_bytes = g_peak_bytes;
    if (prob->verbose)
    {
        print_end_message(&prob->stats);
    }

    /* Free param_nodes array (weak refs, don't free the nodes) */
    sp_free(prob->param_nodes);

    /* Free allocated arrays */
    sp_free(prob->constraint_values);
    sp_free(prob->gradient_values);
    free_CSR_matrix(prob->jacobian);
    free_CSR_matrix(prob->lagrange_hessian);
    free_COO_matrix(prob->jacobian_coo);
    free_COO_matrix(prob->lagrange_hessian_coo);
    sp_free(prob->hess_idx_map);

    /* Release expression references (decrements refcount) */
    free_expr(prob->objective);
    for (int i = 0; i < prob->n_constraints; i++)
    {
        free_expr(prob->constraints[i]);
    }
    sp_free(prob->constraints);

    /* Free problem struct */
    sp_free(prob);
}

void problem_register_params(problem *prob, expr **param_nodes, int n_param_nodes)
{
    prob->n_param_nodes = n_param_nodes;
    prob->param_nodes = (expr **) sp_malloc(n_param_nodes * sizeof(expr *));
    memcpy(prob->param_nodes, param_nodes, n_param_nodes * sizeof(expr *));

    prob->total_parameter_size = 0;
    for (int i = 0; i < n_param_nodes; i++)
    {

        if (((parameter_expr *) param_nodes[i])->param_id == PARAM_FIXED)
        {
            fprintf(stderr, "can this ever happen? please report to developers if "
                            "this happens \n");
            exit(1);
        }

        prob->total_parameter_size += param_nodes[i]->size;
    }
}

void problem_update_params(problem *prob, const double *theta)
{
    /* raise error if there are no parameters */
    if (prob->n_param_nodes == 0)
    {
        fprintf(stderr, "Error: No parameters registered. This is a bug and should "
                        "be reported.\n");
        exit(1);
    }

    for (int i = 0; i < prob->n_param_nodes; i++)
    {
        expr *pnode = prob->param_nodes[i];
        parameter_expr *param = (parameter_expr *) pnode;

        if (param->param_id == PARAM_FIXED)
        {
            fprintf(stderr, "can this ever happen? please report to developers if "
                            "this happens \n");
            exit(1);
        }

        int offset = param->param_id;
        memcpy(pnode->value, theta + offset, pnode->size * sizeof(double));
    }

    /* Propagate needs_parameter_refresh to all expressions */
    expr_set_needs_refresh(prob->objective);
    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr_set_needs_refresh(prob->constraints[i]);
    }

    /* Force re-evaluation of affine Jacobians on next call */
    prob->jacobian_called = false;
}

double problem_objective_forward(problem *prob, const double *u)
{
    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    /* Evaluate objective only */
    prob->objective->forward(prob->objective, u);
    double result = prob->objective->value[0];

    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_forward_obj += GET_ELAPSED_SECONDS(timer);

    return result;
}

void problem_constraint_forward(problem *prob, const double *u)
{
    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    /* Evaluate constraints only and copy values */
    int offset = 0;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];
        c->forward(c, u);
        memcpy(prob->constraint_values + offset, c->value, c->size * sizeof(double));
        offset += c->size;
    }

    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_forward_constraints += GET_ELAPSED_SECONDS(timer);
}

void problem_gradient(problem *prob)
{
    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    /* evaluate jacobian of objective */
    prob->objective->eval_jacobian(prob->objective);

    /* copy sparse jacobian to dense gradient */
    memset(prob->gradient_values, 0, prob->n_vars * sizeof(double));
    CSR_matrix *jac = prob->objective->jacobian->to_csr(prob->objective->jacobian);
    for (int k = jac->p[0]; k < jac->p[1]; k++)
    {
        prob->gradient_values[jac->i[k]] = jac->x[k];
    }

    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_eval_gradient += GET_ELAPSED_SECONDS(timer);
}

void problem_jacobian(problem *prob)
{
    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);
    bool first_call = !prob->jacobian_called;

    CSR_matrix *J = prob->jacobian;
    int nnz_offset = 0;

    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];
        if (!first_call && c->is_affine(c))
        {
            /* skip evaluation for affine constraints after first call */
            nnz_offset += c->jacobian->nnz;
            continue;
        }

        c->eval_jacobian(c);
        memcpy(J->x + nnz_offset, c->jacobian->x, c->jacobian->nnz * sizeof(double));
        nnz_offset += c->jacobian->nnz;
    }

    /* update actual nnz (may be less than allocated) */
    J->nnz = nnz_offset;

    prob->jacobian_called = true;
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_eval_jacobian += GET_ELAPSED_SECONDS(timer);
}

void problem_hessian(problem *prob, double obj_w, const double *w)
{
    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    // ------------------------------------------------------------------------
    //             evaluate hessian of objective and constraints
    // ------------------------------------------------------------------------
    expr *obj = prob->objective;
    obj->eval_wsum_hess(obj, &obj_w);

    int offset = 0;
    expr **constrs = prob->constraints;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        if (constrs[i]->is_affine(constrs[i]))
        {
            /* skip evaluation for affine constraints */
            offset += constrs[i]->size;
            continue;
        }
        constrs[i]->eval_wsum_hess(constrs[i], w + offset);
        offset += constrs[i]->size;
    }

    // ------------------------------------------------------------------------
    //           assemble Lagrange hessian using index map
    // ------------------------------------------------------------------------
    CSR_matrix *H = prob->lagrange_hessian;
    int *idx_map = prob->hess_idx_map;

    /* zero out hessian before adding contribution from obj and constraints */
    memset(H->x, 0, H->nnz * sizeof(double));

    /* accumulate objective function */
    accumulator(obj->wsum_hess->x, obj->wsum_hess->nnz, idx_map, H->x);
    offset = obj->wsum_hess->nnz;

    /* accumulate constraint functions */
    for (int i = 0; i < prob->n_constraints; i++)
    {
        matrix *c_hess = constrs[i]->wsum_hess;
        accumulator(c_hess->x, c_hess->nnz, idx_map + offset, H->x);
        offset += c_hess->nnz;
    }

    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_eval_hessian += GET_ELAPSED_SECONDS(timer);
}
