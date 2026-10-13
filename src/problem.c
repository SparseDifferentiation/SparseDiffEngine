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
#include "sparsediff_linalg/matrix.h"
#include "sparsediff_linalg/matrix_sum.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include "subexpr.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

problem *new_problem(expr *objective, expr **constraints, int n_constraints,
                     bool verbose)
{
#ifdef SP_TRACK_MEMORY
    /* Start the high-water mark at the current live count: allocations made
       before new_problem (e.g. in left_matmul) stay counted, and their frees
       subtract from the live count as usual. */
    sp_memory_reset_peak();
#endif
    problem *prob = (problem *) sp_calloc(1, sizeof(problem));
    if (!prob) return NULL;

    /* objective */
    prob->objective = objective;
    expr_retain(objective);
    prob->n_vars = objective->n_vars;

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

void problem_init_jacobian(problem *prob)
{
    if (prob->jacobian != NULL) return;

    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    // -------------------------------------------------------------------------------
    //                           Jacobian structure
    // -------------------------------------------------------------------------------
    jacobian_init(prob->objective);
    if (prob->n_constraints > 0)
    {
        prob->constraint_jac_seen =
            (uint64_t *) sp_malloc(prob->n_constraints * sizeof(uint64_t));
    }
    int nnz = 0;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];
        jacobian_init(c);
        /* deliberately stale so the first problem_jacobian call copies */
        prob->constraint_jac_seen[i] = c->jacobian->values_version - 1;
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
    int n_ops = 1 + prob->n_constraints;
    matrix **ops = (matrix **) sp_malloc(n_ops * sizeof(matrix *));
    wsum_hess_init(prob->objective);
    ops[0] = prob->objective->wsum_hess;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        wsum_hess_init(prob->constraints[i]);
        ops[i + 1] = prob->constraints[i]->wsum_hess;
    }
    prob->hess_sum = sum_n_matrices_alloc(ops, n_ops);
    sp_free(ops);
    prob->lagrange_hessian = prob->hess_sum->to_csr(prob->hess_sum);
    prob->stats.nnz_hessian = prob->lagrange_hessian->nnz;

    /* affine shortcut */
    sum_n_matrices_fill_zero(prob->hess_sum);

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

void problem_init_derivatives(problem *prob)
{
    problem_init_jacobian(prob);
    problem_init_hessian(prob);
}

#ifdef SP_TRACK_MEMORY
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
#endif

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
#ifdef SP_TRACK_MEMORY
    char mem_buf[64];
    format_memory(stats->memory_bytes, mem_buf, sizeof(mem_buf));
    printf("  Peak memory:                            %s\n", mem_buf);
#endif

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

#ifdef SP_TRACK_MEMORY
    sp_memory_stats(NULL, &prob->stats.memory_bytes);
#endif
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
    free_matrix(prob->hess_sum);
    free_COO_matrix(prob->jacobian_coo);
    free_COO_matrix(prob->lagrange_hessian_coo);
    sp_free(prob->constraint_jac_seen);

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
    eval_jacobian(prob->objective);

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
    CSR_matrix *J = prob->jacobian;
    int nnz_offset = 0;

    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];
        eval_jacobian(c);

        /* copy only when the constraint's jacobian values actually changed
           (an affine constraint's eval is a no-op after its first call and
           leaves the version untouched until the next parameter update).
           Copy through the CSR view: J's rows were laid out from to_csr in
           problem_init_jacobian, and a stacked_pd's own value buffer is
           block-major, which is not row order when its blocks interleave
           rows. For sparse and permuted_dense the view aliases the buffer. */
        if (prob->constraint_jac_seen[i] != c->jacobian->values_version)
        {
            const CSR_matrix *Jc = c->jacobian->to_csr(c->jacobian);
            memcpy(J->x + nnz_offset, Jc->x, Jc->nnz * sizeof(double));
            prob->constraint_jac_seen[i] = c->jacobian->values_version;
        }
        nnz_offset += c->jacobian->nnz;
    }

    assert(nnz_offset == J->nnz);
    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_eval_jacobian += GET_ELAPSED_SECONDS(timer);
}

void problem_hessian(problem *prob, double obj_w, const double *w)
{
    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    // ------------------------------------------------------------------------
    //      evaluate hessians of objective and constraints and sum them
    // ------------------------------------------------------------------------
    /* add each hessian right after evaluating it: the objective and the
       constraints may share a subexpression, whose later evaluation would
       overwrite earlier values */
    matrix *H = prob->hess_sum;
    sum_n_matrices_fill_zero(H);
    expr *obj = prob->objective;
    eval_wsum_hess(obj, &obj_w);
    sum_n_matrices_fill_add(obj->wsum_hess, 0, H);

    int offset = 0;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];
        /* an affine constraint's hessian is zero: skip it */
        if (!c->is_affine(c))
        {
            eval_wsum_hess(c, w + offset);
            sum_n_matrices_fill_add(c->wsum_hess, i + 1, H);
        }
        offset += c->size;
    }

    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    prob->stats.time_eval_hessian += GET_ELAPSED_SECONDS(timer);
}
