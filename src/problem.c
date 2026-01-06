#include "problem.h"
#include <stdlib.h>
#include <string.h>

problem *new_problem(expr *objective, expr **constraints, int n_constraints)
{
    problem *prob = (problem *) calloc(1, sizeof(problem));
    if (!prob) return NULL;

    /* Retain objective (shared ownership with caller) */
    prob->objective = objective;
    expr_retain(objective);

    /* Copy and retain constraints array */
    prob->n_constraints = n_constraints;
    if (n_constraints > 0)
    {
        prob->constraints = (expr **) malloc(n_constraints * sizeof(expr *));
        for (int i = 0; i < n_constraints; i++)
        {
            prob->constraints[i] = constraints[i];
            expr_retain(constraints[i]);
        }
    }
    else
    {
        prob->constraints = NULL;
    }

    /* Compute total constraint size */
    prob->total_constraint_size = 0;
    for (int i = 0; i < n_constraints; i++)
    {
        prob->total_constraint_size += constraints[i]->size;
    }

    prob->n_vars = objective->n_vars;

    /* Initialize allocated pointers to NULL */
    prob->constraint_values = NULL;
    prob->gradient_values = NULL;
    prob->stacked_jac = NULL;

    return prob;
}

void problem_allocate(problem *prob, const double *u)
{
    /* 1. Allocate constraint values array */
    if (prob->total_constraint_size > 0)
    {
        prob->constraint_values = (double *) calloc(prob->total_constraint_size, sizeof(double));
    }

    /* 2. Allocate gradient values array */
    prob->gradient_values = (double *) calloc(prob->n_vars, sizeof(double));

    /* 3. Initialize objective jacobian */
    prob->objective->forward(prob->objective, u);
    prob->objective->jacobian_init(prob->objective);

    /* 4. Initialize constraint jacobians and count total nnz */
    int total_nnz = 0;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];
        c->forward(c, u);
        c->jacobian_init(c);
        total_nnz += c->jacobian->nnz;
    }

    /* 5. Allocate stacked jacobian */
    if (prob->total_constraint_size > 0)
    {
        prob->stacked_jac = new_csr_matrix(prob->total_constraint_size, prob->n_vars, total_nnz);
    }
}

void free_problem(problem *prob)
{
    if (prob == NULL) return;

    /* Free allocated arrays */
    free(prob->constraint_values);
    free(prob->gradient_values);
    free_csr_matrix(prob->stacked_jac);

    /* Release expression references (decrements refcount) */
    free_expr(prob->objective);
    for (int i = 0; i < prob->n_constraints; i++)
    {
        free_expr(prob->constraints[i]);
    }
    free(prob->constraints);

    /* Free problem struct */
    free(prob);
}

double problem_objective_forward(problem *prob, const double *u)
{
    /* Evaluate objective only */
    prob->objective->forward(prob->objective, u);
    return prob->objective->value[0];
}

double *problem_constraint_forward(problem *prob, const double *u)
{
    /* Evaluate constraints only and copy values */
    int offset = 0;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];
        c->forward(c, u);
        memcpy(prob->constraint_values + offset, c->value, c->size * sizeof(double));
        offset += c->size;
    }

    return prob->constraint_values;
}

double *problem_gradient(problem *prob, const double *u)
{
    /* Jacobian on objective */
    prob->objective->eval_jacobian(prob->objective);

    /* Zero gradient array */
    memset(prob->gradient_values, 0, prob->n_vars * sizeof(double));

    /* Copy sparse jacobian row to dense gradient
     * Objective jacobian is 1 x n_vars */
    CSR_Matrix *jac = prob->objective->jacobian;
    for (int k = jac->p[0]; k < jac->p[1]; k++)
    {
        int col = jac->i[k];
        prob->gradient_values[col] = jac->x[k];
    }

    return prob->gradient_values;
}

CSR_Matrix *problem_jacobian(problem *prob, const double *u)
{
    CSR_Matrix *stacked = prob->stacked_jac;

    /* Initialize row pointers */
    stacked->p[0] = 0;

    int row_offset = 0;
    int nnz_offset = 0;

    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];

        /* Evaluate jacobian */
        c->eval_jacobian(c);

        CSR_Matrix *cjac = c->jacobian;

        /* Copy row pointers with offset */
        for (int r = 0; r < cjac->m; r++)
        {
            int row_nnz = cjac->p[r + 1] - cjac->p[r];
            stacked->p[row_offset + r + 1] = stacked->p[row_offset + r] + row_nnz;
        }

        /* Copy column indices and values */
        int constraint_nnz = cjac->p[cjac->m];
        memcpy(stacked->i + nnz_offset, cjac->i, constraint_nnz * sizeof(int));
        memcpy(stacked->x + nnz_offset, cjac->x, constraint_nnz * sizeof(double));

        row_offset += cjac->m;
        nnz_offset += constraint_nnz;
    }

    /* Update actual nnz (may be less than allocated) */
    stacked->nnz = nnz_offset;

    return stacked;
}
