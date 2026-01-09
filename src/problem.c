#include "problem.h"
#include <stdlib.h>
#include <string.h>

// TODO: move this into common file
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* forward declaration */
static void problem_lagrange_hess_fill_sparsity(problem *prob, int *iwork);

/* Helper function to compare integers for qsort */
static int compare_int_asc(const void *a, const void *b)
{
    int ia = *((const int *) a);
    int ib = *((const int *) b);
    return (ia > ib) - (ia < ib);
}

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

    /* Allocate value arrays */
    if (prob->total_constraint_size > 0)
    {
        prob->constraint_values =
            (double *) calloc(prob->total_constraint_size, sizeof(double));
    }
    else
    {
        prob->constraint_values = NULL;
    }
    prob->gradient_values = (double *) calloc(prob->n_vars, sizeof(double));

    /* Derivative structures allocated by problem_init_derivatives */
    prob->jacobian = NULL;

    return prob;
}

void problem_init_derivatives(problem *prob)
{
    // -------------------------------------------------------------------------------
    //                           Jacobian structure
    // -------------------------------------------------------------------------------
    prob->objective->jacobian_init(prob->objective);
    int nnz = 0;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];
        c->jacobian_init(c);
        nnz += c->jacobian->nnz;
    }

    prob->jacobian = new_csr_matrix(prob->total_constraint_size, prob->n_vars, nnz);

    // -------------------------------------------------------------------------------
    //                        Lagrange Hessian structure
    // -------------------------------------------------------------------------------
    prob->objective->wsum_hess_init(prob->objective);
    nnz = prob->objective->wsum_hess->nnz;

    for (int i = 0; i < prob->n_constraints; i++)
    {
        prob->constraints[i]->wsum_hess_init(prob->constraints[i]);
        nnz += prob->constraints[i]->wsum_hess->nnz;
    }

    prob->lagrange_hessian = new_csr_matrix(prob->n_vars, prob->n_vars, nnz);
    prob->hess_idx_map = (int *) malloc(nnz * sizeof(int));
    int *iwork = (int *) malloc(MAX(nnz, prob->n_vars) * sizeof(int));
    problem_lagrange_hess_fill_sparsity(prob, iwork);
    free(iwork);
}

static void problem_lagrange_hess_fill_sparsity(problem *prob, int *iwork)
{
    expr **constrs = prob->constraints;
    int *cols = iwork;
    int *col_to_pos = iwork; /* reused after qsort */
    int nnz = 0;
    CSR_Matrix *H_obj = prob->objective->wsum_hess;
    CSR_Matrix *H_c;
    CSR_Matrix *H = prob->lagrange_hessian;
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
            H_c = constrs[c_idx]->wsum_hess;
            int c_len = H_c->p[row + 1] - H_c->p[row];
            memcpy(cols + count, H_c->i + H_c->p[row], c_len * sizeof(int));
            count += c_len;
        }

        /* find unique columns */
        qsort(cols, count, sizeof(int), compare_int_asc);
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

    /* map constraint hessian entries */
    for (int c_idx = 0; c_idx < prob->n_constraints; c_idx++)
    {
        H_c = constrs[c_idx]->wsum_hess;
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
    }
}

void free_problem(problem *prob)
{
    if (prob == NULL) return;

    /* Free allocated arrays */
    free(prob->constraint_values);
    free(prob->gradient_values);
    free_csr_matrix(prob->jacobian);
    free_csr_matrix(prob->lagrange_hessian);
    free(prob->hess_idx_map);

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

void problem_constraint_forward(problem *prob, const double *u)
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
}

void problem_gradient(problem *prob)
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
}

void problem_jacobian(problem *prob)
{
    CSR_Matrix *stacked = prob->jacobian;

    /* Initialize row pointers */
    stacked->p[0] = 0;

    int row_offset = 0;
    int nnz_offset = 0;

    /* TODO: here we eventually want to evaluate affine jacobians only once */
    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];

        /* Evaluate jacobian */
        c->eval_jacobian(c);

        CSR_Matrix *cjac = c->jacobian;

        /* Copy row pointers with offset */
        for (int r = 1; r <= cjac->m; r++)
        {
            stacked->p[row_offset + r] = nnz_offset + cjac->p[r];
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
}

void problem_hessian(problem *prob, double obj_w, const double *w)
{
    // ------------------------------------------------------------------------
    //             evaluate hessian of objective and constraints
    // ------------------------------------------------------------------------
    expr *obj = prob->objective;
    expr **constrs = prob->constraints;

    obj->eval_wsum_hess(obj, &obj_w);

    int offset = 0;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        constrs[i]->eval_wsum_hess(constrs[i], w + offset);
        offset += constrs[i]->size;
    }

    // ------------------------------------------------------------------------
    //           assemble Lagrange hessian using index map
    // ------------------------------------------------------------------------
    CSR_Matrix *H = prob->lagrange_hessian;
    int *idx_map = prob->hess_idx_map;

    /* zero out hessian before adding contribution from obj and constraints */
    memset(H->x, 0, H->nnz * sizeof(double));

    /* accumulate objective function */
    idx_map_accumulator(obj->wsum_hess, idx_map, H->x);
    offset = obj->wsum_hess->nnz;

    /* accumulate constraint functions */
    for (int i = 0; i < prob->n_constraints; i++)
    {
        idx_map_accumulator(constrs[i]->wsum_hess, idx_map + offset, H->x);
        offset += constrs[i]->wsum_hess->nnz;
    }
}
