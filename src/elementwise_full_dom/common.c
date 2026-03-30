#include "elementwise_full_dom.h"
#include "subexpr.h"
#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"
#include "utils/CSR_sum.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void jacobian_init_elementwise(expr *node)
{
    expr *child = node->left;

    /* if the variable is a child */
    if (child->var_id != NOT_A_VARIABLE)
    {
        node->jacobian = new_csr_matrix(node->size, node->n_vars, node->size);
        for (int j = 0; j < node->size; j++)
        {
            node->jacobian->p[j] = j;
            node->jacobian->i[j] = j + child->var_id;
        }
        node->jacobian->p[node->size] = node->size;
    }
    else
    {
        /* jacobian of h(x) = f(g(x)) is Jf @ Jg, and here Jf is diagonal */
        jacobian_init(child);
        CSR_Matrix *Jg = child->jacobian;
        node->jacobian = new_csr_copy_sparsity(Jg);
        node->work->dwork = (double *) malloc(node->size * sizeof(double));
        node->work->local_jac_diag = (double *) malloc(node->size * sizeof(double));
    }
}

void eval_jacobian_elementwise(expr *node)
{
    expr *child = node->left;

    if (child->var_id != NOT_A_VARIABLE)
    {
        node->local_jacobian(node, node->jacobian->x);
    }
    else
    {
        /* jacobian of h(x) = f(g(x)) is Jf @ Jg, and here Jf is diagonal */
        child->eval_jacobian(child);
        CSR_Matrix *Jg = child->jacobian;
        node->local_jacobian(node, node->work->local_jac_diag);
        memcpy(node->work->dwork, node->work->local_jac_diag,
               node->size * sizeof(double));
        diag_csr_mult_fill_vals(node->work->dwork, Jg, node->jacobian);
    }
}

void wsum_hess_init_elementwise(expr *node)
{
    expr *child = node->left;
    int id = child->var_id;
    int i;

    /* if the variable is a child */
    if (id != NOT_A_VARIABLE)
    {
        node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, node->size);

        for (i = 0; i < node->size; i++)
        {
            node->wsum_hess->p[id + i] = i;
            node->wsum_hess->i[i] = id + i;
        }

        for (i = id + node->size; i <= node->n_vars; i++)
        {
            node->wsum_hess->p[i] = node->size;
        }
    }
    else
    {
        /* Hessian of h(x) = w^T f(g(x) is term1 + term 2 where
            term1 = J_g^T @ D @ J_g with D = sum_i w_i Hf_i,
            term2 = sum_i (J_f^T w)_i^T Hg_i.

            For elementwise functions, D is diagonal. */
        jacobian_csc_init(child);
        CSC_Matrix *Jg = child->work->jacobian_csc;

        if (child->is_affine(child))
        {
            node->wsum_hess = ATA_alloc(Jg);
        }
        else
        {
            /* term1: Jg^T @ D @ Jg */
            node->work->hess_term1 = ATA_alloc(Jg);

            /* term2: child's Hessian */
            wsum_hess_init(child);
            CSR_Matrix *Hg = child->wsum_hess;
            node->work->hess_term2 = new_csr_copy_sparsity(Hg);

            /* wsum_hess = term1 + term2 */
            int max_nnz = node->work->hess_term1->nnz + node->work->hess_term2->nnz;
            node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, max_nnz);
            sum_csr_alloc(node->work->hess_term1, node->work->hess_term2,
                          node->wsum_hess);
        }
    }
}

void eval_wsum_hess_elementwise(expr *node, const double *w)
{
    expr *child = node->left;

    if (child->var_id != NOT_A_VARIABLE)
    {
        node->local_wsum_hess(node, node->wsum_hess->x, w);
    }
    else
    {
        if (child->is_affine(child))
        {
            if (!child->work->jacobian_csc_filled)
            {
                csr_to_csc_fill_vals(child->jacobian, child->work->jacobian_csc,
                                     child->work->csc_work);
                child->work->jacobian_csc_filled = true;
            }

            node->local_wsum_hess(node, node->work->dwork, w);
            ATDA_fill_vals(child->work->jacobian_csc, node->work->dwork,
                           node->wsum_hess);
        }
        else
        {
            /* refresh CSC jacobian values */
            csr_to_csc_fill_vals(child->jacobian, child->work->jacobian_csc,
                                 child->work->csc_work);

            /* term1: Jg^T @ D @ Jg */
            node->local_wsum_hess(node, node->work->dwork, w);
            ATDA_fill_vals(child->work->jacobian_csc, node->work->dwork,
                           node->work->hess_term1);

            /* term2: child Hessian with weight Jf^T w */
            memcpy(node->work->dwork, node->work->local_jac_diag,
                   node->size * sizeof(double));
            for (int k = 0; k < node->size; k++)
            {
                node->work->dwork[k] *= w[k];
            }

            child->eval_wsum_hess(child, node->work->dwork);
            memcpy(node->work->hess_term2->x, child->wsum_hess->x,
                   child->wsum_hess->nnz * sizeof(double));

            /* wsum_hess = term1 + term2 */
            sum_csr_fill_vals(node->work->hess_term1, node->work->hess_term2,
                              node->wsum_hess);
        }
    }
}

bool is_affine_elementwise(const expr *node)
{
    (void) node;
    return false;
}

void init_elementwise(expr *node, expr *child)
{
    /* Initialize base fields */
    init_expr(node, child->d1, child->d2, child->n_vars, NULL,
              jacobian_init_elementwise, eval_jacobian_elementwise,
              is_affine_elementwise, wsum_hess_init_elementwise,
              eval_wsum_hess_elementwise, NULL);

    /* Set left child */
    node->left = child;
    expr_retain(child);
}

expr *new_elementwise(expr *child)
{
    expr *node = (expr *) calloc(1, sizeof(expr));
    if (!node) return NULL;

    init_elementwise(node, child);
    return node;
}
