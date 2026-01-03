#include "affine.h"
#include "utils/int_double_pair.h"
#include "utils/mini_numpy.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    int i, j, end;
    double sum;
    expr *x = node->left;
    sum_expr *snode = (sum_expr *) node;
    int axis = snode->axis;

    /* child's forward pass */
    x->forward(x, u);

    if (axis == -1)
    {
        sum = 0.0;
        end = x->d1 * x->d2;

        /* sum all elements */
        for (i = 0; i < end; i++)
        {
            sum += x->value[i];
        }
        node->value[0] = sum;
    }
    else if (axis == 0)
    {
        /* sum rows together */
        for (j = 0; j < x->d2; j++)
        {
            sum = 0.0;
            end = (j + 1) * x->d1;
            for (i = j * x->d1; i < end; i++)
            {
                sum += x->value[i];
            }
            node->value[j] = sum;
        }
    }
    else if (axis == 1)
    {
        memset(node->value, 0, node->d1 * sizeof(double));

        /* sum columns together */
        for (j = 0; j < x->d2; j++)
        {
            int offset = j * x->d1;
            for (i = 0; i < x->d1; i++)
            {
                node->value[i] += x->value[offset + i];
            }
        }
    }
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    sum_expr *snode = (sum_expr *) node;

    /* initialize child's jacobian */
    x->jacobian_init(x);

    /* we never have to store more than the child's nnz */
    node->jacobian = new_csr_matrix(node->d1, node->n_vars, x->jacobian->nnz);
    snode->int_double_pairs = new_int_double_pair_array(x->jacobian->nnz);
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    sum_expr *snode = (sum_expr *) node;
    int axis = snode->axis;

    /* evaluate child's jacobian */
    x->eval_jacobian(x);

    /* sum rows or columns of child's jacobian */
    if (axis == -1)
    {
        sum_all_rows_csr(x->jacobian, node->jacobian, snode->int_double_pairs);
    }
    else if (axis == 0)
    {
        sum_block_of_rows_csr(x->jacobian, node->jacobian, snode->int_double_pairs,
                              x->d1);
    }

    else if (axis == 1)
    {
        sum_evenly_spaced_rows_csr(x->jacobian, node->jacobian,
                                   snode->int_double_pairs, node->d1);
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;
    /* initialize child's wsum_hess */
    x->wsum_hess_init(x);

    /* we never have to store more than the child's nnz */
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, x->wsum_hess->nnz);
    node->dwork = malloc(x->size * sizeof(double));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    sum_expr *snode = (sum_expr *) node;
    int axis = snode->axis;

    if (axis == -1)
    {
        scaled_ones(node->dwork, x->size, *w);
    }
    else if (axis == 0)
    {
        repeat(node->dwork, w, x->d2, x->d1);
    }
    else if (axis == 1)
    {
        tile(node->dwork, w, x->d1, x->d2);
    }

    x->eval_wsum_hess(x, node->dwork);

    /* todo: is this copy necessary or can we just change pointers? */
    copy_csr_matrix(x->wsum_hess, node->wsum_hess);
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    sum_expr *snode = (sum_expr *) node;
    free_int_double_pair_array(snode->int_double_pairs);
}

expr *new_sum(expr *child, int axis)
{
    int d1 = 0;

    switch (axis)
    {
        case -1:
            /* no axis specified */
            d1 = 1;
            break;
        case 0:
            /* sum rows together */
            d1 = child->d2;
            break;
        case 1:
            /* sum columns together */
            d1 = child->d1;
            break;
    }

    /* Allocate the type-specific struct */
    sum_expr *snode = (sum_expr *) calloc(1, sizeof(sum_expr));
    expr *node = &snode->base;
    init_expr(node, d1, 1, child->n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, free_type_data);
    node->left = child;
    expr_retain(child);

    /* hessian function pointers */
    node->wsum_hess_init = wsum_hess_init;
    node->eval_wsum_hess = eval_wsum_hess;

    /* Set type-specific fields */
    snode->axis = axis;
    snode->int_double_pairs = NULL;

    return node;
}
