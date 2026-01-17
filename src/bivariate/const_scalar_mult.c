#include "bivariate.h"
#include "subexpr.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Constant scalar multiplication: y = a * child where a is a constant double */

static void forward(expr *node, const double *u)
{
    expr *child = node->left;

    /* child's forward pass */
    child->forward(child, u);

    /* local forward pass: multiply each element by scalar a */
    double a = ((const_scalar_mult_expr *) node)->a;
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = a * child->value[i];
    }
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;

    /* initialize child jacobian */
    x->jacobian_init(x);

    /* same sparsity as child */
    node->jacobian = new_csr_matrix(node->size, node->n_vars, x->jacobian->nnz);
    memcpy(node->jacobian->p, x->jacobian->p, (node->size + 1) * sizeof(int));
    memcpy(node->jacobian->i, x->jacobian->i, x->jacobian->nnz * sizeof(int));
}

static void eval_jacobian(expr *node)
{
    expr *child = node->left;
    double a = ((const_scalar_mult_expr *) node)->a;

    /* evaluate child */
    child->eval_jacobian(child);

    /* scale child's jacobian */
    for (int j = 0; j < child->jacobian->nnz; j++)
    {
        node->jacobian->x[j] = a * child->jacobian->x[j];
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;

    /* initialize child's weighted Hessian */
    x->wsum_hess_init(x);

    /* same sparsity as child */
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, x->wsum_hess->nnz);
    memcpy(node->wsum_hess->p, x->wsum_hess->p, (node->n_vars + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, x->wsum_hess->i, x->wsum_hess->nnz * sizeof(int));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    x->eval_wsum_hess(x, w);

    double a = ((const_scalar_mult_expr *) node)->a;
    for (int j = 0; j < x->wsum_hess->nnz; j++)
    {
        node->wsum_hess->x[j] = a * x->wsum_hess->x[j];
    }
}

expr *new_const_scalar_mult(double a, expr *child)
{
    const_scalar_mult_expr *mult_node =
        (const_scalar_mult_expr *) calloc(1, sizeof(const_scalar_mult_expr));
    expr *node = &mult_node->base;

    init_expr(node, child->d1, child->d2, child->n_vars, forward, jacobian_init,
              eval_jacobian, NULL, NULL);
    node->wsum_hess_init = wsum_hess_init;
    node->eval_wsum_hess = eval_wsum_hess;
    node->left = child;
    mult_node->a = a;
    expr_retain(child);

    // just for debugging, should be removed
    strcpy(node->name, "const_scalar_mult");

    return node;
}
