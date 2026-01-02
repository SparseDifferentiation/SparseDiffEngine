#include "expr.h"
#include "utils/int_double_pair.h"
#include <stdlib.h>
#include <string.h>

expr *new_expr(int d1, int d2, int n_vars)
{
    expr *node = (expr *) calloc(1, sizeof(expr));
    if (!node) return NULL;

    node->d1 = d1;
    node->d2 = d2;
    node->size = d1 * d2;
    node->n_vars = n_vars;
    node->refcount = 1;
    node->value = (double *) calloc(d1 * d2, sizeof(double));
    if (!node->value)
    {
        free(node);
        return NULL;
    }

    node->var_id = -1;

    return node;
}

void free_expr(expr *node)
{
    if (node == NULL) return;

    node->refcount--;
    if (node->refcount > 0) return; /* Still referenced elsewhere */

    /* recursively free children */
    free_expr(node->left);
    free_expr(node->right);

    /* free value array and jacobian */
    free(node->value);
    free_csr_matrix(node->jacobian);
    free_csr_matrix(node->wsum_hess);
    free(node->dwork);
    free(node->iwork);

    /* free type-specific data */
    if (node->free_type_data)
    {
        node->free_type_data(node);
    }

    /* free the node itself */
    free(node);
}

void expr_retain(expr *node)
{
    if (node == NULL) return;
    node->refcount++;
}

bool is_affine(expr *node)
{
    bool left_affine = true;
    bool right_affine = true;
    expr *left = node->left;
    expr *right = node->right;

    if (left)
    {
        left_affine = left->is_affine(left);
    }

    if (right)
    {
        right_affine = right->is_affine(right);
    }

    return left_affine && right_affine;
}
