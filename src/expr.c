#include "expr.h"
#include <stdlib.h>
#include <string.h>

expr *new_expr(int m, int n_vars)
{
    expr *node = (expr *) malloc(sizeof(expr));
    if (!node) return NULL;

    node->m = m;
    node->n_vars = n_vars;
    node->value = (double *) calloc(m, sizeof(double));
    if (!node->value)
    {
        free(node);
        return NULL;
    }

    node->left = NULL;
    node->right = NULL;
    node->forward = NULL;
    node->jacobian = NULL;
    node->jacobian_init = NULL;
    node->eval_jacobian = NULL;
    node->is_affine = NULL;
    node->dwork = NULL;
    node->var_id = -1;

    return node;
}

void free_expr(expr *node)
{
    if (!node) return;

    /* recursively free children */
    free_expr(node->left);
    free_expr(node->right);

    /* free value array and jacobian */
    free(node->value);
    free_csr_matrix(node->jacobian);
    free(node->dwork);

    /* free the node itself */
    free(node);
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
