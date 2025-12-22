#include "elementwise_univariate.h"
#include <stdlib.h>

void jacobian_init_elementwise(expr *node)
{
    expr *child = node->left;

    // if the variable is a child
    if (child->var_id != -1)
    {
        node->jacobian = new_csr_matrix(node->m, node->n_vars, node->m);
        node->jacobian->i[0] = JAC_IDXS_NOT_SET;
    }
    // otherwise it should be a linear operator
    else
    {
        node->jacobian = new_csr_matrix(child->jacobian->m, child->jacobian->n,
                                        child->jacobian->nnz);
        node->dwork = (double *) malloc(node->m * sizeof(double));
    }
}

void eval_jacobian_elementwise(expr *node)
{
    expr *child = node->left;

    if (child->var_id != -1)
    {
        if (node->jacobian->i[0] == JAC_IDXS_NOT_SET)
        {
            for (int j = 0; j < node->m; j++)
            {
                node->jacobian->p[j] = j;
                node->jacobian->i[j] = j + child->var_id;
            }
            node->jacobian->p[node->m] = node->m;
        }

        node->eval_local_jacobian(node, node->jacobian->x);
    }
    else
    {
        node->eval_local_jacobian(node, node->dwork);
        diag_csr_mult(node->dwork, child->jacobian, node->jacobian);
    }
}

bool is_affine_elementwise(expr *node)
{
    (void) node;
    return false;
}