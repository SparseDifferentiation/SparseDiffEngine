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
#include "atoms/elementwise_restricted_dom.h"
#include "utils/tracked_alloc.h"
#include <stdlib.h>

void jacobian_init_restricted(expr *node)
{
    expr *child = node->left;

    node->jacobian = new_csr_matrix(node->size, node->n_vars, node->size);
    for (int j = 0; j < node->size; j++)
    {
        node->jacobian->p[j] = j;
        node->jacobian->i[j] = j + child->var_id;
    }
    node->jacobian->p[node->size] = node->size;
}

void wsum_hess_init_restricted(expr *node)
{
    expr *child = node->left;
    int id = child->var_id;
    int i;

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

bool is_affine_restricted(const expr *node)
{
    (void) node;
    return false;
}

expr *new_restricted(expr *child)
{
    expr *node = (expr *) SP_CALLOC(1, sizeof(expr));
    if (!node) return NULL;

    init_expr(node, child->d1, child->d2, child->n_vars, NULL,
              jacobian_init_restricted, NULL, is_affine_restricted,
              wsum_hess_init_restricted, NULL, NULL);

    node->left = child;
    expr_retain(child);
    return node;
}