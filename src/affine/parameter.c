/*
 * Copyright 2026 Daniel Cederberg and William Zhang
 *
 * This file is part of the DNLP-differentiation-engine project.
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

/* Unified parameter/constant leaf node.
 *
 * When param_id == PARAM_FIXED, this is a constant whose values are set at
 * creation and never change. When param_id >= 0, values are updated via
 * problem_update_params.
 *
 * In both cases the derivative behavior is identical: zero Jacobian and
 * Hessian with respect to variables (always affine). */

#include "affine.h"
#include "subexpr.h"
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    /* Values are set at creation (constants) or by problem_update_params */
    (void) node;
    (void) u;
}

static void jacobian_init(expr *node)
{
    /* Parameter/constant jacobian is all zeros: size x n_vars with 0 nonzeros */
    node->jacobian = new_csr_matrix(node->size, node->n_vars, 0);
}

static void eval_jacobian(expr *node)
{
    /* Jacobian never changes */
    (void) node;
}

static void wsum_hess_init(expr *node)
{
    /* Hessian is all zeros */
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, 0);
}

static void eval_wsum_hess(expr *node, const double *w)
{
    (void) node;
    (void) w;
}

static bool is_affine(const expr *node)
{
    (void) node;
    return true;
}

expr *new_parameter(int d1, int d2, int param_id, int n_vars, const double *values)
{
    parameter_expr *pnode = (parameter_expr *) calloc(1, sizeof(parameter_expr));
    expr *node = &pnode->base;
    init_expr(node, d1, d2, n_vars, forward, jacobian_init, eval_jacobian, is_affine,
              wsum_hess_init, eval_wsum_hess, NULL);
    pnode->param_id = param_id;

    /* If values provided (fixed constant), copy them now.
       Otherwise values will be populated by problem_update_params. */
    if (values != NULL)
    {
        memcpy(node->value, values, node->size * sizeof(double));
    }

    return node;
}
