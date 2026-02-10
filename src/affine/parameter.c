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

/* Parameter leaf node: behaviorally identical to constant (zero derivatives
   w.r.t. variables), but its values are updatable via problem_update_params.
   This allows re-solving with different parameter values without rebuilding
   the expression tree. */

#include "affine.h"
#include "subexpr.h"
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    /* Values are set by problem_update_params, not by forward pass */
    (void) node;
    (void) u;
}

static void jacobian_init(expr *node)
{
    /* Parameter jacobian is all zeros: size x n_vars with 0 nonzeros */
    node->jacobian = new_csr_matrix(node->size, node->n_vars, 0);
}

static void eval_jacobian(expr *node)
{
    /* Parameter jacobian never changes */
    (void) node;
}

static void wsum_hess_init(expr *node)
{
    /* Parameter Hessian is all zeros */
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

expr *new_parameter(int d1, int d2, int param_id, int n_vars)
{
    parameter_expr *pnode = (parameter_expr *) calloc(1, sizeof(parameter_expr));
    init_expr(&pnode->base, d1, d2, n_vars, forward, jacobian_init, eval_jacobian,
              is_affine, wsum_hess_init, eval_wsum_hess, NULL);
    pnode->param_id = param_id;
    /* values will be populated by problem_update_params */
    return &pnode->base;
}
