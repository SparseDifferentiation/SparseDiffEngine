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
#include "atoms/elementwise_full_dom.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include "subexpr.h"
#include <math.h>
#include <stdlib.h>

/* out[i] = c * x[i]^e. The exponents that arise from x^2 and x^(-1) and their
   derivatives avoid pow, which is slow. */
static void scaled_pow(double *out, const double *x, int n, double c, double e)
{
    if (e == 0.0)
    {
        for (int i = 0; i < n; i++) out[i] = c;
    }
    else if (e == 1.0)
    {
        for (int i = 0; i < n; i++) out[i] = c * x[i];
    }
    else if (e == 2.0)
    {
        for (int i = 0; i < n; i++) out[i] = c * (x[i] * x[i]);
    }
    else if (e == -1.0)
    {
        for (int i = 0; i < n; i++) out[i] = c / x[i];
    }
    else if (e == -2.0)
    {
        for (int i = 0; i < n; i++) out[i] = c / (x[i] * x[i]);
    }
    else if (e == -3.0)
    {
        for (int i = 0; i < n; i++) out[i] = c / (x[i] * x[i] * x[i]);
    }
    else
    {
        for (int i = 0; i < n; i++) out[i] = c * pow(x[i], e);
    }
}

static void forward(expr *node, const double *u)
{

    /* child's forward pass */
    expr_forward(node->left, u);

    /* local forward pass */
    double *x = node->left->value;
    double p = ((power_expr *) node)->p;
    scaled_pow(node->value, x, node->size, 1.0, p);
}

static void local_jacobian(expr *node, double *vals)
{
    double *x = node->left->value;
    double p = ((power_expr *) node)->p;

    scaled_pow(vals, x, node->size, p, p - 1);
}

static void local_wsum_hess(expr *node, double *out, const double *w)
{
    double *x = node->left->value;
    double p = ((power_expr *) node)->p;

    scaled_pow(out, x, node->size, p * (p - 1), p - 2);
    for (int j = 0; j < node->size; j++)
    {
        out[j] *= w[j];
    }
}

expr *new_power(expr *child, double p)
{
    /* Allocate the type-specific struct */
    power_expr *pnode = (power_expr *) sp_calloc(1, sizeof(power_expr));
    expr *node = &pnode->base;
    init_elementwise(node, child);
    node->forward_impl = forward;
    node->local_jacobian = local_jacobian;
    node->local_wsum_hess = local_wsum_hess;

    /* Set type-specific field */
    pnode->p = p;

    return node;
}
