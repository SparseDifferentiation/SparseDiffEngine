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
#include <math.h>

static void atanh_forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = atanh(node->left->value[i]);
    }
}

static void atanh_eval_jacobian(expr *node)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        node->jacobian->x[j] = 1.0 / (1.0 - x[j] * x[j]);
    }
}

static void atanh_eval_wsum_hess(expr *node, const double *w)
{
    double *x = node->left->value;
    for (int j = 0; j < node->size; j++)
    {
        double c = 1.0 - x[j] * x[j];
        node->wsum_hess->x[j] = w[j] * (2.0 * x[j]) / (c * c);
    }
}

expr *new_atanh(expr *child)
{
    expr *node = new_restricted(child);
    node->forward = atanh_forward;
    node->eval_jacobian = atanh_eval_jacobian;
    node->eval_wsum_hess = atanh_eval_wsum_hess;
    return node;
}