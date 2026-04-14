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
#ifndef ELEMENTWISE_RESTRICTED_DOM_H
#define ELEMENTWISE_RESTRICTED_DOM_H

#include "expr.h"

/* Shared init functions for restricted domain atoms
 * (variable-child only, no linear operator support) */
void jacobian_init_restricted(expr *node);
void wsum_hess_init_restricted(expr *node);
bool is_affine_restricted(const expr *node);
expr *new_restricted(expr *child);

expr *new_log(expr *child);
expr *new_entr(expr *child);
expr *new_atanh(expr *child);
expr *new_tan(expr *child);

#endif /* ELEMENTWISE_RESTRICTED_DOM_H */