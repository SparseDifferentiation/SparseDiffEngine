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
#ifndef MATRIX_SUM_H
#define MATRIX_SUM_H

#include "matrix.h"

/* Polymorphic wrappers for allocating C = A + B. Right now we always
   convert to CSR matrices internally for the sum. */
void sum_matrices_alloc(matrix *A, matrix *B, matrix *C);

/* Fill values of C = A + B. Uses CSR matrices internally. */
void sum_matrices_fill_values(matrix *A, matrix *B, matrix *C);

/* Fill values of C = diag(d1) * A + diag(d2) * B. Uses CSR matrices internally. */
void sum_scaled_matrices_fill_values(matrix *A, matrix *B, matrix *C,
                                     const double *d1, const double *d2);

#endif /* MATRIX_SUM_H */
