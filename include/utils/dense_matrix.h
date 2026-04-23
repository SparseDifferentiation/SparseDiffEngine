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
#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include "matrix.h"

/* Dense matrix (row-major) */
typedef struct Dense_Matrix
{
    Matrix base;
    double *x;
    double *work; /* scratch buffer, length n */
} Dense_Matrix;

/* Constructors. If data is NULL, the value buffer is allocated but left
   uninitialized; otherwise m*n entries are copied from data. */
Matrix *new_dense_matrix(int m, int n, const double *data);

/* Transpose helper */
Matrix *dense_matrix_trans(const Dense_Matrix *self);

void A_transpose(double *AT, const double *A, int m, int n);

#endif /* DENSE_MATRIX_H */
