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
 */
#ifndef MATRIX_BTA_H
#define MATRIX_BTA_H

#include "matrix.h"

/* Polymorphic dispatchers for C = B^T A and C = B^T diag(d) A. The output
   type depends on the input types: (PD, PD) → PD, (Sparse, PD) → PD,
   (PD, Sparse) → PD, (Sparse, Sparse) → Sparse. Dispatched via
   as_permuted_dense() on both operands. */

/* Allocate sparsity for C = B^T A. */
Matrix *BTA_matrices_alloc(Matrix *A, Matrix *B);

/* Fill out->x = B^T diag(d) A (d may be NULL for plain B^T A). out must
   have the structure produced by BTA_matrices_alloc(A, B). For the
   (Sparse, Sparse) path, the caller must ensure both operands' csc_caches
   are fresh (via refresh_csc_values) before calling; the dispatcher does
   not refresh. */
void BTDA_matrices_fill_values(Matrix *A, const double *d, Matrix *B, Matrix *C);

#endif /* MATRIX_BTA_H */
