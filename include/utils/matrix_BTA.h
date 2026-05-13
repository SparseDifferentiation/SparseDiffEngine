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

/* Polymorphic dispatchers for C = BT @ A and C = BT @ diag(d) @ A. The output
   type depends on the input types: (PD, PD) → PD, (Sparse, PD) → PD,
   (PD, Sparse) → PD, (Sparse, Sparse) → Sparse. (Here PD = permuted_dense.)

   Contract: neither function touches sparse_matrix internals. The caller must,
   before calling either function, ensure each Sparse operand's csc_cache
   exists (sparse_matrix_ensure_csc_cache). Before BTDA_matrices_fill_values
   the caller must also refresh the cache values (refresh_csc_values). */

/* Allocate sparsity for C = BT @ A. */
matrix *BTA_matrices_alloc(matrix *A, matrix *B);

/* Fill values of C = BT @ diag(d) @ A. */
void BTDA_matrices_fill_values(matrix *A, const double *d, matrix *B, matrix *C);

#endif /* MATRIX_BTA_H */
