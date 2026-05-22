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
#ifndef OLD_MATRIX_BTA_H
#define OLD_MATRIX_BTA_H

#include "utils/matrix.h"
#include "utils/stacked_pd.h"

/* Legacy polymorphic dispatcher for C = B @ A where B is a stacked_pd and A
   is any matrix type (PD, stacked_pd, or sparse_matrix). C is always a
   stacked_pd. No production atom currently invokes this — see
   include/old-code/old_stacked_pd_linalg.h for the rationale.

   For the sparse-A branch the dispatcher ensures sm_A->csc_cache structure
   exists at alloc time but the caller must refresh values via
   sm_A->refresh_csc_values before calling _fill_values. */
matrix *BA_spd_matrices_alloc(const stacked_pd *B, matrix *A);
void BA_spd_matrices_fill_values(const stacked_pd *B, const matrix *A,
                                 stacked_pd *C);

#endif /* OLD_MATRIX_BTA_H */
