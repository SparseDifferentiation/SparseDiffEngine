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
#include "utils/matrix_sum.h"
#include "utils/CSR_sum.h"

void sum_matrices_alloc(Matrix *A, Matrix *B, Matrix *C)
{
    CSR_Matrix *cc = C->to_csr(C);
    sum_csr_alloc(A->to_csr(A), B->to_csr(B), cc);
    C->nnz = cc->nnz;
}

void sum_matrices_fill_values(Matrix *A, Matrix *B, Matrix *C)
{
    sum_csr_fill_values(A->to_csr(A), B->to_csr(B), C->to_csr(C));
}

void sum_scaled_matrices_fill_values(Matrix *A, Matrix *B, Matrix *C,
                                     const double *d1, const double *d2)
{
    sum_scaled_csr_matrices_fill_values(A->to_csr(A), B->to_csr(B), C->to_csr(C),
                                        d1, d2);
}
