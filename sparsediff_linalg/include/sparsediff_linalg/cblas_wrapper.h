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
#ifndef CBLAS_WRAPPER_H
#define CBLAS_WRAPPER_H

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
/* Apple's Accelerate/LAPACK/Sparse headers trip a large, SDK-dependent set
 * of -Wpedantic/GNU-extension warnings (nullability qualifiers, zero-arg
 * variadic macros, etc.) that we cannot fix and do not want in our build
 * output. Treat everything included below as a system header so all of its
 * warnings are suppressed, while our own code keeps full -Wpedantic
 * strictness. This file contains nothing but the include, so the marker is
 * effectively scoped to Accelerate. */
#pragma clang system_header
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#endif /* CBLAS_WRAPPER_H */
