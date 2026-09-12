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
#ifndef OLD_MINI_NUMPY_H
#define OLD_MINI_NUMPY_H

/* Example: a = [1, 2], len = 2, tiles = 3, result = [1, 2, 1, 2, 1, 2].
   Retired from utils/mini_numpy.h with the CSR broadcast kernel, its last
   caller. */
void tile_int(int *result, const int *a, int len, int tiles);

#endif /* OLD_MINI_NUMPY_H */
