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
#ifndef TRACKED_ALLOC_H
#define TRACKED_ALLOC_H

#include <stddef.h>
#include <stdlib.h>

extern size_t g_allocated_bytes;

static inline void *SP_MALLOC(size_t size)
{
    void *ptr = malloc(size);
    if (ptr) g_allocated_bytes += size;
    return ptr;
}

static inline void *SP_CALLOC(size_t count, size_t size)
{
    void *ptr = calloc(count, size);
    if (ptr) g_allocated_bytes += count * size;
    return ptr;
}

#endif /* TRACKED_ALLOC_H */
