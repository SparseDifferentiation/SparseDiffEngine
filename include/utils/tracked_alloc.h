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

/* Platform shim for "how many usable bytes are at this malloc'd pointer".
   Apple's malloc_size, glibc's malloc_usable_size, MSVC's _msize. The
   returned size is the *usable* size (may exceed the requested size by a
   few bytes of allocator rounding), but alloc and free use the same
   function so the running totals stay symmetric. */
#if defined(__APPLE__)
#include <malloc/malloc.h>
#define TRACKED_BLOCK_SIZE(p) malloc_size(p)
#elif defined(_WIN32) || defined(_WIN64)
#include <malloc.h>
#define TRACKED_BLOCK_SIZE(p) _msize(p)
#else
#include <malloc.h>
#define TRACKED_BLOCK_SIZE(p) malloc_usable_size(p)
#endif

extern size_t g_allocated_bytes; /* current live bytes */
extern size_t g_peak_bytes;      /* high-water mark since last reset */

static inline void *SP_MALLOC(size_t size)
{
    void *ptr = malloc(size);
    if (ptr)
    {
        g_allocated_bytes += TRACKED_BLOCK_SIZE(ptr);
        if (g_allocated_bytes > g_peak_bytes) g_peak_bytes = g_allocated_bytes;
    }
    return ptr;
}

static inline void *SP_CALLOC(size_t count, size_t size)
{
    void *ptr = calloc(count, size);
    if (ptr)
    {
        g_allocated_bytes += TRACKED_BLOCK_SIZE(ptr);
        if (g_allocated_bytes > g_peak_bytes) g_peak_bytes = g_allocated_bytes;
    }
    return ptr;
}

static inline void SP_FREE(void *ptr)
{
    if (ptr)
    {
        g_allocated_bytes -= TRACKED_BLOCK_SIZE(ptr);
        free(ptr);
    }
}

/* Auto-route plain malloc/calloc/free in caller translation units through
   the tracked wrappers. Defined AFTER the wrapper bodies so SP_MALLOC /
   SP_CALLOC / SP_FREE themselves still see the real stdlib symbols. */
#define malloc SP_MALLOC
#define calloc SP_CALLOC
#define free SP_FREE

#endif /* TRACKED_ALLOC_H */
