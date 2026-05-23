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
    Used to track total live bytes */
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

/* All allocations in src/ must go through these wrappers (and pair sp_free
   with sp_malloc / sp_calloc). Tests may use plain malloc/free — those
   bytes are simply not tracked. */
static inline void *sp_malloc(size_t size)
{
    void *ptr = malloc(size);
    if (ptr)
    {
        g_allocated_bytes += TRACKED_BLOCK_SIZE(ptr);
        if (g_allocated_bytes > g_peak_bytes)
        {
            g_peak_bytes = g_allocated_bytes;
        }
    }
    return ptr;
}

static inline void *sp_calloc(size_t count, size_t size)
{
    void *ptr = calloc(count, size);
    if (ptr)
    {
        g_allocated_bytes += TRACKED_BLOCK_SIZE(ptr);
        if (g_allocated_bytes > g_peak_bytes)
        {
            g_peak_bytes = g_allocated_bytes;
        }
    }
    return ptr;
}

static inline void sp_free(void *ptr)
{
    if (ptr)
    {
        g_allocated_bytes -= TRACKED_BLOCK_SIZE(ptr);
        free(ptr);
    }
}

static inline void *sp_realloc(void *ptr, size_t size)
{
    size_t old_block_size = ptr ? TRACKED_BLOCK_SIZE(ptr) : 0;
    void *new_ptr = realloc(ptr, size);
    if (new_ptr)
    {
        g_allocated_bytes =
            g_allocated_bytes - old_block_size + TRACKED_BLOCK_SIZE(new_ptr);
        if (g_allocated_bytes > g_peak_bytes)
        {
            g_peak_bytes = g_allocated_bytes;
        }
    }
    /* realloc returning NULL means failure — original block is still live
       per C standard, so no counter adjustment. */
    return new_ptr;
}

#endif /* TRACKED_ALLOC_H */
