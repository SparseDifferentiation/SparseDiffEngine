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
