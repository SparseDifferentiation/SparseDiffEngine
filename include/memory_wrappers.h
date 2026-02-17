#ifndef MEMORY_WRAPPERS_H
#define MEMORY_WRAPPERS_H

#include <stdlib.h>

#define FREE_AND_NULL(p)                                                            \
    do                                                                              \
    {                                                                               \
        free(p);                                                                    \
        (p) = NULL;                                                                 \
    } while (0)

#endif /* MEMORY_WRAPPERS_H */
