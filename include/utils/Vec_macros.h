/*
 * Copyright 2025 Daniel Cederberg
 *
 * This file is part of the PSLP project (LP Presolver).
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

#ifndef VEC_MACROS_H
#define VEC_MACROS_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Macro to define a generic vector class
#define DEFINE_VECTOR(TYPE, TYPE_NAME)                                              \
    typedef struct TYPE_NAME##Vec                                                   \
    {                                                                               \
        TYPE *data;                                                                 \
        size_t len;                                                                 \
        size_t capacity;                                                            \
    } TYPE_NAME##Vec;                                                               \
                                                                                    \
    __attribute__((unused)) static TYPE_NAME##Vec *TYPE_NAME##Vec_new(              \
        size_t capacity)                                                            \
    {                                                                               \
        assert(capacity > 0);                                                       \
        TYPE_NAME##Vec *vec = (TYPE_NAME##Vec *) malloc(sizeof(TYPE_NAME##Vec));    \
        if (vec == NULL) return NULL;                                               \
        vec->data = (TYPE *) malloc(capacity * sizeof(TYPE));                       \
        if (vec->data == NULL)                                                      \
        {                                                                           \
            free(vec);                                                              \
            return NULL;                                                            \
        }                                                                           \
                                                                                    \
        vec->len = 0;                                                               \
        vec->capacity = capacity;                                                   \
        return vec;                                                                 \
    }                                                                               \
                                                                                    \
    static inline void TYPE_NAME##Vec_free(TYPE_NAME##Vec *vec)                     \
    {                                                                               \
        free(vec->data);                                                            \
        free(vec);                                                                  \
    }                                                                               \
                                                                                    \
    static inline void TYPE_NAME##Vec_clear_no_resize(TYPE_NAME##Vec *vec)          \
    {                                                                               \
        vec->len = 0;                                                               \
    }                                                                               \
                                                                                    \
    static inline void TYPE_NAME##Vec_append(TYPE_NAME##Vec *vec, TYPE value)       \
    {                                                                               \
        if (vec->len >= vec->capacity)                                              \
        {                                                                           \
            vec->capacity *= 2;                                                     \
            assert(vec->capacity > 0);                                              \
            TYPE *temp = (TYPE *) realloc(vec->data, vec->capacity * sizeof(TYPE)); \
            if (temp == NULL)                                                       \
            {                                                                       \
                TYPE_NAME##Vec_free(vec);                                           \
                fprintf(stderr, "Error: realloc failed\n");                         \
                exit(1);                                                            \
            }                                                                       \
                                                                                    \
            vec->data = temp;                                                       \
        }                                                                           \
                                                                                    \
        vec->data[vec->len++] = value;                                              \
    }                                                                               \
                                                                                    \
    static inline void TYPE_NAME##Vec_append_array(TYPE_NAME##Vec *vec,             \
                                                   const TYPE *values, size_t n)    \
    {                                                                               \
        if (vec->len + n > vec->capacity)                                           \
        {                                                                           \
            size_t new_capacity = vec->capacity > 0 ? vec->capacity : 1;            \
            while (vec->len + n > new_capacity)                                     \
            {                                                                       \
                new_capacity *= 2;                                                  \
            }                                                                       \
                                                                                    \
            TYPE *temp = (TYPE *) realloc(vec->data, new_capacity * sizeof(TYPE));  \
            if (temp == NULL)                                                       \
            {                                                                       \
                TYPE_NAME##Vec_free(vec);                                           \
                fprintf(stderr, "Error: realloc failed\n");                         \
                exit(1);                                                            \
            }                                                                       \
                                                                                    \
            vec->data = temp;                                                       \
            vec->capacity = new_capacity;                                           \
        }                                                                           \
                                                                                    \
        memcpy(vec->data + vec->len, values, n * sizeof(TYPE));                     \
        vec->len += n;                                                              \
    }                                                                               \
    __attribute__((unused)) static int TYPE_NAME##Vec_contains(                     \
        const TYPE_NAME##Vec *vec, TYPE value)                                      \
    {                                                                               \
        for (size_t i = 0; i < vec->len; ++i)                                       \
        {                                                                           \
            if (vec->data[i] == value)                                              \
            {                                                                       \
                return 1; /* Element found */                                       \
            }                                                                       \
        }                                                                           \
        return 0; /* Element not found */                                           \
    }

#endif
