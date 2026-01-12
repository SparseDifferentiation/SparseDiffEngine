#ifndef TIMER_H
#define TIMER_H

#include "time.h"

typedef struct
{
    struct timespec start, end;
} Timer;

// Macro to compute elapsed time in seconds
#define GET_ELAPSED_SECONDS(timer)                                                  \
    (((timer).end.tv_sec - (timer).start.tv_sec) +                                  \
     ((double) ((timer).end.tv_nsec - (timer).start.tv_nsec) * 1e-9))

#define RUN_AND_TIME(func, timer, time_variable, result_var, ...)                   \
    do                                                                              \
    {                                                                               \
        clock_gettime(CLOCK_MONOTONIC, &timer.start);                               \
        (result_var) = func(__VA_ARGS__);                                           \
        clock_gettime(CLOCK_MONOTONIC, &timer.end);                                 \
        (time_variable) += GET_ELAPSED_SECONDS(timer);                              \
    } while (0)

#endif // TIMER_H