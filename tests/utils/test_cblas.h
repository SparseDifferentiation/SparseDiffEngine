#ifndef TEST_CBLAS_H
#define TEST_CBLAS_H

#include "minunit.h"
#include "utils/cblas_wrapper.h"
#include <math.h>

static char *test_cblas_ddot(void)
{
    double x[] = {1.0, 2.0, 3.0, 4.0};
    double y[] = {5.0, 6.0, 7.0, 8.0};
    double result = cblas_ddot(4, x, 1, y, 1);
    double expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;
    mu_assert("test_cblas_ddot: wrong dot product", fabs(result - expected) < 1e-12);
    return 0;
}

#endif /* TEST_CBLAS_H */
