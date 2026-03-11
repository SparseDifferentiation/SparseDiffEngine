#ifndef CBLAS_WRAPPER_H
#define CBLAS_WRAPPER_H

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#endif /* CBLAS_WRAPPER_H */
