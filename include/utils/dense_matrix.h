#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include "matrix.h"

/* Dense matrix (row-major) */
typedef struct Dense_Matrix
{
    Matrix base;
    double *x;
    double *work; /* scratch buffer, length n */
} Dense_Matrix;

/* Constructors */
Matrix *new_dense_matrix(int m, int n, const double *data);

/* Transpose helper */
Matrix *dense_matrix_trans(const Dense_Matrix *self);

#endif /* DENSE_MATRIX_H */
