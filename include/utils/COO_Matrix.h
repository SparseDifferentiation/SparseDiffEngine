#ifndef MATRIX_H
#define MATRIX_H

/* Coordinate (COO) format sparse matrix */
typedef struct COO_Matrix
{
    int nnz;
    int *rows;
    int *cols;
    double *vals;
} COO_Matrix;

COO_Matrix *new_coo_matrix(int nnz);
void free_coo_matrix(COO_Matrix *matrix);

#endif /* MATRIX_H */
