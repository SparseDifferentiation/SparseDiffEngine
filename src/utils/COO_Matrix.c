#include "utils/COO_Matrix.h"
#include <stdlib.h>

COO_Matrix *new_coo_matrix(int nnz)
{
    COO_Matrix *matrix = (COO_Matrix *) malloc(sizeof(COO_Matrix));
    if (!matrix) return NULL;

    matrix->nnz = nnz;
    matrix->rows = (int *) calloc(nnz, sizeof(int));
    matrix->cols = (int *) calloc(nnz, sizeof(int));
    matrix->vals = (double *) calloc(nnz, sizeof(double));

    if (!matrix->rows || !matrix->cols || !matrix->vals)
    {
        free(matrix->rows);
        free(matrix->cols);
        free(matrix->vals);
        free(matrix);
        return NULL;
    }

    return matrix;
}

void free_coo_matrix(COO_Matrix *matrix)
{
    if (!matrix) return;
    free(matrix->rows);
    free(matrix->cols);
    free(matrix->vals);
    free(matrix);
}
