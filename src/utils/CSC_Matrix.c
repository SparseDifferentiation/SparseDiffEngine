#include "utils/CSC_Matrix.h"
#include "utils/iVec.h"
#include <stdlib.h>
#include <string.h>

CSC_Matrix *new_csc_matrix(int m, int n, int nnz)
{
    CSC_Matrix *matrix = (CSC_Matrix *) malloc(sizeof(CSC_Matrix));
    if (!matrix) return NULL;

    matrix->p = (int *) malloc((n + 1) * sizeof(int));
    matrix->i = (int *) malloc(nnz * sizeof(int));
    matrix->x = (double *) malloc(nnz * sizeof(double));

    if (!matrix->p || !matrix->i || !matrix->x)
    {
        free(matrix->p);
        free(matrix->i);
        free(matrix->x);
        free(matrix);
        return NULL;
    }

    matrix->m = m;
    matrix->n = n;
    matrix->nnz = nnz;

    return matrix;
}

void free_csc_matrix(CSC_Matrix *matrix)
{
    if (matrix)
    {
        free(matrix->p);
        free(matrix->i);
        free(matrix->x);
        free(matrix);
    }
}

CSR_Matrix *ATA_alloc(const CSC_Matrix *A)
{
    /* A is m x n, A^T A is n x n */
    int n = A->n;
    int m = A->m;
    int nnz = 0;
    int i, j, ii, jj;

    /* row ptr and column idxs for upper triangular part of C = A^T A */
    int *Cp = (int *) malloc((n + 1) * sizeof(int));
    iVec *Ci = iVec_new(m);
    Cp[0] = 0;

    /* compute sparsity pattern, only storing upper triangular part */
    for (i = 0; i < n; i++)
    {
        /* check if Cij != 0 */
        for (j = i; j < n; j++)
        {
            ii = A->p[i];
            jj = A->p[j];

            while (ii < A->p[i + 1] && jj < A->p[j + 1])
            {
                if (A->i[ii] == A->i[jj])
                {
                    nnz += (j != i) ? 2 : 1;
                    iVec_append(Ci, j);
                    break;
                }
                else if (A->i[ii] < A->i[jj])
                {
                    ii++;
                }
                else
                {
                    jj++;
                }
            }
        }
        Cp[i + 1] = Ci->len;
    }

    /* Allocate C and symmetrize it */
    CSR_Matrix *C = new_csr_matrix(n, n, nnz);

    /* TODO: do we need to symmetrize here? If we are a bit careful with symmetry
       throughout the implementation we can skip this step. */
    symmetrize_csr(Cp, Ci->data, n, C);

    /* free workspace */
    free(Cp);
    iVec_free(Ci);

    return C;
}
