#ifndef JMATRIX_H
#define JMATRIX_H

#include "CSR_Matrix.h"

/* A row-major m x n dense block starting at (row0, col0)
   within a larger matrix. */
typedef struct
{
    double *data;
    int m, n;
    int row0, col0;
} Block;

/* A matrix stored as a sparse CSR part plus an array of
   dense sub-blocks. */
typedef struct
{
    int m, n;
    CSR_Matrix *csr;
    Block **blocks;
    int n_blocks;
    int n_blocks_alloc;
    double *dwork;
    int dwork_len;
} JMatrix;

/* To create a JMatrix we use new_jm (which pre-allocates
   n_blocks slots) and then set the CSR part (jm_update_csr)
   and append dense blocks (jm_append_block). Pass n_blocks=0
   if the count is not known upfront. */
JMatrix *new_jm(int m, int n, int n_blocks);
void jm_free(JMatrix *jm);
void jm_update_csr(JMatrix *jm, const CSR_Matrix *csr);
void jm_append_block(JMatrix *jm, const Block *block);
JMatrix *new_jm_from_csr(const CSR_Matrix *csr);

/* Copy sparsity of CSR part and allocate dense blocks with
   matching dimensions. All values are uninitialised. */
JMatrix *new_jm_copy_sparsity(const JMatrix *src);

/* ----------------------------------------------------------------
 *                       Core operations
 * ---------------------------------------------------------------- */

/* Fill values of C = diag(d) @ A. */
void DA_jm_fill_values(const double *d, const JMatrix *A, JMatrix *C);

/* Allocate H = A^T D A (structure only). */
JMatrix *jm_ATDA_alloc(const JMatrix *A);

/* Fill H = A^T D A values. */
void jm_ATDA_fill(const JMatrix *A, const double *d, JMatrix *H);

/* Allocate C = B^T D A (structure only). */
JMatrix *jm_BTDA_alloc(const JMatrix *B, const JMatrix *A);

/* Fill C = B^T D A values. */
void jm_BTDA_fill(const JMatrix *B, const JMatrix *A, const double *d, JMatrix *C);

#endif /* JMATRIX_H */
