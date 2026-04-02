#include "utils/JMatrix.h"
#include "utils/cblas_wrapper.h"
#include "utils/tracked_alloc.h"
#include <stdlib.h>
#include <string.h>

/* Allocate and copy block */
static Block *copy_block(const Block *src)
{
    Block *b = (Block *) SP_MALLOC(sizeof(Block));
    b->m = src->m;
    b->n = src->n;
    b->row0 = src->row0;
    b->col0 = src->col0;
    int nnz = src->m * src->n;
    b->data = (double *) SP_MALLOC(nnz * sizeof(double));
    memcpy(b->data, src->data, nnz * sizeof(double));
    return b;
}

/* Free a single Block (data + struct). */
static void free_block(Block *b)
{
    if (b == NULL) return;
    free(b->data);
    free(b);
}

/* ----------------------------------------------------------------
 *                         Lifecycle
 * ---------------------------------------------------------------- */

JMatrix *new_jm(int m, int n, int n_blocks)
{
    JMatrix *jm = (JMatrix *) SP_CALLOC(1, sizeof(JMatrix));
    jm->m = m;
    jm->n = n;

    if (n_blocks > 0)
    {
        jm->blocks = (Block **) SP_CALLOC(n_blocks, sizeof(Block *));
        jm->n_blocks_alloc = n_blocks;
    }

    return jm;
}

void jm_update_csr(JMatrix *jm, const CSR_Matrix *csr)
{
    /* free to avoid memory leaks */
    free_csr_matrix(jm->csr);
    jm->csr = new_csr(csr);
}

void jm_append_block(JMatrix *jm, const Block *block)
{
    if (jm->n_blocks >= jm->n_blocks_alloc)
    {
        int new_alloc = (jm->n_blocks_alloc == 0) ? 1 : jm->n_blocks_alloc * 2;
        Block **new_arr = (Block **) SP_CALLOC(new_alloc, sizeof(Block *));

        if (jm->blocks != NULL)
        {
            memcpy(new_arr, jm->blocks, jm->n_blocks * sizeof(Block *));
            free(jm->blocks);
        }

        jm->blocks = new_arr;
        jm->n_blocks_alloc = new_alloc;
    }

    jm->blocks[jm->n_blocks] = copy_block(block);
    jm->n_blocks++;
}

JMatrix *new_jm_from_csr(const CSR_Matrix *csr)
{
    JMatrix *jm = new_jm(csr->m, csr->n, 0);
    jm->csr = new_csr(csr);
    return jm;
}

JMatrix *new_jm_copy_sparsity(const JMatrix *src)
{
    JMatrix *jm = new_jm(src->m, src->n, src->n_blocks);

    if (src->csr != NULL)
    {
        jm->csr = new_csr_copy_sparsity(src->csr);
    }

    for (int k = 0; k < src->n_blocks; k++)
    {
        const Block *sb = src->blocks[k];
        Block *b = (Block *) SP_MALLOC(sizeof(Block));
        b->m = sb->m;
        b->n = sb->n;
        b->row0 = sb->row0;
        b->col0 = sb->col0;
        b->data = (double *) SP_MALLOC(b->m * b->n * sizeof(double));

        jm->blocks[k] = b;
        jm->n_blocks++;
    }

    return jm;
}

/* ----------------------------------------------------------------
 *                       Core operations
 * ---------------------------------------------------------------- */

/* C = diag(d) @ A */
void DA_jm_fill_values(const double *d, const JMatrix *A, JMatrix *C)
{
    /* CSR part */
    if (A->csr != NULL && C->csr != NULL)
    {
        DA_fill_values(d, A->csr, C->csr);
    }

    const Block *src;
    Block *dst;

    /* Dense blocks */
    for (int k = 0; k < A->n_blocks; k++)
    {
        src = A->blocks[k];
        dst = C->blocks[k];
        memcpy(dst->data, src->data, src->m * src->n * sizeof(double));
        for (int r = 0; r < src->m; r++)
        {
            cblas_dscal(src->n, d[src->row0 + r], dst->data + r * src->n, 1);
        }
    }
}

/* ----------------------------------------------------------------
 *                         Destructor
 * ---------------------------------------------------------------- */

void jm_free(JMatrix *jm)
{
    if (jm == NULL) return;

    free_csr_matrix(jm->csr);

    for (int k = 0; k < jm->n_blocks; k++)
    {
        free_block(jm->blocks[k]);
    }
    free(jm->blocks);

    free(jm->dwork);
    free(jm);
}
