#include <stdio.h>
#include <string.h>

#include "expr.h"
#include "minunit.h"
#include "other.h"
#include "test_helpers.h"

const char *test_wsum_hess_prod_axis_one_no_zeros()
{
    /* x is 2x3 variable, global index 1, total 8 vars
     * x = [1, 2, 3, 4, 5, 6] (column-major)
     *     [1, 3, 5]
     *     [2, 4, 6]
     * f = prod_axis_one(x) = [15, 48]
     * w = [1, 2]
     *
     * Hessian is block diagonal with two 3x3 blocks (one per row):
     * Row 0 block (scale = 1 * 15 = 15):
     *   cols (c0=1, c1=3, c2=5)
     *   off-diagonals: (0,1)=5, (0,2)=3, (1,0)=5, (1,2)=1, (2,0)=3, (2,1)=1; diag=0
     * Row 1 block (scale = 2 * 48 = 96):
     *   cols (c0=2, c1=4, c2=6)
     *   off-diagonals: (0,1)=12, (0,2)=8, (1,0)=12, (1,2)=4, (2,0)=8, (2,1)=4;
     * diag=0
     */
    double u_vals[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0};
    double w_vals[2] = {1.0, 2.0};
    expr *x = new_variable(2, 3, 1, 8);
    expr *p = new_prod_axis_one(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, w_vals);

    double expected_x[18] = {/* Var 1 (row 0 of matrix): [0, 5, 3] */
                             0.0, 5.0, 3.0,
                             /* Var 2 (row 1 of matrix): [0, 12, 8] */
                             0.0, 12.0, 8.0,
                             /* Var 3 (row 0 of matrix): [5, 0, 1] */
                             5.0, 0.0, 1.0,
                             /* Var 4 (row 1 of matrix): [12, 0, 4] */
                             12.0, 0.0, 4.0,
                             /* Var 5 (row 0 of matrix): [3, 1, 0] */
                             3.0, 1.0, 0.0,
                             /* Var 6 (row 1 of matrix): [8, 4, 0] */
                             8.0, 4.0, 0.0};

    /* Row pointers (monotonically increasing for valid CSR format) */
    int expected_p[9] = {0, 0, 3, 6, 9, 12, 15, 18, 18};

    /* Column indices (each row of the matrix interacts with its own columns) */
    int expected_i[18] = {/* Var 1 (row 0): cols 1,3,5 */
                          1, 3, 5,
                          /* Var 2 (row 1): cols 2,4,6 */
                          2, 4, 6,
                          /* Var 3 (row 0): cols 1,3,5 */
                          1, 3, 5,
                          /* Var 4 (row 1): cols 2,4,6 */
                          2, 4, 6,
                          /* Var 5 (row 0): cols 1,3,5 */
                          1, 3, 5,
                          /* Var 6 (row 1): cols 2,4,6 */
                          2, 4, 6};

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 18));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 9));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 18));

    free_expr(p);
    return 0;
}

const char *test_wsum_hess_prod_axis_one_one_zero()
{
    /* x is 3x3 variable, global index 1, total 10 vars
     * x = [1, 2, 3, 4, 0, 6, 7, 8, 9] (column-major)
     *     [1, 4, 7]
     *     [2, 0, 8]
     *     [3, 6, 9]
     * f = prod_axis_one(x) = [28, 0, 162]
     * w = [1, 2, 3]
     *
     * Blocks are 3x3 (one per row):
     * Row 0 (no zeros, scale=1*28=28):
     *   cols (1,4,7): off-diag (0,1)=7, (0,2)=4, (1,0)=7, (1,2)=2, (2,0)=4, (2,1)=2
     * Row 1 (one zero at col 1, prod_nonzero=16, scale=2*16=32):
     *   only row/col 1 nonzero: (1,0)=32/2=16, (1,2)=32/8=4 (symmetric)
     * Row 2 (no zeros, scale=3*162=486):
     *   cols (3,6,9): off-diag (0,1)=27, (0,2)=18, (1,0)=27, (1,2)=9, (2,0)=18,
     * (2,1)=9
     */
    double u_vals[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 9.0};
    double w_vals[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(3, 3, 1, 10);
    expr *p = new_prod_axis_one(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, w_vals);

    double expected_x[27];
    memset(expected_x, 0, sizeof(expected_x));

    /* Var 1 (row 0): [0, 7, 4] */
    expected_x[0] = 0.0;
    expected_x[1] = 7.0;
    expected_x[2] = 4.0;

    /* Var 2 (row 1, one zero at col 1): [0, 16, 0] */
    expected_x[3] = 0.0;
    expected_x[4] = 16.0;
    expected_x[5] = 0.0;

    /* Var 3 (row 2): [0, 27, 18] */
    expected_x[6] = 0.0;
    expected_x[7] = 27.0;
    expected_x[8] = 18.0;

    /* Var 4 (row 0): [7, 0, 1] */
    expected_x[9] = 7.0;
    expected_x[10] = 0.0;
    expected_x[11] = 1.0;

    /* Var 5 (row 1, one zero at col 1): [16, 0, 4] */
    expected_x[12] = 16.0;
    expected_x[13] = 0.0;
    expected_x[14] = 4.0;

    /* Var 6 (row 2): [27, 0, 9] */
    expected_x[15] = 27.0;
    expected_x[16] = 0.0;
    expected_x[17] = 9.0;

    /* Var 7 (row 0): [4, 1, 0] */
    expected_x[18] = 4.0;
    expected_x[19] = 1.0;
    expected_x[20] = 0.0;

    /* Var 8 (row 1, one zero at col 1): [0, 4, 0] */
    expected_x[21] = 0.0;
    expected_x[22] = 4.0;
    expected_x[23] = 0.0;

    /* Var 9 (row 2): [18, 9, 0] */
    expected_x[24] = 18.0;
    expected_x[25] = 9.0;
    expected_x[26] = 0.0;

    /* Row pointers (monotonically increasing for valid CSR format) */
    int expected_p[11] = {0, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27};

    /* Column indices (each row of the matrix interacts with its own columns) */
    int expected_i[27] = {/* Var 1 (row 0): cols 1,4,7 */
                          1, 4, 7,
                          /* Var 2 (row 1): cols 2,5,8 */
                          2, 5, 8,
                          /* Var 3 (row 2): cols 3,6,9 */
                          3, 6, 9,
                          /* Var 4 (row 0): cols 1,4,7 */
                          1, 4, 7,
                          /* Var 5 (row 1): cols 2,5,8 */
                          2, 5, 8,
                          /* Var 6 (row 2): cols 3,6,9 */
                          3, 6, 9,
                          /* Var 7 (row 0): cols 1,4,7 */
                          1, 4, 7,
                          /* Var 8 (row 1): cols 2,5,8 */
                          2, 5, 8,
                          /* Var 9 (row 2): cols 3,6,9 */
                          3, 6, 9};

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 27));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 11));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 27));

    free_expr(p);
    return 0;
}

const char *test_wsum_hess_prod_axis_one_mixed_zeros()
{
    /* x is 5x3 variable, global index 1, total 16 vars
     * Rows (axis=1 products):
     *   r0: [1, 2, 1] -> no zeros, prod=2
     *   r1: [1, 0, 0] -> two zeros (cols 1,2), prod_nonzero=1
     *   r2: [1, 3, 0] -> one zero (col 2), prod_nonzero=3
     *   r3: [1, 4, 2] -> no zeros, prod=8
     *   r4: [1, 5, 3] -> no zeros, prod=15
     * w = [1, 2, 3, 4, 5]
     * Blocks are 3x3 (one per row, block diagonal):
     */
    double u_vals[16] = {0.0,
                         /* col 0 (rows 0-4): 1,1,1,1,1 */
                         1.0, 1.0, 1.0, 1.0, 1.0,
                         /* col 1: 2,0,3,4,5 */
                         2.0, 0.0, 3.0, 4.0, 5.0,
                         /* col 2: 1,0,0,2,3 */
                         1.0, 0.0, 0.0, 2.0, 3.0};
    /* Actually store column-major:
     * col0: [1,1,1,1,1]
     * col1: [2,0,3,4,5]
     * col2: [1,0,0,2,3]
     */
    double w_vals[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    expr *x = new_variable(5, 3, 1, 16);
    expr *p = new_prod_axis_one(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, w_vals);

    double expected_x[45];
    memset(expected_x, 0, sizeof(expected_x));

    /* For a 5x3 matrix with var_id=1:
     * CSR row pointers: p[i] = (i-1)*3 for i in [1,15]
     * Variables are indexed sequentially:
     *   Var 1 (matrix [0,0]): p[1]=0
     *   Var 2 (matrix [1,0]): p[2]=3
     *   Var 3 (matrix [2,0]): p[3]=6
     *   Var 4 (matrix [3,0]): p[4]=9
     *   Var 5 (matrix [4,0]): p[5]=12
     *   Var 6 (matrix [0,1]): p[6]=15
     *   Var 7 (matrix [1,1]): p[7]=18
     *   Var 8 (matrix [2,1]): p[8]=21
     *   Var 9 (matrix [3,1]): p[9]=24
     *   Var 10 (matrix [4,1]): p[10]=27
     *   Var 11 (matrix [0,2]): p[11]=30
     *   Var 12 (matrix [1,2]): p[12]=33
     *   Var 13 (matrix [2,2]): p[13]=36
     *   Var 14 (matrix [3,2]): p[14]=39
     *   Var 15 (matrix [4,2]): p[15]=42
     */

    /* Row 0 block (no zeros, scale = 1 * 2 = 2), x = [1,2,1] */
    /* Var 1 (matrix [0,0]): [0, 1, 2] */
    expected_x[0] = 0.0;
    expected_x[1] = 1.0; /* 2/(1*2) */
    expected_x[2] = 2.0; /* 2/(1*1) */

    /* Var 6 (matrix [0,1]): [1, 0, 1] */
    expected_x[15] = 1.0; /* 2/(2*1) */
    expected_x[16] = 0.0;
    expected_x[17] = 1.0; /* 2/(2*1) */

    /* Var 11 (matrix [0,2]): [2, 1, 0] */
    expected_x[30] = 2.0; /* 2/(1*1) */
    expected_x[31] = 1.0; /* 2/(1*2) */
    expected_x[32] = 0.0;

    /* Row 1 block (two zeros at cols 1,2), hess = w*prod_nonzero = 2*1 = 2 */
    /* Var 2 (matrix [1,0]): [0, 0, 0] */
    expected_x[3] = 0.0;
    expected_x[4] = 0.0;
    expected_x[5] = 0.0;

    /* Var 7 (matrix [1,1]): [0, 0, 2] */
    expected_x[18] = 0.0;
    expected_x[19] = 0.0;
    expected_x[20] = 2.0; /* (1,2) */

    /* Var 12 (matrix [1,2]): [0, 2, 0] */
    expected_x[33] = 0.0;
    expected_x[34] = 2.0; /* (2,1) */
    expected_x[35] = 0.0;

    /* Row 2 block (one zero at col 2), w_prod = 3 * 3 = 9, x = [1,3,0] */
    /* Var 3 (matrix [2,0]): [0, 0, 9] */
    expected_x[6] = 0.0;
    expected_x[7] = 0.0;
    expected_x[8] = 9.0; /* (0,2): w_prod/x[0] */

    /* Var 8 (matrix [2,1]): [0, 0, 3] */
    expected_x[21] = 0.0;
    expected_x[22] = 0.0;
    expected_x[23] = 3.0; /* (1,2): w_prod/x[1] */

    /* Var 13 (matrix [2,2]): [9, 3, 0] */
    expected_x[36] = 9.0; /* (2,0): w_prod/x[0] */
    expected_x[37] = 3.0; /* (2,1): w_prod/x[1] */
    expected_x[38] = 0.0;

    /* Row 3 block (no zeros, scale = 4 * 8 = 32), x = [1,4,2] */
    /* Var 4 (matrix [3,0]): [0, 8, 16] */
    expected_x[9] = 0.0;
    expected_x[10] = 8.0;  /* 32/(1*4) */
    expected_x[11] = 16.0; /* 32/(1*2) */

    /* Var 9 (matrix [3,1]): [8, 0, 4] */
    expected_x[24] = 8.0; /* 32/(4*1) */
    expected_x[25] = 0.0;
    expected_x[26] = 4.0; /* 32/(4*2) */

    /* Var 14 (matrix [3,2]): [16, 4, 0] */
    expected_x[39] = 16.0; /* 32/(2*1) */
    expected_x[40] = 4.0;  /* 32/(2*4) */
    expected_x[41] = 0.0;

    /* Row 4 block (no zeros, scale = 5 * 15 = 75), x = [1,5,3] */
    /* Var 5 (matrix [4,0]): [0, 15, 25] */
    expected_x[12] = 0.0;
    expected_x[13] = 15.0; /* 75/(1*5) */
    expected_x[14] = 25.0; /* 75/(1*3) */

    /* Var 10 (matrix [4,1]): [15, 0, 5] */
    expected_x[27] = 15.0; /* 75/(5*1) */
    expected_x[28] = 0.0;
    expected_x[29] = 5.0; /* 75/(5*3) */

    /* Var 15 (matrix [4,2]): [25, 5, 0] */
    expected_x[42] = 25.0; /* 75/(3*1) */
    expected_x[43] = 5.0;  /* 75/(3*5) */
    expected_x[44] = 0.0;

    /* Row pointers (monotonically increasing for valid CSR format) */
    int expected_p[17] = {0,  0,  3,  6,  9,  12, 15, 18, 21,
                          24, 27, 30, 33, 36, 39, 42, 45};

    /* Column indices (each row of the matrix interacts with its own columns) */
    int expected_i[45];
    for (int var_idx = 0; var_idx < 15; var_idx++)
    {
        int matrix_row = var_idx % 5; /* which row of the 5x3 matrix */
        int nnz_start = var_idx * 3;
        /* All columns from matrix_row */
        expected_i[nnz_start + 0] = 1 + matrix_row + 0 * 5;
        expected_i[nnz_start + 1] = 1 + matrix_row + 1 * 5;
        expected_i[nnz_start + 2] = 1 + matrix_row + 2 * 5;
    }

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 45));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 17));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 45));

    free_expr(p);
    return 0;
}
const char *test_wsum_hess_prod_axis_one_2x2()
{
    /* x is 2x2 variable, global index 0, total 4 vars
     * x = [2, 1, 3, 2] (column-major)
     *     [2, 3]
     *     [1, 2]
     * f = prod_axis_one(x) = [2*3, 1*2] = [6, 2]
     * w = [1, 1]
     *
     * For a 2x2 matrix with var_id=0:
     *   Var 0 (matrix [0,0]): p[0]=0, stores columns [0,1]
     *   Var 1 (matrix [1,0]): p[1]=2, stores columns [0,1]
     *   Var 2 (matrix [0,1]): p[2]=4, stores columns [0,1]
     *   Var 3 (matrix [1,1]): p[3]=6, stores columns [0,1]
     *
     * Row 0 Hessian (no zeros, x=[2,3], scale=1*6=6):
     *   (0,0)=0, (0,1)=1  -> stored at Var 0: [0, 1]
     *   (1,0)=1, (1,1)=0  -> stored at Var 2: [1, 0]
     *
     * Row 1 Hessian (no zeros, x=[1,2], scale=1*2=2):
     *   (0,0)=0, (0,1)=1  -> stored at Var 1: [0, 1]
     *   (1,0)=1, (1,1)=0  -> stored at Var 3: [1, 0]
     */
    double u_vals[4] = {2.0, 1.0, 3.0, 2.0};
    double w_vals[2] = {1.0, 1.0};
    expr *x = new_variable(2, 2, 0, 4);
    expr *p = new_prod_axis_one(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, w_vals);

    /* Expected sparse structure (nnz = 8) */
    double expected_x[8] = {0.0, 1.0,  /* Var 0 */
                            0.0, 1.0,  /* Var 1 */
                            1.0, 0.0,  /* Var 2 */
                            1.0, 0.0}; /* Var 3 */

    /* Row pointers (each row has 2 nnz) */
    int expected_p[5] = {0, 2, 4, 6, 8};

    /* Column indices (each variable stores columns for its matrix row) */
    int expected_i[8] = {0, 2,  /* Var 0 (row 0) */
                         1, 3,  /* Var 1 (row 1) */
                         0, 2,  /* Var 2 (row 0) */
                         1, 3}; /* Var 3 (row 1) */

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 8));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 5));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 8));

    free_expr(p);
    return 0;
}
