#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "numerical_diff.h"

#define ABS_TOL 1e-6
#define REL_TOL 1e-6

static int is_close(double a, double b)
{
    return fabs(a - b) <= fmax(ABS_TOL, REL_TOL * fmax(fabs(a), fabs(b)));
}

static void csr_to_dense(const CSR_Matrix *A, double *dense)
{
    for (int row = 0; row < A->m; row++)
    {
        for (int idx = A->p[row]; idx < A->p[row + 1]; idx++)
        {
            dense[row * A->n + A->i[idx]] = A->x[idx];
        }
    }
}

double *numerical_jacobian(expr *node, const double *u, double h)
{
    int m = node->size;
    int n = node->n_vars;
    double inv_2h = 1.0 / (2.0 * h);

    double *J = calloc((size_t) m * n, sizeof(double));
    double *u_work = malloc(n * sizeof(double));
    double *f_plus = malloc(m * sizeof(double));
    double *f_minus = malloc(m * sizeof(double));

    memcpy(u_work, u, n * sizeof(double));

    for (int j = 0; j < n; j++)
    {
        u_work[j] = u[j] + h;
        node->forward(node, u_work);
        memcpy(f_plus, node->value, m * sizeof(double));

        u_work[j] = u[j] - h;
        node->forward(node, u_work);
        memcpy(f_minus, node->value, m * sizeof(double));

        u_work[j] = u[j];

        for (int k = 0; k < m; k++)
        {
            J[k * n + j] = (f_plus[k] - f_minus[k]) * inv_2h;
        }
    }

    free(f_minus);
    free(f_plus);
    free(u_work);
    return J;
}

int check_jacobian(expr *node, const double *u, double h)
{
    int m = node->size;
    int n = node->n_vars;

    node->jacobian_init(node);
    node->forward(node, u);
    node->eval_jacobian(node);

    double *J_num = numerical_jacobian(node, u, h);

    /* restore expression state after perturbations */
    node->forward(node, u);

    double *J_analytical = calloc((size_t) m * n, sizeof(double));
    csr_to_dense(node->jacobian, J_analytical);

    int result = 1;
    for (int i = 0; i < m * n; i++)
    {
        if (!is_close(J_analytical[i], J_num[i]))
        {
            int row = i / n;
            int col = i % n;
            printf("  check_jacobian FAILED at (%d, %d):"
                   " analytical=%e, numerical=%e\n",
                   row, col, J_analytical[i], J_num[i]);
            result = 0;
            break;
        }
    }

    free(J_analytical);
    free(J_num);
    return result;
}
