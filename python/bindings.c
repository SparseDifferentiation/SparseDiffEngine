#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

/* Include atom bindings */
#include "atoms/add.h"
#include "atoms/constant.h"
#include "atoms/exp.h"
#include "atoms/linear.h"
#include "atoms/log.h"
#include "atoms/neg.h"
#include "atoms/promote.h"
#include "atoms/sum.h"
#include "atoms/variable.h"

/* Include problem bindings */
#include "problem/constraint_forward.h"
#include "problem/gradient.h"
#include "problem/hessian.h"
#include "problem/init_derivatives.h"
#include "problem/jacobian.h"
#include "problem/make_problem.h"
#include "problem/objective_forward.h"

static int numpy_initialized = 0;

static int ensure_numpy(void)
{
    if (numpy_initialized) return 0;
    import_array1(-1);
    numpy_initialized = 1;
    return 0;
}

static PyMethodDef DNLPMethods[] = {
    {"make_variable", py_make_variable, METH_VARARGS, "Create variable node"},
    {"make_constant", py_make_constant, METH_VARARGS, "Create constant node"},
    {"make_linear", py_make_linear, METH_VARARGS, "Create linear op node"},
    {"make_log", py_make_log, METH_VARARGS, "Create log node"},
    {"make_exp", py_make_exp, METH_VARARGS, "Create exp node"},
    {"make_add", py_make_add, METH_VARARGS, "Create add node"},
    {"make_sum", py_make_sum, METH_VARARGS, "Create sum node"},
    {"make_neg", py_make_neg, METH_VARARGS, "Create neg node"},
    {"make_promote", py_make_promote, METH_VARARGS, "Create promote node"},
    {"make_problem", py_make_problem, METH_VARARGS,
     "Create problem from objective and constraints"},
    {"problem_init_derivatives", py_problem_init_derivatives, METH_VARARGS,
     "Initialize derivative structures"},
    {"problem_objective_forward", py_problem_objective_forward, METH_VARARGS,
     "Evaluate objective only"},
    {"problem_constraint_forward", py_problem_constraint_forward, METH_VARARGS,
     "Evaluate constraints only"},
    {"problem_gradient", py_problem_gradient, METH_VARARGS,
     "Compute objective gradient"},
    {"problem_jacobian", py_problem_jacobian, METH_VARARGS,
     "Compute constraint jacobian"},
    {"problem_hessian", py_problem_hessian, METH_VARARGS,
     "Compute Lagrangian Hessian"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef dnlp_module = {PyModuleDef_HEAD_INIT, "dnlp_diff_engine._core",
                                         NULL, -1, DNLPMethods};

PyMODINIT_FUNC PyInit__core(void)
{
    if (ensure_numpy() < 0) return NULL;
    return PyModule_Create(&dnlp_module);
}
