#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdbool.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"
#include "problem.h"

// Capsule name for expr* pointers
#define EXPR_CAPSULE_NAME "DNLP_EXPR"
#define PROBLEM_CAPSULE_NAME "DNLP_PROBLEM"

static int numpy_initialized = 0;

static int ensure_numpy(void)
{
    if (numpy_initialized) return 0;
    import_array1(-1);
    numpy_initialized = 1;
    return 0;
}

static void expr_capsule_destructor(PyObject *capsule)
{
    expr *node = (expr *) PyCapsule_GetPointer(capsule, EXPR_CAPSULE_NAME);
    if (node)
    {
        free_expr(node);
    }
}

static PyObject *py_make_variable(PyObject *self, PyObject *args)
{
    int d1, d2, var_id, n_vars;
    if (!PyArg_ParseTuple(args, "iiii", &d1, &d2, &var_id, &n_vars))
    {
        return NULL;
    }

    expr *node = new_variable(d1, d2, var_id, n_vars);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create variable node");
        return NULL;
    }
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

static PyObject *py_make_constant(PyObject *self, PyObject *args)
{
    int d1, d2, n_vars;
    PyObject *values_obj;
    if (!PyArg_ParseTuple(args, "iiiO", &d1, &d2, &n_vars, &values_obj))
    {
        return NULL;
    }

    PyArrayObject *values_array =
        (PyArrayObject *) PyArray_FROM_OTF(values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!values_array)
    {
        return NULL;
    }

    expr *node =
        new_constant(d1, d2, n_vars, (const double *) PyArray_DATA(values_array));
    Py_DECREF(values_array);

    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create constant node");
        return NULL;
    }
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

static PyObject *py_make_linear(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    PyObject *data_obj, *indices_obj, *indptr_obj;
    int m, n;
    if (!PyArg_ParseTuple(args, "OOOOii", &child_capsule, &data_obj, &indices_obj,
                          &indptr_obj, &m, &n))
    {
        return NULL;
    }

    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    PyArrayObject *data_array =
        (PyArrayObject *) PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *indices_array =
        (PyArrayObject *) PyArray_FROM_OTF(indices_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *indptr_array =
        (PyArrayObject *) PyArray_FROM_OTF(indptr_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);

    if (!data_array || !indices_array || !indptr_array)
    {
        Py_XDECREF(data_array);
        Py_XDECREF(indices_array);
        Py_XDECREF(indptr_array);
        return NULL;
    }

    int nnz = (int) PyArray_SIZE(data_array);
    CSR_Matrix *A = new_csr_matrix(m, n, nnz);
    memcpy(A->x, PyArray_DATA(data_array), nnz * sizeof(double));
    memcpy(A->i, PyArray_DATA(indices_array), nnz * sizeof(int));
    memcpy(A->p, PyArray_DATA(indptr_array), (m + 1) * sizeof(int));

    Py_DECREF(data_array);
    Py_DECREF(indices_array);
    Py_DECREF(indptr_array);

    expr *node = new_linear(child, A);
    free_csr_matrix(A);

    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create linear node");
        return NULL;
    }
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

static PyObject *py_make_log(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    if (!PyArg_ParseTuple(args, "O", &child_capsule))
    {
        return NULL;
    }
    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_log(child);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create log node");
        return NULL;
    }
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

static PyObject *py_make_exp(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    if (!PyArg_ParseTuple(args, "O", &child_capsule))
    {
        return NULL;
    }
    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_exp(child);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create exp node");
        return NULL;
    }
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

static PyObject *py_make_add(PyObject *self, PyObject *args)
{
    PyObject *left_capsule, *right_capsule;
    if (!PyArg_ParseTuple(args, "OO", &left_capsule, &right_capsule))
    {
        return NULL;
    }
    expr *left = (expr *) PyCapsule_GetPointer(left_capsule, EXPR_CAPSULE_NAME);
    expr *right = (expr *) PyCapsule_GetPointer(right_capsule, EXPR_CAPSULE_NAME);
    if (!left || !right)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_add(left, right);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create add node");
        return NULL;
    }
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

static PyObject *py_make_sum(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    int axis;
    if (!PyArg_ParseTuple(args, "Oi", &child_capsule, &axis))
    {
        return NULL;
    }
    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_sum(child, axis);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create sum node");
        return NULL;
    }
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

static PyObject *py_make_neg(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    if (!PyArg_ParseTuple(args, "O", &child_capsule))
    {
        return NULL;
    }
    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_neg(child);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create neg node");
        return NULL;
    }
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

static PyObject *py_make_promote(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    int d1, d2;
    if (!PyArg_ParseTuple(args, "Oii", &child_capsule, &d1, &d2))
    {
        return NULL;
    }
    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_promote(child, d1, d2);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create promote node");
        return NULL;
    }
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

static PyObject *py_forward(PyObject *self, PyObject *args)
{
    PyObject *node_capsule;
    PyObject *u_obj;
    if (!PyArg_ParseTuple(args, "OO", &node_capsule, &u_obj))
    {
        return NULL;
    }

    expr *node = (expr *) PyCapsule_GetPointer(node_capsule, EXPR_CAPSULE_NAME);
    if (!node)
    {
        PyErr_SetString(PyExc_ValueError, "invalid node capsule");
        return NULL;
    }

    PyArrayObject *u_array =
        (PyArrayObject *) PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!u_array)
    {
        return NULL;
    }

    node->forward(node, (const double *) PyArray_DATA(u_array));

    npy_intp size = node->size;
    PyObject *out = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
    if (!out)
    {
        Py_DECREF(u_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject *) out), node->value, size * sizeof(double));

    Py_DECREF(u_array);
    return out;
}

static PyObject *py_jacobian(PyObject *self, PyObject *args)
{
    PyObject *node_capsule;
    PyObject *u_obj;
    if (!PyArg_ParseTuple(args, "OO", &node_capsule, &u_obj))
    {
        return NULL;
    }

    expr *node = (expr *) PyCapsule_GetPointer(node_capsule, EXPR_CAPSULE_NAME);
    if (!node)
    {
        PyErr_SetString(PyExc_ValueError, "invalid node capsule");
        return NULL;
    }

    PyArrayObject *u_array =
        (PyArrayObject *) PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!u_array)
    {
        return NULL;
    }

    // Run forward pass first (required before jacobian)
    node->forward(node, (const double *) PyArray_DATA(u_array));

    // Initialize and evaluate jacobian
    node->jacobian_init(node);
    node->eval_jacobian(node);

    CSR_Matrix *jac = node->jacobian;

    // Create numpy arrays for CSR components
    npy_intp nnz = jac->nnz;
    npy_intp m_plus_1 = jac->m + 1;

    PyObject *data = PyArray_SimpleNew(1, &nnz, NPY_DOUBLE);
    PyObject *indices = PyArray_SimpleNew(1, &nnz, NPY_INT32);
    PyObject *indptr = PyArray_SimpleNew(1, &m_plus_1, NPY_INT32);

    if (!data || !indices || !indptr)
    {
        Py_XDECREF(data);
        Py_XDECREF(indices);
        Py_XDECREF(indptr);
        Py_DECREF(u_array);
        return NULL;
    }

    memcpy(PyArray_DATA((PyArrayObject *) data), jac->x, nnz * sizeof(double));
    memcpy(PyArray_DATA((PyArrayObject *) indices), jac->i, nnz * sizeof(int));
    memcpy(PyArray_DATA((PyArrayObject *) indptr), jac->p, m_plus_1 * sizeof(int));

    Py_DECREF(u_array);

    // Return tuple: (data, indices, indptr, shape)
    return Py_BuildValue("(OOO(ii))", data, indices, indptr, jac->m, jac->n);
}

/* ========== Problem bindings ========== */

static void problem_capsule_destructor(PyObject *capsule)
{
    problem *prob = (problem *) PyCapsule_GetPointer(capsule, PROBLEM_CAPSULE_NAME);
    if (prob)
    {
        free_problem(prob);
    }
}

static PyObject *py_make_problem(PyObject *self, PyObject *args)
{
    PyObject *obj_capsule;
    PyObject *constraints_list;
    if (!PyArg_ParseTuple(args, "OO", &obj_capsule, &constraints_list))
    {
        return NULL;
    }

    expr *objective = (expr *) PyCapsule_GetPointer(obj_capsule, EXPR_CAPSULE_NAME);
    if (!objective)
    {
        PyErr_SetString(PyExc_ValueError, "invalid objective capsule");
        return NULL;
    }

    if (!PyList_Check(constraints_list))
    {
        PyErr_SetString(PyExc_TypeError, "constraints must be a list");
        return NULL;
    }

    Py_ssize_t n_constraints = PyList_Size(constraints_list);
    expr **constraints = NULL;
    if (n_constraints > 0)
    {
        constraints = (expr **) malloc(n_constraints * sizeof(expr *));
        if (!constraints)
        {
            PyErr_NoMemory();
            return NULL;
        }
        for (Py_ssize_t i = 0; i < n_constraints; i++)
        {
            PyObject *c_capsule = PyList_GetItem(constraints_list, i);
            constraints[i] =
                (expr *) PyCapsule_GetPointer(c_capsule, EXPR_CAPSULE_NAME);
            if (!constraints[i])
            {
                free(constraints);
                PyErr_SetString(PyExc_ValueError, "invalid constraint capsule");
                return NULL;
            }
        }
    }

    problem *prob = new_problem(objective, constraints, (int) n_constraints);
    free(constraints);

    if (!prob)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create problem");
        return NULL;
    }

    return PyCapsule_New(prob, PROBLEM_CAPSULE_NAME, problem_capsule_destructor);
}

static PyObject *py_problem_init_derivatives(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    if (!PyArg_ParseTuple(args, "O", &prob_capsule))
    {
        return NULL;
    }

    problem *prob =
        (problem *) PyCapsule_GetPointer(prob_capsule, PROBLEM_CAPSULE_NAME);
    if (!prob)
    {
        PyErr_SetString(PyExc_ValueError, "invalid problem capsule");
        return NULL;
    }

    problem_init_derivatives(prob);

    Py_RETURN_NONE;
}

static PyObject *py_problem_objective_forward(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    PyObject *u_obj;
    if (!PyArg_ParseTuple(args, "OO", &prob_capsule, &u_obj))
    {
        return NULL;
    }

    problem *prob =
        (problem *) PyCapsule_GetPointer(prob_capsule, PROBLEM_CAPSULE_NAME);
    if (!prob)
    {
        PyErr_SetString(PyExc_ValueError, "invalid problem capsule");
        return NULL;
    }

    PyArrayObject *u_array =
        (PyArrayObject *) PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!u_array)
    {
        return NULL;
    }

    double obj_val = problem_objective_forward(prob, (const double *) PyArray_DATA(u_array));

    Py_DECREF(u_array);
    return Py_BuildValue("d", obj_val);
}

static PyObject *py_problem_constraint_forward(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    PyObject *u_obj;
    if (!PyArg_ParseTuple(args, "OO", &prob_capsule, &u_obj))
    {
        return NULL;
    }

    problem *prob =
        (problem *) PyCapsule_GetPointer(prob_capsule, PROBLEM_CAPSULE_NAME);
    if (!prob)
    {
        PyErr_SetString(PyExc_ValueError, "invalid problem capsule");
        return NULL;
    }

    PyArrayObject *u_array =
        (PyArrayObject *) PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!u_array)
    {
        return NULL;
    }

    double *constraint_vals =
        problem_constraint_forward(prob, (const double *) PyArray_DATA(u_array));

    PyObject *out = NULL;
    if (prob->total_constraint_size > 0)
    {
        npy_intp size = prob->total_constraint_size;
        out = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
        if (!out)
        {
            Py_DECREF(u_array);
            return NULL;
        }
        memcpy(PyArray_DATA((PyArrayObject *) out), constraint_vals,
               size * sizeof(double));
    }
    else
    {
        npy_intp size = 0;
        out = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
    }

    Py_DECREF(u_array);
    return out;
}

static PyObject *py_problem_gradient(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    PyObject *u_obj;
    if (!PyArg_ParseTuple(args, "OO", &prob_capsule, &u_obj))
    {
        return NULL;
    }

    problem *prob =
        (problem *) PyCapsule_GetPointer(prob_capsule, PROBLEM_CAPSULE_NAME);
    if (!prob)
    {
        PyErr_SetString(PyExc_ValueError, "invalid problem capsule");
        return NULL;
    }

    PyArrayObject *u_array =
        (PyArrayObject *) PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!u_array)
    {
        return NULL;
    }

    double *grad = problem_gradient(prob, (const double *) PyArray_DATA(u_array));

    npy_intp size = prob->n_vars;
    PyObject *out = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
    if (!out)
    {
        Py_DECREF(u_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject *) out), grad, size * sizeof(double));

    Py_DECREF(u_array);
    return out;
}

static PyObject *py_problem_jacobian(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    PyObject *u_obj;
    if (!PyArg_ParseTuple(args, "OO", &prob_capsule, &u_obj))
    {
        return NULL;
    }

    problem *prob =
        (problem *) PyCapsule_GetPointer(prob_capsule, PROBLEM_CAPSULE_NAME);
    if (!prob)
    {
        PyErr_SetString(PyExc_ValueError, "invalid problem capsule");
        return NULL;
    }

    if (prob->n_constraints == 0)
    {
        // Return empty CSR components
        npy_intp zero = 0;
        npy_intp one = 1;
        PyObject *data = PyArray_SimpleNew(1, &zero, NPY_DOUBLE);
        PyObject *indices = PyArray_SimpleNew(1, &zero, NPY_INT32);
        PyObject *indptr = PyArray_SimpleNew(1, &one, NPY_INT32);
        ((int *) PyArray_DATA((PyArrayObject *) indptr))[0] = 0;
        return Py_BuildValue("(OOO(ii))", data, indices, indptr, 0, prob->n_vars);
    }

    PyArrayObject *u_array =
        (PyArrayObject *) PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!u_array)
    {
        return NULL;
    }

    CSR_Matrix *jac = problem_jacobian(prob, (const double *) PyArray_DATA(u_array));

    npy_intp nnz = jac->nnz;
    npy_intp m_plus_1 = jac->m + 1;

    PyObject *data = PyArray_SimpleNew(1, &nnz, NPY_DOUBLE);
    PyObject *indices = PyArray_SimpleNew(1, &nnz, NPY_INT32);
    PyObject *indptr = PyArray_SimpleNew(1, &m_plus_1, NPY_INT32);

    if (!data || !indices || !indptr)
    {
        Py_XDECREF(data);
        Py_XDECREF(indices);
        Py_XDECREF(indptr);
        Py_DECREF(u_array);
        return NULL;
    }

    memcpy(PyArray_DATA((PyArrayObject *) data), jac->x, nnz * sizeof(double));
    memcpy(PyArray_DATA((PyArrayObject *) indices), jac->i, nnz * sizeof(int));
    memcpy(PyArray_DATA((PyArrayObject *) indptr), jac->p, m_plus_1 * sizeof(int));

    Py_DECREF(u_array);
    return Py_BuildValue("(OOO(ii))", data, indices, indptr, jac->m, jac->n);
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
    {"forward", py_forward, METH_VARARGS, "Run forward pass and return values"},
    {"jacobian", py_jacobian, METH_VARARGS,
     "Compute jacobian and return CSR components"},
    {"make_problem", py_make_problem, METH_VARARGS, "Create problem from objective and constraints"},
    {"problem_init_derivatives", py_problem_init_derivatives, METH_VARARGS, "Initialize derivative structures"},
    {"problem_objective_forward", py_problem_objective_forward, METH_VARARGS, "Evaluate objective only"},
    {"problem_constraint_forward", py_problem_constraint_forward, METH_VARARGS, "Evaluate constraints only"},
    {"problem_gradient", py_problem_gradient, METH_VARARGS, "Compute objective gradient"},
    {"problem_jacobian", py_problem_jacobian, METH_VARARGS, "Compute constraint jacobian"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef dnlp_module = {PyModuleDef_HEAD_INIT, "DNLP_diff_engine",
                                         NULL, -1, DNLPMethods};

PyMODINIT_FUNC PyInit_DNLP_diff_engine(void)
{
    if (ensure_numpy() < 0) return NULL;
    return PyModule_Create(&dnlp_module);
}
