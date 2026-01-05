#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdbool.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"

// Capsule name for expr* pointers
#define EXPR_CAPSULE_NAME "DNLP_EXPR"

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

static PyMethodDef DNLPMethods[] = {
    {"make_variable", py_make_variable, METH_VARARGS, "Create variable node"},
    {"make_log", py_make_log, METH_VARARGS, "Create log node"},
    {"make_exp", py_make_exp, METH_VARARGS, "Create exp node"},
    {"make_add", py_make_add, METH_VARARGS, "Create add node"},
    {"make_sum", py_make_sum, METH_VARARGS, "Create sum node"},
    {"forward", py_forward, METH_VARARGS, "Run forward pass and return values"},
    {"jacobian", py_jacobian, METH_VARARGS, "Compute jacobian and return CSR components"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef dnlp_module = {PyModuleDef_HEAD_INIT, "DNLP_diff_engine",
                                         NULL, -1, DNLPMethods};

PyMODINIT_FUNC PyInit_DNLP_diff_engine(void)
{
    if (ensure_numpy() < 0) return NULL;
    return PyModule_Create(&dnlp_module);
}
