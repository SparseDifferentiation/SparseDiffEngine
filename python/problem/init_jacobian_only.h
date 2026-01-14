#ifndef PROBLEM_INIT_JACOBIAN_ONLY_H
#define PROBLEM_INIT_JACOBIAN_ONLY_H

#include "common.h"

static PyObject *py_problem_init_jacobian_only(PyObject *self, PyObject *args)
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

    problem_init_jacobian_only(prob);

    Py_RETURN_NONE;
}

#endif /* PROBLEM_INIT_JACOBIAN_ONLY_H */
