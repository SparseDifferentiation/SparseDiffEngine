# Design: C Problem Struct

## Summary

Create a native C `problem` struct that encapsulates objective + constraints, with methods for:
- `forward(u)` - evaluate objective and all constraints
- `gradient(u)` - return objective gradient (jacobian.T)
- `jacobian(u)` - return single stacked CSR matrix of constraint jacobians

## Files to Create/Modify

1. **`include/problem.h`** - New header defining problem struct
2. **`src/problem.c`** - Implementation
3. **`python/bindings.c`** - Python bindings for problem
4. **`python/convert.py`** - Update to return problem capsule
5. **`tests/problem/test_problem.h`** - C tests

---

## Step 1: Create `include/problem.h`

```c
#ifndef PROBLEM_H
#define PROBLEM_H

#include "expr.h"
#include "utils/CSR_Matrix.h"

typedef struct problem
{
    expr *objective;           /* Objective expression (scalar) */
    expr **constraints;        /* Array of constraint expressions */
    int n_constraints;
    int n_vars;
    int total_constraint_size; /* Sum of all constraint sizes */

    /* Pre-allocated storage */
    double *constraint_values;
    double *gradient_values;   /* Dense gradient array */
    CSR_Matrix *jacobian;
} problem;

problem *new_problem(expr *objective, expr **constraints, int n_constraints);
void problem_allocate(problem *prob, const double *u);
void free_problem(problem *prob);
double problem_forward(problem *prob, const double *u);
double *problem_gradient(problem *prob, const double *u);
CSR_Matrix *problem_jacobian(problem *prob, const double *u);

#endif
```

---

## Step 2: Create `src/problem.c`

Key functions:

### `new_problem`
- Retain (increment refcount) on objective and all constraints
- Compute `total_constraint_size = sum(constraints[i]->size)`
- Does NOT allocate storage arrays (use `problem_allocate` separately)

### `problem_allocate`
Separate function to allocate memory for constraint values and jacobian:

```c
void problem_allocate(problem *prob, const double *u)
{
    /* 1. Allocate constraint values array */
    prob->constraint_values = malloc(prob->total_constraint_size * sizeof(double));

    /* 2. Allocate jacobian:
     *    - First, initialize all constraint jacobians
     *    - Count total nnz across all constraints
     *    - Allocate CSR matrix with this nnz (may be slight overestimate)
     */
    int total_nnz = 0;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];
        c->forward(c, u);
        c->jacobian_init(c);
        total_nnz += c->jacobian->nnz;
    }

    /* Allocate stacked jacobian with total_constraint_size rows */
    prob->jacobian = alloc_csr(prob->total_constraint_size, prob->n_vars, total_nnz);

    /* Note: The actual nnz may be smaller after evaluation due to
     * cancellations. Update jacobian->nnz after problem_jacobian(). */
}
```

### `free_problem`
- Call `free_expr` on objective and all constraints (decrements refcount)
- Free allocated arrays and jacobian

### `problem_forward`
```c
double problem_forward(problem *prob, const double *u)
{
    prob->objective->forward(prob->objective, u);
    double obj_val = prob->objective->value[0];

    int offset = 0;
    for (int i = 0; i < prob->n_constraints; i++)
    {
        expr *c = prob->constraints[i];
        c->forward(c, u);
        memcpy(prob->constraint_values + offset, c->value, c->size * sizeof(double));
        offset += c->size;
    }
    return obj_val;
}
```

### `problem_gradient`
- Run forward pass on objective
- Call jacobian_init + eval_jacobian
- Objective jacobian is 1 x n_vars row vector
- Copy sparse row to dense `gradient_values` array
- Return pointer to internal array

### `problem_jacobian`
- Forward + jacobian for each constraint
- Stack CSR matrices vertically:
  - Total rows = `total_constraint_size`
  - Copy row pointers with offset, copy col indices and values
- Lazy allocate/reallocate `jacobian` based on total nnz

---

## Step 3: Update `python/bindings.c`

Add capsule and functions:

```c
#define PROBLEM_CAPSULE_NAME "DNLP_PROBLEM"

static void problem_capsule_destructor(PyObject *capsule) { ... }

static PyObject *py_make_problem(PyObject *self, PyObject *args)
{
    PyObject *obj_capsule, *constraints_list;
    // Parse objective capsule and list of constraint capsules
    // Extract expr* pointers, call new_problem
    // Return PyCapsule
}

static PyObject *py_problem_forward(PyObject *self, PyObject *args)
{
    // Returns: (obj_value, constraint_values_array)
}

static PyObject *py_problem_gradient(PyObject *self, PyObject *args)
{
    // Returns: numpy array of size n_vars
}

static PyObject *py_problem_jacobian(PyObject *self, PyObject *args)
{
    // Returns: (data, indices, indptr, (m, n)) for scipy CSR
}
```

Add to DNLPMethods:
```c
{"make_problem", py_make_problem, METH_VARARGS, "Create problem"},
{"problem_forward", py_problem_forward, METH_VARARGS, "Evaluate problem"},
{"problem_gradient", py_problem_gradient, METH_VARARGS, "Compute gradient"},
{"problem_jacobian", py_problem_jacobian, METH_VARARGS, "Compute constraint jacobian"},
```

---

## Step 4: Update `python/convert.py`

```python
def convert_problem(problem: cp.Problem):
    """Convert CVXPY Problem to C problem struct."""
    var_dict = build_variable_dict(problem.variables())

    c_objective = _convert_expr(problem.objective.expr, var_dict)
    c_constraints = [_convert_expr(c.expr, var_dict) for c in problem.constraints]

    return diffengine.make_problem(c_objective, c_constraints)


class Problem:
    """Wrapper for C problem struct with clean Python API."""

    def __init__(self, cvxpy_problem: cp.Problem):
        self._capsule = convert_problem(cvxpy_problem)

    def forward(self, u: np.ndarray) -> tuple[float, np.ndarray]:
        return diffengine.problem_forward(self._capsule, u)

    def gradient(self, u: np.ndarray) -> np.ndarray:
        return diffengine.problem_gradient(self._capsule, u)

    def jacobian(self, u: np.ndarray) -> sparse.csr_matrix:
        data, indices, indptr, shape = diffengine.problem_jacobian(self._capsule, u)
        return sparse.csr_matrix((data, indices, indptr), shape=shape)
```

---

## Step 5: Add Tests

### C tests in `tests/problem/test_problem.h`:
- `test_problem_forward` - verify objective and constraint values
- `test_problem_gradient` - verify gradient matches manual calculation
- `test_problem_jacobian_stacking` - verify stacked matrix structure

### Python tests in `convert.py`:
- `test_problem_forward` - compare with numpy
- `test_problem_gradient` - gradient of sum(log(x)) = 1/x
- `test_problem_jacobian` - verify stacked jacobian shape and values

---

## Implementation Order

1. Create `include/problem.h`
2. Create `src/problem.c` with new_problem, free_problem, problem_forward
3. Add problem_gradient and problem_jacobian
4. Add Python bindings to `bindings.c`
5. Rebuild: `cmake --build build`
6. Update `convert.py` with Problem class
7. Add and run tests

## Key Design Notes

- **Memory**: Uses expr refcounting - new_problem retains, free_problem releases
- **Two-phase init**: `new_problem` creates struct, `problem_allocate` allocates arrays
  - Constraint values array: size = `total_constraint_size`
  - Jacobian: initialize all constraint jacobians first, count total nnz, allocate CSR
  - The allocated nnz may be a slight overestimate; update `jacobian->nnz` after evaluation
- **Hessian**: Deferred - not allocated in this design (to be added later)
- **API**: Returns internal pointers (caller should NOT free)
