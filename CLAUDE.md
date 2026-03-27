# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SparseDiffEngine is a C library implementing the differentiation engine for CVXPY's DNLP extension. It computes sparse Jacobians and Hessians for nonlinear programming solvers. Used as a backend via Python bindings in the parent SparseDiffPy package.

## Build & Test Commands

```bash
# Build standalone (CMake)
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make

# Run all tests
cd build && ctest --output-on-failure

# Run tests directly (more verbose output)
cd build && ./all_tests

# Rebuild after changes (from build/)
make && ctest --output-on-failure

# Format code
clang-format -i src/**/*.c include/**/*.h
```

**Platform dependencies**: macOS uses Accelerate framework (automatic), Linux needs `libopenblas-dev`, Windows uses OpenBLAS via vcpkg.

## Architecture

### Expression DAG with Function-Pointer Polymorphism

Every node is an `expr` struct (`include/expr.h`) containing function pointers that act as a vtable:
- `forward()` — evaluate node given variable values `u`
- `jacobian_init()` / `eval_jacobian()` — sparsity pattern allocation and numeric fill
- `wsum_hess_init()` / `eval_wsum_hess()` — weighted-sum Hessian
- `is_affine()` — used for caching optimizations
- `free_type_data()` — type-specific cleanup

Type-specific nodes use C struct inheritance: embed `expr base` as the first field, then cast. Defined in `include/subexpr.h` (e.g., `parameter_expr`, `scalar_mult_expr`).

All node constructors call `init_expr()` to wire up the dispatch table. Children are owned via `left`/`right` pointers with reference counting (`expr_retain`/`free_expr`).

### Parameter System

`parameter_expr` (in `subexpr.h`) unifies constants and updatable parameters:
- `param_id == PARAM_FIXED (-1)`: fixed constant, not updatable
- `param_id >= 0`: offset into the parameter vector `theta`

Bivariate ops that involve parameters (scalar_mult, vector_mult, left_matmul, right_matmul) store a `param_source` pointer and read values during forward/jacobian/hessian evaluation.

**Workflow**: `new_parameter()` → `problem_register_params()` → `problem_update_params(prob, theta)` copies from `theta` into each parameter node's `value` array using `param_id` as offset.

### Problem Interface (`include/problem.h`)

The `problem` struct owns an objective, constraints array, and parameter nodes. Key lifecycle:
1. `new_problem()` — allocates problem, output arrays
2. `problem_register_params()` — registers parameter nodes
3. `problem_init_derivatives()` — allocates sparsity patterns (Jacobian + Hessian)
4. `problem_objective_forward()` / `problem_constraint_forward()` — evaluate
5. `problem_gradient()` / `problem_jacobian()` / `problem_hessian()` — compute derivatives
6. `problem_update_params()` — update parameter values, reset caches

### Sparse Matrix Formats (`include/utils/`)

CSR is primary. CSC is intermediate during computation. COO for Python interop; lower-triangular COO for symmetric Hessians.

## Adding a New Operation

1. Create `src/<category>/new_op.c` implementing the function pointer interface
2. Add constructor declaration to the appropriate `include/<category>.h`
3. Write test as a `.h` file in `tests/<test_category>/`
4. Register test in `tests/all_tests.c` with `mu_run_test(test_name, tests_run)`

## Test Framework

Uses minunit.h (header-only). Tests are `const char *test_name(void)` functions returning NULL on success. Tolerance: `ABS_TOL=1e-6, REL_TOL=1e-6` via `cmp_double_array()` in `test_helpers.c`. No built-in test filtering — all tests run via single `all_tests` binary.

## C Code Style

C99, Allman braces, 4-space indent, 85-column limit, right-aligned pointers (`int *ptr`). Enforced by `.clang-format`. Strict warnings: `-Wall -Wextra -Wpedantic -Wshadow -Wformat=2 -Wcast-qual -Wcast-align -Wdouble-promotion -Wnull-dereference`.
