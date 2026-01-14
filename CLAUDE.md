# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DNLP-diff-engine is a C library with Python bindings that provides automatic differentiation for nonlinear optimization problems. It builds expression trees (ASTs) from CVXPY problems and computes first and second derivatives (gradients, Jacobians, Hessians) needed by NLP solvers like IPOPT.

## Build Commands

### Python Package (Recommended)

```bash
# Install in development mode with uv (recommended)
uv pip install -e ".[test]"

# Or with pip
pip install -e ".[test]"

# Run all Python tests (tests are in python/tests/)
pytest

# Run specific test file
pytest python/tests/test_unconstrained.py

# Run specific test
pytest python/tests/test_unconstrained.py::test_sum_log

# Lint with ruff
ruff check src/

# Auto-fix lint issues
ruff check --fix src/
```

### Standalone C Library

```bash
# Build core C library and tests
cmake -B build -S .
cmake --build build

# Run C tests
./build/all_tests
```

## Architecture

### Expression Tree System

The core abstraction is the `expr` struct (in `include/expr.h`) representing a node in an expression AST. Each node stores:
- Shape information (`d1`, `d2`, `size`, `n_vars`)
- Function pointers for evaluation: `forward`, `jacobian_init`, `eval_jacobian`, `eval_wsum_hess`
- Computed values: `value`, `jacobian` (CSR), `wsum_hess` (CSR)
- Child pointers (`left`, `right`) and reference counting (`refcount`)

### Atom Categories

Atoms are organized by mathematical properties in `src/`:

- **`affine/`** - Linear operations: `variable`, `constant`, `add`, `neg`, `sum`, `promote`, `hstack`, `trace`, `linear_op`, `index`
- **`elementwise_univariate/`** - Scalar functions applied elementwise: `log`, `exp`, `entr`, `power`, `logistic`, `xexp`, trigonometric (`sin`, `cos`, `tan`), hyperbolic (`sinh`, `tanh`, `asinh`, `atanh`)
- **`bivariate/`** - Two-argument operations: `multiply`, `quad_over_lin`, `rel_entr`, `const_scalar_mult`, `const_vector_mult`, `left_matmul`, `right_matmul`
- **`other/`** - Special atoms: `quad_form`, `prod`

Each atom implements its own `forward`, `jacobian_init`, `eval_jacobian`, and `eval_wsum_hess` functions following a consistent pattern.

### Problem Struct

The `problem` struct (in `include/problem.h`) encapsulates an optimization problem:
- Single `objective` expression (scalar)
- Array of `constraints` expressions
- Pre-allocated storage for `constraint_values`, `gradient_values`, `jacobian`, `lagrange_hessian`

Key oracle methods:
- `problem_objective_forward(prob, u)` - Evaluate objective at point u
- `problem_constraint_forward(prob, u)` - Evaluate all constraints at u
- `problem_gradient(prob)` - Compute objective gradient (after forward)
- `problem_jacobian(prob)` - Compute stacked constraint Jacobian (after forward)
- `problem_hessian(prob, obj_w, lambda)` - Compute Lagrangian Hessian

### Python Bindings

The Python package `dnlp_diff_engine` (in `src/dnlp_diff_engine/`) provides:

**High-level API** (`__init__.py`):
- `C_problem` class wraps the C problem struct
- `convert_problem()` builds expression trees from CVXPY Problem objects
- Atoms are mapped via `ATOM_CONVERTERS` dictionary (maps CVXPY atom names → converter functions)
- Special converters handle: matrix multiplication (`_convert_matmul`), multiply with constants (`_convert_multiply`), indexing, reshape (Fortran order only)

**Low-level C extension** (`_core` module, built from `python/bindings.c`):
- Atom constructors: `make_variable`, `make_constant`, `make_log`, `make_exp`, `make_add`, etc.
- Problem interface: `make_problem`, `problem_init_derivatives`, `problem_objective_forward`, `problem_gradient`, `problem_jacobian`, `problem_hessian`

### Derivative Computation Flow

1. Call `problem_init_derivatives()` to allocate Jacobian/Hessian storage and compute sparsity patterns
2. Call forward pass (`objective_forward` / `constraint_forward`) to propagate values through tree
3. Call derivative functions (`gradient`, `jacobian`, `hessian`) which traverse tree computing derivatives

Jacobian uses chain rule: each node computes local Jacobian, combined via sparse matrix operations.
Hessian computes weighted sum: `obj_w * H_obj + sum(lambda_i * H_constraint_i)`

### Sparse Matrix Utilities

`include/utils/` contains CSR and CSC sparse matrix implementations used throughout for efficient derivative storage and computation.

## Key Directories

- `include/` - Header files defining public API (`expr.h`, `problem.h`, atom headers)
- `src/` - C implementation files organized by atom category
- `src/dnlp_diff_engine/` - Python package with high-level API
- `python/` - Python bindings C code (`bindings.c`)
- `python/atoms/` - Python binding headers for each atom type
- `python/problem/` - Python binding headers for problem interface
- `python/tests/` - Python integration tests (run via pytest)
- `tests/` - C tests using minunit framework
- `tests/forward_pass/` - Forward evaluation tests (C)
- `tests/jacobian_tests/` - Jacobian correctness tests (C)
- `tests/wsum_hess/` - Hessian correctness tests (C)

## Adding a New Atom

1. Create header in `include/` declaring the constructor function
2. Create implementation in appropriate `src/` subdirectory
3. Implement: `forward`, `jacobian_init`, `eval_jacobian`, `eval_wsum_hess` (optional), `free_type_data` (if needed)
4. Add Python binding header in `python/atoms/`
5. Register in `python/bindings.c` (both include and method table)
6. Add converter entry in `src/dnlp_diff_engine/__init__.py` `ATOM_CONVERTERS` dict
7. Rebuild: `pip install -e .`
8. Add tests in `tests/` (C) and `tests/python/` (Python)

## License Header

```c
// SPDX-License-Identifier: Apache-2.0
```
