# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DNLP-diff-engine is a pure C library that provides automatic differentiation for nonlinear optimization problems. It builds expression trees (ASTs) from CVXPY problems and computes first and second derivatives (gradients, Jacobians, Hessians) needed by NLP solvers like IPOPT.

**Note:** This library is designed to be used as a git submodule in CVXPY. Python packaging is handled by the CVXPY build system, not this repository.

## Build Commands

### Standalone C Library (for testing/development)

```bash
# Build core C library and tests
cmake -B build -S .
cmake --build build

# Run C tests
./build/all_tests
```

### Building with CVXPY

This library is included as a git submodule in CVXPY. To build:

```bash
# From the CVXPY repository root
pip install -e .  # or: uv pip install -e .

# The _diffengine Python extension is built automatically
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

- **`affine/`** - Linear operations: `variable`, `constant`, `add`, `neg`, `sum`, `promote`, `hstack`, `trace`, `linear_op`, `index`, `reshape`
- **`elementwise_univariate/`** - Scalar functions applied elementwise: `log`, `exp`, `entr`, `power`, `logistic`, `xexp`, trigonometric (`sin`, `cos`, `tan`), hyperbolic (`sinh`, `tanh`, `asinh`, `atanh`). Uses `common.c` for shared chain-rule patterns.
- **`bivariate/`** - Two-argument operations: `multiply`, `quad_over_lin`, `rel_entr`, `const_scalar_mult`, `const_vector_mult`, `left_matmul` (A @ f(x)), `right_matmul` (f(x) @ A)
- **`other/`** - Special atoms: `quad_form` (x'Px), `prod` (product of elements)

Each atom implements: `forward`, `jacobian_init`, `eval_jacobian`, and optionally `eval_wsum_hess` (defaults to zero for affine atoms).

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

The Python C extension (`_diffengine` module, built from `python/bindings.c`) provides:
- Atom constructors: `make_variable`, `make_constant`, `make_log`, `make_exp`, `make_add`, etc.
- Problem interface: `make_problem`, `problem_init_derivatives`, `problem_objective_forward`, `problem_gradient`, `problem_jacobian`, `problem_hessian`

The high-level Python API (converters, C_problem class) is in CVXPY at `cvxpy/reductions/solvers/nlp_solvers/diff_engine/`.

### Derivative Computation Flow

1. **Initialization**: `problem_init_derivatives()` allocates storage and computes sparsity patterns for all Jacobians and Hessians. This is done once per problem.
2. **Forward pass**: `objective_forward(u)` / `constraint_forward(u)` propagate values through expression tree, storing results in each node's `value` field.
3. **Derivative computation**: `gradient()`, `jacobian()`, `hessian()` traverse tree computing derivatives via chain rule.

**Key invariant**: Forward pass must be called before corresponding derivative functions. The derivatives are computed using values cached during forward pass.

Jacobian uses chain rule: `J_composite = J_outer * J_inner` via sparse matrix operations.
Hessian computes weighted sum: `obj_w * H_obj + sum(lambda_i * H_constraint_i)`, returning lower triangular.

### Sparse Matrix Utilities

`include/utils/` contains CSR and CSC sparse matrix implementations used throughout for efficient derivative storage and computation.

## Key Directories

- `include/` - Header files defining public API (`expr.h`, `problem.h`, atom headers)
- `src/` - C implementation files organized by atom category
- `python/` - Python bindings C code (`bindings.c`)
- `python/atoms/` - Python binding headers for each atom type
- `python/problem/` - Python binding headers for problem interface
- `python/tests/` - Python integration tests (run via pytest from CVXPY)
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
6. Add converter in CVXPY: `cvxpy/reductions/solvers/nlp_solvers/diff_engine/converters.py`
7. Rebuild CVXPY: `pip install -e .`
8. Add tests in `tests/` (C, register in `tests/all_tests.c`) and CVXPY `cvxpy/tests/nlp_tests/`

## Known Limitations

- **Bivariate matmul not supported**: `f(x) @ g(x)` where both sides depend on variables is not implemented. Only `A @ f(x)` and `f(x) @ A` with constant A work.
- **Reshape order**: Only Fortran order (`order='F'`) is supported. C order would require permutation logic.
- **Hessian sparsity**: Some atoms (`hstack`, `trace`) don't compute hessian sparsity patterns during initialization (see TODO.md).

## License Header

```c
// SPDX-License-Identifier: Apache-2.0
```
