# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
# Build (from repo root)
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make

# Run all tests
./build/all_tests

# Run tests via CTest
cd build && ctest --output-on-failure

# Format check (requires clang-format)
find src include -name '*.c' -o -name '*.h' | xargs clang-format --dry-run -Werror

# Build with sanitizers (for debugging)
cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS="-fsanitize=address,undefined" && make
```

There is no way to run a single test in isolation — `all_tests` runs the full suite. To test selectively, temporarily comment out `mu_run_test()` calls in `tests/all_tests.c`.

## Architecture

### Expr Node (Virtual Dispatch via Function Pointers)

Every operation is an `expr` node (`include/expr.h`). Polymorphism is achieved through function pointers set at construction time via `init_expr()`:

- `forward` — evaluate node value from input vector `u`
- `jacobian_init` / `eval_jacobian` — allocate then fill sparse Jacobian (CSR)
- `wsum_hess_init` / `eval_wsum_hess` — allocate then fill weighted-sum Hessian (CSR)
- `is_affine` — used to skip zero Hessian computation
- `local_jacobian` / `local_wsum_hess` — element-wise ops only: diagonal f'(g(x)) and f''(g(x))
- `free_type_data` — cleanup for type-specific allocations (e.g., param_source pointers)

Type-specific data uses C struct inheritance: cast the first field (`expr base`) to a subtype defined in `include/subexpr.h` (e.g., `parameter_expr`, `scalar_mult_expr`, `left_matmul_expr`).

### Operation Categories

| Directory | Description | Chain rule pattern |
|-----------|-------------|-------------------|
| `src/affine/` | Linear ops (variable, parameter, add, index, matmul, etc.) | J = structural combination of child Jacobians |
| `src/elementwise_full_dom/` | Univariate ops (exp, sin, log, power, etc.) | J = diag(f'(g(x))) · J_child; common chain rule in `common.c` |
| `src/elementwise_restricted_dom/` | Univariate ops with domain restrictions (log, entropy) | Same pattern as full_dom |
| `src/bivariate_full_dom/` | Two-arg ops (elementwise multiply) | Product rule on two children |
| `src/bivariate_restricted_dom/` | Two-arg restricted domain (quad_over_lin, rel_entr) | Custom chain rules |
| `src/other/` | Special ops (prod, quad_form) | Custom implementations |

### Init vs Eval Split

Derivative computation is split into two phases:
1. **Init** (`jacobian_init`, `wsum_hess_init`) — allocates CSR matrices and sets sparsity patterns. Called once per expression tree. Guarded by `jacobian_init()` / `wsum_hess_init()` wrappers in `expr.c` to handle DAGs where nodes are visited multiple times.
2. **Eval** (`eval_jacobian`, `eval_wsum_hess`) — fills numerical values into pre-allocated structures. Called each time input changes.

### Elementwise Common Chain Rule

`src/elementwise_full_dom/common.c` and `src/elementwise_restricted_dom/common.c` implement shared chain rule logic for all univariate ops. Each op only provides `local_jacobian` (diagonal of f') and `local_wsum_hess` (diagonal of f''). The common code handles:
- Variable child: Jacobian is diagonal
- Composite child: J = diag(f') · J_child; H = J_child^T · diag(w·f'') · J_child + f' · H_child

### Parameter System

Parameters (`src/affine/parameter.c`) unify constants and updatable values:
- `param_id == PARAM_FIXED` (-1): fixed constant, value set at construction
- `param_id >= 0`: updatable parameter, indexed into a global theta array

Problem-level API (`src/problem.c`):
- `problem_register_params()` — register parameter nodes
- `problem_update_params()` — update all parameter values from theta vector

Operations that use parameters (scalar_mult, vector_mult, left/right_matmul) store a `param_source` pointer to the parameter node and read its value during forward/jacobian/hessian evaluation.

### Workspace (`Expr_Work`)

Each node has a `work` pointer for cached intermediate results: CSC conversion of Jacobian, local Jacobian diagonal, Hessian term1/term2 matrices. Allocated during init, reused across evals.

## Adding a New Operation

1. If needed, add a subtype struct in `include/subexpr.h`
2. Declare the constructor in the appropriate header (`include/affine.h`, `include/elementwise_full_dom.h`, etc.)
3. Implement in `src/<category>/<op_name>.c` with static functions for forward, jacobian_init_impl, eval_jacobian, wsum_hess_init_impl, eval_wsum_hess, is_affine
4. For elementwise univariate ops: only implement `local_jacobian` and `local_wsum_hess`, then use `common_jacobian_init`/`common_eval_jacobian`/etc. from `common.c`
5. Add tests in `tests/forward_pass/`, `tests/jacobian_tests/`, `tests/wsum_hess/` and register them in `tests/all_tests.c`
6. Use `check_jacobian()` and `check_wsum_hess()` from `tests/numerical_diff.h` to validate against finite differences

## Code Style

C99. Enforced by `.clang-format`: Allman braces, 4-space indent, 85-column limit, right-aligned pointers. Run `clang-format -i <file>` before committing.

## CI Workflows (`.github/workflows/`)

- `cmake.yml` — build and test on Linux, macOS, Windows
- `formatting.yml` — clang-format check
- `sanitizer.yml` — ASan, MSan, UBSan
- `valgrind.yml` — memory leak detection
- `release.yml` — release packaging

## Platform Notes

- **macOS**: Links Accelerate for BLAS
- **Linux**: Links OpenBLAS (`libopenblas-dev`)
- **Windows**: Links OpenBLAS via vcpkg; no dense matmul in some configurations

## Test Tolerances

`ABS_TOL = 1e-6`, `REL_TOL = 1e-6` (defined in test headers). Numerical differentiation step size: `NUMERICAL_DIFF_DEFAULT_H = 1e-7`.
