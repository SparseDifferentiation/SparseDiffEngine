# Plan: Implement `vstack` atom

## Context

The engine needs a `vstack` (vertical stack) operation that concatenates expressions along the row dimension. Unlike `hstack` — where column-major storage means output values are a simple flat concatenation of children — `vstack` interleaves children's data across columns, making every operation (forward, Jacobian, Hessian) require row-level remapping.

## Core challenge: column-major interleaving

For `vstack` of children with shapes `(r_0, d2), (r_1, d2), ...`:
- Output shape: `(R, d2)` where `R = sum(r_i)`
- Column `j` of the output = `[child_0 col j, child_1 col j, ...]` stacked
- Output flat index `k` maps to: `global_row = k % R`, `col = k / R`
- Finding which child owns `global_row` requires a precomputed lookup

## Implementation

### 1. Type-specific struct (`include/subexpr.h`)

```c
typedef struct vstack_expr
{
    expr base;
    expr **args;
    int n_args;
    int *row_offsets;     /* row_offsets[i] = sum of args[0..i-1]->d1 */
    int *row_to_child;    /* length R: maps global row -> child index */
    int *row_to_local;    /* length R: maps global row -> local row in child */
    CSR_Matrix *CSR_work; /* for Hessian accumulation */
} vstack_expr;
```

Precompute `row_to_child[R]` and `row_to_local[R]` once in the constructor. These let every later operation do O(1) lookups instead of scanning children.

### 2. Constructor (`src/affine/vstack.c` — `new_vstack`)

- Assert all children have same `d2`
- Compute `R = sum(r_i)`, output is `(R, d2)`
- Allocate `vstack_expr`, call `init_expr`
- Build `row_offsets`, `row_to_child`, `row_to_local` arrays
- Retain all children

### 3. Forward pass

Loop over `d2` columns; within each column, loop over children and `memcpy` each child's column slice:

```
offset = 0
for j in 0..d2-1:
    for i in 0..n_args-1:
        memcpy(value + offset, args[i]->value + j * r_i, r_i * sizeof(double))
        offset += r_i
```

### 4. Jacobian init

The output Jacobian is `(size, n_vars)` in CSR. Each output row `k` corresponds to a specific child Jacobian row. Iterate output rows in order, use the precomputed mapping to copy column indices from the right child row:

```
for k in 0..size-1:
    global_row = k % R
    col = k / R
    child_idx = row_to_child[global_row]
    local_row = row_to_local[global_row]
    child_flat = col * args[child_idx]->d1 + local_row
    copy column indices from child's Jacobian row child_flat
    set J->p[k+1]
```

### 5. eval_jacobian

Same loop as jacobian_init but copies values (`x`) instead of column indices (`i`).

### 6. Hessian (wsum_hess_init / eval)

**Sparsity init**: Identical to hstack — iteratively union children's Hessian patterns using `sum_csr_matrices_fill_sparsity`.

**Eval**: The weight vector `w` has one entry per output element. Each child needs a gathered weight vector. Allocate `dwork` of size `max(child->size)`. For each child, gather its weights:

```
for j in 0..d2-1:
    for r in 0..r_i-1:
        dwork[j * r_i + r] = w[j * R + row_offsets[i] + r]
```

Then call `child->eval_wsum_hess(child, dwork)` and accumulate with `sum_csr_matrices_fill_values`. This follows the same pattern transpose uses for weight permutation (`transpose.c:111-117`).

### 7. is_affine / free_type_data

Same pattern as hstack — check all children / free all children + auxiliary arrays.

## Files to modify

| File | Change |
|------|--------|
| `include/subexpr.h` | Add `vstack_expr` struct |
| `include/affine.h` | Declare `new_vstack` |
| `src/affine/vstack.c` | New file — full implementation |
| `src/affine/hstack.c` | Reference only (pattern to follow) |
| `src/affine/transpose.c` | Reference only (weight permutation pattern) |
| `tests/forward_pass/affine/test_vstack.h` | New — forward pass tests |
| `tests/jacobian_tests/test_vstack.h` | New — Jacobian tests |
| `tests/wsum_hess/test_vstack.h` | New — Hessian tests |
| `tests/all_tests.c` | Include new test headers, add `mu_run_test` calls |

## Verification

1. `cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug && cmake --build build`
2. `./build/all_tests` — all existing + new vstack tests pass
3. Test cases should cover:
   - Vectors: `vstack([log(x), exp(x)])` where x is `(3,1)` — output `(6,1)` (degenerate case: behaves like hstack for vectors)
   - Matrices: `vstack([log(x), exp(y)])` where x is `(2,3)`, y is `(1,3)` — output `(3,3)` — verifies interleaving
   - Hessian with shared variables: `vstack([log(x), exp(x)])` — verifies weight gathering and Hessian accumulation
