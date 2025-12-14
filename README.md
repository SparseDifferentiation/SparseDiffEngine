# DNLP Diff Engine

A minimalistic C-based expression tree system for nonlinear programming with automatic differentiation capabilities.

## Structure

```
.
├── include/              # Header files
│   ├── expr.h           # Base expression node definition
│   ├── affine/          # Affine operations
│   │   ├── variable.h
│   │   ├── constant.h
│   │   └── add.h
│   └── elementwise/     # Elementwise operations
│       ├── exp.h
│       └── log.h
├── src/
│   ├── expr.c           # Base node implementation
│   ├── affine/          # Affine operations
│   │   ├── variable.c
│   │   ├── constant.c
│   │   └── add.c
│   └── elementwise/     # Elementwise operations
│       ├── exp.c
│       └── log.c
├── tests/
│   └── forward_pass/    # Forward pass tests
│       └── test_forward_pass.c
└── CMakeLists.txt       # Build configuration
```

## Features

- **Expression nodes**: Each node has dimension `m`, preallocated value memory, and up to two children
- **Leaf nodes**: Variable (reads from input `u`) and Constant (fixed values)
- **Affine operations**: Addition
- **Elementwise operations**: Exponential (`exp`) and Logarithm (`log`)
- **Forward pass**: Compute output values through the expression tree

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Running Tests

```bash
# Run the test executable
./test_forward_pass

# Or use CTest
ctest --verbose
```

## Example Usage

```c
#include "expr.h"
#include "affine/variable.h"
#include "affine/constant.h"
#include "affine/add.h"
#include "elementwise/exp.h"
#include "elementwise/log.h"

// Build expression tree: log(exp(x) + c)
double u[2] = {1.0, 2.0};
double c[2] = {1.0, 1.0};

expr* var = new_variable(2);
expr* exp_node = new_exp(var);
expr* const_node = new_constant(2, c);
expr* sum = new_add(exp_node, const_node);
expr* log_node = new_log(sum);

// Compute forward pass
log_node->forward(log_node, u);

// Result is in log_node->value
```

## Design

- Each node allocates its own output memory
- Forward pass recursively evaluates children before computing node output
- Clean separation: affine operations in `src/affine/`, elementwise in `src/elementwise/`
- Memory management: nodes must be freed manually (children are not auto-freed)
# DNLP-diff-engine
