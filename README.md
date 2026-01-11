# DNLP Diff Engine

A C library with Python bindings for automatic differentiation of nonlinear optimization problems. Builds expression trees from CVXPY problems and computes gradients, Jacobians, and Hessians needed by NLP solvers like IPOPT.

## Installation

### Using uv (recommended)

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[test]"
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Running Tests

```bash
# first go to python folder

# Run all tests
pytest
```

## Usage

```python
import cvxpy as cp
import numpy as np
from dnlp_diff_engine import C_problem

# Define a CVXPY problem
x = cp.Variable(3)
problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x))))

# Convert to C problem struct
prob = C_problem(problem)
prob.init_derivatives()

# Evaluate at a point
u = np.array([1.0, 2.0, 3.0])
obj_val = prob.objective_forward(u)
gradient = prob.gradient()

print(f"Objective: {obj_val}")
print(f"Gradient: {gradient}")
```

## Building the C Library (standalone)

```bash
cmake -B build -S .
cmake --build build
./build/all_tests
```
