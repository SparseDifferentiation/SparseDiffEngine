"""Tests for linear_op with constant offset: A @ x + b.

Tests verify that linear_op correctly handles:
1. Forward pass: y = A @ x + b
2. Gradient computation via chain rule

Copyright 2025, the CVXPY developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import dnlp_diff_engine as de


def test_linear_op_with_offset_forward():
    """Test linear_op(x, A, b) computes A @ x + b correctly in forward pass.

    Note: linear_op is an internal expression type and must be wrapped in
    another expression (like log) to be used as an objective.
    """
    n_vars = 2
    x = de.make_variable(n_vars, 1, 0, n_vars)  # Column vector (n_vars x 1)

    # A @ x + b where A = [[1, 1]], b = [5]
    # Should compute x[0] + x[1] + 5
    A_data = np.array([1.0, 1.0])
    A_indices = np.array([0, 1], dtype=np.int32)
    A_indptr = np.array([0, 2], dtype=np.int32)
    b = np.array([5.0])

    linear_with_offset = de.make_linear(x, A_data, A_indices, A_indptr, 1, 2, b)
    # Wrap in log to create a valid objective
    log_expr = de.make_log(linear_with_offset)

    prob = de.make_problem(log_expr, [])
    de.problem_init_derivatives(prob)

    # Test at u = [2.0, 3.0]
    u = np.array([2.0, 3.0])

    # Forward: log(2 + 3 + 5) = log(10)
    obj = de.problem_objective_forward(prob, u)
    assert np.isclose(obj, np.log(10.0)), f"Expected log(10), got {obj}"


def test_linear_op_with_offset_gradient():
    """Test gradient of log(A @ x + b)."""
    n_vars = 2
    x = de.make_variable(n_vars, 1, 0, n_vars)  # Column vector (n_vars x 1)

    # A @ x + b where A = [[1, 1]], b = [5]
    # log(x[0] + x[1] + 5)
    A_data = np.array([1.0, 1.0])
    A_indices = np.array([0, 1], dtype=np.int32)
    A_indptr = np.array([0, 2], dtype=np.int32)
    b = np.array([5.0])

    linear_with_offset = de.make_linear(x, A_data, A_indices, A_indptr, 1, 2, b)
    log_expr = de.make_log(linear_with_offset)

    prob = de.make_problem(log_expr, [])
    de.problem_init_derivatives(prob)

    # Test at u = [2.0, 3.0]
    u = np.array([2.0, 3.0])

    # Forward: log(2 + 3 + 5) = log(10)
    obj = de.problem_objective_forward(prob, u)
    assert np.isclose(obj, np.log(10.0))

    # Gradient: d/dx log(x+y+5) = 1/(x+y+5) for both
    grad = de.problem_gradient(prob)
    expected = 1.0 / 10.0
    np.testing.assert_allclose(grad, [expected, expected], rtol=1e-5)


def test_linear_op_without_offset():
    """Test linear_op(x, A) still works (no b parameter)."""
    n_vars = 2
    x = de.make_variable(n_vars, 1, 0, n_vars)  # Column vector (n_vars x 1)

    # A @ x where A = [[2, 3]]
    A_data = np.array([2.0, 3.0])
    A_indices = np.array([0, 1], dtype=np.int32)
    A_indptr = np.array([0, 2], dtype=np.int32)

    # No b parameter - pass None explicitly
    linear_no_offset = de.make_linear(x, A_data, A_indices, A_indptr, 1, 2, None)
    log_expr = de.make_log(linear_no_offset)

    prob = de.make_problem(log_expr, [])
    de.problem_init_derivatives(prob)

    u = np.array([1.0, 1.0])
    obj = de.problem_objective_forward(prob, u)
    assert np.isclose(obj, np.log(5.0))  # log(2*1 + 3*1)

    grad = de.problem_gradient(prob)
    # d/dx log(2x + 3y) = [2, 3] / (2x + 3y) = [2/5, 3/5]
    np.testing.assert_allclose(grad, [2.0/5.0, 3.0/5.0], rtol=1e-5)


def test_linear_op_with_offset_hessian():
    """Test Hessian of log(A @ x + b)."""
    n_vars = 2
    x = de.make_variable(n_vars, 1, 0, n_vars)  # Column vector (n_vars x 1)

    # log(x[0] + x[1] + 5)
    A_data = np.array([1.0, 1.0])
    A_indices = np.array([0, 1], dtype=np.int32)
    A_indptr = np.array([0, 2], dtype=np.int32)
    b = np.array([5.0])

    linear_with_offset = de.make_linear(x, A_data, A_indices, A_indptr, 1, 2, b)
    log_expr = de.make_log(linear_with_offset)

    prob = de.make_problem(log_expr, [])
    de.problem_init_derivatives(prob)

    # Test at u = [2.0, 3.0]
    u = np.array([2.0, 3.0])
    de.problem_objective_forward(prob, u)

    # Hessian: d²/dx² log(x+y+5) = -1/(x+y+5)² for all entries
    obj_factor = 1.0
    hess_data, hess_indices, hess_indptr, hess_shape = de.problem_hessian(
        prob, obj_factor, np.array([])
    )

    # The Hessian should be [[h, h], [h, h]] where h = -1/100
    expected_h = -1.0 / 100.0

    # Check that entries are correct
    assert len(hess_data) >= 3, f"Expected at least 3 Hessian entries, got {len(hess_data)}"

    # Check values
    for val in hess_data:
        np.testing.assert_allclose(val, expected_h, rtol=1e-5)


def test_linear_op_vector_with_offset():
    """Test linear_op with vector output: y = A @ x + b where y is a vector."""
    n_vars = 3
    x = de.make_variable(n_vars, 1, 0, n_vars)  # Column vector (n_vars x 1)

    # A is 2x3, b is 2-vector
    # A = [[1, 2, 0], [0, 1, 3]]
    # b = [1, 2]
    # y[0] = x[0] + 2*x[1] + 1
    # y[1] = x[1] + 3*x[2] + 2
    A_data = np.array([1.0, 2.0, 1.0, 3.0])
    A_indices = np.array([0, 1, 1, 2], dtype=np.int32)
    A_indptr = np.array([0, 2, 4], dtype=np.int32)
    b = np.array([1.0, 2.0])

    linear_with_offset = de.make_linear(x, A_data, A_indices, A_indptr, 2, 3, b)

    # sum(log(A @ x + b))
    log_expr = de.make_log(linear_with_offset)
    sum_expr = de.make_sum(log_expr, -1)

    prob = de.make_problem(sum_expr, [])
    de.problem_init_derivatives(prob)

    # Test at u = [1, 1, 1]
    u = np.array([1.0, 1.0, 1.0])

    # y[0] = 1 + 2 + 1 = 4
    # y[1] = 1 + 3 + 2 = 6
    # sum(log(y)) = log(4) + log(6)
    obj = de.problem_objective_forward(prob, u)
    expected_obj = np.log(4.0) + np.log(6.0)
    np.testing.assert_allclose(obj, expected_obj, rtol=1e-5)

    # Gradient:
    # d/dx[0] = 1/y[0] * 1 = 1/4
    # d/dx[1] = 1/y[0] * 2 + 1/y[1] * 1 = 2/4 + 1/6 = 0.5 + 0.1667
    # d/dx[2] = 1/y[1] * 3 = 3/6 = 0.5
    grad = de.problem_gradient(prob)
    expected_grad = np.array([1.0/4.0, 2.0/4.0 + 1.0/6.0, 3.0/6.0])
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-5)


def test_linear_op_sparse_matrix_with_offset():
    """Test linear_op with sparse matrix and offset."""
    n_vars = 5
    x = de.make_variable(n_vars, 1, 0, n_vars)  # Column vector (n_vars x 1)

    # A is 3x5 sparse matrix (only some entries non-zero)
    # A = [[1, 0, 2, 0, 0],
    #      [0, 3, 0, 4, 0],
    #      [0, 0, 0, 0, 5]]
    # b = [10, 20, 30]
    A_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    A_indices = np.array([0, 2, 1, 3, 4], dtype=np.int32)
    A_indptr = np.array([0, 2, 4, 5], dtype=np.int32)
    b = np.array([10.0, 20.0, 30.0])

    linear_with_offset = de.make_linear(x, A_data, A_indices, A_indptr, 3, 5, b)
    log_expr = de.make_log(linear_with_offset)
    sum_expr = de.make_sum(log_expr, -1)

    prob = de.make_problem(sum_expr, [])
    de.problem_init_derivatives(prob)

    # Test at u = [1, 1, 1, 1, 1]
    u = np.ones(5)

    # y[0] = 1*1 + 2*1 + 10 = 13
    # y[1] = 3*1 + 4*1 + 20 = 27
    # y[2] = 5*1 + 30 = 35
    obj = de.problem_objective_forward(prob, u)
    expected_obj = np.log(13.0) + np.log(27.0) + np.log(35.0)
    np.testing.assert_allclose(obj, expected_obj, rtol=1e-5)

    grad = de.problem_gradient(prob)
    # d/dx[0] = 1/13
    # d/dx[1] = 3/27
    # d/dx[2] = 2/13
    # d/dx[3] = 4/27
    # d/dx[4] = 5/35
    expected_grad = np.array([1.0/13.0, 3.0/27.0, 2.0/13.0, 4.0/27.0, 5.0/35.0])
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-5)
