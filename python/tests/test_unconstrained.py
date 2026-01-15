"""Tests for unconstrained optimization problems."""

import cvxpy as cp
import numpy as np

from cvxpy.reductions.solvers.nlp_solvers.diff_engine import C_problem


def test_sum_log():
    """Test sum(log(x)) objective and gradient."""
    x = cp.Variable(4)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x))))
    prob = C_problem(problem)

    u = np.array([1.0, 2.0, 3.0, 4.0])
    prob.init_derivatives()

    # Objective
    obj_val = prob.objective_forward(u)
    expected = np.sum(np.log(u))
    assert np.allclose(obj_val, expected)

    # Gradient: d/dx sum(log(x)) = 1/x
    grad = prob.gradient()
    expected_grad = 1.0 / u
    assert np.allclose(grad, expected_grad)


def test_sum_exp():
    """Test sum(exp(x)) objective and gradient."""
    x = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.exp(x))))
    prob = C_problem(problem)

    u = np.array([0.0, 1.0, 2.0])
    prob.init_derivatives()

    # Objective
    obj_val = prob.objective_forward(u)
    expected = np.sum(np.exp(u))
    assert np.allclose(obj_val, expected)

    # Gradient: d/dx sum(exp(x)) = exp(x)
    grad = prob.gradient()
    expected_grad = np.exp(u)
    assert np.allclose(grad, expected_grad)


def test_variable_reuse():
    """Test sum(log(x) + exp(x)) - same variable used twice."""
    x = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x) + cp.exp(x))))
    prob = C_problem(problem)

    u = np.array([1.0, 2.0])
    prob.init_derivatives()

    # Objective
    obj_val = prob.objective_forward(u)
    expected = np.sum(np.log(u) + np.exp(u))
    assert np.allclose(obj_val, expected)

    # Gradient: d/dx_i = 1/x_i + exp(x_i)
    grad = prob.gradient()
    expected_grad = 1.0 / u + np.exp(u)
    assert np.allclose(grad, expected_grad)


def test_variable_used_multiple_times():
    """Test sum(log(x) + exp(x) + log(x)) - variable used 3 times."""
    x = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x) + cp.exp(x) + cp.log(x))))
    prob = C_problem(problem)

    u = np.array([1.0, 2.0, 3.0])
    prob.init_derivatives()

    # Objective
    obj_val = prob.objective_forward(u)
    expected = np.sum(2 * np.log(u) + np.exp(u))
    assert np.allclose(obj_val, expected)

    # Gradient: d/dx_i = 2/x_i + exp(x_i)
    grad = prob.gradient()
    expected_grad = 2.0 / u + np.exp(u)
    assert np.allclose(grad, expected_grad)


def test_larger_variable():
    """Test sum(log(x)) with larger variable (100 elements)."""
    x = cp.Variable(100)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x))))
    prob = C_problem(problem)

    u = np.linspace(1.0, 10.0, 100)
    prob.init_derivatives()

    # Objective
    obj_val = prob.objective_forward(u)
    expected = np.sum(np.log(u))
    assert np.allclose(obj_val, expected)

    # Gradient
    grad = prob.gradient()
    expected_grad = 1.0 / u
    assert np.allclose(grad, expected_grad)


def test_matrix_variable():
    """Test sum(log(X)) with 2D matrix variable (3x4)."""
    X = cp.Variable((3, 4))
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(X))))
    prob = C_problem(problem)

    u = np.arange(1.0, 13.0)  # 12 elements
    prob.init_derivatives()

    # Objective
    obj_val = prob.objective_forward(u)
    expected = np.sum(np.log(u))
    assert np.allclose(obj_val, expected)

    # Gradient
    grad = prob.gradient()
    expected_grad = 1.0 / u
    assert np.allclose(grad, expected_grad)


def test_two_variables_separate_ops():
    """Test sum(log(x)) + sum(exp(y)) - two variables with separate ops."""
    x = cp.Variable(2)
    y = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x)) + cp.sum(cp.exp(y))))
    prob = C_problem(problem)

    x_vals = np.array([1.0, 2.0])
    y_vals = np.array([0.5, 1.0])
    u = np.concatenate([x_vals, y_vals])
    prob.init_derivatives()

    # Objective
    obj_val = prob.objective_forward(u)
    expected = np.sum(np.log(x_vals)) + np.sum(np.exp(y_vals))
    assert np.allclose(obj_val, expected)

    # Gradient
    grad = prob.gradient()
    expected_grad = np.concatenate([1.0 / x_vals, np.exp(y_vals)])
    assert np.allclose(grad, expected_grad)


def test_two_variables_same_sum():
    """Test sum(log(x) + log(y)) - two variables added before sum."""
    x = cp.Variable(2)
    y = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x) + cp.log(y))))
    prob = C_problem(problem)

    x_vals = np.array([1.0, 2.0])
    y_vals = np.array([3.0, 4.0])
    u = np.concatenate([x_vals, y_vals])
    prob.init_derivatives()

    # Objective
    obj_val = prob.objective_forward(u)
    expected = np.sum(np.log(x_vals) + np.log(y_vals))
    assert np.allclose(obj_val, expected)

    # Gradient
    grad = prob.gradient()
    expected_grad = np.concatenate([1.0 / x_vals, 1.0 / y_vals])
    assert np.allclose(grad, expected_grad)


def test_mixed_sizes():
    """Test sum(log(a)) + sum(log(b)) + sum(log(c)) with different sized variables."""
    a = cp.Variable(2)
    b = cp.Variable(5)
    c = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(
        cp.sum(cp.log(a)) + cp.sum(cp.log(b)) + cp.sum(cp.log(c))
    ))
    prob = C_problem(problem)

    a_vals = np.array([1.0, 2.0])
    b_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c_vals = np.array([1.0, 2.0, 3.0])
    u = np.concatenate([a_vals, b_vals, c_vals])
    prob.init_derivatives()

    # Objective
    obj_val = prob.objective_forward(u)
    expected = np.sum(np.log(a_vals)) + np.sum(np.log(b_vals)) + np.sum(np.log(c_vals))
    assert np.allclose(obj_val, expected)

    # Gradient
    grad = prob.gradient()
    expected_grad = 1.0 / u
    assert np.allclose(grad, expected_grad)


def test_repeated_evaluations():
    """Test repeated evaluations at different points."""
    x = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x))))
    prob = C_problem(problem)

    u1 = np.array([1.0, 2.0, 3.0])
    prob.init_derivatives()

    # First evaluation
    obj1 = prob.objective_forward(u1)
    grad1 = prob.gradient()

    # Second evaluation
    u2 = np.array([2.0, 3.0, 4.0])
    obj2 = prob.objective_forward(u2)
    grad2 = prob.gradient()

    assert np.allclose(obj1, np.sum(np.log(u1)))
    assert np.allclose(obj2, np.sum(np.log(u2)))
    assert np.allclose(grad1, 1.0 / u1)
    assert np.allclose(grad2, 1.0 / u2)
