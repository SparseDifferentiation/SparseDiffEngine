"""Tests for constrained optimization problems."""

import cvxpy as cp
import numpy as np

from cvxpy.reductions.solvers.nlp_solvers.diff_engine import C_problem

# Note: CVXPY converts constraints A >= B to B - A <= 0
# So constr.expr for "log(x) >= 0" is "0 - log(x)" = -log(x)
# All constraint values and jacobians are negated compared to the LHS


def test_single_constraint():
    """Test C_problem with single constraint."""
    x = cp.Variable(3)
    obj = cp.sum(cp.log(x))
    constraints = [cp.log(x) >= 0]  # becomes 0 - log(x) <= 0

    problem = cp.Problem(cp.Minimize(obj), constraints)
    prob = C_problem(problem)

    u = np.array([1.0, 2.0, 3.0])
    prob.init_derivatives()

    # Test constraint_forward: constr.expr = -log(x)
    constraint_vals = prob.constraint_forward(u)
    assert np.allclose(constraint_vals, -np.log(u))

    # Test jacobian: d/dx(-log(x)) = -1/x
    jac = prob.jacobian()
    expected_jac = np.diag(-1.0 / u)
    assert np.allclose(jac.toarray(), expected_jac)


def test_two_constraints():
    """Test C_problem with two constraints."""
    x = cp.Variable(2)
    obj = cp.sum(cp.log(x))
    constraints = [
        cp.log(x) >= 0,  # becomes -log(x) <= 0
        cp.exp(x) >= 0,  # becomes -exp(x) <= 0
    ]

    problem = cp.Problem(cp.Minimize(obj), constraints)
    prob = C_problem(problem)

    u = np.array([1.0, 2.0])
    prob.init_derivatives()

    # Test constraint_forward: [-log(u), -exp(u)]
    expected_constraint_vals = np.concatenate([-np.log(u), -np.exp(u)])
    constraint_vals = prob.constraint_forward(u)
    assert np.allclose(constraint_vals, expected_constraint_vals)

    # Test jacobian - stacked vertically
    jac = prob.jacobian()
    assert jac.shape == (4, 2)
    expected_jac = np.vstack([np.diag(-1.0 / u), np.diag(-np.exp(u))])
    assert np.allclose(jac.toarray(), expected_jac)


def test_three_constraints_different_sizes():
    """Test C_problem with three constraints of different types."""
    x = cp.Variable(3)
    obj = cp.sum(cp.exp(x))
    constraints = [
        cp.log(x) >= 0,  # 3 constraints, becomes -log(x) <= 0
        cp.exp(x) >= 0,  # 3 constraints, becomes -exp(x) <= 0
        cp.sum(cp.log(x)) >= 0,  # 1 constraint, becomes -sum(log(x)) <= 0
    ]

    problem = cp.Problem(cp.Minimize(obj), constraints)
    prob = C_problem(problem)

    u = np.array([1.0, 2.0, 3.0])
    prob.init_derivatives()

    # Test constraint_forward
    expected_constraint_vals = np.concatenate([
        -np.log(u),
        -np.exp(u),
        [-np.sum(np.log(u))]
    ])
    constraint_vals = prob.constraint_forward(u)
    assert np.allclose(constraint_vals, expected_constraint_vals)

    # Test jacobian shape and values
    jac = prob.jacobian()
    assert jac.shape == (7, 3)
    # First 3 rows: -diag(1/u), next 3 rows: -diag(exp(u)), last row: -1/u
    expected_jac = np.zeros((7, 3))
    expected_jac[:3, :] = np.diag(-1.0 / u)
    expected_jac[3:6, :] = np.diag(-np.exp(u))
    expected_jac[6, :] = -1.0 / u
    assert np.allclose(jac.toarray(), expected_jac)


def test_multiple_variables():
    """Test C_problem with multiple CVXPY variables."""
    x = cp.Variable(2)
    y = cp.Variable(2)
    obj = cp.sum(cp.log(x)) + cp.sum(cp.exp(y))
    constraints = [
        cp.log(x) >= 0,  # becomes -log(x) <= 0
        cp.exp(y) >= 0,  # becomes -exp(y) <= 0
    ]

    problem = cp.Problem(cp.Minimize(obj), constraints)
    prob = C_problem(problem)

    x_vals = np.array([1.0, 2.0])
    y_vals = np.array([0.5, 1.0])
    u = np.concatenate([x_vals, y_vals])
    prob.init_derivatives()

    # Test constraint_forward
    expected_constraint_vals = np.concatenate([-np.log(x_vals), -np.exp(y_vals)])
    constraint_vals = prob.constraint_forward(u)
    assert np.allclose(constraint_vals, expected_constraint_vals)

    # Test jacobian
    jac = prob.jacobian()
    assert jac.shape == (4, 4)
    expected_jac = np.zeros((4, 4))
    expected_jac[0, 0] = -1.0 / x_vals[0]
    expected_jac[1, 1] = -1.0 / x_vals[1]
    expected_jac[2, 2] = -np.exp(y_vals[0])
    expected_jac[3, 3] = -np.exp(y_vals[1])
    assert np.allclose(jac.toarray(), expected_jac)


def test_larger_scale():
    """Test C_problem with larger variables and multiple constraints."""
    n = 50
    x = cp.Variable(n)
    obj = cp.sum(cp.log(x))
    constraints = [
        cp.log(x) >= 0,  # n constraints
        cp.exp(x) >= 0,  # n constraints
        cp.sum(cp.log(x)) >= 0,  # 1 constraint
        cp.sum(cp.exp(x)) >= 0,  # 1 constraint
    ]

    problem = cp.Problem(cp.Minimize(obj), constraints)
    prob = C_problem(problem)

    u = np.linspace(1.0, 5.0, n)
    prob.init_derivatives()

    # Test constraint_forward
    expected_constraint_vals = np.concatenate([
        -np.log(u),
        -np.exp(u),
        [-np.sum(np.log(u))],
        [-np.sum(np.exp(u))],
    ])
    constraint_vals = prob.constraint_forward(u)
    assert np.allclose(constraint_vals, expected_constraint_vals)

    # Test jacobian shape
    jac = prob.jacobian()
    assert jac.shape == (n + n + 1 + 1, n)


def test_repeated_evaluations():
    """Test C_problem with repeated evaluations at different points."""
    x = cp.Variable(3)
    obj = cp.sum(cp.log(x))
    constraints = [cp.exp(x) >= 0]  # becomes -exp(x) <= 0

    problem = cp.Problem(cp.Minimize(obj), constraints)
    prob = C_problem(problem)

    u1 = np.array([1.0, 2.0, 3.0])
    prob.init_derivatives()

    # First evaluation
    constraint_vals1 = prob.constraint_forward(u1)
    jac1 = prob.jacobian()

    # Second evaluation at different point
    u2 = np.array([2.0, 3.0, 4.0])
    constraint_vals2 = prob.constraint_forward(u2)
    jac2 = prob.jacobian()

    assert np.allclose(constraint_vals1, -np.exp(u1))
    assert np.allclose(constraint_vals2, -np.exp(u2))
    assert np.allclose(jac1.toarray(), np.diag(-np.exp(u1)))
    assert np.allclose(jac2.toarray(), np.diag(-np.exp(u2)))


def test_hessian_objective_only():
    """Test Hessian of sum(log(x)) with no constraints.

    For f(x) = sum(log(x)), Hessian is diagonal: d²f/dx_i² = -1/x_i²
    """
    x = cp.Variable(3)
    obj = cp.sum(cp.log(x))
    problem = cp.Problem(cp.Minimize(obj))
    prob = C_problem(problem)

    u = np.array([1.0, 2.0, 4.0])
    prob.init_derivatives()

    prob.objective_forward(u)
    # No constraints, so pass empty lagrange array
    H = prob.hessian(obj_factor=1.0, lagrange=np.array([]))

    # Expected: diag([-1/1², -1/2², -1/4²]) = diag([-1, -0.25, -0.0625])
    expected_H = np.diag([-1.0, -0.25, -0.0625])
    assert np.allclose(H.toarray(), expected_H)


def test_hessian_with_constraint():
    """Test Lagrangian Hessian: sum(log(x)) + lagrange * (-exp(x)).

    Objective Hessian: diag([-1/x²])
    Constraint (-exp(x)) Hessian: diag([-exp(x)])

    Lagrangian = obj + lagrange * constraint
    """
    x = cp.Variable(2)
    obj = cp.sum(cp.log(x))
    constraints = [cp.exp(x) >= 0]  # becomes -exp(x) <= 0

    problem = cp.Problem(cp.Minimize(obj), constraints)
    prob = C_problem(problem)

    u = np.array([1.0, 2.0])
    prob.init_derivatives()

    prob.objective_forward(u)
    prob.constraint_forward(u)

    # Lagrange multipliers for the constraint (size 2)
    lagrange = np.array([0.5, 0.5])
    H = prob.hessian(obj_factor=1.0, lagrange=lagrange)

    # Constraint expr is -exp(x), so its Hessian is -exp(x)
    # H[0,0] = -1/1² + 0.5*(-exp(1))
    # H[1,1] = -1/4 + 0.5*(-exp(2))
    expected_00 = -1.0 + 0.5 * (-np.exp(1.0))
    expected_11 = -0.25 + 0.5 * (-np.exp(2.0))
    expected_H = np.diag([expected_00, expected_11])

    assert np.allclose(H.toarray(), expected_H)


def test_hessian_obj_factor_zero():
    """Test Hessian with obj_factor=0 (constraint Hessian only)."""
    x = cp.Variable(2)
    obj = cp.sum(cp.log(x))
    constraints = [cp.exp(x) >= 0]  # becomes -exp(x) <= 0

    problem = cp.Problem(cp.Minimize(obj), constraints)
    prob = C_problem(problem)

    u = np.array([1.0, 2.0])
    prob.init_derivatives()

    prob.objective_forward(u)
    prob.constraint_forward(u)

    # With obj_factor=0, only constraint Hessian contributes
    # Constraint is -exp(x), so Hessian is -exp(x)
    lagrange = np.array([1.0, 1.0])
    H = prob.hessian(obj_factor=0.0, lagrange=lagrange)

    expected_H = np.diag([-np.exp(1.0), -np.exp(2.0)])
    assert np.allclose(H.toarray(), expected_H)


def test_hessian_two_constraints():
    """Test Hessian with two constraints."""
    x = cp.Variable(2)
    obj = cp.sum(cp.log(x))
    constraints = [
        cp.log(x) >= 0,  # becomes -log(x) <= 0, Hessian: 1/x²
        cp.exp(x) >= 0,  # becomes -exp(x) <= 0, Hessian: -exp(x)
    ]

    problem = cp.Problem(cp.Minimize(obj), constraints)
    prob = C_problem(problem)

    u = np.array([2.0, 3.0])
    prob.init_derivatives()

    prob.objective_forward(u)
    prob.constraint_forward(u)

    # Lagrange: [lambda1_1, lambda1_2, lambda2_1, lambda2_2]
    # First constraint (-log): 2 elements, second constraint (-exp): 2 elements
    lagrange = np.array([1.0, 1.0, 0.5, 0.5])
    H = prob.hessian(obj_factor=1.0, lagrange=lagrange)

    # Objective Hessian: diag([-1/x²])
    # Constraint 1 (-log) Hessian: diag([1/x²]) (second derivative of -log is 1/x²)
    # Constraint 2 (-exp) Hessian: diag([-exp(x)])
    # Total: (-1/x²) + 1.0*(1/x²) + 0.5*(-exp(x)) = -0.5*exp(x)
    expected_00 = -1.0/(2.0**2) + 1.0*(1.0/(2.0**2)) + 0.5*(-np.exp(2.0))
    expected_11 = -1.0/(3.0**2) + 1.0*(1.0/(3.0**2)) + 0.5*(-np.exp(3.0))
    expected_H = np.diag([expected_00, expected_11])

    assert np.allclose(H.toarray(), expected_H)


def test_hessian_repeated_evaluations():
    """Test Hessian with repeated evaluations at different points."""
    x = cp.Variable(2)
    obj = cp.sum(cp.log(x))
    problem = cp.Problem(cp.Minimize(obj))
    prob = C_problem(problem)

    prob.init_derivatives()

    # First evaluation
    u1 = np.array([1.0, 2.0])
    prob.objective_forward(u1)
    H1 = prob.hessian(obj_factor=1.0, lagrange=np.array([]))

    # Second evaluation at different point
    u2 = np.array([2.0, 4.0])
    prob.objective_forward(u2)
    H2 = prob.hessian(obj_factor=1.0, lagrange=np.array([]))

    expected_H1 = np.diag([-1.0, -0.25])
    expected_H2 = np.diag([-0.25, -0.0625])

    assert np.allclose(H1.toarray(), expected_H1)
    assert np.allclose(H2.toarray(), expected_H2)
