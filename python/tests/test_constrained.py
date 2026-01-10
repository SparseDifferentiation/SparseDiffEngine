import cvxpy as cp
import numpy as np
from convert import C_problem

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


if __name__ == "__main__":
    test_single_constraint()
    print("test_single_constraint passed!")
    test_two_constraints()
    print("test_two_constraints passed!")
    test_three_constraints_different_sizes()
    print("test_three_constraints_different_sizes passed!")
    test_multiple_variables()
    print("test_multiple_variables passed!")
    test_larger_scale()
    print("test_larger_scale passed!")
    test_repeated_evaluations()
    print("test_repeated_evaluations passed!")
    print("\nAll constrained tests passed!")
