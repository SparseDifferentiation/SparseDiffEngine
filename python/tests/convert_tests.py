import cvxpy as cp
import numpy as np
import DNLP_diff_engine as diffengine
from convert import convert_problem, get_jacobian


def test_sum_log():
    """Test sum(log(x)) forward and jacobian."""
    x = cp.Variable(4)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x))))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([1.0, 2.0, 3.0, 4.0])

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(test_values))
    assert np.allclose(result, expected)

    # Jacobian: d/dx sum(log(x)) = [1/x_1, 1/x_2, ...]
    jac = get_jacobian(c_obj, test_values)
    expected_jac = (1.0 / test_values).reshape(1, -1)
    assert np.allclose(jac.toarray(), expected_jac)


def test_sum_exp():
    """Test sum(exp(x)) forward and jacobian."""
    x = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.exp(x))))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([0.0, 1.0, 2.0])

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.exp(test_values))
    assert np.allclose(result, expected)

    # Jacobian: d/dx sum(exp(x)) = [exp(x_1), exp(x_2), ...]
    jac = get_jacobian(c_obj, test_values)
    expected_jac = np.exp(test_values).reshape(1, -1)
    assert np.allclose(jac.toarray(), expected_jac)


def test_two_variables_elementwise_add():
    """Test sum(log(x + y)) - elementwise after add."""
    x = cp.Variable(2)
    y = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x + y))))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([1.0, 2.0, 3.0, 4.0])

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(np.array([1+3, 2+4])))
    assert np.allclose(result, expected)

    # TODO: Jacobian for elementwise(add(...)) patterns not yet supported


def test_variable_reuse():
    """Test sum(log(x) + exp(x)) - same variable used twice."""
    x = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x) + cp.exp(x))))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([1.0, 2.0])

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(test_values) + np.exp(test_values))
    assert np.allclose(result, expected)

    # Jacobian: d/dx_i = 1/x_i + exp(x_i)
    jac = get_jacobian(c_obj, test_values)
    expected_jac = (1.0 / test_values + np.exp(test_values)).reshape(1, -1)
    assert np.allclose(jac.toarray(), expected_jac)


def test_four_variables_elementwise_add():
    """Test sum(log(a + b) + exp(c + d)) - elementwise after add."""
    a = cp.Variable(3)
    b = cp.Variable(3)
    c = cp.Variable(3)
    d = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(a + b) + cp.exp(c + d))))
    c_obj, _ = convert_problem(problem)

    a_vals = np.array([1.0, 2.0, 3.0])
    b_vals = np.array([0.5, 1.0, 1.5])
    c_vals = np.array([0.1, 0.2, 0.3])
    d_vals = np.array([0.1, 0.1, 0.1])
    test_values = np.concatenate([a_vals, b_vals, c_vals, d_vals])

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(a_vals + b_vals) + np.exp(c_vals + d_vals))
    assert np.allclose(result, expected)

    # TODO: Jacobian for elementwise(add(...)) patterns not yet supported


def test_deep_nesting():
    """Test sum(log(exp(log(exp(x))))) - deeply nested elementwise."""
    x = cp.Variable(4)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(cp.exp(cp.log(cp.exp(x)))))))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([0.5, 1.0, 1.5, 2.0])

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(np.exp(np.log(np.exp(test_values)))))
    assert np.allclose(result, expected)

    # TODO: Jacobian for nested elementwise compositions not yet supported


def test_chained_additions():
    """Test sum(x + y + z + w) - chained additions."""
    x = cp.Variable(2)
    y = cp.Variable(2)
    z = cp.Variable(2)
    w = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum(x + y + z + w)))
    c_obj, _ = convert_problem(problem)

    x_vals = np.array([1.0, 2.0])
    y_vals = np.array([3.0, 4.0])
    z_vals = np.array([5.0, 6.0])
    w_vals = np.array([7.0, 8.0])
    test_values = np.concatenate([x_vals, y_vals, z_vals, w_vals])

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(x_vals + y_vals + z_vals + w_vals)
    assert np.allclose(result, expected)

    # TODO: Jacobian for sum(add(...)) patterns not yet supported


def test_variable_used_multiple_times():
    """Test sum(log(x) + exp(x) + x) - variable used 3+ times."""
    x = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x) + cp.exp(x) + x)))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([1.0, 2.0, 3.0])

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(test_values) + np.exp(test_values) + test_values)
    assert np.allclose(result, expected)

    # TODO: Jacobian for expressions with sum(variable) not yet supported


def test_larger_variable():
    """Test sum(log(x)) with larger variable (100 elements)."""
    x = cp.Variable(100)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x))))
    c_obj, _ = convert_problem(problem)

    test_values = np.linspace(1.0, 10.0, 100)

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(test_values))
    assert np.allclose(result, expected)

    # Jacobian
    jac = get_jacobian(c_obj, test_values)
    expected_jac = (1.0 / test_values).reshape(1, -1)
    assert np.allclose(jac.toarray(), expected_jac)


def test_matrix_variable():
    """Test sum(log(X)) with 2D matrix variable (3x4)."""
    X = cp.Variable((3, 4))
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(X))))
    c_obj, _ = convert_problem(problem)

    test_values = np.arange(1.0, 13.0)  # 12 elements

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(test_values))
    assert np.allclose(result, expected)

    # Jacobian
    jac = get_jacobian(c_obj, test_values)
    expected_jac = (1.0 / test_values).reshape(1, -1)
    assert np.allclose(jac.toarray(), expected_jac)


def test_mixed_sizes():
    """Test sum(log(a)) + sum(log(b)) + sum(log(c)) with different sized variables."""
    a = cp.Variable(2)
    b = cp.Variable(5)
    c = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(a)) + cp.sum(cp.log(b)) + cp.sum(cp.log(c))))
    c_obj, _ = convert_problem(problem)

    a_vals = np.array([1.0, 2.0])
    b_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c_vals = np.array([1.0, 2.0, 3.0])
    test_values = np.concatenate([a_vals, b_vals, c_vals])

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(a_vals)) + np.sum(np.log(b_vals)) + np.sum(np.log(c_vals))
    assert np.allclose(result, expected)

    # Jacobian
    jac = get_jacobian(c_obj, test_values)
    expected_jac = (1.0 / test_values).reshape(1, -1)
    assert np.allclose(jac.toarray(), expected_jac)


def test_multiple_variables_log_exp():
    """Test sum(log(a)) + sum(log(b)) + sum(exp(c)) + sum(exp(d))."""
    a = cp.Variable(2)
    b = cp.Variable(2)
    c = cp.Variable(2)
    d = cp.Variable(2)
    obj = cp.sum(cp.log(a)) + cp.sum(cp.log(b)) + cp.sum(cp.exp(c)) + cp.sum(cp.exp(d))
    problem = cp.Problem(cp.Minimize(obj))
    c_obj, _ = convert_problem(problem)

    a_vals = np.array([1.0, 2.0])
    b_vals = np.array([0.5, 1.0])
    c_vals = np.array([0.1, 0.2])
    d_vals = np.array([0.1, 0.1])
    test_values = np.concatenate([a_vals, b_vals, c_vals, d_vals])

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = (np.sum(np.log(a_vals)) + np.sum(np.log(b_vals)) +
                np.sum(np.exp(c_vals)) + np.sum(np.exp(d_vals)))
    assert np.allclose(result, expected)

    # Jacobian
    jac = get_jacobian(c_obj, test_values)
    df_da = 1.0 / a_vals
    df_db = 1.0 / b_vals
    df_dc = np.exp(c_vals)
    df_dd = np.exp(d_vals)
    expected_jac = np.concatenate([df_da, df_db, df_dc, df_dd]).reshape(1, -1)
    assert np.allclose(jac.toarray(), expected_jac)


def test_two_variables_separate_sums():
    """Test sum(log(x) + log(y)) - two variables with separate elementwise ops."""
    x = cp.Variable(2)
    y = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x) + cp.log(y))))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([1.0, 2.0, 3.0, 4.0])

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(test_values[:2]) + np.log(test_values[2:]))
    assert np.allclose(result, expected)

    # Jacobian
    jac = get_jacobian(c_obj, test_values)
    expected_jac = np.array([[1/1, 1/2, 1/3, 1/4]])
    assert np.allclose(jac.toarray(), expected_jac)


def test_complex_objective_elementwise_add():
    """Test sum(log(x + y)) + sum(exp(y + z)) + sum(log(z + x)) - elementwise after add."""
    x = cp.Variable(3)
    y = cp.Variable(3)
    z = cp.Variable(3)
    obj = cp.sum(cp.log(x + y)) + cp.sum(cp.exp(y + z)) + cp.sum(cp.log(z + x))
    problem = cp.Problem(cp.Minimize(obj))
    c_obj, _ = convert_problem(problem)

    x_vals = np.array([1.0, 2.0, 3.0])
    y_vals = np.array([0.5, 1.0, 1.5])
    z_vals = np.array([0.2, 0.3, 0.4])
    test_values = np.concatenate([x_vals, y_vals, z_vals])

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = (np.sum(np.log(x_vals + y_vals)) +
                np.sum(np.exp(y_vals + z_vals)) +
                np.sum(np.log(z_vals + x_vals)))
    assert np.allclose(result, expected)

    # TODO: Jacobian for elementwise(add(...)) patterns not yet supported


def test_complex_objective_no_add():
    """Test sum(log(x) + exp(y) + log(z)) - multiple elementwise ops without add composition."""
    x = cp.Variable(2)
    y = cp.Variable(2)
    z = cp.Variable(2)
    obj = cp.sum(cp.log(x) + cp.exp(y) + cp.log(z))
    problem = cp.Problem(cp.Minimize(obj))
    c_obj, _ = convert_problem(problem)

    x_vals = np.array([1.0, 2.0])
    y_vals = np.array([0.5, 1.0])
    z_vals = np.array([0.2, 0.3])
    test_values = np.concatenate([x_vals, y_vals, z_vals])

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(x_vals) + np.exp(y_vals) + np.log(z_vals))
    assert np.allclose(result, expected)

    # Jacobian
    jac = get_jacobian(c_obj, test_values)
    df_dx = 1.0 / x_vals
    df_dy = np.exp(y_vals)
    df_dz = 1.0 / z_vals
    expected_jac = np.concatenate([df_dx, df_dy, df_dz]).reshape(1, -1)
    assert np.allclose(jac.toarray(), expected_jac)


def test_log_exp_identity():
    """Test sum(log(exp(x))) = sum(x) identity - nested elementwise."""
    x = cp.Variable(5)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(cp.exp(x)))))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])

    # Forward
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(test_values)  # log(exp(x)) = x
    assert np.allclose(result, expected)

    # TODO: Jacobian for nested elementwise compositions not yet supported


if __name__ == "__main__":
    test_sum_log()
    test_sum_exp()
    test_two_variables_elementwise_add()
    test_variable_reuse()
    test_four_variables_elementwise_add()
    test_deep_nesting()
    test_chained_additions()
    test_variable_used_multiple_times()
    test_larger_variable()
    test_matrix_variable()
    test_mixed_sizes()
    test_multiple_variables_log_exp()
    test_two_variables_separate_sums()
    test_complex_objective_elementwise_add()
    test_complex_objective_no_add()
    test_log_exp_identity()
    print("All tests passed!")
