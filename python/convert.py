import cvxpy as cp
import numpy as np
from cvxpy.reductions.inverse_data import InverseData
import DNLP_diff_engine as diffengine


def _chain_add(children):
    """Chain multiple children with binary adds: a + b + c -> add(add(a, b), c)."""
    result = children[0]
    for child in children[1:]:
        result = diffengine.make_add(result, child)
    return result


# Mapping from CVXPY atom names to C diff engine functions
ATOM_CONVERTERS = {
    # Elementwise unary
    "log": lambda child: diffengine.make_log(child),
    "exp": lambda child: diffengine.make_exp(child),

    # N-ary (handles 2+ args)
    "AddExpression": _chain_add,

    # Reductions
    "Sum": lambda child: diffengine.make_sum(child, -1),  # axis=-1 = sum all
}


def build_variable_dict(variables: list) -> tuple[dict, int]:
    """
    Build dictionary mapping CVXPY variable ids to C variables.

    Args:
        variables: list of CVXPY Variable objects

    Returns:
        var_dict: {var.id: c_variable} mapping
        n_vars: total number of scalar variables
    """
    id_map, _, n_vars, var_shapes = InverseData.get_var_offsets(variables)

    var_dict = {}
    for var in variables:
        offset, size = id_map[var.id]
        shape = var_shapes[var.id]
        if len(shape) == 2:
            d1, d2 = shape[0], shape[1]
        elif len(shape) == 1:
            d1, d2 = shape[0], 1
        else:  # scalar
            d1, d2 = 1, 1
        c_var = diffengine.make_variable(d1, d2, offset, n_vars)
        var_dict[var.id] = c_var

    return var_dict, n_vars


def _convert_expr(expr, var_dict: dict):
    """Convert CVXPY expression using pre-built variable dictionary."""
    # Base case: variable lookup
    if isinstance(expr, cp.Variable):
        return var_dict[expr.id]

    # Recursive case: atoms
    atom_name = type(expr).__name__
    if atom_name in ATOM_CONVERTERS:
        children = [_convert_expr(arg, var_dict) for arg in expr.args]
        converter = ATOM_CONVERTERS[atom_name]
        # N-ary ops (like AddExpression) take list, unary ops take single arg
        if atom_name == "AddExpression":
            return converter(children)
        return converter(*children) if len(children) > 1 else converter(children[0])

    raise NotImplementedError(f"Atom '{atom_name}' not supported")


def convert_problem(problem: cp.Problem) -> tuple:
    """
    Convert CVXPY Problem to C expressions.

    Args:
        problem: CVXPY Problem object

    Returns:
        c_objective: C expression for objective
        c_constraints: list of C expressions for constraints
    """
    var_dict, _ = build_variable_dict(problem.variables())

    # Convert objective
    c_objective = _convert_expr(problem.objective.expr, var_dict)

    # Convert constraints (expression part only for now)
    c_constraints = []
    for constr in problem.constraints:
        c_expr = _convert_expr(constr.expr, var_dict)
        c_constraints.append(c_expr)

    return c_objective, c_constraints


def cvxpy_expr_to_C(expr):
    """
    Convert a standalone CVXPY expression to C.
    Collects variables from the expression and builds variable dict.
    """
    # Collect variables
    variables = []
    seen_ids = set()

    def collect_vars(e):
        if isinstance(e, cp.Variable):
            if e.id not in seen_ids:
                variables.append(e)
                seen_ids.add(e.id)
        elif hasattr(e, 'args'):
            for arg in e.args:
                collect_vars(arg)

    collect_vars(expr)

    var_dict, _ = build_variable_dict(variables)
    return _convert_expr(expr, var_dict)


# === Problem-based tests ===

def test_problem_simple_sum_log():
    """Test converting cp.sum(cp.log(x))."""
    x = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x))))
    c_obj, c_constrs = convert_problem(problem)

    test_values = np.array([1.0, 2.0, 3.0])
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(test_values))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_problem_simple_sum_log passed")


def test_problem_two_variables():
    """Test problem with two variables: sum(log(x + y))."""
    x = cp.Variable(2)
    y = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x + y))))
    c_obj, c_constrs = convert_problem(problem)

    # Variables flattened: [x0, x1, y0, y1]
    test_values = np.array([1.0, 2.0, 3.0, 4.0])
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(np.array([1+3, 2+4])))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_problem_two_variables passed")


def test_variable_reuse():
    """Test that same variable used twice works correctly."""
    x = cp.Variable(2)
    # log(x) + exp(x) uses x twice
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x) + cp.exp(x))))
    c_obj, c_constrs = convert_problem(problem)

    test_values = np.array([1.0, 2.0])
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(test_values) + np.exp(test_values))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_variable_reuse passed")


# === Standalone expression tests ===

def test_log_conversion():
    """Test converting CVXPY log expression to C diff engine."""
    x = cp.Variable(3)
    c_expr = cvxpy_expr_to_C(cp.log(x))

    test_values = np.array([1.0, 2.0, 3.0])
    result = diffengine.forward(c_expr, test_values)
    expected = np.log(test_values)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_log_conversion passed")


def test_exp_conversion():
    """Test converting CVXPY exp expression to C diff engine."""
    x = cp.Variable(3)
    c_expr = cvxpy_expr_to_C(cp.exp(x))

    test_values = np.array([0.0, 1.0, 2.0])
    result = diffengine.forward(c_expr, test_values)
    expected = np.exp(test_values)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_exp_conversion passed")


def test_add_conversion():
    """Test converting CVXPY addition expression to C diff engine."""
    x = cp.Variable(3)
    y = cp.Variable(3)
    c_expr = cvxpy_expr_to_C(x + y)

    test_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = diffengine.forward(c_expr, test_values)
    expected = test_values[:3] + test_values[3:]
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_add_conversion passed")


def test_composite_expression():
    """Test converting a composite CVXPY expression: log(exp(x) + y)."""
    x = cp.Variable(2)
    y = cp.Variable(2)
    c_expr = cvxpy_expr_to_C(cp.log(cp.exp(x) + y))

    test_values = np.array([1.0, 2.0, 0.5, 1.0])
    result = diffengine.forward(c_expr, test_values)
    expected = np.log(np.exp(test_values[:2]) + test_values[2:])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_composite_expression passed")


# === Complex tests ===

def test_four_variables():
    """Test problem with 4 variables: sum(log(a + b) + exp(c + d))."""
    a = cp.Variable(3)
    b = cp.Variable(3)
    c = cp.Variable(3)
    d = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(a + b) + cp.exp(c + d))))
    c_obj, _ = convert_problem(problem)

    # Variables flattened: [a0,a1,a2, b0,b1,b2, c0,c1,c2, d0,d1,d2]
    a_vals = np.array([1.0, 2.0, 3.0])
    b_vals = np.array([0.5, 1.0, 1.5])
    c_vals = np.array([0.1, 0.2, 0.3])
    d_vals = np.array([0.1, 0.1, 0.1])
    test_values = np.concatenate([a_vals, b_vals, c_vals, d_vals])

    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(a_vals + b_vals) + np.exp(c_vals + d_vals))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_four_variables passed")


def test_deep_nesting():
    """Test deeply nested composition: log(exp(log(exp(x))))."""
    x = cp.Variable(4)
    expr = cp.log(cp.exp(cp.log(cp.exp(x))))
    c_expr = cvxpy_expr_to_C(expr)

    test_values = np.array([0.5, 1.0, 1.5, 2.0])
    result = diffengine.forward(c_expr, test_values)
    # log(exp(log(exp(x)))) = log(exp(x)) = x (for positive x)
    expected = np.log(np.exp(np.log(np.exp(test_values))))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_deep_nesting passed")


def test_chained_additions():
    """Test multiple chained additions: x + y + z + w."""
    x = cp.Variable(2)
    y = cp.Variable(2)
    z = cp.Variable(2)
    w = cp.Variable(2)
    c_expr = cvxpy_expr_to_C(x + y + z + w)

    x_vals = np.array([1.0, 2.0])
    y_vals = np.array([3.0, 4.0])
    z_vals = np.array([5.0, 6.0])
    w_vals = np.array([7.0, 8.0])
    test_values = np.concatenate([x_vals, y_vals, z_vals, w_vals])

    result = diffengine.forward(c_expr, test_values)
    expected = x_vals + y_vals + z_vals + w_vals
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_chained_additions passed")


def test_variable_used_multiple_times():
    """Test variable used 3+ times: log(x) + exp(x) + x."""
    x = cp.Variable(3)
    expr = cp.log(x) + cp.exp(x) + x
    c_expr = cvxpy_expr_to_C(expr)

    test_values = np.array([1.0, 2.0, 3.0])
    result = diffengine.forward(c_expr, test_values)
    expected = np.log(test_values) + np.exp(test_values) + test_values
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_variable_used_multiple_times passed")


def test_larger_variable_size():
    """Test with larger variable (100 elements)."""
    x = cp.Variable(100)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(cp.exp(x)))))
    c_obj, _ = convert_problem(problem)

    test_values = np.linspace(0.1, 2.0, 100)
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(np.exp(test_values)))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_larger_variable_size passed")


def test_matrix_variable():
    """Test with 2D matrix variable (3x4)."""
    X = cp.Variable((3, 4))
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(X))))
    c_obj, _ = convert_problem(problem)

    # Matrix flattened in column-major (Fortran) order by CVXPY
    test_values = np.arange(1.0, 13.0)  # 12 elements
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(test_values))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_matrix_variable passed")


def test_mixed_sizes():
    """Test with variables of different sizes."""
    a = cp.Variable(2)
    b = cp.Variable(5)
    c = cp.Variable(3)

    # sum of each variable's log, then sum all
    expr = cp.sum(cp.log(a)) + cp.sum(cp.log(b)) + cp.sum(cp.log(c))
    c_expr = cvxpy_expr_to_C(expr)

    a_vals = np.array([1.0, 2.0])
    b_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c_vals = np.array([1.0, 2.0, 3.0])
    test_values = np.concatenate([a_vals, b_vals, c_vals])

    result = diffengine.forward(c_expr, test_values)
    expected = np.sum(np.log(a_vals)) + np.sum(np.log(b_vals)) + np.sum(np.log(c_vals))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_mixed_sizes passed")


def test_complex_objective():
    """Test complex objective: sum(log(x + y)) + sum(exp(y + z)) + sum(log(z + x))."""
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

    result = diffengine.forward(c_obj, test_values)
    expected = (np.sum(np.log(x_vals + y_vals)) +
                np.sum(np.exp(y_vals + z_vals)) +
                np.sum(np.log(z_vals + x_vals)))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_complex_objective passed")


def test_nested_sums():
    """Test sum of sum (should collapse to single sum)."""
    x = cp.Variable(4)
    # sum(sum(log(x))) - outer sum is over a scalar, so effectively just sum(log(x))
    expr = cp.sum(cp.log(x))
    c_expr = cvxpy_expr_to_C(expr)

    test_values = np.array([1.0, 2.0, 3.0, 4.0])
    result = diffengine.forward(c_expr, test_values)
    expected = np.sum(np.log(test_values))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_nested_sums passed")


if __name__ == "__main__":
    # Problem-based tests
    test_problem_simple_sum_log()
    test_problem_two_variables()
    test_variable_reuse()

    # Standalone expression tests
    test_log_conversion()
    test_exp_conversion()
    test_add_conversion()
    test_composite_expression()

    # Complex tests
    test_four_variables()
    test_deep_nesting()
    test_chained_additions()
    test_variable_used_multiple_times()
    test_larger_variable_size()
    test_matrix_variable()
    test_mixed_sizes()
    test_complex_objective()
    test_nested_sums()

    print("\nAll tests passed!")
