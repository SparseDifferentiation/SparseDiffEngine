import numpy as np
from scipy import sparse
import DNLP_diff_engine as diffengine


def test_problem_objective_forward():
    """Test problem_objective_forward and problem_constraint_forward (low-level)."""
    n_vars = 3
    x = diffengine.make_variable(n_vars, 1, 0, n_vars)
    log_x = diffengine.make_log(x)
    objective = diffengine.make_sum(log_x, -1)
    constraints = [log_x]

    prob = diffengine.make_problem(objective, constraints)
    u = np.array([1.0, 2.0, 3.0])
    diffengine.problem_init_derivatives(prob)

    obj_val = diffengine.problem_objective_forward(prob, u)
    constraint_vals = diffengine.problem_constraint_forward(prob, u)

    expected_obj = np.sum(np.log(u))
    assert np.allclose(obj_val, expected_obj)
    assert np.allclose(constraint_vals, np.log(u))


def test_problem_constraint_forward():
    """Test problem_constraint_forward for constraint values only (low-level)."""
    n_vars = 2
    x = diffengine.make_variable(n_vars, 1, 0, n_vars)

    log_obj = diffengine.make_log(x)
    objective = diffengine.make_sum(log_obj, -1)

    log_c = diffengine.make_log(x)
    exp_c = diffengine.make_exp(x)
    constraints = [log_c, exp_c]

    prob = diffengine.make_problem(objective, constraints)
    u = np.array([2.0, 4.0])
    diffengine.problem_init_derivatives(prob)

    constraint_vals = diffengine.problem_constraint_forward(prob, u)

    # Expected: [log(2), log(4), exp(2), exp(4)]
    expected = np.concatenate([np.log(u), np.exp(u)])
    assert np.allclose(constraint_vals, expected)


def test_problem_gradient():
    """Test problem_gradient for objective gradient (low-level)."""
    n_vars = 3
    x = diffengine.make_variable(n_vars, 1, 0, n_vars)
    log_x = diffengine.make_log(x)
    objective = diffengine.make_sum(log_x, -1)

    prob = diffengine.make_problem(objective, [])
    u = np.array([1.0, 2.0, 4.0])
    diffengine.problem_init_derivatives(prob)

    diffengine.problem_objective_forward(prob, u)
    grad = diffengine.problem_gradient(prob, u)
    expected_grad = 1.0 / u
    assert np.allclose(grad, expected_grad)


def test_problem_jacobian():
    """Test problem_jacobian for constraint jacobian (low-level)."""
    n_vars = 2
    x = diffengine.make_variable(n_vars, 1, 0, n_vars)
    log_x = diffengine.make_log(x)
    objective = diffengine.make_sum(log_x, -1)
    constraints = [log_x]

    prob = diffengine.make_problem(objective, constraints)
    u = np.array([2.0, 4.0])
    diffengine.problem_init_derivatives(prob)

    diffengine.problem_constraint_forward(prob, u)
    data, indices, indptr, shape = diffengine.problem_jacobian(prob, u)
    jac = sparse.csr_matrix((data, indices, indptr), shape=shape)

    expected_jac = np.diag(1.0 / u)
    assert np.allclose(jac.toarray(), expected_jac)


def test_problem_no_constraints():
    """Test Problem with no constraints (low-level)."""
    n_vars = 3
    x = diffengine.make_variable(n_vars, 1, 0, n_vars)
    log_x = diffengine.make_log(x)
    objective = diffengine.make_sum(log_x, -1)

    prob = diffengine.make_problem(objective, [])
    u = np.array([1.0, 2.0, 3.0])
    diffengine.problem_init_derivatives(prob)

    obj_val = diffengine.problem_objective_forward(prob, u)
    constraint_vals = diffengine.problem_constraint_forward(prob, u)
    assert np.allclose(obj_val, np.sum(np.log(u)))
    assert len(constraint_vals) == 0

    diffengine.problem_objective_forward(prob, u)
    grad = diffengine.problem_gradient(prob, u)
    assert np.allclose(grad, 1.0 / u)

    diffengine.problem_constraint_forward(prob, u)
    data, indices, indptr, shape = diffengine.problem_jacobian(prob, u)
    jac = sparse.csr_matrix((data, indices, indptr), shape=shape)
    assert jac.shape == (0, 3)


def test_problem_multiple_constraints():
    """Test problem with multiple different constraints (low-level)."""
    n_vars = 3
    x = diffengine.make_variable(n_vars, 1, 0, n_vars)
    
    # Objective: sum(log(x))
    log_x = diffengine.make_log(x)
    objective = diffengine.make_sum(log_x, -1)
    
    # Multiple constraints using the same variable:
    # Constraint 1: log(x) - reused from objective
    # Constraint 2: exp(x)
    exp_x = diffengine.make_exp(x)
    constraints = [log_x, exp_x]
    
    prob = diffengine.make_problem(objective, constraints)
    u = np.array([1.0, 2.0, 3.0])
    diffengine.problem_init_derivatives(prob)
    
    # Test forward pass
    obj_val = diffengine.problem_objective_forward(prob, u)
    constraint_vals = diffengine.problem_constraint_forward(prob, u)
    expected_obj = np.sum(np.log(u))
    expected_constraints = np.concatenate([np.log(u), np.exp(u)])
    assert np.allclose(obj_val, expected_obj)
    assert np.allclose(constraint_vals, expected_constraints)

    # Test gradient
    diffengine.problem_objective_forward(prob, u)
    grad = diffengine.problem_gradient(prob, u)
    expected_grad = 1.0 / u
    assert np.allclose(grad, expected_grad)

    # Test Jacobian
    diffengine.problem_constraint_forward(prob, u)
    data, indices, indptr, shape = diffengine.problem_jacobian(prob, u)
    jac = sparse.csr_matrix((data, indices, indptr), shape=shape)
    
    # Expected Jacobian: 
    # log(x): diag(1/u)
    # exp(x): diag(exp(u))
    expected_jac = np.vstack([
        np.diag(1.0 / u),
        np.diag(np.exp(u))
    ])
    assert jac.shape == (6, 3)
    assert np.allclose(jac.toarray(), expected_jac)


if __name__ == "__main__":
    test_problem_objective_forward()
    test_problem_constraint_forward()
    test_problem_gradient()
    test_problem_jacobian()
    test_problem_no_constraints()
    test_problem_multiple_constraints()
    print("All native problem tests passed!")
