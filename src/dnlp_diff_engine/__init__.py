"""
DNLP Diff Engine - Automatic differentiation for nonlinear optimization.

This package provides automatic differentiation capabilities for CVXPY problems,
computing gradients, Jacobians, and Hessians needed by NLP solvers.
"""

import cvxpy as cp
import numpy as np
from cvxpy.reductions.inverse_data import InverseData
from scipy import sparse

from . import _core as _diffengine

__all__ = ["C_problem", "convert_problem", "build_variable_dict"]


def _chain_add(children):
    """Chain multiple children with binary adds: a + b + c -> add(add(a, b), c)."""
    result = children[0]
    for child in children[1:]:
        result = _diffengine.make_add(result, child)
    return result


def _convert_matmul(expr, children):
    """Convert matrix multiplication A @ f(x) or f(x) @ A."""
    # MulExpression has args: [left, right]
    # One of them should be a Constant, the other a variable expression
    left_arg, right_arg = expr.args

    if isinstance(left_arg, cp.Constant):
        # A @ f(x) -> left_matmul
        A = np.asarray(left_arg.value, dtype=np.float64)
        if A.ndim == 1:
            A = A.reshape(1, -1)  # Convert 1D to row vector
        A_csr = sparse.csr_matrix(A)
        m, n = A_csr.shape
        return _diffengine.make_left_matmul(
            children[1],  # right child is the variable expression
            A_csr.data.astype(np.float64),
            A_csr.indices.astype(np.int32),
            A_csr.indptr.astype(np.int32),
            m,
            n,
        )
    elif isinstance(right_arg, cp.Constant):
        # f(x) @ A -> right_matmul
        A = np.asarray(right_arg.value, dtype=np.float64)
        if A.ndim == 1:
            A = A.reshape(-1, 1)  # Convert 1D to column vector
        A_csr = sparse.csr_matrix(A)
        m, n = A_csr.shape
        return _diffengine.make_right_matmul(
            children[0],  # left child is the variable expression
            A_csr.data.astype(np.float64),
            A_csr.indices.astype(np.int32),
            A_csr.indptr.astype(np.int32),
            m,
            n,
        )
    else:
        raise NotImplementedError("MulExpression with two non-constant args not supported")


def _convert_multiply(expr, children):
    """Convert multiplication based on argument types."""
    # multiply has args: [left, right]
    left_arg, right_arg = expr.args

    # Check if left is a constant
    if isinstance(left_arg, cp.Constant):
        value = np.asarray(left_arg.value, dtype=np.float64)

        # Scalar constant
        if value.size == 1:
            scalar = float(value.flat[0])
            return _diffengine.make_const_scalar_mult(children[1], scalar)

        # Vector constant
        if value.ndim == 1 or (value.ndim == 2 and min(value.shape) == 1):
            vector = value.flatten()
            return _diffengine.make_const_vector_mult(children[1], vector)

    # Check if right is a constant
    elif isinstance(right_arg, cp.Constant):
        value = np.asarray(right_arg.value, dtype=np.float64)

        # Scalar constant
        if value.size == 1:
            scalar = float(value.flat[0])
            return _diffengine.make_const_scalar_mult(children[0], scalar)

        # Vector constant
        if value.ndim == 1 or (value.ndim == 2 and min(value.shape) == 1):
            vector = value.flatten()
            return _diffengine.make_const_vector_mult(children[0], vector)

    # Neither is constant, use general multiply
    return _diffengine.make_multiply(children[0], children[1])


def _extract_flat_indices_from_index(expr):
    """Extract flattened indices from CVXPY index expression."""
    parent_shape = expr.args[0].shape
    indices_per_dim = [np.arange(s.start, s.stop, s.step) for s in expr.key]

    if len(indices_per_dim) == 1:
        return indices_per_dim[0].astype(np.int32)
    elif len(indices_per_dim) == 2:
        # Fortran order: idx = row + col * n_rows
        return np.add.outer(
            indices_per_dim[0], indices_per_dim[1] * parent_shape[0]
        ).flatten(order="F").astype(np.int32)
    else:
        raise NotImplementedError("index with >2 dimensions not supported")


def _extract_flat_indices_from_special_index(expr):
    """Extract flattened indices from CVXPY special_index expression."""
    return np.reshape(expr._select_mat, expr._select_mat.size, order="F").astype(np.int32)


def _convert_reshape(expr, children):
    """Convert reshape - only Fortran order is supported.

    For Fortran order, reshape is a no-op since the underlying data layout
    is unchanged. We just pass through the child expression.

    Note: Only order='F' (Fortran/column-major) is supported. This is the
    default and most common case in CVXPY. C order would require permutation.
    """
    if expr.order != 'F':
        raise NotImplementedError(
            f"reshape with order='{expr.order}' not supported. "
            "Only order='F' (Fortran) is currently supported."
        )
    # Pass through - data layout is unchanged with Fortran order
    return children[0]


def _convert_quad_form(expr, children):
    """Convert quadratic form x.T @ P @ x."""

    P_arg = expr.args[1]

    if not isinstance(P_arg, cp.Constant):
        raise NotImplementedError("quad_form requires P to be a constant matrix")

    P = np.asarray(P_arg.value, dtype=np.float64)
    if P.ndim == 1:
        P = P.reshape(-1, 1)

    P_csr = sparse.csr_matrix(P)
    m, n = P_csr.shape

    return _diffengine.make_quad_form(
        children[0],  # x expression
        P_csr.data.astype(np.float64),
        P_csr.indices.astype(np.int32),
        P_csr.indptr.astype(np.int32),
        m,
        n,
    )


# Mapping from CVXPY atom names to C diff engine functions
# Converters receive (expr, children) where expr is the CVXPY expression
ATOM_CONVERTERS = {
    # Elementwise unary
    "log": lambda _expr, children: _diffengine.make_log(children[0]),
    "exp": lambda _expr, children: _diffengine.make_exp(children[0]),
    # Affine unary
    "NegExpression": lambda _expr, children: _diffengine.make_neg(children[0]),
    "Promote": lambda expr, children: _diffengine.make_promote(
        children[0],
        expr.shape[0] if len(expr.shape) >= 1 else 1,
        expr.shape[1] if len(expr.shape) >= 2 else 1,
    ),
    # N-ary (handles 2+ args)
    "AddExpression": lambda _expr, children: _chain_add(children),
    # Reductions
    "Sum": lambda _expr, children: _diffengine.make_sum(children[0], -1),
    # Bivariate
    "multiply": _convert_multiply,
    "QuadForm": _convert_quad_form,
    # Matrix multiplication
    "MulExpression": _convert_matmul,
    # Elementwise univariate with parameter
    "power": lambda expr, children: _diffengine.make_power(children[0], float(expr.p.value)),
    # Trigonometric
    "sin": lambda _expr, children: _diffengine.make_sin(children[0]),
    "cos": lambda _expr, children: _diffengine.make_cos(children[0]),
    "tan": lambda _expr, children: _diffengine.make_tan(children[0]),
    # Hyperbolic
    "sinh": lambda _expr, children: _diffengine.make_sinh(children[0]),
    "tanh": lambda _expr, children: _diffengine.make_tanh(children[0]),
    "asinh": lambda _expr, children: _diffengine.make_asinh(children[0]),
    "atanh": lambda _expr, children: _diffengine.make_atanh(children[0]),
    # Other elementwise
    "entr": lambda _expr, children: _diffengine.make_entr(children[0]),
    "logistic": lambda _expr, children: _diffengine.make_logistic(children[0]),
    "xexp": lambda _expr, children: _diffengine.make_xexp(children[0]),
    # Indexing/slicing
    "index": lambda expr, children: _diffengine.make_index(
        children[0], _extract_flat_indices_from_index(expr)
    ),
    "special_index": lambda expr, children: _diffengine.make_index(
        children[0], _extract_flat_indices_from_special_index(expr)
    ),
    # Reshape (Fortran order only - pass-through since data layout unchanged)
    "reshape": _convert_reshape,
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
        offset, _ = id_map[var.id]
        shape = var_shapes[var.id]
        if len(shape) == 2:
            d1, d2 = shape[0], shape[1]
        elif len(shape) == 1:
            d1, d2 = shape[0], 1
        else:  # scalar
            d1, d2 = 1, 1
        c_var = _diffengine.make_variable(d1, d2, offset, n_vars)
        var_dict[var.id] = c_var

    return var_dict, n_vars


def _convert_expr(expr, var_dict: dict, n_vars: int):
    """Convert CVXPY expression using pre-built variable dictionary."""
    # Base case: variable lookup
    if isinstance(expr, cp.Variable):
        return var_dict[expr.id]

    # Base case: constant
    if isinstance(expr, cp.Constant):
        value = np.asarray(expr.value, dtype=np.float64).flatten()
        d1 = expr.shape[0] if len(expr.shape) >= 1 else 1
        d2 = expr.shape[1] if len(expr.shape) >= 2 else 1
        return _diffengine.make_constant(d1, d2, n_vars, value)

    # Recursive case: atoms
    atom_name = type(expr).__name__

    if atom_name in ATOM_CONVERTERS:
        children = [_convert_expr(arg, var_dict, n_vars) for arg in expr.args]
        return ATOM_CONVERTERS[atom_name](expr, children)

    raise NotImplementedError(f"Atom '{atom_name}' not supported")


def convert_expressions(problem: cp.Problem) -> tuple:
    """
    Convert CVXPY Problem to C expressions (low-level).

    Args:
        problem: CVXPY Problem object

    Returns:
        c_objective: C expression for objective
        c_constraints: list of C expressions for constraints
    """
    var_dict, n_vars = build_variable_dict(problem.variables())

    # Convert objective
    c_objective = _convert_expr(problem.objective.expr, var_dict, n_vars)

    # Convert constraints (expression part only for now)
    c_constraints = []
    for constr in problem.constraints:
        c_expr = _convert_expr(constr.expr, var_dict, n_vars)
        c_constraints.append(c_expr)

    return c_objective, c_constraints


def convert_problem(problem: cp.Problem) -> "C_problem":
    """
    Convert CVXPY Problem to C problem struct.

    Args:
        problem: CVXPY Problem object

    Returns:
        C_Problem wrapper around C problem struct
    """
    return C_problem(problem)


class C_problem:
    """Wrapper around C problem struct for CVXPY problems."""

    def __init__(self, cvxpy_problem: cp.Problem):
        var_dict, n_vars = build_variable_dict(cvxpy_problem.variables())
        c_obj = _convert_expr(cvxpy_problem.objective.expr, var_dict, n_vars)
        c_constraints = [_convert_expr(c.expr, var_dict, n_vars) for c in cvxpy_problem.constraints]
        self._capsule = _diffengine.make_problem(c_obj, c_constraints)
        self._allocated = False

    def init_derivatives(self):
        """Initialize derivative structures. Must be called before forward/gradient/jacobian."""
        _diffengine.problem_init_derivatives(self._capsule)
        self._allocated = True

    def objective_forward(self, u: np.ndarray) -> float:
        """Evaluate objective. Returns obj_value float."""
        return _diffengine.problem_objective_forward(self._capsule, u)

    def constraint_forward(self, u: np.ndarray) -> np.ndarray:
        """Evaluate constraints only. Returns constraint_values array."""
        return _diffengine.problem_constraint_forward(self._capsule, u)

    def gradient(self) -> np.ndarray:
        """Compute gradient of objective. Call objective_forward first. Returns gradient array."""
        return _diffengine.problem_gradient(self._capsule)

    def jacobian(self) -> sparse.csr_matrix:
        """Compute constraint Jacobian. Call constraint_forward first."""
        data, indices, indptr, shape = _diffengine.problem_jacobian(self._capsule)
        return sparse.csr_matrix((data, indices, indptr), shape=shape)

    def hessian(self, obj_factor: float, lagrange: np.ndarray) -> sparse.csr_matrix:
        """Compute Lagrangian Hessian.

        Computes: obj_factor * H_obj + sum(lagrange_i * H_constraint_i)

        Call objective_forward and constraint_forward before this.

        Args:
            obj_factor: Weight for objective Hessian
            lagrange: Array of Lagrange multipliers (length = total_constraint_size)

        Returns:
            scipy CSR matrix of shape (n_vars, n_vars)
        """
        data, indices, indptr, shape = _diffengine.problem_hessian(
            self._capsule, obj_factor, lagrange
        )
        return sparse.csr_matrix((data, indices, indptr), shape=shape)
