import cvxpy as cp
import numpy as np
from scipy import sparse
from cvxpy.reductions.inverse_data import InverseData
import DNLP_diff_engine as diffengine


def get_jacobian(c_expr, values):
    """Compute jacobian and return as scipy sparse CSR matrix."""
    data, indices, indptr, shape = diffengine.jacobian(c_expr, values)
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


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
        offset, _ = id_map[var.id]
        shape = var_shapes[var.id]
        if len(shape) == 2:
            d1, d2 = shape[0], shape[1]
        elif len(shape) == 1:
            d1, d2 = shape[0], 1
        else:  # scalar
            d1, d2 = 1, 1
        c_var = diffengine.make_variable(d1, d2, offset, n_vars)
        var_dict[var.id] = c_var

    return var_dict


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


def convert_expressions(problem: cp.Problem) -> tuple:
    """
    Convert CVXPY Problem to C expressions (low-level).

    Args:
        problem: CVXPY Problem object

    Returns:
        c_objective: C expression for objective
        c_constraints: list of C expressions for constraints
    """
    var_dict = build_variable_dict(problem.variables())

    # Convert objective
    c_objective = _convert_expr(problem.objective.expr, var_dict)

    # Convert constraints (expression part only for now)
    c_constraints = []
    for constr in problem.constraints:
        c_expr = _convert_expr(constr.expr, var_dict)
        c_constraints.append(c_expr)

    return c_objective, c_constraints


def convert_problem(problem: cp.Problem) -> "Problem":
    """
    Convert CVXPY Problem to C problem struct.

    Args:
        problem: CVXPY Problem object

    Returns:
        Problem wrapper around C problem struct
    """
    return Problem(problem)


class Problem:
    """Wrapper around C problem struct for CVXPY problems."""

    def __init__(self, cvxpy_problem: cp.Problem):
        var_dict = build_variable_dict(cvxpy_problem.variables())
        c_obj = _convert_expr(cvxpy_problem.objective.expr, var_dict)
        c_constraints = [_convert_expr(c.expr, var_dict) for c in cvxpy_problem.constraints]
        self._capsule = diffengine.make_problem(c_obj, c_constraints)
        self._allocated = False

    def allocate(self, u: np.ndarray):
        """Allocate internal buffers. Must be called before forward/gradient/jacobian."""
        diffengine.problem_allocate(self._capsule, u)
        self._allocated = True

    def forward(self, u: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate objective and constraints. Returns (obj_value, constraint_values)."""
        return diffengine.problem_forward(self._capsule, u)

    def gradient(self, u: np.ndarray) -> np.ndarray:
        """Compute gradient of objective. Returns gradient array."""
        return diffengine.problem_gradient(self._capsule, u)

    def jacobian(self, u: np.ndarray) -> sparse.csr_matrix:
        """Compute jacobian of constraints. Returns scipy CSR matrix."""
        data, indices, indptr, shape = diffengine.problem_jacobian(self._capsule, u)
        return sparse.csr_matrix((data, indices, indptr), shape=shape)
