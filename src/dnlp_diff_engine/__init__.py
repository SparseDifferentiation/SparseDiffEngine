"""
DNLP Diff Engine - Low-level C bindings for automatic differentiation.

This package provides the raw C extension for building expression trees
and computing derivatives. For CVXPY integration, use:
    from cvxpy.reductions.solvers.nlp_solvers.diff_engine import C_problem
"""

# Re-export all C functions directly from the _core extension
from ._core import *  # noqa: F401, F403

__all__ = [
    # Atom constructors
    "make_variable",
    "make_constant",
    "make_add",
    "make_neg",
    "make_sum",
    "make_promote",
    "make_index",
    "make_reshape",
    "make_log",
    "make_exp",
    "make_power",
    "make_entr",
    "make_logistic",
    "make_xexp",
    "make_sin",
    "make_cos",
    "make_tan",
    "make_sinh",
    "make_tanh",
    "make_asinh",
    "make_atanh",
    "make_multiply",
    "make_const_scalar_mult",
    "make_const_vector_mult",
    "make_left_matmul",
    "make_right_matmul",
    "make_quad_form",
    "make_quad_over_lin",
    "make_rel_entr",
    "make_prod",
    # Problem interface
    "make_problem",
    "problem_init_derivatives",
    "problem_objective_forward",
    "problem_constraint_forward",
    "problem_gradient",
    "problem_jacobian",
    "problem_hessian",
]
