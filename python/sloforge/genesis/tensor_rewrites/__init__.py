"""Semantics-aware, explicitly bounded tensor rewrite surface."""

from .builtins import BUILTIN_RULES
from .engine import apply_rule, explore_rewrites, structural_key, validate_graph
from .model import (
    Dimension,
    DType,
    Layout,
    NumericalContract,
    OperatorParameters,
    RewriteApplication,
    RewriteCandidate,
    RewriteError,
    RewriteKind,
    RewriteRule,
    SemanticCategory,
    TensorGraph,
    TensorNode,
    TensorSpec,
)

__all__ = [
    "BUILTIN_RULES",
    "DType",
    "Dimension",
    "Layout",
    "NumericalContract",
    "OperatorParameters",
    "RewriteApplication",
    "RewriteCandidate",
    "RewriteError",
    "RewriteKind",
    "RewriteRule",
    "SemanticCategory",
    "TensorGraph",
    "TensorNode",
    "TensorSpec",
    "apply_rule",
    "explore_rewrites",
    "structural_key",
    "validate_graph",
]
