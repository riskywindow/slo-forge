"""Typed tensor-expression and rewrite contracts for Genesis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


class DType(StrEnum):
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    INT64 = "int64"
    INT32 = "int32"
    INT8 = "int8"


class Layout(StrEnum):
    CONTIGUOUS = "contiguous"
    STRIDED = "strided"
    CHANNELS_LAST = "channels_last"
    PAGED = "paged"


class SemanticCategory(StrEnum):
    EXACT = "semantics_preserving"
    APPROXIMATE = "approximate_within_quality_budget"
    EXPERIMENTAL = "experimental_operator_review"


class RewriteKind(StrEnum):
    REDUNDANT_CAST = "redundant_cast"
    DOUBLE_TRANSPOSE = "double_transpose"
    ADD_ZERO = "add_zero"
    MUL_ONE = "mul_one"
    REASSOCIATE_ADD = "reassociate_add"
    STATE_UPDATE_FUSION = "state_update_fusion"
    QUANTIZE_OUTPUT = "quantize_output"


class StateOwnershipEvidence(StrEnum):
    EXCLUSIVE = "exclusive"


class StateAtomicityEvidence(StrEnum):
    PER_TOKEN = "per_token"


Dimension: TypeAlias = int | str


@dataclass(frozen=True, slots=True)
class NumericalContract:
    maximum_absolute_error: float = 0.0
    maximum_relative_error: float = 0.0
    minimum_finite_value: float | None = None
    maximum_finite_value: float | None = None
    preserve_nan: bool = True
    preserve_infinity: bool = True
    preserve_signed_zero: bool = True
    deterministic: bool = True


@dataclass(frozen=True, slots=True)
class TensorSpec:
    shape: tuple[Dimension, ...]
    dtype: DType
    layout: Layout = Layout.CONTIGUOUS
    strides: tuple[int | str, ...] | None = None
    alias_group: str | None = None
    state_dependency: str | None = None
    numerical: NumericalContract = NumericalContract()


@dataclass(frozen=True, slots=True)
class OperatorParameters:
    target_dtype: DType | None = None
    permutation: tuple[int, ...] | None = None
    constant_value: int | float | bool | None = None
    activation: str | None = None
    axis: int | None = None
    scale: float | None = None
    state_key: str | None = None
    state_ownership: StateOwnershipEvidence | None = None
    state_atomicity: StateAtomicityEvidence | None = None


@dataclass(frozen=True, slots=True)
class TensorNode:
    node_id: str
    operator: str
    inputs: tuple[str, ...]
    output: TensorSpec
    parameters: OperatorParameters = OperatorParameters()


@dataclass(frozen=True, slots=True)
class TensorGraph:
    nodes: tuple[TensorNode, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RewriteRule:
    rule_id: str
    kind: RewriteKind
    semantic_category: SemanticCategory
    supported_dtypes: tuple[DType, ...]
    maximum_quality_cost: float
    requires_signed_zero_insensitive: bool
    verification_obligations: tuple[str, ...]
    fallback_required: bool = True


@dataclass(frozen=True, slots=True)
class RewriteApplication:
    rule_id: str
    matched_nodes: tuple[str, ...]
    semantic_category: SemanticCategory
    quality_cost: float
    verification_obligations: tuple[str, ...]
    preconditions_checked: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RewriteCandidate:
    graph: TensorGraph
    history: tuple[RewriteApplication, ...]
    structural_key: str


class RewriteError(ValueError):
    """Raised for invalid graphs or unsatisfied rewrite preconditions."""
