"""Typed intermediate representation for the restricted Genesis policy DSL."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


class ScalarType(StrEnum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"


Scalar: TypeAlias = int | float | bool


@dataclass(frozen=True, slots=True)
class VariableSpec:
    name: str
    scalar_type: ScalarType
    lower: Scalar
    upper: Scalar


@dataclass(frozen=True, slots=True)
class Literal:
    value: Scalar


@dataclass(frozen=True, slots=True)
class Variable:
    name: str


@dataclass(frozen=True, slots=True)
class Unary:
    operator: str
    operand: Expr


@dataclass(frozen=True, slots=True)
class Binary:
    operator: str
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class Conditional:
    condition: Expr
    when_true: Expr
    when_false: Expr


@dataclass(frozen=True, slots=True)
class Clamp:
    value: Expr
    lower: Expr
    upper: Expr


Expr: TypeAlias = Literal | Variable | Unary | Binary | Conditional | Clamp


@dataclass(frozen=True, slots=True)
class PolicyProgram:
    name: str
    inputs: tuple[VariableSpec, ...]
    output: VariableSpec
    expression: Expr
    maximum_operations: int = 256


@dataclass(frozen=True, slots=True)
class Analysis:
    scalar_type: ScalarType
    lower: Scalar
    upper: Scalar
    operation_count: int


@dataclass(frozen=True, slots=True)
class Instruction:
    opcode: str
    operand: Scalar | str | None = None


@dataclass(frozen=True, slots=True)
class BytecodeProgram:
    name: str
    inputs: tuple[VariableSpec, ...]
    output: VariableSpec
    instructions: tuple[Instruction, ...]
    maximum_operations: int


@dataclass(frozen=True, slots=True)
class PolicyGraphNode:
    node_id: int
    label: str


@dataclass(frozen=True, slots=True)
class PolicyGraphEdge:
    source: int
    target: int
    role: str


@dataclass(frozen=True, slots=True)
class PolicyGraph:
    nodes: tuple[PolicyGraphNode, ...]
    edges: tuple[PolicyGraphEdge, ...]


@dataclass(frozen=True, slots=True)
class EquivalenceResult:
    equivalent: bool
    states_checked: int
    counterexample: tuple[tuple[str, Scalar], ...] | None = None


class PolicyError(ValueError):
    """Raised when a policy is not valid in the restricted language."""
