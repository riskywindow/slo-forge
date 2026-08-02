"""Simplification, deterministic mutation, and bounded equivalence checking."""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator
from dataclasses import replace

from .checker import check_policy
from .model import (
    Binary,
    Clamp,
    Conditional,
    EquivalenceResult,
    Expr,
    Literal,
    PolicyError,
    PolicyProgram,
    Scalar,
    ScalarType,
    Unary,
    Variable,
    VariableSpec,
)
from .runtime import evaluate


def _literal_binary(operator: str, left: Literal, right: Literal) -> Literal | None:
    a, b = left.value, right.value
    if operator == "add":
        return Literal(a + b)
    if operator == "sub":
        return Literal(a - b)
    if operator == "mul":
        return Literal(a * b)
    if operator == "floor_div" and b != 0:
        return Literal(a // b)
    if operator == "min":
        return Literal(min(a, b))
    if operator == "max":
        return Literal(max(a, b))
    if operator == "and":
        return Literal(bool(a) and bool(b))
    if operator == "or":
        return Literal(bool(a) or bool(b))
    if operator == "eq":
        return Literal(a == b)
    if operator == "ne":
        return Literal(a != b)
    if operator == "lt":
        return Literal(a < b)
    if operator == "le":
        return Literal(a <= b)
    if operator == "gt":
        return Literal(a > b)
    if operator == "ge":
        return Literal(a >= b)
    return None


def simplify_expression(expression: Expr) -> Expr:
    if isinstance(expression, (Literal, Variable)):
        return expression
    if isinstance(expression, Unary):
        operand = simplify_expression(expression.operand)
        if isinstance(operand, Literal):
            return Literal(
                not bool(operand.value) if expression.operator == "not" else -operand.value
            )
        return Unary(expression.operator, operand)
    if isinstance(expression, Binary):
        left = simplify_expression(expression.left)
        right = simplify_expression(expression.right)
        if isinstance(left, Literal) and isinstance(right, Literal):
            folded = _literal_binary(expression.operator, left, right)
            if folded is not None:
                return folded
        if left == right and expression.operator in {"min", "max"}:
            return left
        return Binary(expression.operator, left, right)
    if isinstance(expression, Conditional):
        condition = simplify_expression(expression.condition)
        when_true = simplify_expression(expression.when_true)
        when_false = simplify_expression(expression.when_false)
        if isinstance(condition, Literal) and type(condition.value) is bool:
            return when_true if condition.value else when_false
        if when_true == when_false:
            return when_true
        return Conditional(condition, when_true, when_false)
    value = simplify_expression(expression.value)
    lower = simplify_expression(expression.lower)
    upper = simplify_expression(expression.upper)
    if isinstance(value, Literal) and isinstance(lower, Literal) and isinstance(upper, Literal):
        return Literal(min(max(value.value, lower.value), upper.value))
    return Clamp(value, lower, upper)


def simplify_policy(program: PolicyProgram) -> PolicyProgram:
    simplified = replace(program, expression=simplify_expression(program.expression))
    check_policy(simplified)
    return simplified


def _mutate_expression(expression: Expr, target: int, current: list[int], delta: int) -> Expr:
    index = current[0]
    current[0] += 1
    if (
        index == target
        and isinstance(expression, Literal)
        and type(expression.value) in {int, float}
    ):
        return Literal(expression.value + delta)
    if isinstance(expression, Unary):
        return Unary(
            expression.operator,
            _mutate_expression(expression.operand, target, current, delta),
        )
    if isinstance(expression, Binary):
        return Binary(
            expression.operator,
            _mutate_expression(expression.left, target, current, delta),
            _mutate_expression(expression.right, target, current, delta),
        )
    if isinstance(expression, Conditional):
        return Conditional(
            _mutate_expression(expression.condition, target, current, delta),
            _mutate_expression(expression.when_true, target, current, delta),
            _mutate_expression(expression.when_false, target, current, delta),
        )
    if isinstance(expression, Clamp):
        return Clamp(
            _mutate_expression(expression.value, target, current, delta),
            _mutate_expression(expression.lower, target, current, delta),
            _mutate_expression(expression.upper, target, current, delta),
        )
    return expression


def _node_count(expression: Expr) -> int:
    if isinstance(expression, (Literal, Variable)):
        return 1
    if isinstance(expression, Unary):
        return 1 + _node_count(expression.operand)
    if isinstance(expression, Binary):
        return 1 + _node_count(expression.left) + _node_count(expression.right)
    if isinstance(expression, Conditional):
        return (
            1
            + _node_count(expression.condition)
            + _node_count(expression.when_true)
            + _node_count(expression.when_false)
        )
    return (
        1
        + _node_count(expression.value)
        + _node_count(expression.lower)
        + _node_count(expression.upper)
    )


def mutate_policy(program: PolicyProgram, *, seed: int, attempts: int = 32) -> PolicyProgram:
    """Return the first valid deterministic literal mutation.

    Invalid mutations are discarded by the same checker used at admission.
    """

    check_policy(program)
    if attempts <= 0:
        raise PolicyError("mutation attempts must be positive")
    generator = random.Random(seed)
    count = _node_count(program.expression)
    for _ in range(attempts):
        candidate = replace(
            program,
            expression=_mutate_expression(
                program.expression,
                generator.randrange(count),
                [0],
                generator.choice((-2, -1, 1, 2)),
            ),
        )
        if candidate == program:
            continue
        try:
            check_policy(candidate)
        except PolicyError:
            continue
        return candidate
    raise PolicyError("no valid literal mutation found within the attempt budget")


def _domain(spec: VariableSpec) -> tuple[Scalar, ...]:
    if spec.scalar_type is ScalarType.BOOL:
        return tuple(value for value in (False, True) if spec.lower <= value <= spec.upper)
    if spec.scalar_type is ScalarType.INT:
        return tuple(range(int(spec.lower), int(spec.upper) + 1))
    raise PolicyError("bounded exhaustive equivalence does not enumerate floating inputs")


def _assignments(inputs: tuple[VariableSpec, ...]) -> Iterator[dict[str, Scalar]]:
    domains = tuple(_domain(spec) for spec in inputs)
    for values in itertools.product(*domains):
        yield {spec.name: value for spec, value in zip(inputs, values, strict=True)}


def check_equivalent(
    left: PolicyProgram,
    right: PolicyProgram,
    *,
    maximum_states: int = 100_000,
) -> EquivalenceResult:
    check_policy(left)
    check_policy(right)
    if left.inputs != right.inputs or left.output != right.output:
        raise PolicyError("equivalence requires identical input and output domains")
    if maximum_states <= 0:
        raise PolicyError("maximum_states must be positive")
    total = 1
    for spec in left.inputs:
        total *= len(_domain(spec))
        if total > maximum_states:
            raise PolicyError(f"bounded domain contains {total} states, limit is {maximum_states}")
    checked = 0
    for assignment in _assignments(left.inputs):
        checked += 1
        if evaluate(left, assignment) != evaluate(right, assignment):
            return EquivalenceResult(
                equivalent=False,
                states_checked=checked,
                counterexample=tuple(sorted(assignment.items())),
            )
    return EquivalenceResult(equivalent=True, states_checked=checked)
