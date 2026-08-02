"""Static type, range, and execution-bound checking for Genesis policies."""

from __future__ import annotations

import math
import re

from .limits import (
    MAX_POLICY_FLOAT_ABS,
    MAX_POLICY_INSTRUCTIONS,
    MAX_POLICY_INTEGER_ABS,
    MAX_POLICY_NAME_BYTES,
    MAX_POLICY_VARIABLES,
)
from .model import (
    Analysis,
    Binary,
    Clamp,
    Conditional,
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

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _checked_scalar(value: object, label: str) -> Scalar:
    if type(value) is bool:
        return value
    if type(value) is int:
        if abs(value) > MAX_POLICY_INTEGER_ABS:
            raise PolicyError(f"{label} integer exceeds the absolute policy bound")
        return value
    if type(value) is float:
        if not math.isfinite(value) or abs(value) > MAX_POLICY_FLOAT_ABS:
            raise PolicyError(f"{label} must be a bounded finite number")
        return value
    raise PolicyError(f"{label} must be a boolean, integer, or float")


def _validate_spec(spec: VariableSpec) -> None:
    if (
        type(spec) is not VariableSpec
        or type(spec.scalar_type) is not ScalarType
        or not isinstance(spec.name, str)
        or not spec.name
        or len(spec.name.encode("utf-8")) > MAX_POLICY_NAME_BYTES
        or _NAME.fullmatch(spec.name) is None
    ):
        raise PolicyError(f"invalid variable name {spec.name!r}")
    expected = {
        ScalarType.BOOL: bool,
        ScalarType.INT: int,
        ScalarType.FLOAT: float,
    }[spec.scalar_type]
    lower = _checked_scalar(spec.lower, f"{spec.name} lower bound")
    upper = _checked_scalar(spec.upper, f"{spec.name} upper bound")
    if type(lower) is not expected or type(upper) is not expected:
        raise PolicyError(f"{spec.name} bounds do not match {spec.scalar_type.value}")
    if lower > upper:
        raise PolicyError(f"{spec.name} lower bound exceeds upper bound")


def _numeric(analysis: Analysis, operator: str) -> None:
    if analysis.scalar_type not in {ScalarType.INT, ScalarType.FLOAT}:
        raise PolicyError(f"{operator} requires numeric operands")


def _promote(left: ScalarType, right: ScalarType) -> ScalarType:
    if ScalarType.FLOAT in {left, right}:
        return ScalarType.FLOAT
    return ScalarType.INT


def _binary_range(operator: str, left: Analysis, right: Analysis) -> tuple[Scalar, Scalar]:
    if operator == "add":
        return left.lower + right.lower, left.upper + right.upper
    if operator == "sub":
        return left.lower - right.upper, left.upper - right.lower
    if operator == "mul":
        products = (
            left.lower * right.lower,
            left.lower * right.upper,
            left.upper * right.lower,
            left.upper * right.upper,
        )
        return min(products), max(products)
    if operator == "floor_div":
        if right.lower <= 0 <= right.upper:
            raise PolicyError("floor_div denominator range includes zero")
        quotients = (
            left.lower // right.lower,
            left.lower // right.upper,
            left.upper // right.lower,
            left.upper // right.upper,
        )
        return min(quotients), max(quotients)
    if operator == "min":
        return min(left.lower, right.lower), min(left.upper, right.upper)
    return max(left.lower, right.lower), max(left.upper, right.upper)


def analyze_expression(expression: Expr, variables: dict[str, VariableSpec]) -> Analysis:
    if isinstance(expression, Literal):
        value = _checked_scalar(expression.value, "literal")
        scalar_type = (
            ScalarType.BOOL
            if type(value) is bool
            else ScalarType.INT
            if type(value) is int
            else ScalarType.FLOAT
        )
        return Analysis(scalar_type, value, value, 1)
    if isinstance(expression, Variable):
        try:
            spec = variables[expression.name]
        except KeyError as error:
            raise PolicyError(f"unknown input {expression.name!r}") from error
        return Analysis(spec.scalar_type, spec.lower, spec.upper, 1)
    if isinstance(expression, Unary):
        operand = analyze_expression(expression.operand, variables)
        if expression.operator == "not":
            if operand.scalar_type is not ScalarType.BOOL:
                raise PolicyError("not requires a boolean operand")
            return Analysis(ScalarType.BOOL, False, True, operand.operation_count + 1)
        if expression.operator != "neg":
            raise PolicyError(f"unary operator {expression.operator!r} is not allowed")
        _numeric(operand, "neg")
        lower = _checked_scalar(-operand.upper, "neg lower range")
        upper = _checked_scalar(-operand.lower, "neg upper range")
        return Analysis(
            operand.scalar_type,
            lower,
            upper,
            operand.operation_count + 1,
        )
    if isinstance(expression, Binary):
        left = analyze_expression(expression.left, variables)
        right = analyze_expression(expression.right, variables)
        count = left.operation_count + right.operation_count + 1
        if expression.operator in {"and", "or"}:
            if left.scalar_type is not ScalarType.BOOL or right.scalar_type is not ScalarType.BOOL:
                raise PolicyError(f"{expression.operator} requires boolean operands")
            return Analysis(ScalarType.BOOL, False, True, count)
        if expression.operator in {"eq", "ne"}:
            if left.scalar_type is not right.scalar_type:
                raise PolicyError(f"{expression.operator} requires operands of the same type")
            return Analysis(ScalarType.BOOL, False, True, count)
        if expression.operator in {"lt", "le", "gt", "ge"}:
            _numeric(left, expression.operator)
            _numeric(right, expression.operator)
            return Analysis(ScalarType.BOOL, False, True, count)
        if expression.operator not in {"add", "sub", "mul", "floor_div", "min", "max"}:
            raise PolicyError(f"binary operator {expression.operator!r} is not allowed")
        _numeric(left, expression.operator)
        _numeric(right, expression.operator)
        lower, upper = _binary_range(expression.operator, left, right)
        lower = _checked_scalar(lower, f"{expression.operator} lower range")
        upper = _checked_scalar(upper, f"{expression.operator} upper range")
        return Analysis(_promote(left.scalar_type, right.scalar_type), lower, upper, count)
    if isinstance(expression, Conditional):
        condition = analyze_expression(expression.condition, variables)
        when_true = analyze_expression(expression.when_true, variables)
        when_false = analyze_expression(expression.when_false, variables)
        if condition.scalar_type is not ScalarType.BOOL:
            raise PolicyError("if condition must be boolean")
        if when_true.scalar_type is not when_false.scalar_type:
            raise PolicyError("if branches must have the same type")
        return Analysis(
            when_true.scalar_type,
            min(when_true.lower, when_false.lower),
            max(when_true.upper, when_false.upper),
            condition.operation_count + when_true.operation_count + when_false.operation_count + 1,
        )
    if not isinstance(expression, Clamp):
        raise PolicyError("expression contains an unsupported node")
    value_analysis = analyze_expression(expression.value, variables)
    lower_analysis = analyze_expression(expression.lower, variables)
    upper_analysis = analyze_expression(expression.upper, variables)
    _numeric(value_analysis, "clamp")
    if not isinstance(expression.lower, Literal) or not isinstance(expression.upper, Literal):
        raise PolicyError("clamp bounds must be numeric literals")
    _numeric(lower_analysis, "clamp")
    _numeric(upper_analysis, "clamp")
    if lower_analysis.lower > upper_analysis.upper:
        raise PolicyError("clamp lower bound exceeds upper bound")
    # Clamp is monotonic, so independently clamp both ordered endpoints.  The
    # older max(lower), min(upper) calculation inverted disjoint intervals.
    clamp_lower = min(max(value_analysis.lower, lower_analysis.lower), upper_analysis.upper)
    clamp_upper = min(max(value_analysis.upper, lower_analysis.lower), upper_analysis.upper)
    return Analysis(
        _promote(
            value_analysis.scalar_type,
            _promote(lower_analysis.scalar_type, upper_analysis.scalar_type),
        ),
        _checked_scalar(clamp_lower, "clamp lower range"),
        _checked_scalar(clamp_upper, "clamp upper range"),
        value_analysis.operation_count
        + lower_analysis.operation_count
        + upper_analysis.operation_count
        + 1,
    )


def check_policy(program: PolicyProgram) -> Analysis:
    if (
        not isinstance(program.name, str)
        or not program.name
        or len(program.name.encode("utf-8")) > MAX_POLICY_NAME_BYTES
        or _NAME.fullmatch(program.name) is None
    ):
        raise PolicyError(f"invalid policy name {program.name!r}")
    if type(program.maximum_operations) is not int or not (
        1 <= program.maximum_operations <= MAX_POLICY_INSTRUCTIONS
    ):
        raise PolicyError(f"operation limit must be in [1, {MAX_POLICY_INSTRUCTIONS}]")
    if len(program.inputs) > MAX_POLICY_VARIABLES:
        raise PolicyError(f"policy has more than {MAX_POLICY_VARIABLES} inputs")
    variables: dict[str, VariableSpec] = {}
    for spec in program.inputs:
        _validate_spec(spec)
        if spec.name in variables:
            raise PolicyError(f"duplicate input {spec.name!r}")
        variables[spec.name] = spec
    _validate_spec(program.output)
    if program.output.name in variables:
        raise PolicyError(f"output name collides with input {program.output.name!r}")
    analysis = analyze_expression(program.expression, variables)
    if analysis.operation_count > program.maximum_operations:
        raise PolicyError(
            f"expression needs {analysis.operation_count} operations, limit is "
            f"{program.maximum_operations}"
        )
    if analysis.scalar_type is not program.output.scalar_type:
        raise PolicyError(
            f"return type {analysis.scalar_type.value} does not match "
            f"{program.output.scalar_type.value}"
        )
    if analysis.lower < program.output.lower or analysis.upper > program.output.upper:
        raise PolicyError(
            f"return range [{analysis.lower}, {analysis.upper}] exceeds output bounds "
            f"[{program.output.lower}, {program.output.upper}]"
        )
    return analysis
