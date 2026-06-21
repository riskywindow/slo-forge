"""Static type, range, and execution-bound checking for Genesis policies."""

from __future__ import annotations

from .model import (
    Analysis,
    Binary,
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


def _validate_spec(spec: VariableSpec) -> None:
    if not spec.name or not spec.name.replace("_", "a").isalnum():
        raise PolicyError(f"invalid variable name {spec.name!r}")
    expected = {
        ScalarType.BOOL: bool,
        ScalarType.INT: int,
        ScalarType.FLOAT: float,
    }[spec.scalar_type]
    if type(spec.lower) is not expected or type(spec.upper) is not expected:
        raise PolicyError(f"{spec.name} bounds do not match {spec.scalar_type.value}")
    if spec.lower > spec.upper:
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
        value = expression.value
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
        _numeric(operand, "neg")
        return Analysis(
            operand.scalar_type,
            -operand.upper,
            -operand.lower,
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
        _numeric(left, expression.operator)
        _numeric(right, expression.operator)
        lower, upper = _binary_range(expression.operator, left, right)
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
    return Analysis(
        _promote(
            value_analysis.scalar_type,
            _promote(lower_analysis.scalar_type, upper_analysis.scalar_type),
        ),
        max(value_analysis.lower, lower_analysis.lower),
        min(value_analysis.upper, upper_analysis.upper),
        value_analysis.operation_count
        + lower_analysis.operation_count
        + upper_analysis.operation_count
        + 1,
    )


def check_policy(program: PolicyProgram) -> Analysis:
    if not 1 <= program.maximum_operations <= 4096:
        raise PolicyError("operation limit must be in [1, 4096]")
    variables: dict[str, VariableSpec] = {}
    for spec in program.inputs:
        _validate_spec(spec)
        if spec.name in variables:
            raise PolicyError(f"duplicate input {spec.name!r}")
        variables[spec.name] = spec
    _validate_spec(program.output)
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
