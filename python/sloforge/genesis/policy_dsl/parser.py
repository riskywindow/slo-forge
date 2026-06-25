"""Parser and canonical formatter for the bounded, S-expression policy language."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator

from .limits import (
    MAX_POLICY_DOCUMENT_BYTES,
    MAX_POLICY_FLOAT_ABS,
    MAX_POLICY_INTEGER_ABS,
    MAX_POLICY_NAME_BYTES,
    MAX_POLICY_VARIABLES,
)
from .model import (
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
_TOKEN = re.compile(
    r"\s*(?:(?P<number>-?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))|"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)|(?P<open>\()|(?P<close>\)))"
)
_UNARY = {"not", "neg"}
_BINARY = {
    "add",
    "sub",
    "mul",
    "floor_div",
    "min",
    "max",
    "and",
    "or",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
}


def _scalar(text: str, scalar_type: ScalarType | None = None) -> Scalar:
    lowered = text.lower()
    if lowered in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        raise PolicyError("non-finite numbers are forbidden")
    if lowered in {"true", "false"}:
        value: Scalar = lowered == "true"
    elif "." in text:
        value = float(text)
        if not math.isfinite(value) or abs(value) > MAX_POLICY_FLOAT_ABS:
            raise PolicyError("float must be a bounded finite number")
    else:
        if len(text.removeprefix("-")) > 19:
            raise PolicyError("integer exceeds the absolute policy bound")
        value = int(text)
        if abs(value) > MAX_POLICY_INTEGER_ABS:
            raise PolicyError("integer exceeds the absolute policy bound")
    if scalar_type is ScalarType.BOOL and type(value) is not bool:
        raise PolicyError(f"expected boolean bound, got {text!r}")
    if scalar_type is ScalarType.INT and type(value) is not int:
        raise PolicyError(f"expected integer bound, got {text!r}")
    if scalar_type is ScalarType.FLOAT and type(value) not in {int, float}:
        raise PolicyError(f"expected numeric bound, got {text!r}")
    return float(value) if scalar_type is ScalarType.FLOAT else value


def _tokens(source: str) -> tuple[str, ...]:
    result: list[str] = []
    position = 0
    while position < len(source):
        match = _TOKEN.match(source, position)
        if match is None:
            raise PolicyError(f"invalid expression token at offset {position}")
        result.append(match.group(match.lastgroup or ""))
        position = match.end()
    return tuple(result)


class _TokenStream:
    def __init__(self, tokens: tuple[str, ...]) -> None:
        self._tokens = tokens
        self._position = 0

    def pop(self) -> str:
        if self._position >= len(self._tokens):
            raise PolicyError("unexpected end of expression")
        token = self._tokens[self._position]
        self._position += 1
        return token

    def at_end(self) -> bool:
        return self._position == len(self._tokens)


def _parse_expr(stream: _TokenStream) -> Expr:
    token = stream.pop()
    if token != "(":
        if token.lower() in {"nan", "inf", "infinity"}:
            raise PolicyError("non-finite numbers are forbidden")
        if token in {"true", "false"}:
            return Literal(token == "true")
        if token[0].isdigit() or token[0] in {"-", "."}:
            return Literal(_scalar(token))
        if _NAME.fullmatch(token) is None:
            raise PolicyError(f"invalid variable name {token!r}")
        return Variable(token)
    operator = stream.pop()
    if operator in _UNARY:
        expression: Expr = Unary(operator, _parse_expr(stream))
    elif operator in _BINARY:
        expression = Binary(operator, _parse_expr(stream), _parse_expr(stream))
    elif operator == "if":
        expression = Conditional(_parse_expr(stream), _parse_expr(stream), _parse_expr(stream))
    elif operator == "clamp":
        expression = Clamp(_parse_expr(stream), _parse_expr(stream), _parse_expr(stream))
    else:
        raise PolicyError(f"operator {operator!r} is not allowed")
    if stream.pop() != ")":
        raise PolicyError(f"operator {operator!r} has too many arguments")
    return expression


def _meaningful_lines(source: str) -> Iterator[str]:
    for raw in source.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            yield line


def parse_policy(source: str) -> PolicyProgram:
    if not 1 <= len(source.encode("utf-8")) <= MAX_POLICY_DOCUMENT_BYTES:
        raise PolicyError("policy source exceeds the absolute document bound")
    lines = list(_meaningful_lines(source))
    if len(lines) < 3:
        raise PolicyError("policy requires a header, output, and return expression")
    header = lines.pop(0).split()
    if (
        len(header) != 2
        or header[0] != "policy"
        or _NAME.fullmatch(header[1]) is None
        or len(header[1].encode("utf-8")) > MAX_POLICY_NAME_BYTES
    ):
        raise PolicyError("first line must be 'policy <name>'")
    inputs: list[VariableSpec] = []
    output: VariableSpec | None = None
    return_source: str | None = None
    maximum_operations = 256
    output_seen = False
    return_seen = False
    for line in lines:
        fields = line.split()
        if fields[0] == "input" and len(fields) == 5:
            try:
                scalar_type = ScalarType(fields[2])
            except ValueError as error:
                raise PolicyError(f"unknown scalar type {fields[2]!r}") from error
            if len(inputs) >= MAX_POLICY_VARIABLES:
                raise PolicyError(f"policy has more than {MAX_POLICY_VARIABLES} inputs")
            inputs.append(
                VariableSpec(
                    name=fields[1],
                    scalar_type=scalar_type,
                    lower=_scalar(fields[3], scalar_type),
                    upper=_scalar(fields[4], scalar_type),
                )
            )
        elif fields[0] == "output" and len(fields) == 4:
            if output_seen:
                raise PolicyError("policy has more than one output declaration")
            output_seen = True
            try:
                scalar_type = ScalarType(fields[1])
            except ValueError as error:
                raise PolicyError(f"unknown scalar type {fields[1]!r}") from error
            output = VariableSpec(
                name="output",
                scalar_type=scalar_type,
                lower=_scalar(fields[2], scalar_type),
                upper=_scalar(fields[3], scalar_type),
            )
        elif fields[0] == "limit" and len(fields) == 2:
            parsed_limit = _scalar(fields[1], ScalarType.INT)
            assert type(parsed_limit) is int
            maximum_operations = parsed_limit
        elif line.startswith("return "):
            if return_seen:
                raise PolicyError("policy has more than one return declaration")
            return_seen = True
            return_source = line.removeprefix("return ").strip()
        else:
            raise PolicyError(f"invalid policy declaration {line!r}")
    if output is None or return_source is None:
        raise PolicyError("policy requires exactly one output and return declaration")
    tokens = _TokenStream(_tokens(return_source))
    try:
        expression = _parse_expr(tokens)
    except RecursionError as error:
        raise PolicyError("policy expression nesting exceeds the parser bound") from error
    if not tokens.at_end():
        raise PolicyError("trailing tokens after return expression")
    return PolicyProgram(
        name=header[1],
        inputs=tuple(inputs),
        output=output,
        expression=expression,
        maximum_operations=maximum_operations,
    )


def _format_scalar(value: Scalar) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    return str(value)


def format_expression(expression: Expr) -> str:
    if isinstance(expression, Literal):
        return _format_scalar(expression.value)
    if isinstance(expression, Variable):
        return expression.name
    if isinstance(expression, Unary):
        return f"({expression.operator} {format_expression(expression.operand)})"
    if isinstance(expression, Binary):
        return (
            f"({expression.operator} {format_expression(expression.left)} "
            f"{format_expression(expression.right)})"
        )
    if isinstance(expression, Conditional):
        return (
            f"(if {format_expression(expression.condition)} "
            f"{format_expression(expression.when_true)} "
            f"{format_expression(expression.when_false)})"
        )
    return (
        f"(clamp {format_expression(expression.value)} "
        f"{format_expression(expression.lower)} {format_expression(expression.upper)})"
    )


def format_policy(program: PolicyProgram) -> str:
    lines = [f"policy {program.name}"]
    lines.extend(
        f"input {item.name} {item.scalar_type.value} "
        f"{_format_scalar(item.lower)} {_format_scalar(item.upper)}"
        for item in program.inputs
    )
    lines.append(
        f"output {program.output.scalar_type.value} "
        f"{_format_scalar(program.output.lower)} {_format_scalar(program.output.upper)}"
    )
    lines.append(f"limit {program.maximum_operations}")
    lines.append(f"return {format_expression(program.expression)}")
    return "\n".join(lines) + "\n"
