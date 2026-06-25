"""Independent, fail-closed validation for untrusted policy bytecode.

This checker consumes only the bytecode IR.  It does not trust the source
checker or compiler, and it never executes the program while admitting it.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .limits import (
    MAX_POLICY_DOCUMENT_BYTES,
    MAX_POLICY_FLOAT_ABS,
    MAX_POLICY_INSTRUCTIONS,
    MAX_POLICY_INTEGER_ABS,
    MAX_POLICY_NAME_BYTES,
    MAX_POLICY_VARIABLES,
)
from .model import (
    Analysis,
    BytecodeProgram,
    Instruction,
    PolicyError,
    Scalar,
    ScalarType,
    VariableSpec,
)

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UNARY = frozenset({"not", "neg"})
_BINARY = frozenset(
    {
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
)
_OPERAND_FREE = _UNARY | _BINARY | frozenset({"select", "clamp"})
_OPCODES = _OPERAND_FREE | frozenset({"literal", "input"})
_DOCUMENT_FIELDS = frozenset({"name", "inputs", "output", "instructions", "maximum_operations"})
_VARIABLE_FIELDS = frozenset({"name", "scalar_type", "lower", "upper"})
_INSTRUCTION_FIELDS = frozenset({"opcode", "operand"})


@dataclass(frozen=True, slots=True)
class _AbstractValue:
    scalar_type: ScalarType
    lower: Scalar
    upper: Scalar


def _valid_name(name: object, *, label: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or len(name.encode("utf-8")) > MAX_POLICY_NAME_BYTES
        or _NAME.fullmatch(name) is None
    ):
        raise PolicyError(f"invalid {label} name {name!r}")
    return name


def _checked_scalar(value: object, *, label: str) -> Scalar:
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


def _validate_spec(spec: VariableSpec, *, label: str) -> None:
    if type(spec) is not VariableSpec or type(spec.scalar_type) is not ScalarType:
        raise PolicyError(f"{label} must be a strictly typed variable specification")
    _valid_name(spec.name, label=label)
    expected = {
        ScalarType.BOOL: bool,
        ScalarType.INT: int,
        ScalarType.FLOAT: float,
    }[spec.scalar_type]
    lower = _checked_scalar(spec.lower, label=f"{label} lower bound")
    upper = _checked_scalar(spec.upper, label=f"{label} upper bound")
    if type(lower) is not expected or type(upper) is not expected:
        raise PolicyError(f"{label} bounds do not match {spec.scalar_type.value}")
    if lower > upper:
        raise PolicyError(f"{label} lower bound exceeds upper bound")


def _bounded(value: _AbstractValue, *, opcode: str) -> _AbstractValue:
    lower = _checked_scalar(value.lower, label=f"{opcode} lower range")
    upper = _checked_scalar(value.upper, label=f"{opcode} upper range")
    if lower > upper:
        raise PolicyError(f"{opcode} produced an invalid abstract range")
    if value.scalar_type is ScalarType.INT and (type(lower) is not int or type(upper) is not int):
        raise PolicyError(f"{opcode} produced a non-integer integer range")
    if value.scalar_type is ScalarType.FLOAT and (
        type(lower) not in {int, float} or type(upper) not in {int, float}
    ):
        raise PolicyError(f"{opcode} produced a non-numeric float range")
    return value


def _numeric(value: _AbstractValue, opcode: str) -> None:
    if value.scalar_type not in {ScalarType.INT, ScalarType.FLOAT}:
        raise PolicyError(f"{opcode} requires numeric operands")


def _promote(left: ScalarType, right: ScalarType) -> ScalarType:
    return ScalarType.FLOAT if ScalarType.FLOAT in {left, right} else ScalarType.INT


def _binary_range(
    opcode: str, left: _AbstractValue, right: _AbstractValue
) -> tuple[Scalar, Scalar]:
    if opcode == "add":
        return left.lower + right.lower, left.upper + right.upper
    if opcode == "sub":
        return left.lower - right.upper, left.upper - right.lower
    if opcode == "mul":
        products = (
            left.lower * right.lower,
            left.lower * right.upper,
            left.upper * right.lower,
            left.upper * right.upper,
        )
        return min(products), max(products)
    if opcode == "floor_div":
        if right.lower <= 0 <= right.upper:
            raise PolicyError("floor_div denominator range includes zero")
        quotients = (
            left.lower // right.lower,
            left.lower // right.upper,
            left.upper // right.lower,
            left.upper // right.upper,
        )
        return min(quotients), max(quotients)
    if opcode == "min":
        return min(left.lower, right.lower), min(left.upper, right.upper)
    return max(left.lower, right.lower), max(left.upper, right.upper)


def validate_bytecode(program: BytecodeProgram) -> Analysis:
    """Statically validate types, ranges, stack shape, and absolute bounds."""

    if type(program) is not BytecodeProgram:
        raise PolicyError("bytecode must be a BytecodeProgram")
    _valid_name(program.name, label="policy")
    if type(program.maximum_operations) is not int or not (
        1 <= program.maximum_operations <= MAX_POLICY_INSTRUCTIONS
    ):
        raise PolicyError(f"bytecode operation limit must be in [1, {MAX_POLICY_INSTRUCTIONS}]")
    if len(program.inputs) > MAX_POLICY_VARIABLES:
        raise PolicyError(f"bytecode has more than {MAX_POLICY_VARIABLES} inputs")
    variables: dict[str, VariableSpec] = {}
    for spec in program.inputs:
        _validate_spec(spec, label="input")
        if spec.name in variables:
            raise PolicyError(f"duplicate input {spec.name!r}")
        variables[spec.name] = spec
    _validate_spec(program.output, label="output")
    if program.output.name in variables:
        raise PolicyError(f"output name collides with input {program.output.name!r}")
    instruction_count = len(program.instructions)
    if not 1 <= instruction_count <= MAX_POLICY_INSTRUCTIONS:
        raise PolicyError(f"bytecode instruction count must be in [1, {MAX_POLICY_INSTRUCTIONS}]")
    if instruction_count > program.maximum_operations:
        raise PolicyError("bytecode exceeds its declared operation limit")

    stack: list[_AbstractValue] = []
    for index, instruction in enumerate(program.instructions):
        if type(instruction) is not Instruction:
            raise PolicyError(f"instruction {index} is not a typed instruction")
        opcode = instruction.opcode
        if type(opcode) is not str or opcode not in _OPCODES:
            raise PolicyError(f"instruction {index} has forbidden opcode {opcode!r}")
        operand = instruction.operand
        if opcode in _OPERAND_FREE and operand is not None:
            raise PolicyError(f"instruction {index} opcode {opcode!r} forbids an operand")
        if opcode == "literal":
            if isinstance(operand, str) or operand is None:
                raise PolicyError(f"instruction {index} has an invalid literal operand")
            literal_value = _checked_scalar(operand, label=f"instruction {index} literal")
            scalar_type = (
                ScalarType.BOOL
                if type(literal_value) is bool
                else ScalarType.INT
                if type(literal_value) is int
                else ScalarType.FLOAT
            )
            stack.append(_AbstractValue(scalar_type, literal_value, literal_value))
            continue
        if opcode == "input":
            if type(operand) is not str or operand not in variables:
                raise PolicyError(f"instruction {index} references an unknown input")
            spec = variables[operand]
            stack.append(_AbstractValue(spec.scalar_type, spec.lower, spec.upper))
            continue
        if opcode in _UNARY:
            if not stack:
                raise PolicyError(f"instruction {index} causes bytecode stack underflow")
            unary_value = stack.pop()
            if opcode == "not":
                if unary_value.scalar_type is not ScalarType.BOOL:
                    raise PolicyError("not requires a boolean operand")
                stack.append(_AbstractValue(ScalarType.BOOL, False, True))
            else:
                _numeric(unary_value, opcode)
                stack.append(
                    _bounded(
                        _AbstractValue(
                            unary_value.scalar_type, -unary_value.upper, -unary_value.lower
                        ),
                        opcode=opcode,
                    )
                )
            continue
        if opcode == "select":
            if len(stack) < 3:
                raise PolicyError(f"instruction {index} causes bytecode stack underflow")
            when_false = stack.pop()
            when_true = stack.pop()
            condition = stack.pop()
            if condition.scalar_type is not ScalarType.BOOL:
                raise PolicyError("select condition must be boolean")
            if when_true.scalar_type is not when_false.scalar_type:
                raise PolicyError("select branches must have the same type")
            stack.append(
                _bounded(
                    _AbstractValue(
                        when_true.scalar_type,
                        min(when_true.lower, when_false.lower),
                        max(when_true.upper, when_false.upper),
                    ),
                    opcode=opcode,
                )
            )
            continue
        if opcode == "clamp":
            if len(stack) < 3:
                raise PolicyError(f"instruction {index} causes bytecode stack underflow")
            upper = stack.pop()
            lower = stack.pop()
            clamp_value = stack.pop()
            _numeric(clamp_value, opcode)
            _numeric(lower, opcode)
            _numeric(upper, opcode)
            if lower.lower != lower.upper or upper.lower != upper.upper:
                raise PolicyError("clamp bounds must be numeric literals")
            if lower.lower > upper.upper:
                raise PolicyError("clamp lower bound exceeds upper bound")
            # Clamp is monotonic.  Clamp each ordered endpoint independently;
            # max(value.lower, lower), min(value.upper, upper) is unsound for
            # disjoint intervals such as [0, 1] clamped to [2, 3].
            clamp_lower = min(max(clamp_value.lower, lower.lower), upper.upper)
            clamp_upper = min(max(clamp_value.upper, lower.lower), upper.upper)
            result_type = _promote(
                clamp_value.scalar_type, _promote(lower.scalar_type, upper.scalar_type)
            )
            stack.append(
                _bounded(_AbstractValue(result_type, clamp_lower, clamp_upper), opcode=opcode)
            )
            continue

        if len(stack) < 2:
            raise PolicyError(f"instruction {index} causes bytecode stack underflow")
        right = stack.pop()
        left = stack.pop()
        if opcode in {"and", "or"}:
            if left.scalar_type is not ScalarType.BOOL or right.scalar_type is not ScalarType.BOOL:
                raise PolicyError(f"{opcode} requires boolean operands")
            stack.append(_AbstractValue(ScalarType.BOOL, False, True))
            continue
        if opcode in {"eq", "ne"}:
            if left.scalar_type is not right.scalar_type:
                raise PolicyError(f"{opcode} requires operands of the same type")
            stack.append(_AbstractValue(ScalarType.BOOL, False, True))
            continue
        if opcode in {"lt", "le", "gt", "ge"}:
            _numeric(left, opcode)
            _numeric(right, opcode)
            stack.append(_AbstractValue(ScalarType.BOOL, False, True))
            continue
        _numeric(left, opcode)
        _numeric(right, opcode)
        lower_bound, upper_bound = _binary_range(opcode, left, right)
        stack.append(
            _bounded(
                _AbstractValue(
                    _promote(left.scalar_type, right.scalar_type), lower_bound, upper_bound
                ),
                opcode=opcode,
            )
        )

    if len(stack) != 1:
        raise PolicyError("bytecode did not leave exactly one abstract result")
    result = stack[0]
    if result.scalar_type is not program.output.scalar_type:
        raise PolicyError(
            f"bytecode return type {result.scalar_type.value} does not match "
            f"{program.output.scalar_type.value}"
        )
    if result.lower < program.output.lower or result.upper > program.output.upper:
        raise PolicyError(
            f"bytecode return range [{result.lower}, {result.upper}] exceeds output bounds "
            f"[{program.output.lower}, {program.output.upper}]"
        )
    return Analysis(result.scalar_type, result.lower, result.upper, instruction_count)


def _reject_constant(value: str) -> None:
    raise PolicyError(f"non-finite JSON number {value!r} is forbidden")


def _parse_integer(value: str) -> int:
    if len(value.removeprefix("-")) > 19:
        raise PolicyError("JSON integer exceeds the absolute policy bound")
    parsed = int(value)
    _checked_scalar(parsed, label="JSON integer")
    return parsed


def _parse_float(value: str) -> float:
    parsed = float(value)
    _checked_scalar(parsed, label="JSON float")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _variable_from_document(value: object, *, label: str) -> VariableSpec:
    if not isinstance(value, dict) or set(value) != _VARIABLE_FIELDS:
        raise PolicyError(f"{label} must contain exactly {sorted(_VARIABLE_FIELDS)}")
    name = _valid_name(value["name"], label=label)
    scalar_type_value = value["scalar_type"]
    if type(scalar_type_value) is not str:
        raise PolicyError(f"{label} scalar_type must be a string")
    try:
        scalar_type = ScalarType(scalar_type_value)
    except ValueError as error:
        raise PolicyError(f"{label} has an unknown scalar type") from error
    spec = VariableSpec(
        name=name,
        scalar_type=scalar_type,
        lower=_checked_scalar(value["lower"], label=f"{label} lower bound"),
        upper=_checked_scalar(value["upper"], label=f"{label} upper bound"),
    )
    _validate_spec(spec, label=label)
    return spec


def load_bytecode_document(payload: bytes) -> BytecodeProgram:
    """Parse and validate a bounded, strict JSON bytecode document."""

    if not 1 <= len(payload) <= MAX_POLICY_DOCUMENT_BYTES:
        raise PolicyError(
            f"policy bytecode document must be in [1, {MAX_POLICY_DOCUMENT_BYTES}] bytes"
        )
    try:
        document = json.loads(
            payload,
            parse_constant=_reject_constant,
            parse_int=_parse_integer,
            parse_float=_parse_float,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise PolicyError("policy bytecode is not valid strict JSON") from error
    if not isinstance(document, dict) or set(document) != _DOCUMENT_FIELDS:
        raise PolicyError(f"policy bytecode must contain exactly {sorted(_DOCUMENT_FIELDS)}")
    inputs_value = document["inputs"]
    instructions_value = document["instructions"]
    if not isinstance(inputs_value, list):
        raise PolicyError("policy bytecode inputs must be a list")
    if len(inputs_value) > MAX_POLICY_VARIABLES:
        raise PolicyError(f"policy bytecode has more than {MAX_POLICY_VARIABLES} inputs")
    if not isinstance(instructions_value, list):
        raise PolicyError("policy bytecode instructions must be a list")
    if len(instructions_value) > MAX_POLICY_INSTRUCTIONS:
        raise PolicyError(f"policy bytecode has more than {MAX_POLICY_INSTRUCTIONS} instructions")
    instructions: list[Instruction] = []
    for index, item in enumerate(instructions_value):
        if not isinstance(item, dict) or set(item) != _INSTRUCTION_FIELDS:
            raise PolicyError(
                f"instruction {index} must contain exactly {sorted(_INSTRUCTION_FIELDS)}"
            )
        opcode = item["opcode"]
        if type(opcode) is not str:
            raise PolicyError(f"instruction {index} opcode must be a string")
        operand = item["operand"]
        if operand is not None and type(operand) not in {bool, int, float, str}:
            raise PolicyError(f"instruction {index} operand has a forbidden type")
        instructions.append(Instruction(opcode=opcode, operand=operand))
    name = _valid_name(document["name"], label="policy")
    maximum_operations = document["maximum_operations"]
    if type(maximum_operations) is not int:
        raise PolicyError("maximum_operations must be an integer")
    program = BytecodeProgram(
        name=name,
        inputs=tuple(
            _variable_from_document(item, label=f"input {index}")
            for index, item in enumerate(inputs_value)
        ),
        output=_variable_from_document(document["output"], label="output"),
        instructions=tuple(instructions),
        maximum_operations=maximum_operations,
    )
    validate_bytecode(program)
    return program


def authenticate_bytecode_source(program: BytecodeProgram, source: bytes) -> None:
    """Require a policy source file to compile to exactly the admitted bytecode."""

    if not 1 <= len(source) <= MAX_POLICY_DOCUMENT_BYTES:
        raise PolicyError("policy source exceeds the absolute document bound")
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PolicyError("policy source is not UTF-8") from error
    # Local imports keep the bytecode checker independent from both modules.
    from .parser import parse_policy
    from .runtime import compile_policy

    expected = compile_policy(parse_policy(text))
    if expected != program:
        raise PolicyError("policy source does not authenticate the bytecode program")
