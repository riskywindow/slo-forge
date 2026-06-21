"""Deterministic interpreter, bytecode compiler, and graph lowering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .checker import check_policy
from .model import (
    Binary,
    BytecodeProgram,
    Conditional,
    Expr,
    Instruction,
    Literal,
    PolicyError,
    PolicyGraph,
    PolicyGraphEdge,
    PolicyGraphNode,
    PolicyProgram,
    Scalar,
    ScalarType,
    Unary,
    Variable,
    VariableSpec,
)


def _validate_value(spec: VariableSpec, value: Scalar) -> None:
    expected = {
        ScalarType.BOOL: bool,
        ScalarType.INT: int,
        ScalarType.FLOAT: float,
    }[spec.scalar_type]
    if type(value) is not expected:
        raise PolicyError(f"input {spec.name!r} must be {spec.scalar_type.value}")
    if value < spec.lower or value > spec.upper:
        raise PolicyError(f"input {spec.name!r}={value} is outside [{spec.lower}, {spec.upper}]")


def _binary(operator: str, left: Scalar, right: Scalar) -> Scalar:
    if operator == "add":
        return left + right
    if operator == "sub":
        return left - right
    if operator == "mul":
        return left * right
    if operator == "floor_div":
        return left // right
    if operator == "min":
        return min(left, right)
    if operator == "max":
        return max(left, right)
    if operator == "and":
        return bool(left) and bool(right)
    if operator == "or":
        return bool(left) or bool(right)
    if operator == "eq":
        return left == right
    if operator == "ne":
        return left != right
    if operator == "lt":
        return left < right
    if operator == "le":
        return left <= right
    if operator == "gt":
        return left > right
    if operator == "ge":
        return left >= right
    raise PolicyError(f"unknown bytecode operator {operator!r}")


def _compile(expression: Expr, instructions: list[Instruction]) -> None:
    if isinstance(expression, Literal):
        instructions.append(Instruction("literal", expression.value))
    elif isinstance(expression, Variable):
        instructions.append(Instruction("input", expression.name))
    elif isinstance(expression, Unary):
        _compile(expression.operand, instructions)
        instructions.append(Instruction(expression.operator))
    elif isinstance(expression, Binary):
        _compile(expression.left, instructions)
        _compile(expression.right, instructions)
        instructions.append(Instruction(expression.operator))
    elif isinstance(expression, Conditional):
        _compile(expression.condition, instructions)
        _compile(expression.when_true, instructions)
        _compile(expression.when_false, instructions)
        instructions.append(Instruction("select"))
    else:
        _compile(expression.value, instructions)
        _compile(expression.lower, instructions)
        _compile(expression.upper, instructions)
        instructions.append(Instruction("clamp"))


def compile_policy(program: PolicyProgram) -> BytecodeProgram:
    check_policy(program)
    instructions: list[Instruction] = []
    _compile(program.expression, instructions)
    if len(instructions) > program.maximum_operations:
        raise PolicyError("compiled instruction count exceeds declared limit")
    return BytecodeProgram(
        name=program.name,
        inputs=program.inputs,
        output=program.output,
        instructions=tuple(instructions),
        maximum_operations=program.maximum_operations,
    )


def execute_bytecode(program: BytecodeProgram, inputs: Mapping[str, Scalar]) -> Scalar:
    expected_names = {item.name for item in program.inputs}
    if set(inputs) != expected_names:
        raise PolicyError(f"inputs must be exactly {sorted(expected_names)}")
    for spec in program.inputs:
        _validate_value(spec, inputs[spec.name])
    if len(program.instructions) > program.maximum_operations:
        raise PolicyError("bytecode exceeds its operation limit")
    stack: list[Scalar] = []
    for instruction in program.instructions:
        if instruction.opcode == "literal":
            if instruction.operand is None or isinstance(instruction.operand, str):
                raise PolicyError("invalid literal instruction")
            stack.append(instruction.operand)
        elif instruction.opcode == "input":
            if not isinstance(instruction.operand, str):
                raise PolicyError("invalid input instruction")
            stack.append(inputs[instruction.operand])
        elif instruction.opcode in {"not", "neg"}:
            if not stack:
                raise PolicyError("bytecode stack underflow")
            operand = stack.pop()
            stack.append(not bool(operand) if instruction.opcode == "not" else -operand)
        elif instruction.opcode == "select":
            if len(stack) < 3:
                raise PolicyError("bytecode stack underflow")
            when_false = stack.pop()
            when_true = stack.pop()
            condition = stack.pop()
            stack.append(when_true if bool(condition) else when_false)
        elif instruction.opcode == "clamp":
            if len(stack) < 3:
                raise PolicyError("bytecode stack underflow")
            upper = stack.pop()
            lower = stack.pop()
            value = stack.pop()
            stack.append(min(max(value, lower), upper))
        else:
            if len(stack) < 2:
                raise PolicyError("bytecode stack underflow")
            right = stack.pop()
            left = stack.pop()
            stack.append(_binary(instruction.opcode, left, right))
    if len(stack) != 1:
        raise PolicyError("bytecode did not leave exactly one result")
    result = stack[0]
    _validate_value(program.output, result)
    return result


def evaluate(program: PolicyProgram, inputs: Mapping[str, Scalar]) -> Scalar:
    return execute_bytecode(compile_policy(program), inputs)


@dataclass(slots=True)
class _GraphBuilder:
    nodes: list[PolicyGraphNode]
    edges: list[PolicyGraphEdge]

    def add(self, expression: Expr) -> int:
        node_id = len(self.nodes)
        if isinstance(expression, Literal):
            self.nodes.append(PolicyGraphNode(node_id, f"literal:{expression.value}"))
            return node_id
        if isinstance(expression, Variable):
            self.nodes.append(PolicyGraphNode(node_id, f"input:{expression.name}"))
            return node_id
        label = (
            expression.operator
            if isinstance(expression, (Unary, Binary))
            else "if"
            if isinstance(expression, Conditional)
            else "clamp"
        )
        self.nodes.append(PolicyGraphNode(node_id, label))
        children: tuple[tuple[str, Expr], ...]
        if isinstance(expression, Unary):
            children = (("operand", expression.operand),)
        elif isinstance(expression, Binary):
            children = (("left", expression.left), ("right", expression.right))
        elif isinstance(expression, Conditional):
            children = (
                ("condition", expression.condition),
                ("true", expression.when_true),
                ("false", expression.when_false),
            )
        else:
            children = (
                ("value", expression.value),
                ("lower", expression.lower),
                ("upper", expression.upper),
            )
        for role, child in children:
            self.edges.append(PolicyGraphEdge(node_id, self.add(child), role))
        return node_id


def policy_graph(program: PolicyProgram) -> PolicyGraph:
    check_policy(program)
    builder = _GraphBuilder([], [])
    builder.add(program.expression)
    return PolicyGraph(tuple(builder.nodes), tuple(builder.edges))
