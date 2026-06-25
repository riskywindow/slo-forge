from __future__ import annotations

import dataclasses
import json
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sloforge.genesis.policy_dsl import (
    MAX_POLICY_DOCUMENT_BYTES,
    MAX_POLICY_INTEGER_ABS,
    Binary,
    BytecodeProgram,
    Instruction,
    Literal,
    PolicyError,
    ScalarType,
    VariableSpec,
    authenticate_bytecode_source,
    check_equivalent,
    check_policy,
    compile_policy,
    evaluate,
    execute_bytecode,
    format_policy,
    load_bytecode_document,
    mutate_policy,
    parse_policy,
    policy_graph,
    simplify_policy,
    validate_bytecode,
)

SOURCE = """\
policy slack_batch
input queue_length int 0 16
input priority int 0 10
input overloaded bool false true
output int 0 8
limit 64
return (clamp (if (or overloaded (ge priority 7)) (min queue_length 2) (min queue_length 8)) 0 8)
"""


def test_parse_typecheck_format_bytecode_and_graph() -> None:
    program = parse_policy(SOURCE)
    analysis = check_policy(program)
    assert analysis.lower == 0
    assert analysis.upper == 8
    assert format_policy(parse_policy(format_policy(program))) == format_policy(program)
    bytecode = compile_policy(program)
    inputs = {"queue_length": 7, "priority": 9, "overloaded": False}
    assert evaluate(program, inputs) == 2
    assert execute_bytecode(bytecode, inputs) == 2
    graph = policy_graph(program)
    assert graph.nodes[0].label == "clamp"
    assert len(graph.edges) == len(graph.nodes) - 1


def test_policy_rejects_unbounded_invalid_and_unknown_behavior() -> None:
    with pytest.raises(PolicyError, match="outside"):
        evaluate(parse_policy(SOURCE), {"queue_length": 17, "priority": 1, "overloaded": False})
    with pytest.raises(PolicyError, match="not allowed"):
        parse_policy(SOURCE.replace("(min queue_length 2)", "(read_file queue_length 2)"))
    with pytest.raises(PolicyError, match="operation limit"):
        check_policy(replace(parse_policy(SOURCE), maximum_operations=0))
    with pytest.raises(PolicyError, match="denominator range includes zero"):
        check_policy(
            replace(
                parse_policy(SOURCE),
                expression=Binary("floor_div", Literal(4), Binary("sub", Literal(1), Literal(1))),
            )
        )


def test_clamp_range_analysis_preserves_order_for_disjoint_intervals() -> None:
    program = parse_policy(
        """\
policy disjoint_clamp
input value int 0 1
output int 2 2
limit 8
return (clamp value 2 3)
"""
    )
    analysis = check_policy(program)
    admitted = validate_bytecode(compile_policy(program))
    assert (analysis.lower, analysis.upper) == (2, 2)
    assert admitted == analysis
    assert [evaluate(program, {"value": value}) for value in range(2)] == [2, 2]


def test_numeric_promotion_is_reflected_by_the_interpreter() -> None:
    clamped = parse_policy(
        """\
policy promoted_clamp
input value int 0 3
output float 0.0 3.0
limit 8
return (clamp value 0.0 3.0)
"""
    )
    minimum = parse_policy(
        """\
policy promoted_min
input value int 0 3
output float 0.0 2.0
limit 8
return (min value 2.0)
"""
    )
    for program in (clamped, minimum):
        validate_bytecode(compile_policy(program))
        for value in range(4):
            assert type(evaluate(program, {"value": value})) is float


@given(
    value_lower=st.integers(min_value=-4, max_value=4),
    value_width=st.integers(min_value=0, max_value=4),
    clamp_lower=st.integers(min_value=-6, max_value=6),
    clamp_width=st.integers(min_value=0, max_value=4),
)
def test_checker_bytecode_validator_and_interpreter_agree_exhaustively_for_clamp(
    value_lower: int, value_width: int, clamp_lower: int, clamp_width: int
) -> None:
    value_upper = value_lower + value_width
    clamp_upper = clamp_lower + clamp_width
    expected = [
        min(max(value, clamp_lower), clamp_upper) for value in range(value_lower, value_upper + 1)
    ]
    output_lower, output_upper = min(expected), max(expected)
    program = parse_policy(
        f"""\
policy exhaustive_clamp
input value int {value_lower} {value_upper}
output int {output_lower} {output_upper}
limit 8
return (clamp value {clamp_lower} {clamp_upper})
"""
    )
    checked = check_policy(program)
    bytecode = compile_policy(program)
    admitted = validate_bytecode(bytecode)
    observed = [
        execute_bytecode(bytecode, {"value": value})
        for value in range(value_lower, value_upper + 1)
    ]
    assert (checked.lower, checked.upper) == (output_lower, output_upper)
    assert admitted == checked
    assert observed == expected


def test_strict_bytecode_loader_rejects_hostile_documents_and_numbers() -> None:
    payload = json.dumps(
        dataclasses.asdict(compile_policy(parse_policy(SOURCE))),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert load_bytecode_document(payload) == compile_policy(parse_policy(SOURCE))

    hostile = json.loads(payload)
    hostile["instructions"][-1]["opcode"] = "dynamic_import"
    with pytest.raises(PolicyError, match="forbidden opcode"):
        load_bytecode_document(json.dumps(hostile).encode())
    with pytest.raises(PolicyError, match="non-finite"):
        load_bytecode_document(payload.replace(b'"lower":0', b'"lower":NaN', 1))
    with pytest.raises(PolicyError, match="integer exceeds"):
        load_bytecode_document(
            payload.replace(b'"lower":0', f'"lower":{MAX_POLICY_INTEGER_ABS + 1}'.encode(), 1)
        )
    with pytest.raises(PolicyError, match="document"):
        load_bytecode_document(b" " * (MAX_POLICY_DOCUMENT_BYTES + 1))
    with pytest.raises(PolicyError, match="non-finite"):
        parse_policy(SOURCE.replace("return ", "return NaN # "))
    with pytest.raises(PolicyError, match="integer exceeds"):
        parse_policy(SOURCE.replace("output int 0 8", f"output int 0 {10**20}"))


def test_policy_source_authenticates_exact_compiler_output() -> None:
    bytecode = compile_policy(parse_policy(SOURCE))
    authenticate_bytecode_source(bytecode, SOURCE.encode())
    changed_source = SOURCE.replace("(min queue_length 2)", "(min queue_length 3)")
    with pytest.raises(PolicyError, match="does not authenticate"):
        authenticate_bytecode_source(bytecode, changed_source.encode())


def test_bytecode_instruction_count_has_an_absolute_ceiling() -> None:
    program = BytecodeProgram(
        name="too_many_instructions",
        inputs=(),
        output=VariableSpec("output", ScalarType.INT, 0, 0),
        instructions=tuple(Instruction("literal", 0) for _ in range(4097)),
        maximum_operations=4096,
    )
    with pytest.raises(PolicyError, match="instruction count"):
        validate_bytecode(program)


@pytest.mark.parametrize(
    "program, message",
    [
        (
            BytecodeProgram(
                name="unknown",
                inputs=(),
                output=VariableSpec("output", ScalarType.INT, 0, 1),
                instructions=(Instruction("open_socket"),),
                maximum_operations=1,
            ),
            "forbidden opcode",
        ),
        (
            BytecodeProgram(
                name="underflow",
                inputs=(),
                output=VariableSpec("output", ScalarType.INT, 0, 1),
                instructions=(Instruction("add"),),
                maximum_operations=1,
            ),
            "stack underflow",
        ),
        (
            BytecodeProgram(
                name="out_of_range",
                inputs=(),
                output=VariableSpec("output", ScalarType.INT, 0, 1),
                instructions=(Instruction("literal", 2),),
                maximum_operations=1,
            ),
            "return range",
        ),
        (
            BytecodeProgram(
                name="duplicate_inputs",
                inputs=(
                    VariableSpec("value", ScalarType.INT, 0, 1),
                    VariableSpec("value", ScalarType.INT, 0, 1),
                ),
                output=VariableSpec("output", ScalarType.INT, 0, 1),
                instructions=(Instruction("literal", 0),),
                maximum_operations=1,
            ),
            "duplicate input",
        ),
        (
            BytecodeProgram(
                name="operand_smuggling",
                inputs=(),
                output=VariableSpec("output", ScalarType.INT, 0, 1),
                instructions=(
                    Instruction("literal", 0),
                    Instruction("literal", 1),
                    Instruction("add", "open_socket"),
                ),
                maximum_operations=3,
            ),
            "forbids an operand",
        ),
        (
            BytecodeProgram(
                name="unbounded_limit",
                inputs=(),
                output=VariableSpec("output", ScalarType.INT, 0, 1),
                instructions=(Instruction("literal", 0),),
                maximum_operations=4097,
            ),
            "operation limit",
        ),
    ],
)
def test_independent_bytecode_validator_rejects_hostile_ir(
    program: BytecodeProgram, message: str
) -> None:
    with pytest.raises(PolicyError, match=message):
        validate_bytecode(program)


def test_checker_rejects_non_finite_and_huge_constructed_literals() -> None:
    program = parse_policy(SOURCE)
    with pytest.raises(PolicyError, match="bounded finite"):
        check_policy(replace(program, expression=Literal(float("nan"))))
    with pytest.raises(PolicyError, match="absolute policy bound"):
        check_policy(replace(program, expression=Literal(MAX_POLICY_INTEGER_ABS + 1)))


def test_simplification_equivalence_and_real_counterexample() -> None:
    source = SOURCE.replace(
        "(min queue_length 2)", "(if true (min queue_length 2) (min queue_length 2))"
    )
    original = parse_policy(source)
    simplified = simplify_policy(original)
    result = check_equivalent(original, simplified)
    assert result.equivalent
    assert result.states_checked == 17 * 11 * 2
    changed = parse_policy(SOURCE.replace("(min queue_length 2)", "(min queue_length 3)"))
    failure = check_equivalent(parse_policy(SOURCE), changed)
    assert not failure.equivalent
    assert failure.counterexample is not None


def test_mutation_is_seeded_and_admitted_by_checker() -> None:
    program = parse_policy(SOURCE)
    first = mutate_policy(program, seed=73129)
    second = mutate_policy(program, seed=73129)
    assert first == second
    assert first != program
    check_policy(first)


@given(
    queue_length=st.integers(min_value=0, max_value=16),
    priority=st.integers(min_value=0, max_value=10),
    overloaded=st.booleans(),
)
def test_interpreter_is_deterministic_and_bounded(
    queue_length: int, priority: int, overloaded: bool
) -> None:
    program = parse_policy(SOURCE)
    inputs = {
        "queue_length": queue_length,
        "priority": priority,
        "overloaded": overloaded,
    }
    first = evaluate(program, inputs)
    assert first == evaluate(program, inputs)
    assert type(first) is int
    assert 0 <= first <= 8
