from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sloforge.genesis.policy_dsl import (
    Binary,
    Literal,
    PolicyError,
    check_equivalent,
    check_policy,
    compile_policy,
    evaluate,
    execute_bytecode,
    format_policy,
    mutate_policy,
    parse_policy,
    policy_graph,
    simplify_policy,
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
