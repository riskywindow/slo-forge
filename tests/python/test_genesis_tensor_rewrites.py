from __future__ import annotations

from dataclasses import replace

import pytest

from sloforge.genesis.tensor_rewrites import (
    BUILTIN_RULES,
    DType,
    NumericalContract,
    OperatorParameters,
    RewriteError,
    RewriteKind,
    TensorGraph,
    TensorNode,
    TensorSpec,
    apply_rule,
    explore_rewrites,
    structural_key,
    validate_graph,
)


def _rule(kind: RewriteKind):
    return next(rule for rule in BUILTIN_RULES if rule.kind is kind)


def _integer_identity_graph() -> TensorGraph:
    spec = TensorSpec(("batch", 8), DType.INT32)
    return TensorGraph(
        (
            TensorNode("input", "input", (), spec),
            TensorNode(
                "zero",
                "constant",
                (),
                spec,
                OperatorParameters(constant_value=0),
            ),
            TensorNode("sum", "add", ("input", "zero"), spec),
            TensorNode(
                "cast",
                "cast",
                ("sum",),
                spec,
                OperatorParameters(target_dtype=DType.INT32),
            ),
        ),
        ("cast",),
    )


def test_exact_rewrite_exploration_is_deterministic_and_typed() -> None:
    graph = _integer_identity_graph()
    validate_graph(graph)
    candidates = explore_rewrites(
        graph,
        BUILTIN_RULES,
        quality_budget=0.0,
        maximum_candidates=16,
        maximum_depth=3,
    )
    assert candidates == explore_rewrites(
        graph,
        BUILTIN_RULES,
        quality_budget=0.0,
        maximum_candidates=16,
        maximum_depth=3,
    )
    assert len({item.structural_key for item in candidates}) == len(candidates)
    assert any(
        application.rule_id == "tensor/add-zero-integer/v1"
        for candidate in candidates
        for application in candidate.history
    )
    assert len(candidates[-1].graph.nodes) < len(graph.nodes)


def test_double_transpose_eliminates_inverse_pair() -> None:
    source = TensorSpec((2, 3), DType.FLOAT32)
    transposed = replace(source, shape=(3, 2))
    graph = TensorGraph(
        (
            TensorNode("x", "input", (), source),
            TensorNode(
                "t1",
                "transpose",
                ("x",),
                transposed,
                OperatorParameters(permutation=(1, 0)),
            ),
            TensorNode(
                "t2",
                "transpose",
                ("t1",),
                source,
                OperatorParameters(permutation=(1, 0)),
            ),
        ),
        ("t2",),
    )
    results = apply_rule(graph, _rule(RewriteKind.DOUBLE_TRANSPOSE))
    assert len(results) == 1
    assert results[0][0].outputs == ("x",)
    assert tuple(node.node_id for node in results[0][0].nodes) == ("x",)


def test_float_reassociation_is_never_treated_as_exact() -> None:
    spec = TensorSpec(
        (4,),
        DType.FLOAT32,
        numerical=NumericalContract(maximum_absolute_error=1e-4),
    )
    graph = TensorGraph(
        (
            TensorNode("a", "input", (), spec),
            TensorNode("b", "input", (), spec),
            TensorNode("c", "input", (), spec),
            TensorNode("ab", "add", ("a", "b"), spec),
            TensorNode("out", "add", ("ab", "c"), spec),
        ),
        ("out",),
    )
    rule = _rule(RewriteKind.REASSOCIATE_ADD)
    assert not apply_rule(graph, rule, quality_budget=0.0)
    results = apply_rule(graph, rule, quality_budget=1e-4)
    assert len(results) == 1
    assert results[0][1].quality_cost == 1e-5
    assert "quality_budget" in results[0][1].verification_obligations


def test_float_add_zero_respects_signed_zero_contract() -> None:
    spec = TensorSpec((2,), DType.FLOAT32)
    graph = TensorGraph(
        (
            TensorNode("x", "input", (), spec),
            TensorNode("zero", "constant", (), spec, OperatorParameters(constant_value=0.0)),
            TensorNode("out", "add", ("x", "zero"), spec),
        ),
        ("out",),
    )
    float_rule = next(
        rule
        for rule in BUILTIN_RULES
        if rule.kind is RewriteKind.ADD_ZERO and DType.FLOAT32 in rule.supported_dtypes
    )
    assert not apply_rule(graph, float_rule)
    relaxed = replace(spec, numerical=replace(spec.numerical, preserve_signed_zero=False))
    relaxed_graph = TensorGraph(
        tuple(replace(node, output=relaxed) for node in graph.nodes), graph.outputs
    )
    assert len(apply_rule(relaxed_graph, float_rule)) == 1


def test_invalid_graphs_and_bounds_fail_closed() -> None:
    spec = TensorSpec((2,), DType.INT32)
    graph = TensorGraph((TensorNode("x", "add", ("missing", "missing"), spec),), ("x",))
    with pytest.raises(RewriteError, match="non-topological"):
        validate_graph(graph)
    with pytest.raises(RewriteError, match="non-negative"):
        apply_rule(_integer_identity_graph(), _rule(RewriteKind.ADD_ZERO), quality_budget=-1)
    assert structural_key(_integer_identity_graph()) == structural_key(_integer_identity_graph())
