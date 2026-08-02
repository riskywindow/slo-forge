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
    StateAtomicityEvidence,
    StateOwnershipEvidence,
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
    with pytest.raises(RewriteError, match="finite"):
        explore_rewrites(_integer_identity_graph(), BUILTIN_RULES, quality_budget=float("nan"))
    assert structural_key(_integer_identity_graph()) == structural_key(_integer_identity_graph())


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1.0])
def test_graph_rejects_invalid_numerical_tolerances(tolerance: float) -> None:
    spec = TensorSpec(
        (4,),
        DType.FLOAT32,
        numerical=NumericalContract(maximum_absolute_error=tolerance),
    )
    with pytest.raises(RewriteError, match="finite and non-negative"):
        validate_graph(TensorGraph((TensorNode("out", "input", (), spec),), ("out",)))


def test_quantization_requires_a_finite_representable_value_domain() -> None:
    rule = _rule(RewriteKind.QUANTIZE_OUTPUT)

    def graph(contract: NumericalContract) -> TensorGraph:
        spec = TensorSpec((4,), DType.FLOAT32, numerical=contract)
        return TensorGraph((TensorNode("out", "input", (), spec),), ("out",))

    missing_domain = NumericalContract(
        maximum_absolute_error=0.02,
        preserve_nan=False,
        preserve_infinity=False,
        preserve_signed_zero=False,
    )
    assert not apply_rule(graph(missing_domain), rule, quality_budget=0.01)

    outside_int8_scale = replace(
        missing_domain,
        minimum_finite_value=-3.0,
        maximum_finite_value=3.0,
    )
    assert not apply_rule(graph(outside_int8_scale), rule, quality_budget=0.01)

    malformed_domain = replace(
        missing_domain,
        minimum_finite_value=float("nan"),
        maximum_finite_value=1.0,
    )
    with pytest.raises(RewriteError, match="finite and ordered"):
        validate_graph(graph(malformed_domain))

    bounded = replace(
        missing_domain,
        minimum_finite_value=-1.0,
        maximum_finite_value=1.0,
    )
    source = TensorSpec((4,), DType.FLOAT32, numerical=bounded)
    encoded = replace(source, dtype=DType.INT8)
    missing_scale = TensorGraph(
        (
            TensorNode("source", "input", (), source),
            TensorNode(
                "encoded",
                "quantize",
                ("source",),
                encoded,
                OperatorParameters(target_dtype=DType.INT8),
            ),
        ),
        ("encoded",),
    )
    with pytest.raises(RewriteError, match="finite positive symmetric scale"):
        validate_graph(missing_scale)

    underscaled = replace(
        missing_scale,
        nodes=(
            missing_scale.nodes[0],
            replace(
                missing_scale.nodes[1],
                parameters=OperatorParameters(target_dtype=DType.INT8, scale=0.001),
            ),
        ),
    )
    with pytest.raises(RewriteError, match="exceeds the int8 scale domain"):
        validate_graph(underscaled)


def test_exact_eliminations_require_the_full_output_contract() -> None:
    source = TensorSpec((2,), DType.INT32, alias_group="source")
    cast_output = replace(source, alias_group="cast-output")
    cast_graph = TensorGraph(
        (
            TensorNode("x", "input", (), source),
            TensorNode(
                "cast",
                "cast",
                ("x",),
                cast_output,
                OperatorParameters(target_dtype=DType.INT32),
            ),
        ),
        ("cast",),
    )
    assert not apply_rule(cast_graph, _rule(RewriteKind.REDUNDANT_CAST))

    zero = TensorNode(
        "zero",
        "constant",
        (),
        source,
        OperatorParameters(constant_value=0),
    )
    identity_graph = TensorGraph(
        (
            TensorNode("x", "input", (), source),
            zero,
            TensorNode("out", "add", ("x", "zero"), cast_output),
        ),
        ("out",),
    )
    assert not apply_rule(identity_graph, _rule(RewriteKind.ADD_ZERO))


def test_state_update_fusion_requires_explicit_ownership_and_atomicity_evidence() -> None:
    spec = TensorSpec((2,), DType.FLOAT32, state_dependency="kv-state")
    graph = TensorGraph(
        (
            TensorNode("state", "input", (), spec),
            TensorNode("delta", "input", (), spec),
            TensorNode("update", "add", ("state", "delta"), spec),
            TensorNode(
                "write",
                "state_write",
                ("update",),
                spec,
                OperatorParameters(state_key="kv-state"),
            ),
        ),
        ("write",),
    )
    rule = _rule(RewriteKind.STATE_UPDATE_FUSION)
    assert not apply_rule(graph, rule)
    evidenced = replace(
        graph,
        nodes=tuple(
            replace(
                node,
                parameters=replace(
                    node.parameters,
                    state_ownership=StateOwnershipEvidence.EXCLUSIVE,
                    state_atomicity=StateAtomicityEvidence.PER_TOKEN,
                ),
            )
            if node.node_id == "write"
            else node
            for node in graph.nodes
        ),
    )
    results = apply_rule(evidenced, rule)
    assert len(results) == 1
    assert next(node for node in results[0][0].nodes if node.node_id == "write").operator == (
        "fused_state_add"
    )


def test_quantization_inserts_a_consumer_without_destroying_the_original_output() -> None:
    spec = TensorSpec(
        (4,),
        DType.FLOAT32,
        alias_group="reference-output",
        numerical=NumericalContract(
            maximum_absolute_error=0.02,
            minimum_finite_value=-1.0,
            maximum_finite_value=1.0,
            preserve_nan=False,
            preserve_infinity=False,
            preserve_signed_zero=False,
        ),
    )
    graph = TensorGraph((TensorNode("out", "input", (), spec),), ("out",))
    strict_contract = replace(
        spec,
        numerical=NumericalContract(maximum_absolute_error=0.02),
    )
    assert not apply_rule(
        TensorGraph((TensorNode("out", "input", (), strict_contract),), ("out",)),
        _rule(RewriteKind.QUANTIZE_OUTPUT),
        quality_budget=0.01,
    )
    results = apply_rule(
        graph,
        _rule(RewriteKind.QUANTIZE_OUTPUT),
        quality_budget=0.01,
    )
    assert len(results) == 1
    candidate = results[0][0]
    original = next(node for node in candidate.nodes if node.node_id == "out")
    quantized = next(node for node in candidate.nodes if node.node_id == "out.quantized")
    assert original.operator == "input"
    assert original.output == spec
    assert quantized.inputs == ("out",)
    assert quantized.output.dtype is DType.INT8
    assert quantized.output.alias_group is None
    assert quantized.parameters.scale == 0.02
    assert candidate.outputs == ("out.quantized",)


def test_search_enforces_cumulative_quality_and_preserves_topological_order() -> None:
    spec = TensorSpec(
        (4,),
        DType.FLOAT32,
        numerical=NumericalContract(
            maximum_absolute_error=0.02,
            minimum_finite_value=-1.0,
            maximum_finite_value=1.0,
            preserve_nan=False,
            preserve_infinity=False,
            preserve_signed_zero=False,
        ),
    )
    multi_output = TensorGraph(
        (
            TensorNode("a", "input", (), spec),
            TensorNode("b", "input", (), spec),
        ),
        ("a", "b"),
    )
    quantize = (_rule(RewriteKind.QUANTIZE_OUTPUT),)
    bounded = explore_rewrites(
        multi_output,
        quantize,
        quality_budget=0.01,
        maximum_depth=2,
    )
    assert all(
        sum(item.quality_cost for item in candidate.history) <= 0.01 for candidate in bounded
    )
    assert all(len(candidate.history) <= 1 for candidate in bounded)
    expanded = explore_rewrites(
        multi_output,
        quantize,
        quality_budget=0.02,
        maximum_depth=2,
    )
    assert any(len(candidate.history) == 2 for candidate in expanded)

    graph = TensorGraph(
        (
            TensorNode("a", "input", (), spec),
            TensorNode("b", "input", (), spec),
            TensorNode("c", "input", (), spec),
            TensorNode("d", "input", (), spec),
            TensorNode("ab", "add", ("a", "b"), spec),
            TensorNode("abc", "add", ("ab", "c"), spec),
            TensorNode("out", "add", ("abc", "d"), spec),
        ),
        ("out",),
    )
    reassociated = apply_rule(
        graph,
        _rule(RewriteKind.REASSOCIATE_ADD),
        quality_budget=0.01,
    )
    target = next(item for item in reassociated if item[1].matched_nodes == ("ab", "abc"))[0]
    validate_graph(target)
    order = {node.node_id: index for index, node in enumerate(target.nodes)}
    assert order["abc.reassociated"] < order["abc"] < order["out"]
