"""Validation and bounded deterministic tensor rewrite exploration."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import asdict, replace

from .model import (
    DType,
    OperatorParameters,
    RewriteApplication,
    RewriteCandidate,
    RewriteError,
    RewriteKind,
    RewriteRule,
    SemanticCategory,
    StateAtomicityEvidence,
    StateOwnershipEvidence,
    TensorGraph,
    TensorNode,
    TensorSpec,
)

_FLOAT_DTYPES = {DType.FLOAT64, DType.FLOAT32, DType.BFLOAT16, DType.FLOAT16}


def _validate_spec(spec: TensorSpec) -> None:
    if not spec.shape:
        raise RewriteError("rank-zero tensors must be represented with shape (1,)")
    for dimension in spec.shape:
        if type(dimension) is int and dimension <= 0:
            raise RewriteError("static dimensions must be positive")
        if type(dimension) is str and not dimension:
            raise RewriteError("symbolic dimensions must be non-empty")
    if spec.strides is not None and len(spec.strides) != len(spec.shape):
        raise RewriteError("stride rank must match shape rank")
    numerical = spec.numerical
    if numerical.maximum_absolute_error < 0 or numerical.maximum_relative_error < 0:
        raise RewriteError("numerical tolerances must be non-negative")


def validate_graph(graph: TensorGraph) -> None:
    known: dict[str, TensorNode] = {}
    for node in graph.nodes:
        if not node.node_id or node.node_id in known:
            raise RewriteError(f"duplicate or empty node id {node.node_id!r}")
        _validate_spec(node.output)
        unknown = [input_id for input_id in node.inputs if input_id not in known]
        if unknown:
            raise RewriteError(f"node {node.node_id!r} has non-topological inputs {unknown}")
        if node.operator == "constant" and node.parameters.constant_value is None:
            raise RewriteError("constant node requires constant_value")
        if node.operator == "cast":
            if len(node.inputs) != 1 or node.parameters.target_dtype is None:
                raise RewriteError("cast requires one input and target_dtype")
            if node.output.dtype is not node.parameters.target_dtype:
                raise RewriteError("cast output dtype must equal target_dtype")
        if node.operator == "quantize":
            if len(node.inputs) != 1 or node.parameters.target_dtype is None:
                raise RewriteError("quantize requires one input and target_dtype")
            source = known[node.inputs[0]].output
            if source.dtype not in _FLOAT_DTYPES:
                raise RewriteError("quantize source must have a floating dtype")
            expected = replace(
                source,
                dtype=node.parameters.target_dtype,
                alias_group=None,
            )
            if node.output != expected:
                raise RewriteError("quantize output must preserve the full source contract")
        if node.operator == "transpose":
            permutation = node.parameters.permutation
            if len(node.inputs) != 1 or permutation is None:
                raise RewriteError("transpose requires one input and a permutation")
            source = known[node.inputs[0]].output
            if tuple(sorted(permutation)) != tuple(range(len(source.shape))):
                raise RewriteError("transpose permutation must cover every axis")
            if node.output.shape != tuple(source.shape[index] for index in permutation):
                raise RewriteError("transpose output shape does not match permutation")
        if node.operator in {"add", "mul"}:
            if len(node.inputs) != 2:
                raise RewriteError(f"{node.operator} requires two inputs")
            left, right = (known[item].output for item in node.inputs)
            if left.shape != right.shape or node.output.shape != left.shape:
                raise RewriteError(f"{node.operator} requires identical shapes")
            if left.dtype is not right.dtype or node.output.dtype is not left.dtype:
                raise RewriteError(f"{node.operator} requires identical dtypes")
        known[node.node_id] = node
    missing_outputs = [item for item in graph.outputs if item not in known]
    if missing_outputs:
        raise RewriteError(f"unknown graph outputs {missing_outputs}")


def structural_key(graph: TensorGraph) -> str:
    validate_graph(graph)
    payload = json.dumps(asdict(graph), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _consumers(graph: TensorGraph) -> dict[str, int]:
    counts = {node.node_id: 0 for node in graph.nodes}
    for node in graph.nodes:
        for input_id in node.inputs:
            counts[input_id] += 1
    for output in graph.outputs:
        counts[output] += 1
    return counts


def _replace_reference(graph: TensorGraph, old: str, new: str, *, remove: set[str]) -> TensorGraph:
    nodes = tuple(
        replace(node, inputs=tuple(new if item == old else item for item in node.inputs))
        for node in graph.nodes
        if node.node_id not in remove
    )
    outputs = tuple(new if item == old else item for item in graph.outputs)
    result = TensorGraph(nodes, outputs)
    return _prune(result)


def _prune(graph: TensorGraph) -> TensorGraph:
    by_id = {node.node_id: node for node in graph.nodes}
    needed = set(graph.outputs)
    pending = list(graph.outputs)
    while pending:
        current = pending.pop()
        for dependency in by_id[current].inputs:
            if dependency not in needed:
                needed.add(dependency)
                pending.append(dependency)
    result = TensorGraph(
        tuple(node for node in graph.nodes if node.node_id in needed), graph.outputs
    )
    validate_graph(result)
    return result


def _application(
    rule: RewriteRule,
    matched: tuple[str, ...],
    quality_cost: float,
    checks: tuple[str, ...],
) -> RewriteApplication:
    return RewriteApplication(
        rule_id=rule.rule_id,
        matched_nodes=matched,
        semantic_category=rule.semantic_category,
        quality_cost=quality_cost,
        verification_obligations=rule.verification_obligations,
        preconditions_checked=checks,
    )


def _is_identity_constant(node: TensorNode, identity: int) -> bool:
    return node.operator == "constant" and node.parameters.constant_value == identity


def _apply_at(
    graph: TensorGraph,
    rule: RewriteRule,
    node: TensorNode,
    *,
    quality_budget: float,
) -> tuple[TensorGraph, RewriteApplication] | None:
    by_id = {item.node_id: item for item in graph.nodes}
    if node.output.dtype not in rule.supported_dtypes:
        return None
    if rule.semantic_category is not SemanticCategory.EXACT and (
        rule.maximum_quality_cost <= 0 or quality_budget < rule.maximum_quality_cost
    ):
        return None
    if rule.requires_signed_zero_insensitive and node.output.numerical.preserve_signed_zero:
        return None
    if rule.kind is RewriteKind.REDUNDANT_CAST and node.operator == "cast":
        source = by_id[node.inputs[0]]
        if source.output == node.output:
            return (
                _replace_reference(graph, node.node_id, source.node_id, remove={node.node_id}),
                _application(rule, (node.node_id,), 0.0, ("full_output_contract_equal",)),
            )
    if rule.kind is RewriteKind.DOUBLE_TRANSPOSE and node.operator == "transpose":
        inner = by_id[node.inputs[0]]
        if inner.operator != "transpose":
            return None
        outer_perm = node.parameters.permutation
        inner_perm = inner.parameters.permutation
        if outer_perm is None or inner_perm is None:
            return None
        composed = tuple(inner_perm[index] for index in outer_perm)
        source = by_id[inner.inputs[0]]
        if composed == tuple(range(len(composed))) and source.output == node.output:
            return (
                _replace_reference(graph, node.node_id, source.node_id, remove={node.node_id}),
                _application(
                    rule,
                    (inner.node_id, node.node_id),
                    0.0,
                    ("inverse_permutations", "full_output_contract_equal"),
                ),
            )
    if rule.kind in {RewriteKind.ADD_ZERO, RewriteKind.MUL_ONE} and node.operator in {
        "add",
        "mul",
    }:
        expected_operator = "add" if rule.kind is RewriteKind.ADD_ZERO else "mul"
        identity = 0 if rule.kind is RewriteKind.ADD_ZERO else 1
        if node.operator != expected_operator:
            return None
        left, right = (by_id[item] for item in node.inputs)
        constant = right if _is_identity_constant(right, identity) else left
        source = left if constant is right else right
        if not _is_identity_constant(constant, identity):
            return None
        if (
            node.output.dtype in _FLOAT_DTYPES
            and rule.kind is RewriteKind.ADD_ZERO
            and node.output.numerical.preserve_signed_zero
        ):
            return None
        if source.output != node.output:
            return None
        return (
            _replace_reference(graph, node.node_id, source.node_id, remove={node.node_id}),
            _application(
                rule,
                (constant.node_id, node.node_id),
                0.0,
                ("identity_constant", "full_output_contract_equal"),
            ),
        )
    if rule.kind is RewriteKind.REASSOCIATE_ADD and node.operator == "add":
        left = by_id[node.inputs[0]]
        if left.operator != "add" or left.output.dtype not in _FLOAT_DTYPES:
            return None
        a, b = left.inputs
        c = node.inputs[1]
        right_id = f"{node.node_id}.reassociated"
        if right_id in by_id:
            return None
        right = TensorNode(right_id, "add", (b, c), node.output)
        rewritten = replace(node, inputs=(a, right_id))
        ordered: list[TensorNode] = []
        for item in graph.nodes:
            if item.node_id == node.node_id:
                ordered.extend((right, rewritten))
            else:
                ordered.append(item)
        nodes = tuple(ordered)
        candidate = _prune(TensorGraph(nodes, graph.outputs))
        return (
            candidate,
            _application(
                rule,
                (left.node_id, node.node_id),
                rule.maximum_quality_cost,
                ("floating_dtype", "quality_budget_available"),
            ),
        )
    if rule.kind is RewriteKind.STATE_UPDATE_FUSION and node.operator == "state_write":
        if len(node.inputs) != 1:
            return None
        update = by_id[node.inputs[0]]
        if update.operator != "add" or node.output.state_dependency is None:
            return None
        if node.parameters.state_key != node.output.state_dependency:
            return None
        if node.parameters.state_ownership is not StateOwnershipEvidence.EXCLUSIVE:
            return None
        if node.parameters.state_atomicity is not StateAtomicityEvidence.PER_TOKEN:
            return None
        fused = replace(node, operator="fused_state_add", inputs=update.inputs)
        candidate = _prune(
            TensorGraph(
                tuple(fused if item.node_id == node.node_id else item for item in graph.nodes),
                graph.outputs,
            )
        )
        return (
            candidate,
            _application(
                rule,
                (update.node_id, node.node_id),
                0.0,
                (
                    "exclusive_state_ownership_evidence",
                    "per_token_atomicity_evidence",
                    "state_key_equal",
                ),
            ),
        )
    if rule.kind is RewriteKind.QUANTIZE_OUTPUT and node.node_id in graph.outputs:
        if (
            node.output.dtype not in _FLOAT_DTYPES
            or node.output.numerical.maximum_absolute_error <= 0
            or rule.maximum_quality_cost
            > node.output.numerical.maximum_absolute_error
            or node.output.numerical.preserve_nan
            or node.output.numerical.preserve_infinity
            or node.output.numerical.preserve_signed_zero
        ):
            return None
        quantized_id = f"{node.node_id}.quantized"
        if quantized_id in by_id:
            return None
        quantized = TensorNode(
            node_id=quantized_id,
            operator="quantize",
            inputs=(node.node_id,),
            output=replace(node.output, dtype=DType.INT8, alias_group=None),
            parameters=OperatorParameters(target_dtype=DType.INT8),
        )
        candidate = TensorGraph(
            (*graph.nodes, quantized),
            tuple(quantized_id if output == node.node_id else output for output in graph.outputs),
        )
        validate_graph(candidate)
        return (
            candidate,
            _application(
                rule,
                (node.node_id, quantized_id),
                rule.maximum_quality_cost,
                ("quality_tolerance_nonzero", "int8_domain"),
            ),
        )
    return None


def apply_rule(
    graph: TensorGraph,
    rule: RewriteRule,
    *,
    quality_budget: float = 0.0,
) -> tuple[tuple[TensorGraph, RewriteApplication], ...]:
    validate_graph(graph)
    if not math.isfinite(quality_budget) or quality_budget < 0:
        raise RewriteError("quality budget must be finite and non-negative")
    if (
        not rule.rule_id
        or not rule.supported_dtypes
        or not rule.verification_obligations
        or not math.isfinite(rule.maximum_quality_cost)
        or rule.maximum_quality_cost < 0
    ):
        raise RewriteError("rewrite rule metadata is incomplete or invalid")
    if rule.semantic_category is SemanticCategory.EXACT and rule.maximum_quality_cost != 0:
        raise RewriteError("exact rewrite rules cannot declare quality cost")
    results: list[tuple[TensorGraph, RewriteApplication]] = []
    seen: set[str] = set()
    for node in graph.nodes:
        applied = _apply_at(graph, rule, node, quality_budget=quality_budget)
        if applied is None:
            continue
        key = structural_key(applied[0])
        if key not in seen:
            seen.add(key)
            results.append(applied)
    return tuple(results)


def explore_rewrites(
    graph: TensorGraph,
    rules: tuple[RewriteRule, ...],
    *,
    quality_budget: float,
    maximum_candidates: int = 128,
    maximum_depth: int = 4,
) -> tuple[RewriteCandidate, ...]:
    validate_graph(graph)
    if not math.isfinite(quality_budget) or quality_budget < 0:
        raise RewriteError("quality budget must be finite and non-negative")
    if maximum_candidates <= 0 or maximum_depth < 0:
        raise RewriteError("candidate and depth bounds must be valid")
    root_key = structural_key(graph)
    archive = [RewriteCandidate(graph, (), root_key)]
    queue: deque[tuple[TensorGraph, tuple[RewriteApplication, ...], int]] = deque([(graph, (), 0)])
    seen = {root_key}
    while queue and len(archive) < maximum_candidates:
        current, history, depth = queue.popleft()
        if depth >= maximum_depth:
            continue
        spent = math.fsum(application.quality_cost for application in history)
        remaining_quality = max(0.0, quality_budget - spent)
        for rule in rules:
            for candidate, application in apply_rule(
                current,
                rule,
                quality_budget=remaining_quality,
            ):
                if spent + application.quality_cost > quality_budget + 1e-15:
                    continue
                key = structural_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                candidate_history = (*history, application)
                archive.append(RewriteCandidate(candidate, candidate_history, key))
                queue.append((candidate, candidate_history, depth + 1))
                if len(archive) >= maximum_candidates:
                    break
            if len(archive) >= maximum_candidates:
                break
    return tuple(archive)
