"""Compile restricted distributed mutations through existing Fabric validators."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from sloforge.fabric.ir import PhysicalExecutionPlan

from .models import (
    CollectiveMutation,
    DistributedMutation,
    DistributedSynthesisResult,
    ExpertPlacementMutation,
    KVTransferMutation,
    OverlapMutation,
    RankPlacementMutation,
)


class DistributedSynthesisError(ValueError):
    """A mutation cannot be applied to the declared physical plan."""


def _single_index(items: list[dict[str, Any]], field: str, value: str) -> int:
    indices = [index for index, item in enumerate(items) if item.get(field) == value]
    if len(indices) != 1:
        raise DistributedSynthesisError(
            f"expected exactly one {field}={value!r}, observed {len(indices)}"
        )
    return indices[0]


def _mutate_collective(payload: dict[str, Any], mutation: CollectiveMutation) -> None:
    operations = cast(list[dict[str, Any]], payload["collectives"]["operations"])
    target = operations[_single_index(operations, "operation_id", mutation.operation_id)]
    participating = target["participating_ranks"]
    if set(mutation.rank_order) != set(participating):
        raise DistributedSynthesisError("collective rank_order must permute participating ranks")
    target.update(
        algorithm=mutation.algorithm,
        transport=mutation.transport,
        channel_count=mutation.channel_count,
        rank_order=list(mutation.rank_order),
    )


def _mutate_kv(payload: dict[str, Any], mutation: KVTransferMutation) -> None:
    kv_transfer = payload.get("kv_transfer")
    if not isinstance(kv_transfer, dict):
        raise DistributedSynthesisError("physical plan has no KV transfer surface")
    routes = cast(list[dict[str, Any]], kv_transfer["routes"])
    target = routes[_single_index(routes, "route_id", mutation.route_id)]
    target.update(
        chunk_bytes=mutation.chunk_bytes,
        maximum_inflight_chunks=mutation.maximum_inflight_chunks,
        overlap_with_decode=mutation.overlap_with_decode,
        eviction_policy=mutation.eviction_policy,
    )


def _mutate_overlap(payload: dict[str, Any], mutation: OverlapMutation) -> None:
    windows = cast(list[dict[str, Any]], payload["communication_overlap"]["windows"])
    target = windows[_single_index(windows, "window_id", mutation.window_id)]
    target.update(
        expected_overlap_fraction=mutation.expected_overlap_fraction,
        stream=mutation.stream,
        fallback_serialization=mutation.fallback_serialization,
    )


def _mutate_expert(payload: dict[str, Any], mutation: ExpertPlacementMutation) -> None:
    placement = payload.get("expert_placement")
    if not isinstance(placement, dict):
        raise DistributedSynthesisError("physical plan has no expert placement surface")
    assignments = cast(list[dict[str, Any]], placement["assignments"])
    target = assignments[_single_index(assignments, "expert_id", mutation.expert_id)]
    rank_count = len(cast(list[object], payload["rank_placement"]["bindings"]))
    if not mutation.rank_ids or len(set(mutation.rank_ids)) != len(mutation.rank_ids):
        raise DistributedSynthesisError("expert rank_ids must be non-empty and unique")
    if any(rank_id >= rank_count for rank_id in mutation.rank_ids):
        raise DistributedSynthesisError("expert placement references rank outside the plan")
    target.update(rank_ids=list(mutation.rank_ids), capacity_factor=mutation.capacity_factor)


def _mutate_rank_placement(payload: dict[str, Any], mutation: RankPlacementMutation) -> None:
    bindings = cast(list[dict[str, Any]], payload["rank_placement"]["bindings"])
    allocations = cast(list[dict[str, Any]], payload["memory"]["allocations"])
    rank_count = len(bindings)
    if tuple(sorted(mutation.source_rank_order)) != tuple(range(rank_count)):
        raise DistributedSynthesisError("source_rank_order must permute every logical rank")
    binding_by_rank = {int(item["rank_id"]): item for item in bindings}
    allocation_by_rank = {int(item["rank_id"]): item for item in allocations}
    reordered_bindings: list[dict[str, Any]] = []
    reordered_allocations: list[dict[str, Any]] = []
    for destination_rank, source_rank in enumerate(mutation.source_rank_order):
        binding = dict(binding_by_rank[source_rank])
        binding["rank_id"] = destination_rank
        reordered_bindings.append(binding)
        allocation = dict(allocation_by_rank[source_rank])
        allocation["rank_id"] = destination_rank
        reordered_allocations.append(allocation)
    payload["rank_placement"]["bindings"] = reordered_bindings
    payload["memory"]["allocations"] = reordered_allocations


def compile_distributed_mutation(
    source: PhysicalExecutionPlan, mutation: DistributedMutation, *, seed: int
) -> DistributedSynthesisResult:
    """Apply one mutation, then reparse through canonical Fabric validation."""

    if seed < 0:
        raise DistributedSynthesisError("seed must be non-negative")
    payload = source.model_dump(mode="json")
    if isinstance(mutation, CollectiveMutation):
        _mutate_collective(payload, mutation)
    elif isinstance(mutation, KVTransferMutation):
        _mutate_kv(payload, mutation)
    elif isinstance(mutation, OverlapMutation):
        _mutate_overlap(payload, mutation)
    elif isinstance(mutation, ExpertPlacementMutation):
        _mutate_expert(payload, mutation)
    elif isinstance(mutation, RankPlacementMutation):
        _mutate_rank_placement(payload, mutation)
    else:
        raise DistributedSynthesisError(f"unsupported mutation type {type(mutation).__name__}")

    identity = hashlib.sha256(
        json.dumps(
            {
                "source_plan_id": source.plan_id,
                "mutation": mutation.model_dump(mode="json"),
                "seed": seed,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    payload["plan_id"] = f"genesis-{identity[:24]}"
    history = cast(list[dict[str, Any]], payload["optimizer_history"])
    next_sequence = max((int(item["sequence"]) for item in history), default=-1) + 1
    history.append(
        {
            "sequence": next_sequence,
            "candidate_id": payload["plan_id"],
            "phase": "feasibility",
            "decision": "evaluate",
            "reason_code": f"genesis_{mutation.kind}_mutation",
            "simulator_calls": 0,
            "solver_time_ms": 0.0,
        }
    )
    extensions = cast(dict[str, Any], payload["extensions"])
    extensions["sloforge.dev/genesis-candidate"] = {
        "performance_evidence_valid": False,
        "requires_bounded_model_check": True,
        "seed": seed,
        "source_plan_id": source.plan_id,
        "transformation_id": mutation.transformation_id,
    }
    candidate = PhysicalExecutionPlan.model_validate_json(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False), strict=True
    )
    return DistributedSynthesisResult(
        source_plan_id=source.plan_id,
        candidate_plan=candidate,
        transformation_id=mutation.transformation_id,
        affected_surface=mutation.kind,
        fabric_schema_validated=True,
        performance_evidence_valid=False,
        required_verifier_stages=(
            "fabric-static-validation",
            "digital-twin",
            "bounded-protocol-model-check",
            "end-to-end-benchmark",
        ),
        invalidated_evidence_uris=tuple(sorted(item.uri for item in source.evidence)),
        bounded_model_check_required=True,
    )


__all__ = ["DistributedSynthesisError", "compile_distributed_mutation"]
