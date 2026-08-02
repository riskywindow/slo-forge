"""Conservative state transformation compilation and resource analysis."""

from __future__ import annotations

import math

from .model import (
    CompiledStateTransition,
    Consistency,
    Ownership,
    StateRegion,
    StateTransformation,
    StateTransformError,
    StorageTier,
)


def region_capacity_bytes(region: StateRegion) -> int:
    validate_region(region)
    return region.bytes_per_item * region.maximum_items * region.replication_factor


def validate_region(region: StateRegion) -> None:
    if not region.region_id or not region.semantic_contract:
        raise StateTransformError("state region identity and semantic contract are required")
    if region.bytes_per_item <= 0 or region.maximum_items <= 0:
        raise StateTransformError("state capacity dimensions must be positive")
    if region.replication_factor <= 0:
        raise StateTransformError("replication factor must be positive")
    if len(set(region.owners)) != len(region.owners) or not region.owners:
        raise StateTransformError("state owners must be non-empty and unique")
    if region.ownership is Ownership.EXCLUSIVE:
        if region.replication_factor != 1 or len(region.owners) != 1:
            raise StateTransformError("exclusive state must have exactly one owner and replica")
    elif len(region.owners) != region.replication_factor:
        raise StateTransformError("replicated state owners must match replication factor")
    if region.ownership is Ownership.REPLICATED_READ_ONLY and region.mutable:
        raise StateTransformError("replicated read-only state cannot be mutable")
    if (
        region.ownership is Ownership.REPLICATED_CONSENSUS
        and region.consistency is not Consistency.LINEARIZABLE
    ):
        raise StateTransformError("mutable replicated state requires linearizable consistency")
    if region.storage is StorageTier.REMOTE and region.consistency is Consistency.LINEARIZABLE:
        raise StateTransformError("remote state cannot claim local linearizable semantics")


def compile_transformation(
    transformation: StateTransformation,
    *,
    memory_capacity_bytes: int,
    safety_margin_fraction: float = 0.1,
) -> CompiledStateTransition:
    validate_region(transformation.source)
    validate_region(transformation.target)
    if transformation.source.region_id != transformation.target.region_id:
        raise StateTransformError("state transformation cannot change stable region identity")
    if transformation.source.semantic_contract != transformation.target.semantic_contract:
        raise StateTransformError("state semantic contract changes require a new region")
    if transformation.expected_quality_cost < 0:
        raise StateTransformError("quality cost must be non-negative")
    if transformation.exact and transformation.expected_quality_cost != 0:
        raise StateTransformError("exact state transformations cannot declare quality loss")
    if transformation.migration_chunk_bytes <= 0:
        raise StateTransformError("migration chunk size must be positive")
    if memory_capacity_bytes <= 0 or not 0 <= safety_margin_fraction < 1:
        raise StateTransformError("resource capacity and safety margin are invalid")
    source_bytes = region_capacity_bytes(transformation.source)
    target_bytes = region_capacity_bytes(transformation.target)
    coexistence = source_bytes + target_bytes
    usable_capacity = math.floor(memory_capacity_bytes * (1 - safety_margin_fraction))
    if coexistence > usable_capacity:
        raise StateTransformError(
            f"champion/challenger coexistence requires {coexistence} bytes; "
            f"usable capacity is {usable_capacity}"
        )
    if transformation.target.mutable and transformation.target.ownership not in {
        Ownership.EXCLUSIVE,
        Ownership.REPLICATED_CONSENSUS,
    }:
        raise StateTransformError("mutable target state has ambiguous ownership")
    active_compatible = (
        transformation.source.dtype == transformation.target.dtype
        and transformation.source.layout == transformation.target.layout
        and transformation.source.consistency == transformation.target.consistency
        and transformation.source.ownership == transformation.target.ownership
    )
    return CompiledStateTransition(
        transformation_id=transformation.transformation_id,
        coexistence_peak_bytes=coexistence,
        migration_chunks=math.ceil(target_bytes / transformation.migration_chunk_bytes),
        active_stream_compatible=active_compatible,
        requires_request_boundary=not active_compatible,
        checked_preconditions=(
            "stable_region_identity",
            "semantic_contract_equal",
            "unambiguous_ownership",
            "coexistence_memory_bound",
            "bounded_migration_chunks",
        ),
        proof_obligations=transformation.proof_obligations,
    )
