"""Conservative state transformation compilation and resource analysis."""

from __future__ import annotations

import math

from .model import (
    CompiledStateTransition,
    Consistency,
    Ownership,
    RollbackStrategy,
    StatePrecondition,
    StateRegion,
    StateTransformation,
    StateTransformError,
    StorageTier,
    TransformationKind,
)


def region_capacity_bytes(region: StateRegion) -> int:
    validate_region(region)
    return region.bytes_per_item * region.maximum_items * region.replication_factor


def validate_region(region: StateRegion) -> None:
    if not region.region_id or not region.semantic_contract:
        raise StateTransformError("state region identity and semantic contract are required")
    if not region.dtype:
        raise StateTransformError("state dtype is required")
    if (
        type(region.bytes_per_item) is not int
        or type(region.maximum_items) is not int
        or region.bytes_per_item <= 0
        or region.maximum_items <= 0
    ):
        raise StateTransformError("state capacity dimensions must be positive")
    if type(region.replication_factor) is not int or region.replication_factor <= 0:
        raise StateTransformError("replication factor must be positive")
    if (
        len(set(region.owners)) != len(region.owners)
        or not region.owners
        or any(not owner for owner in region.owners)
    ):
        raise StateTransformError("state owners must be non-empty and unique")
    if len(set(region.migration_target_owners)) != len(region.migration_target_owners) or any(
        not owner for owner in region.migration_target_owners
    ):
        raise StateTransformError("migration target owners must be non-empty and unique")
    if len(set(region.compatible_genome_hashes)) != len(region.compatible_genome_hashes) or any(
        not genome_hash for genome_hash in region.compatible_genome_hashes
    ):
        raise StateTransformError("compatible genome hashes must be non-empty and unique")
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


def _normalize_preconditions(
    transformation: StateTransformation,
) -> tuple[StatePrecondition, ...]:
    if not transformation.preconditions:
        raise StateTransformError("state transformation must declare preconditions")
    try:
        declared = tuple(StatePrecondition(item) for item in transformation.preconditions)
    except ValueError as exc:
        raise StateTransformError("state transformation declares an unknown precondition") from exc
    if len(declared) != len(set(declared)):
        raise StateTransformError("state transformation preconditions must be unique")
    return declared


def _validate_kind(transformation: StateTransformation) -> TransformationKind:
    try:
        kind = TransformationKind(transformation.kind)
    except ValueError as exc:
        raise StateTransformError("unsupported state transformation kind") from exc
    source = transformation.source
    target = transformation.target
    if source.dtype != target.dtype and kind is not TransformationKind.PRECISION:
        raise StateTransformError("dtype changes require a precision transformation kind")
    meaningful = {
        TransformationKind.LAYOUT: source.layout != target.layout,
        TransformationKind.PRECISION: (
            source.dtype != target.dtype or source.bytes_per_item != target.bytes_per_item
        ),
        TransformationKind.OFFLOAD: source.storage != target.storage,
        TransformationKind.REPLICATION: (
            source.ownership != target.ownership
            or source.owners != target.owners
            or source.replication_factor != target.replication_factor
            or source.consistency != target.consistency
        ),
        TransformationKind.MIGRATION: (
            source.owners != target.owners or source.storage != target.storage
        ),
        TransformationKind.PREFETCH: source.storage != target.storage,
        TransformationKind.EVICTION: (
            target.maximum_items < source.maximum_items or source.storage != target.storage
        ),
        TransformationKind.CHECKPOINT: (not source.checkpointed and target.checkpointed),
        TransformationKind.RECOMPUTE: (
            region_capacity_bytes(target) < region_capacity_bytes(source)
        ),
    }
    if not meaningful[kind]:
        raise StateTransformError(
            f"state transformation kind {kind.value!r} has no matching change"
        )
    return kind


def compile_transformation(
    transformation: StateTransformation,
    *,
    memory_capacity_bytes: int,
    safety_margin_fraction: float = 0.1,
    quality_budget: float = 0.0,
) -> CompiledStateTransition:
    validate_region(transformation.source)
    validate_region(transformation.target)
    if transformation.source.region_id != transformation.target.region_id:
        raise StateTransformError("state transformation cannot change stable region identity")
    if transformation.source.semantic_contract != transformation.target.semantic_contract:
        raise StateTransformError("state semantic contract changes require a new region")
    if not transformation.transformation_id:
        raise StateTransformError("state transformation identity is required")
    kind = _validate_kind(transformation)
    declared_preconditions = _normalize_preconditions(transformation)
    if (
        not math.isfinite(transformation.expected_quality_cost)
        or transformation.expected_quality_cost < 0
    ):
        raise StateTransformError("quality cost must be non-negative")
    if not math.isfinite(quality_budget) or quality_budget < 0:
        raise StateTransformError("quality budget must be finite and non-negative")
    if transformation.exact and transformation.expected_quality_cost != 0:
        raise StateTransformError("exact state transformations cannot declare quality loss")
    if transformation.expected_quality_cost > quality_budget:
        raise StateTransformError("state transformation exceeds the quality budget")
    if (
        not transformation.exact
        and StatePrecondition.QUALITY_CONTRACT not in declared_preconditions
    ):
        raise StateTransformError("approximate state transformations require a quality contract")
    if transformation.migration_chunk_bytes <= 0:
        raise StateTransformError("migration chunk size must be positive")
    if memory_capacity_bytes <= 0 or not 0 <= safety_margin_fraction < 1:
        raise StateTransformError("resource capacity and safety margin are invalid")
    source_bytes = region_capacity_bytes(transformation.source)
    target_bytes = region_capacity_bytes(transformation.target)
    expected_delta = target_bytes - source_bytes
    if (
        type(transformation.expected_memory_delta_bytes) is not int
        or transformation.expected_memory_delta_bytes != expected_delta
    ):
        raise StateTransformError(
            "declared memory delta does not match conservative region capacity analysis"
        )
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
        and transformation.source.storage == transformation.target.storage
        and transformation.source.consistency == transformation.target.consistency
        and transformation.source.ownership == transformation.target.ownership
        and transformation.source.owners == transformation.target.owners
        and transformation.source.bytes_per_item == transformation.target.bytes_per_item
        and transformation.source.replication_factor == transformation.target.replication_factor
    )
    if not active_compatible and StatePrecondition.REQUEST_BOUNDARY not in declared_preconditions:
        raise StateTransformError(
            "incompatible state changes require a request-boundary precondition"
        )
    ownership_changed = (
        transformation.source.ownership != transformation.target.ownership
        or transformation.source.owners != transformation.target.owners
        or transformation.source.replication_factor != transformation.target.replication_factor
    )
    if ownership_changed and not {
        StatePrecondition.QUIESCENT_STATE,
        StatePrecondition.OWNER_ALLOWLIST,
    }.issubset(declared_preconditions):
        raise StateTransformError(
            "ownership changes require quiescent-state and owner-allowlist preconditions"
        )
    conversion_required = (
        transformation.source.dtype != transformation.target.dtype
        or transformation.source.bytes_per_item != transformation.target.bytes_per_item
    )
    if conversion_required:
        if StatePrecondition.STATE_CONVERSION_VERIFIED not in declared_preconditions:
            raise StateTransformError("representation changes require conversion verification")
        if (
            not transformation.conversion_evidence
            or len(set(transformation.conversion_evidence))
            != len(transformation.conversion_evidence)
            or any(not evidence for evidence in transformation.conversion_evidence)
        ):
            raise StateTransformError("representation changes require unique conversion evidence")
    elif transformation.conversion_evidence:
        raise StateTransformError("conversion evidence is invalid without a representation change")
    if not transformation.proof_obligations or any(
        not obligation for obligation in transformation.proof_obligations
    ):
        raise StateTransformError("state transformation proof obligations are required")
    if len(set(transformation.proof_obligations)) != len(transformation.proof_obligations):
        raise StateTransformError("state transformation proof obligations must be unique")
    try:
        rollback = RollbackStrategy(transformation.rollback_strategy)
    except ValueError as exc:
        raise StateTransformError("unsupported state rollback strategy") from exc
    if rollback is RollbackStrategy.RESTORE_CHECKPOINT and not transformation.source.checkpointed:
        raise StateTransformError("checkpoint rollback requires a source checkpoint contract")
    if rollback is RollbackStrategy.REVERSE_CONVERSION and not conversion_required:
        raise StateTransformError("reverse-conversion rollback requires a representation change")
    checked = [
        "stable_region_identity",
        f"kind:{kind.value}",
        "semantic_contract_equal",
        "declared_preconditions_validated",
        "unambiguous_ownership",
        "resource_delta_equal",
        "coexistence_memory_bound",
        "bounded_migration_chunks",
        f"rollback:{rollback.value}",
        "quality_budget",
    ]
    if conversion_required:
        checked.append("conversion_evidence")
    return CompiledStateTransition(
        transformation_id=transformation.transformation_id,
        coexistence_peak_bytes=coexistence,
        migration_chunks=math.ceil(target_bytes / transformation.migration_chunk_bytes),
        active_stream_compatible=active_compatible,
        requires_request_boundary=not active_compatible,
        checked_preconditions=tuple(checked),
        proof_obligations=transformation.proof_obligations,
    )
