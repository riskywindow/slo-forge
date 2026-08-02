"""Independent explicit trace checker for state ownership and migration safety."""

from __future__ import annotations

from dataclasses import dataclass

from .compiler import validate_region
from .model import (
    Ownership,
    StateAction,
    StateEvent,
    StateRegion,
    StateTraceResult,
    StateTransformError,
    StateViolation,
)


@dataclass(slots=True)
class _Allocation:
    owner: str
    epoch: int
    item_count: int
    acquired_by: set[tuple[str, str]]
    pending_target: str | None = None
    pending_genome_hash: str | None = None
    checkpoint_epoch: int | None = None
    released: bool = False


def _violation(event: StateEvent, invariant: str, detail: str) -> StateViolation:
    return StateViolation(event.sequence, invariant, detail)


def verify_state_trace(
    region: StateRegion,
    events: tuple[StateEvent, ...],
    *,
    memory_capacity_bytes: int,
    require_quiescent: bool = True,
) -> StateTraceResult:
    validate_region(region)
    if memory_capacity_bytes <= 0:
        raise StateTransformError("memory capacity must be positive")
    allocations: dict[tuple[str, str], _Allocation] = {}
    cancelled: set[str] = set()
    violations: list[StateViolation] = []
    current_bytes = 0
    peak_bytes = 0
    previous_sequence = -1
    cleanup_actions = {
        StateAction.ABORT_MIGRATION,
        StateAction.CANCEL_REQUEST,
        StateAction.RELEASE,
    }
    for event in events:
        if event.sequence <= previous_sequence:
            violations.append(_violation(event, "event_order", "sequence is not increasing"))
        previous_sequence = event.sequence
        if event.region_id != region.region_id:
            violations.append(_violation(event, "region_identity", "event targets another region"))
            continue
        key = (event.request_id, event.region_id)
        allocation = allocations.get(key)
        if event.request_id in cancelled and event.action not in cleanup_actions:
            violations.append(
                _violation(
                    event,
                    "post_cancel_action",
                    "cancelled requests may perform cleanup actions only",
                )
            )
            continue
        if event.action is StateAction.ALLOCATE:
            if allocation is not None and not allocation.released:
                violations.append(_violation(event, "single_allocation", "state already allocated"))
                continue
            if event.actor not in region.owners:
                violations.append(
                    _violation(event, "declared_owner", "actor is not a declared owner")
                )
                continue
            if not 0 < event.item_count <= region.maximum_items:
                violations.append(_violation(event, "item_bound", "allocation exceeds item domain"))
                continue
            allocation = _Allocation(event.actor, event.epoch, event.item_count, set())
            allocations[key] = allocation
            current_bytes += event.item_count * region.bytes_per_item * region.replication_factor
            peak_bytes = max(peak_bytes, current_bytes)
            if current_bytes > memory_capacity_bytes:
                violations.append(_violation(event, "memory_bound", "capacity exceeded"))
            continue
        if allocation is None or allocation.released:
            violations.append(_violation(event, "no_use_after_free", "state is absent or released"))
            continue
        if event.epoch != allocation.epoch and event.action not in {
            StateAction.ROLLBACK,
            StateAction.COMMIT_MIGRATION,
        }:
            violations.append(_violation(event, "epoch_consistency", "stale state epoch"))
            continue
        if event.action is StateAction.ACQUIRE:
            if event.request_id in cancelled:
                violations.append(
                    _violation(event, "cancelled_request", "cancelled request reacquired state")
                )
            elif (region.ownership is Ownership.EXCLUSIVE and event.actor != allocation.owner) or (
                region.ownership is not Ownership.EXCLUSIVE and event.actor not in region.owners
            ):
                violations.append(
                    _violation(event, "lease_owner", "actor is not an authorized state owner")
                )
            else:
                allocation.acquired_by.add((event.request_id, event.actor))
        elif event.action is StateAction.READ:
            if allocation.pending_target is not None:
                violations.append(
                    _violation(event, "no_partial_visibility", "migration is not committed")
                )
            if (event.request_id, event.actor) not in allocation.acquired_by:
                violations.append(_violation(event, "acquire_before_read", "actor has no lease"))
        elif event.action is StateAction.WRITE:
            if (event.request_id, event.actor) not in allocation.acquired_by:
                violations.append(_violation(event, "acquire_before_write", "actor has no lease"))
            if not region.mutable:
                violations.append(_violation(event, "immutability", "write to immutable state"))
            if region.ownership is Ownership.EXCLUSIVE and event.actor != allocation.owner:
                violations.append(_violation(event, "single_writer", "non-owner attempted write"))
            if allocation.pending_target is not None:
                violations.append(
                    _violation(event, "migration_atomicity", "write during migration")
                )
        elif event.action is StateAction.BEGIN_MIGRATION:
            if allocation.pending_target is not None:
                violations.append(
                    _violation(event, "single_migration", "migration already pending")
                )
            elif event.target_actor is None or event.target_genome_hash is None:
                violations.append(
                    _violation(event, "migration_target", "target metadata is incomplete")
                )
            elif event.target_actor not in region.migration_target_owners:
                violations.append(
                    _violation(
                        event,
                        "migration_target_owner",
                        "target actor is not in the migration owner allowlist",
                    )
                )
            elif event.target_genome_hash not in region.compatible_genome_hashes:
                violations.append(
                    _violation(event, "state_compatibility", "target genome is incompatible")
                )
            else:
                allocation.pending_target = event.target_actor
                allocation.pending_genome_hash = event.target_genome_hash
        elif event.action is StateAction.COMMIT_MIGRATION:
            if allocation.pending_target is None:
                violations.append(_violation(event, "migration_commit", "no migration is pending"))
            elif allocation.acquired_by:
                violations.append(
                    _violation(event, "active_stream_migration", "leases remain during commit")
                )
            else:
                allocation.owner = allocation.pending_target
                allocation.pending_target = None
                allocation.pending_genome_hash = None
                allocation.epoch = event.epoch
        elif event.action is StateAction.ABORT_MIGRATION:
            allocation.pending_target = None
            allocation.pending_genome_hash = None
        elif event.action is StateAction.CHECKPOINT:
            if not region.checkpointed:
                violations.append(
                    _violation(event, "checkpoint_contract", "checkpointing disabled")
                )
            else:
                allocation.checkpoint_epoch = allocation.epoch
        elif event.action is StateAction.ROLLBACK:
            if allocation.checkpoint_epoch is None:
                violations.append(_violation(event, "rollback_checkpoint", "no checkpoint exists"))
            else:
                allocation.epoch = allocation.checkpoint_epoch
                allocation.pending_target = None
                allocation.pending_genome_hash = None
        elif event.action is StateAction.CANCEL_REQUEST:
            cancelled.add(event.request_id)
            allocation.acquired_by = {
                lease for lease in allocation.acquired_by if lease[0] != event.request_id
            }
        elif event.action is StateAction.RELEASE:
            if allocation.acquired_by:
                violations.append(_violation(event, "release_leases", "active leases remain"))
            if allocation.pending_target is not None:
                violations.append(
                    _violation(event, "release_migration", "migration remains pending")
                )
            if not allocation.acquired_by and allocation.pending_target is None:
                allocation.released = True
                current_bytes -= (
                    allocation.item_count * region.bytes_per_item * region.replication_factor
                )
    if require_quiescent:
        for (request_id, _), allocation in allocations.items():
            if not allocation.released:
                violations.append(
                    StateViolation(
                        previous_sequence + 1,
                        "bounded_release",
                        f"request {request_id!r} retained state at trace end",
                    )
                )
    return StateTraceResult(not violations, peak_bytes, current_bytes, tuple(violations))
