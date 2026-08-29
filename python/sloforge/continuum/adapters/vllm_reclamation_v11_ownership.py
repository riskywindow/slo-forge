"""Post-free ownership proof for BranchFabric Experiment 004 v11.

This module is intentionally separate from the frozen v10 reclamation worker.
It binds the v11 release proof to :class:`VllmLiveStateAdapter`, the runtime's
allocator-issued allocation epochs, and the production engine-step critical
section.  A scoped layout disappearance or an HBM counter change is never
accepted as release proof on its own.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from sloforge.continuum.adapters.real_runtime import GpuMemoryState
from sloforge.continuum.adapters.vllm_allocator_epochs import (
    VllmAllocatorEpochError,
    VllmAllocatorEpochProof,
    VllmAllocatorIssuedEpoch,
    require_vllm_allocator_epoch_source,
)
from sloforge.continuum.adapters.vllm_live import (
    VLLM_LIVE_RUNTIME_VERSION,
    VllmLiveStateAdapter,
)
from sloforge.continuum.adapters.vllm_reclamation_v11_sync import (
    V11AdmissionGateWitness,
    require_production_gate_v11,
)

if TYPE_CHECKING:
    from sloforge.continuum.adapters.vllm_reclamation import CanonicalCapturePlan

V11_OWNERSHIP_PROOF_SCHEMA_VERSION = "sloforge.continuum.vllm-v11-post-free-ownership-proof/v1"
V11_SOURCE_CAPTURE_COMMIT_SCHEMA_VERSION = "sloforge.continuum.vllm-v11-source-capture-commit/v1"
V11_ALLOCATOR_QUIESCENCE_SCHEMA_VERSION = "sloforge.continuum.vllm-v11-allocator-quiescence/v1"
_V11_SOURCE_OWNERSHIP_SNAPSHOT_SEAL = object()
_V11_SOURCE_CAPTURE_COMMIT_SEAL = object()
_V11_POST_FREE_OWNERSHIP_PROOF_SEAL = object()


class V11OwnershipProofError(RuntimeError):
    """The exact source allocation set was not proven released."""


@dataclass(frozen=True, slots=True)
class V11PreReleaseBlockOwnership:
    runtime_block_id: str
    block_id: int
    allocation_epoch: int
    pre_release_owner: tuple[str, ...]
    pre_release_runtime_owner: tuple[str, ...]
    pre_release_refcount: int
    physical_bytes: int


@dataclass(frozen=True, slots=True)
class V11SourceOwnershipSnapshot:
    schema_version: str
    runtime: str
    runtime_version: str
    device: str
    root_session_id: str
    root_reference_id: str
    branch_session_ids: tuple[str, ...]
    runtime_request_ids: tuple[tuple[str, str], ...]
    blocks: tuple[V11PreReleaseBlockOwnership, ...]
    captured_at_monotonic_ns: int
    kv_pool_reserved_bytes: int
    kv_assigned_bytes: int
    kv_unassigned_bytes: int
    source_physical_bytes: int
    _adapter_object_id: int = field(repr=False, compare=False)
    _allocator_epoch_proof: VllmAllocatorEpochProof = field(repr=False, compare=False)
    _proof_seal: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class V11AllocatorQuiescenceEvidence:
    schema_version: str
    observed_at_monotonic_ns: int
    branch_session_ids: tuple[str, ...]
    runtime_request_ids: tuple[str, ...]
    paused_branch_count: int
    scheduler_request_ids: tuple[str, ...]
    native_request_table_ids: tuple[str, ...]
    scheduler_running_request_ids: tuple[str, ...]
    scheduler_waiting_count: int
    scheduler_skipped_waiting_count: int
    asynchronous_scheduling: bool
    gate_binding_id: str
    gate_acquisition_generation: int
    engine_step_excluded: bool
    allocator_mutations_excluded: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class V11SourceSemanticAllocation:
    logical_page_id: str
    block_table_slot: int
    logical_token_start: int
    logical_token_end: int
    valid_tokens: int
    physical_block_id: int
    allocation_epoch: int
    logical_owner_set: tuple[str, ...]
    runtime_owner_set: tuple[str, ...]
    expected_refcount: int
    shared: bool
    physical_bytes: int


@dataclass(frozen=True, slots=True)
class V11SourceCaptureBranch:
    logical_branch_id: str
    parent_logical_branch_id: str
    runtime_request_id: str
    computed_tokens: int
    token_history_sha256: str
    logical_page_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceCaptureCommit:
    """Sealed snapshot-time semantic identity for one source branch group.

    The semantic digest deliberately excludes timestamps, allocator notification
    history, inspection counters, and diagnostic generations.  Those values are
    useful provenance, but they are not a live allocation identity.
    """

    schema_version: str
    committed_at_monotonic_ns: int
    semantic_sha256: str
    runtime_instance_id: str
    runtime_model_identity: tuple[tuple[str, str], ...]
    runtime_model_identity_sha256: str
    device: str
    root_session_id: str
    branch_group: tuple[str, ...]
    branches: tuple[V11SourceCaptureBranch, ...]
    allocations: tuple[V11SourceSemanticAllocation, ...]
    logical_state_bytes: int
    physical_source_bytes: int
    gate_binding_id: str
    gate_acquisition_generation: int
    allocator_quiescence: V11AllocatorQuiescenceEvidence
    _adapter_object_id: int = field(repr=False, compare=False)
    _ownership_snapshot: V11SourceOwnershipSnapshot = field(repr=False, compare=False)
    _proof_seal: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class V11SourceIdentityValidation:
    schema_version: str
    validated_at_monotonic_ns: int
    semantic_sha256: str
    allocation_count: int
    same_engine_step_witness: bool
    exact_logical_mapping: bool
    exact_block_epoch_identity: bool
    exact_owner_sets: bool
    exact_refcounts: bool
    all_allocations_live: bool
    no_post_commit_mutation: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class V11BlockOwnershipTransition:
    runtime_block_id: str
    block_id: int
    allocation_epoch: int
    pre_release_owner: tuple[str, ...]
    pre_release_runtime_owner: tuple[str, ...]
    pre_release_refcount: int
    release_operation: str
    post_release_owner: tuple[str, ...]
    post_release_refcount: int
    allocator_available: bool
    block_hash_present: bool
    no_live_branch_reference: bool
    old_session_inaccessible: bool
    allocator_epoch_tombstoned: bool
    native_request_tables_absent: bool
    physical_bytes: int

    @property
    def passed(self) -> bool:
        return (
            bool(self.pre_release_owner)
            and self.pre_release_refcount == len(self.pre_release_owner)
            and bool(self.release_operation)
            and not self.post_release_owner
            and self.post_release_refcount == 0
            and self.allocator_available
            and not self.block_hash_present
            and self.no_live_branch_reference
            and self.old_session_inaccessible
            and self.allocator_epoch_tombstoned
            and self.native_request_tables_absent
        )


@dataclass(frozen=True, slots=True)
class V11PostFreeOwnershipProof:
    schema_version: str
    runtime: str
    runtime_version: str
    device: str
    root_session_id: str
    root_reference_id: str
    branch_session_ids: tuple[str, ...]
    release_started_at_monotonic_ns: int
    release_completed_at_monotonic_ns: int
    records: tuple[V11BlockOwnershipTransition, ...]
    source_block_count: int
    source_physical_bytes: int
    pool_free_block_count: int
    pool_usable_block_count: int
    kv_pool_reserved_bytes_before: int
    kv_pool_reserved_bytes_after: int
    kv_assigned_bytes_before: int
    kv_assigned_bytes_after: int
    kv_unassigned_bytes_before: int
    kv_unassigned_bytes_after: int
    exact_source_coverage: bool
    all_old_sessions_destroyed: bool
    root_reference_released: bool
    old_session_layout_inaccessible: bool
    no_live_branch_reference: bool
    native_manager_request_tables_absent: bool
    all_allocator_epochs_tombstoned: bool
    allocator_hbm_agreement: bool
    full_free_pool_recovered: bool
    passed: bool
    _proof_seal: object = field(repr=False, compare=False)

    def require_passed(self) -> None:
        if self._proof_seal is not _V11_POST_FREE_OWNERSHIP_PROOF_SEAL or not self.passed:
            raise V11OwnershipProofError("v11 post-free ownership proof did not pass")


def _remaining_seconds(deadline_ns: int) -> float:
    remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
    if remaining <= 0:
        raise TimeoutError("v11 ownership proof exceeded its bounded timeout")
    return remaining


def _require_production_adapter(adapter: VllmLiveStateAdapter) -> None:
    # Exact type deliberately rejects proxy/test adapter substitution at the
    # production integration boundary. Tests exercise this type with a bounded
    # structural vLLM fixture, not a replacement adapter implementation.
    if type(adapter) is not VllmLiveStateAdapter:
        raise TypeError("v11 ownership proof requires the production VllmLiveStateAdapter")
    view = getattr(adapter, "_view", None)
    if view is None or getattr(view, "runtime_version", None) != VLLM_LIVE_RUNTIME_VERSION:
        raise V11OwnershipProofError("v11 ownership proof lacks the validated vLLM 0.23 view")


def _parse_runtime_block_id(runtime_block_id: str) -> int:
    parts = runtime_block_id.split(":")
    if (
        len(parts) != 5
        or parts[:2] != ["vllm", "kv-group"]
        or parts[2] != "0"
        or parts[3] != "block"
    ):
        raise V11OwnershipProofError("v11 ownership proof found an invalid runtime block ID")
    try:
        block_id = int(parts[4])
    except ValueError as error:
        raise V11OwnershipProofError("v11 ownership proof found an invalid block index") from error
    if block_id < 0:
        raise V11OwnershipProofError("v11 ownership proof found a negative block index")
    return block_id


def _runtime_request_ids(
    adapter: VllmLiveStateAdapter,
    branch_session_ids: tuple[str, ...],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for session_id in branch_session_ids:
        session = adapter._sessions.get(session_id)
        runtime_id = None if session is None else session.runtime_request_id
        if (
            session is None
            or session.phase.value == "destroyed"
            or not isinstance(runtime_id, str)
            or not runtime_id
            or adapter._view.request_object(runtime_id) is None
        ):
            raise V11OwnershipProofError(
                f"source branch {session_id!r} lacks a live runtime request table"
            )
        result[session_id] = runtime_id
    if len(set(result.values())) != len(result):
        raise V11OwnershipProofError("source branches alias one runtime request identity")
    return result


def _live_referrers_by_block(adapter: VllmLiveStateAdapter) -> dict[int, set[str]]:
    scheduler_requests = getattr(adapter._view.scheduler, "requests", None)
    if not isinstance(scheduler_requests, dict):
        raise V11OwnershipProofError("vLLM scheduler request table is unavailable")
    referrers: dict[int, set[str]] = {}
    for runtime_request_id in tuple(scheduler_requests):
        try:
            groups = adapter._view.request_blocks(runtime_request_id)
        except (KeyError, RuntimeError) as error:
            raise V11OwnershipProofError(
                "a live scheduler request lacks an allocator block table"
            ) from error
        for group in groups:
            for block in group:
                if block.is_null:
                    continue
                referrers.setdefault(int(block.block_id), set()).add(runtime_request_id)
    return referrers


def _native_request_table_keys(adapter: VllmLiveStateAdapter) -> frozenset[str]:
    """Read the exact single-group vLLM request-to-block ownership map.

    Scheduler absence is insufficient: ``KVCacheManager.free`` must also have
    removed every old request from ``SingleTypeKVCacheManager.req_to_blocks``.
    """

    manager = adapter._view.manager
    coordinator = getattr(manager, "coordinator", None)
    single_type_managers = getattr(coordinator, "single_type_managers", None)
    if not isinstance(single_type_managers, (list, tuple)) or len(single_type_managers) != 1:
        raise V11OwnershipProofError(
            "v11 ownership proof requires exactly one native KV cache manager"
        )
    request_tables = getattr(single_type_managers[0], "req_to_blocks", None)
    if not isinstance(request_tables, dict):
        raise V11OwnershipProofError(
            "v11 ownership proof cannot read the native request-to-block table"
        )
    if any(not isinstance(request_id, str) or not request_id for request_id in request_tables):
        raise V11OwnershipProofError("native request-to-block table has an invalid request ID")
    return frozenset(request_tables)


def _memory_tuple(memory: GpuMemoryState) -> tuple[int, int, int]:
    return (
        int(memory.kv_pool_reserved_bytes),
        int(memory.kv_assigned_bytes),
        int(memory.kv_unassigned_bytes),
    )


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def expected_owner_set(
    plan: CanonicalCapturePlan,
    logical_page_id: str,
) -> tuple[str, ...]:
    """Return the declared logical owners for one canonical source page."""

    page_rows = tuple(item for item in plan.page_order if item[0] == logical_page_id)
    if len(page_rows) != 1:
        raise V11OwnershipProofError(
            f"canonical source plan does not define logical page {logical_page_id!r} exactly once"
        )
    owners = tuple(
        sorted(
            str(branch.logical_branch_id)
            for branch in plan.branch_tables
            if logical_page_id in branch.logical_page_ids
        )
    )
    if not owners or len(owners) != len(set(owners)):
        raise V11OwnershipProofError("canonical source page has an invalid expected owner set")
    return owners


def expected_refcount(plan: CanonicalCapturePlan, logical_page_id: str) -> int:
    """Derive native refcount from the canonical owner relationship."""

    return len(expected_owner_set(plan, logical_page_id))


def _queue_size(queue: Any, *, label: str) -> int:
    try:
        size = len(queue)
    except TypeError:
        for name in ("rows", "requests", "request_ids"):
            rows = getattr(queue, name, None)
            if isinstance(rows, (list, tuple, dict, set, frozenset)):
                size = len(rows)
                break
        else:
            raise V11OwnershipProofError(f"cannot inspect scheduler {label} queue")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise V11OwnershipProofError(f"scheduler {label} queue size is invalid")
    return size


def _request_ids(rows: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(rows, (list, tuple)):
        raise V11OwnershipProofError(f"scheduler {label} request list is unavailable")
    result: list[str] = []
    for request in rows:
        request_id = getattr(request, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            raise V11OwnershipProofError(f"scheduler {label} contains an invalid request")
        result.append(request_id)
    if len(result) != len(set(result)):
        raise V11OwnershipProofError(f"scheduler {label} aliases a runtime request")
    return tuple(sorted(result))


def prove_allocator_quiescence_v11(
    adapter: VllmLiveStateAdapter,
    *,
    branch_session_ids: tuple[str, ...],
    admission_gate: V11AdmissionGateWitness,
) -> V11AllocatorQuiescenceEvidence:
    """Prove the source allocator is stable while its live tables stay installed.

    Queue-empty alone is deliberately insufficient.  The live source requests
    remain in ``scheduler.running`` and in the native request tables; quiescence
    means they are paused at a committed token boundary while the production
    engine-step gate excludes every source-affecting mutation surface.
    """

    _require_production_adapter(adapter)
    view = adapter._view
    require_production_gate_v11(
        admission_gate,
        view.scheduler,
        view.manager,
        operation="EXPORT_CAPTURE",
    )
    runtime_ids_by_session = _runtime_request_ids(adapter, branch_session_ids)
    expected_runtime_ids = tuple(sorted(runtime_ids_by_session.values()))
    not_paused = tuple(
        session_id
        for session_id in branch_session_ids
        if adapter._sessions[session_id].phase.value != "paused"
    )
    if not_paused:
        raise V11OwnershipProofError(
            "allocator quiescence requires every source branch to be paused: "
            + ", ".join(not_paused)
        )
    scheduler_requests = getattr(view.scheduler, "requests", None)
    if not isinstance(scheduler_requests, dict):
        raise V11OwnershipProofError("allocator quiescence cannot inspect scheduler requests")
    scheduler_request_ids = tuple(sorted(scheduler_requests))
    native_request_ids = tuple(sorted(_native_request_table_keys(adapter)))
    running_request_ids = _request_ids(getattr(view.scheduler, "running", None), label="running")
    waiting_count = _queue_size(getattr(view.scheduler, "waiting", None), label="waiting")
    skipped_count = _queue_size(
        getattr(view.scheduler, "skipped_waiting", None), label="skipped-waiting"
    )
    runtime_config = getattr(view.vllm_config, "scheduler_config", None)
    internal_config = getattr(view.scheduler, "scheduler_config", runtime_config)
    asynchronous = bool(getattr(runtime_config, "async_scheduling", False)) or bool(
        getattr(internal_config, "async_scheduling", False)
    )
    passed = bool(
        not asynchronous
        and waiting_count == 0
        and skipped_count == 0
        and scheduler_request_ids == expected_runtime_ids
        and native_request_ids == expected_runtime_ids
        and running_request_ids == expected_runtime_ids
    )
    if not passed:
        raise V11OwnershipProofError(
            "allocator quiescence requires exact live source request tables and empty pending queues"
        )
    require_production_gate_v11(
        admission_gate,
        view.scheduler,
        view.manager,
        operation="EXPORT_CAPTURE",
    )
    return V11AllocatorQuiescenceEvidence(
        schema_version=V11_ALLOCATOR_QUIESCENCE_SCHEMA_VERSION,
        observed_at_monotonic_ns=time.monotonic_ns(),
        branch_session_ids=tuple(sorted(branch_session_ids)),
        runtime_request_ids=expected_runtime_ids,
        paused_branch_count=len(branch_session_ids),
        scheduler_request_ids=scheduler_request_ids,
        native_request_table_ids=native_request_ids,
        scheduler_running_request_ids=running_request_ids,
        scheduler_waiting_count=waiting_count,
        scheduler_skipped_waiting_count=skipped_count,
        asynchronous_scheduling=asynchronous,
        gate_binding_id=admission_gate.binding_id,
        gate_acquisition_generation=admission_gate.acquisition_generation,
        engine_step_excluded=True,
        allocator_mutations_excluded=True,
        passed=True,
    )


def _source_commit_semantic_payload(
    *,
    runtime_instance_id: str,
    runtime_model_identity: tuple[tuple[str, str], ...],
    device: str,
    root_session_id: str,
    branch_group: tuple[str, ...],
    branches: tuple[V11SourceCaptureBranch, ...],
    allocations: tuple[V11SourceSemanticAllocation, ...],
    logical_state_bytes: int,
    physical_source_bytes: int,
    gate_binding_id: str,
    gate_acquisition_generation: int,
) -> dict[str, Any]:
    return {
        "schema_version": V11_SOURCE_CAPTURE_COMMIT_SCHEMA_VERSION,
        "runtime_instance_id": runtime_instance_id,
        "runtime_model_identity": runtime_model_identity,
        "device": device,
        "root_session_id": root_session_id,
        "branch_group": branch_group,
        "branches": tuple(
            (
                branch.logical_branch_id,
                branch.parent_logical_branch_id,
                branch.runtime_request_id,
                branch.computed_tokens,
                branch.token_history_sha256,
                branch.logical_page_ids,
            )
            for branch in branches
        ),
        "allocations": tuple(
            (
                allocation.logical_page_id,
                allocation.block_table_slot,
                allocation.logical_token_start,
                allocation.logical_token_end,
                allocation.valid_tokens,
                allocation.physical_block_id,
                allocation.allocation_epoch,
                allocation.logical_owner_set,
                allocation.runtime_owner_set,
                allocation.expected_refcount,
                allocation.shared,
                allocation.physical_bytes,
            )
            for allocation in allocations
        ),
        "logical_state_bytes": logical_state_bytes,
        "physical_source_bytes": physical_source_bytes,
        "gate_binding_id": gate_binding_id,
        "gate_acquisition_generation": gate_acquisition_generation,
    }


def create_source_capture_commit_v11(
    adapter: VllmLiveStateAdapter,
    plan: CanonicalCapturePlan,
    ownership_snapshot: V11SourceOwnershipSnapshot,
    *,
    block_size_tokens: int,
    admission_gate: V11AdmissionGateWitness,
) -> SourceCaptureCommit:
    """Seal the current quiesced logical-to-physical source allocation map."""

    _require_production_adapter(adapter)
    if block_size_tokens <= 0:
        raise ValueError("source capture commit block size must be positive")
    if (
        not isinstance(ownership_snapshot, V11SourceOwnershipSnapshot)
        or ownership_snapshot._proof_seal is not _V11_SOURCE_OWNERSHIP_SNAPSHOT_SEAL
        or ownership_snapshot._adapter_object_id != id(adapter)
    ):
        raise V11OwnershipProofError("source capture commit received a forged ownership snapshot")
    view = adapter._view
    require_production_gate_v11(
        admission_gate,
        view.scheduler,
        view.manager,
        operation="EXPORT_CAPTURE",
    )
    quiescence = prove_allocator_quiescence_v11(
        adapter,
        branch_session_ids=ownership_snapshot.branch_session_ids,
        admission_gate=admission_gate,
    )
    branch_group = tuple(sorted(branch.logical_branch_id for branch in plan.branch_tables))
    if branch_group != tuple(sorted(ownership_snapshot.branch_session_ids)):
        raise V11OwnershipProofError("capture plan branch group differs from live source ownership")
    runtime_id_by_branch = dict(ownership_snapshot.runtime_request_ids)
    ownership_by_block = {record.block_id: record for record in ownership_snapshot.blocks}
    binding_by_page = {
        binding.logical_page_id: binding for binding in plan.capture_evidence.bindings
    }
    slots_by_page: dict[str, set[int]] = {}
    for branch in plan.branch_tables:
        for slot, logical_page_id in enumerate(branch.logical_page_ids):
            slots_by_page.setdefault(logical_page_id, set()).add(slot)
    allocations: list[V11SourceSemanticAllocation] = []
    for logical_page_id, block_id, valid_tokens, _owners in plan.page_order:
        owners = expected_owner_set(plan, logical_page_id)
        slots = slots_by_page.get(logical_page_id, set())
        if len(slots) != 1:
            raise V11OwnershipProofError(
                f"logical page {logical_page_id!r} does not occupy one canonical block-table slot"
            )
        slot = next(iter(slots))
        binding = binding_by_page.get(logical_page_id)
        before = ownership_by_block.get(int(block_id))
        if binding is None or before is None:
            raise V11OwnershipProofError("source commit lacks allocation or ownership evidence")
        runtime_owners = tuple(sorted(runtime_id_by_branch[owner] for owner in owners))
        refcount = expected_refcount(plan, logical_page_id)
        if (
            int(binding.source.block_index) != int(block_id)
            or int(binding.source.allocation_epoch) != before.allocation_epoch
            or before.pre_release_owner != owners
            or before.pre_release_runtime_owner != runtime_owners
            or before.pre_release_refcount != refcount
        ):
            raise V11OwnershipProofError(
                f"semantic owner/refcount/block/epoch mismatch for {logical_page_id!r}"
            )
        token_start = slot * block_size_tokens
        allocations.append(
            V11SourceSemanticAllocation(
                logical_page_id=logical_page_id,
                block_table_slot=slot,
                logical_token_start=token_start,
                logical_token_end=token_start + int(valid_tokens),
                valid_tokens=int(valid_tokens),
                physical_block_id=int(block_id),
                allocation_epoch=before.allocation_epoch,
                logical_owner_set=owners,
                runtime_owner_set=runtime_owners,
                expected_refcount=refcount,
                shared=len(owners) > 1,
                physical_bytes=before.physical_bytes,
            )
        )
    ordered_allocations = tuple(sorted(allocations, key=lambda item: item.logical_page_id))
    if len(ordered_allocations) != len(ownership_snapshot.blocks):
        raise V11OwnershipProofError("source commit does not cover every live source allocation")
    branches = tuple(
        sorted(
            (
                V11SourceCaptureBranch(
                    logical_branch_id=branch.logical_branch_id,
                    parent_logical_branch_id=branch.parent_logical_branch_id,
                    runtime_request_id=runtime_id_by_branch[branch.logical_branch_id],
                    computed_tokens=int(branch.computed_tokens),
                    token_history_sha256=str(branch.token_history_sha256),
                    logical_page_ids=tuple(branch.logical_page_ids),
                )
                for branch in plan.branch_tables
            ),
            key=lambda item: item.logical_branch_id,
        )
    )
    identity_payload = adapter._identity.model_dump(mode="json")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in identity_payload.items()
    ):
        raise V11OwnershipProofError("runtime model identity is not canonical string data")
    runtime_model_identity = tuple(sorted(identity_payload.items()))
    identity_sha = hashlib.sha256(_canonical_bytes(runtime_model_identity)).hexdigest()
    runtime_instance_id = (
        f"adapter:{id(adapter)}:scheduler:{id(view.scheduler)}:manager:{id(view.manager)}"
    )
    semantic_payload = _source_commit_semantic_payload(
        runtime_instance_id=runtime_instance_id,
        runtime_model_identity=runtime_model_identity,
        device=ownership_snapshot.device,
        root_session_id=ownership_snapshot.root_session_id,
        branch_group=branch_group,
        branches=branches,
        allocations=ordered_allocations,
        logical_state_bytes=int(plan.logical_state_bytes),
        physical_source_bytes=int(plan.physical_source_bytes),
        gate_binding_id=admission_gate.binding_id,
        gate_acquisition_generation=admission_gate.acquisition_generation,
    )
    semantic_sha = hashlib.sha256(_canonical_bytes(semantic_payload)).hexdigest()
    require_production_gate_v11(
        admission_gate,
        view.scheduler,
        view.manager,
        operation="EXPORT_CAPTURE",
    )
    return SourceCaptureCommit(
        schema_version=V11_SOURCE_CAPTURE_COMMIT_SCHEMA_VERSION,
        committed_at_monotonic_ns=time.monotonic_ns(),
        semantic_sha256=semantic_sha,
        runtime_instance_id=runtime_instance_id,
        runtime_model_identity=runtime_model_identity,
        runtime_model_identity_sha256=identity_sha,
        device=ownership_snapshot.device,
        root_session_id=ownership_snapshot.root_session_id,
        branch_group=branch_group,
        branches=branches,
        allocations=ordered_allocations,
        logical_state_bytes=int(plan.logical_state_bytes),
        physical_source_bytes=int(plan.physical_source_bytes),
        gate_binding_id=admission_gate.binding_id,
        gate_acquisition_generation=admission_gate.acquisition_generation,
        allocator_quiescence=quiescence,
        _adapter_object_id=id(adapter),
        _ownership_snapshot=ownership_snapshot,
        _proof_seal=_V11_SOURCE_CAPTURE_COMMIT_SEAL,
    )


def validate_source_capture_commit_v11(
    adapter: VllmLiveStateAdapter,
    commit: SourceCaptureCommit,
    current_plan: CanonicalCapturePlan,
    *,
    block_size_tokens: int,
    admission_gate: V11AdmissionGateWitness,
    timeout_s: float,
) -> V11SourceIdentityValidation:
    """Re-prove a commit and reject any post-commit semantic mutation."""

    _require_production_adapter(adapter)
    if (
        not isinstance(commit, SourceCaptureCommit)
        or commit._proof_seal is not _V11_SOURCE_CAPTURE_COMMIT_SEAL
        or commit._adapter_object_id != id(adapter)
    ):
        raise V11OwnershipProofError("source capture commit is forged or foreign")
    if (
        admission_gate.binding_id != commit.gate_binding_id
        or admission_gate.acquisition_generation != commit.gate_acquisition_generation
    ):
        raise V11OwnershipProofError("source capture commit witness is stale or foreign")
    expected_epochs = {
        allocation.physical_block_id: allocation.allocation_epoch
        for allocation in commit.allocations
    }
    current_ownership = capture_source_ownership_v11(
        adapter,
        root_session_id=commit.root_session_id,
        branch_session_ids=commit.branch_group,
        expected_allocation_epochs=expected_epochs,
        expected_source_block_count=len(commit.allocations),
        admission_gate=admission_gate,
        timeout_s=timeout_s,
    )
    candidate = create_source_capture_commit_v11(
        adapter,
        current_plan,
        current_ownership,
        block_size_tokens=block_size_tokens,
        admission_gate=admission_gate,
    )
    if candidate.semantic_sha256 != commit.semantic_sha256:
        raise V11OwnershipProofError("source allocation identity changed after SourceCaptureCommit")
    return V11SourceIdentityValidation(
        schema_version="sloforge.continuum.vllm-v11-source-identity-validation/v1",
        validated_at_monotonic_ns=time.monotonic_ns(),
        semantic_sha256=commit.semantic_sha256,
        allocation_count=len(commit.allocations),
        same_engine_step_witness=True,
        exact_logical_mapping=True,
        exact_block_epoch_identity=True,
        exact_owner_sets=True,
        exact_refcounts=True,
        all_allocations_live=True,
        no_post_commit_mutation=True,
        passed=True,
    )


def capture_source_ownership_v11(
    adapter: VllmLiveStateAdapter,
    *,
    root_session_id: str,
    branch_session_ids: tuple[str, ...],
    expected_allocation_epochs: dict[int, int],
    expected_source_block_count: int,
    admission_gate: V11AdmissionGateWitness,
    timeout_s: float,
) -> V11SourceOwnershipSnapshot:
    """Capture exact pre-release owners/refcounts under the production gate."""

    _require_production_adapter(adapter)
    if (
        not root_session_id
        or not branch_session_ids
        or len(branch_session_ids) != len(set(branch_session_ids))
    ):
        raise ValueError("v11 source ownership requires unique branch sessions and one root")
    if root_session_id in branch_session_ids:
        raise ValueError("v11 source root cannot also be a branch session")
    if (
        expected_source_block_count <= 0
        or len(expected_allocation_epochs) != expected_source_block_count
    ):
        raise ValueError("v11 source allocation expectation must be exact and non-empty")
    if not 0 < timeout_s <= 300:
        raise ValueError("v11 ownership timeout must be in (0, 300] seconds")
    deadline = time.monotonic_ns() + int(timeout_s * 1_000_000_000)
    view = adapter._view
    require_production_gate_v11(
        admission_gate,
        view.scheduler,
        view.manager,
        operation="EXPORT_CAPTURE",
    )
    root = adapter._sessions.get(root_session_id)
    if root is None or root.phase.value == "destroyed":
        raise V11OwnershipProofError("v11 source root session is absent or destroyed")
    runtime_ids = _runtime_request_ids(adapter, branch_session_ids)
    native_request_keys = _native_request_table_keys(adapter)
    if not set(runtime_ids.values()).issubset(native_request_keys):
        raise V11OwnershipProofError(
            "source runtime ownership is absent from the native request-to-block table"
        )
    branch_sessions = [adapter._sessions[session_id] for session_id in branch_session_ids]
    root_reference_ids = {session.root_reference_id for session in branch_sessions}
    if None in root_reference_ids or len(root_reference_ids) != 1:
        raise V11OwnershipProofError("source branches do not share one live root reference")
    root_reference_id = cast(str, root_reference_ids.pop())
    root_reference = adapter._roots.get(root_reference_id)
    if (
        root_reference is None
        or root_reference.source_session_id != root_session_id
        or any(session.parent_session_id != root_session_id for session in branch_sessions)
    ):
        raise V11OwnershipProofError("source branch ownership is not bound to the declared root")
    layout = adapter.inspect_physical_kv_layout(
        branch_session_ids,
        timeout_s=_remaining_seconds(deadline),
    )
    if len(layout.blocks) != expected_source_block_count:
        raise V11OwnershipProofError(
            "pre-release layout does not contain the exact expected source block count"
        )
    by_block_id = {
        _parse_runtime_block_id(block.runtime_block_id): block for block in layout.blocks
    }
    if len(by_block_id) != len(layout.blocks) or set(by_block_id) != set(
        expected_allocation_epochs
    ):
        raise V11OwnershipProofError("pre-release layout differs from captured source allocations")

    epoch_source = require_vllm_allocator_epoch_source(view.manager)
    ordered_block_ids = tuple(sorted(by_block_id))
    try:
        allocator_proof = epoch_source.proof_for_live_blocks(
            ordered_block_ids,
            owner_request_ids=tuple(runtime_ids.values()),
        )
        epoch_source.require_current(allocator_proof, expected_block_ids=ordered_block_ids)
    except VllmAllocatorEpochError as error:
        raise V11OwnershipProofError("pre-release allocator epoch proof is stale") from error
    issued_by_block = {record.block_id: record for record in allocator_proof.records}
    global_referrers = _live_referrers_by_block(adapter)
    records: list[V11PreReleaseBlockOwnership] = []
    for block_id in ordered_block_ids:
        block = by_block_id[block_id]
        issued = issued_by_block[block_id]
        expected_epoch = expected_allocation_epochs[block_id]
        logical_owners = tuple(sorted(block.branch_ids))
        runtime_owners = tuple(sorted(runtime_ids[owner] for owner in logical_owners))
        observed_referrers = tuple(sorted(global_referrers.get(block_id, ())))
        if (
            not logical_owners
            or any(owner not in runtime_ids for owner in logical_owners)
            or observed_referrers != runtime_owners
            or int(block.refcount) != len(logical_owners)
            or int(block.allocation_epoch) != expected_epoch
            or issued.allocation_epoch != expected_epoch
        ):
            raise V11OwnershipProofError(
                f"pre-release owner/refcount/epoch proof failed for block {block_id}"
            )
        records.append(
            V11PreReleaseBlockOwnership(
                runtime_block_id=block.runtime_block_id,
                block_id=block_id,
                allocation_epoch=expected_epoch,
                pre_release_owner=logical_owners,
                pre_release_runtime_owner=runtime_owners,
                pre_release_refcount=int(block.refcount),
                physical_bytes=int(block.bytes),
            )
        )

    memory = adapter.inspect_gpu_memory_state(timeout_s=_remaining_seconds(deadline))
    pool_bytes, assigned_bytes, unassigned_bytes = _memory_tuple(memory)
    source_bytes = sum(record.physical_bytes for record in records)
    if (
        int(layout.physical_assigned_bytes) != source_bytes
        or assigned_bytes != source_bytes
        or pool_bytes != int(layout.kv_pool_reserved_bytes)
    ):
        raise V11OwnershipProofError(
            "allocator-visible pre-release ownership disagrees with HBM assignment accounting"
        )
    require_production_gate_v11(
        admission_gate,
        view.scheduler,
        view.manager,
        operation="EXPORT_CAPTURE",
    )
    return V11SourceOwnershipSnapshot(
        schema_version=V11_OWNERSHIP_PROOF_SCHEMA_VERSION,
        runtime="vllm",
        runtime_version=VLLM_LIVE_RUNTIME_VERSION,
        device=str(layout.blocks[0].device),
        root_session_id=root_session_id,
        root_reference_id=root_reference_id,
        branch_session_ids=branch_session_ids,
        runtime_request_ids=tuple(sorted(runtime_ids.items())),
        blocks=tuple(records),
        captured_at_monotonic_ns=time.monotonic_ns(),
        kv_pool_reserved_bytes=pool_bytes,
        kv_assigned_bytes=assigned_bytes,
        kv_unassigned_bytes=unassigned_bytes,
        source_physical_bytes=source_bytes,
        _adapter_object_id=id(adapter),
        _allocator_epoch_proof=allocator_proof,
        _proof_seal=_V11_SOURCE_OWNERSHIP_SNAPSHOT_SEAL,
    )


def _require_allocator_epochs_released(
    epoch_source: Any,
    records: tuple[VllmAllocatorIssuedEpoch, ...],
) -> None:
    checker = getattr(epoch_source, "require_released", None)
    if not callable(checker):
        raise V11OwnershipProofError(
            "runtime allocator epoch source lacks the mandatory post-free check"
        )
    checker(records)


def release_source_and_prove_v11(
    adapter: VllmLiveStateAdapter,
    snapshot: V11SourceOwnershipSnapshot,
    *,
    admission_gate: V11AdmissionGateWitness,
    timeout_s: float,
) -> V11PostFreeOwnershipProof:
    """Release the source through vLLM and prove exact post-free ownership."""

    _require_production_adapter(adapter)
    if (
        not isinstance(snapshot, V11SourceOwnershipSnapshot)
        or snapshot._proof_seal is not _V11_SOURCE_OWNERSHIP_SNAPSHOT_SEAL
        or snapshot._adapter_object_id != id(adapter)
    ):
        raise V11OwnershipProofError("v11 source ownership snapshot is forged or foreign")
    if not 0 < timeout_s <= 300:
        raise ValueError("v11 ownership timeout must be in (0, 300] seconds")
    deadline = time.monotonic_ns() + int(timeout_s * 1_000_000_000)
    view = adapter._view
    require_production_gate_v11(
        admission_gate,
        view.scheduler,
        view.manager,
        operation="EXPORT_CAPTURE",
    )
    epoch_source = require_vllm_allocator_epoch_source(view.manager)
    epoch_source.require_current(
        snapshot._allocator_epoch_proof,
        expected_block_ids=tuple(record.block_id for record in snapshot.blocks),
    )
    release_started_ns = time.monotonic_ns()
    for branch_session_id in snapshot.branch_session_ids:
        adapter.destroy_branch(
            branch_session_id,
            timeout_s=_remaining_seconds(deadline),
        )
        require_production_gate_v11(
            admission_gate,
            view.scheduler,
            view.manager,
            operation="EXPORT_CAPTURE",
        )
    adapter.destroy_session(
        snapshot.root_session_id,
        timeout_s=_remaining_seconds(deadline),
    )
    release_completed_ns = time.monotonic_ns()
    require_production_gate_v11(
        admission_gate,
        view.scheduler,
        view.manager,
        operation="EXPORT_CAPTURE",
    )

    source_runtime_ids = tuple(record.runtime_block_id for record in snapshot.blocks)
    release_evidence = adapter.inspect_block_release_evidence(
        source_runtime_ids,
        timeout_s=_remaining_seconds(deadline),
    )
    observed = {record.runtime_block_id: record for record in release_evidence.blocks}
    exact_coverage = (
        set(release_evidence.requested_block_ids) == set(source_runtime_ids)
        and set(observed) == set(source_runtime_ids)
        and len(observed) == len(snapshot.blocks)
    )
    if not exact_coverage:
        raise V11OwnershipProofError("post-free evidence does not cover every source block once")

    session_ids = (snapshot.root_session_id, *snapshot.branch_session_ids)
    old_layout = adapter.inspect_physical_kv_layout(
        session_ids,
        timeout_s=_remaining_seconds(deadline),
    )
    old_layout_inaccessible = not old_layout.blocks and old_layout.physical_assigned_bytes == 0
    all_destroyed = all(
        adapter._sessions[session_id].phase.value == "destroyed" for session_id in session_ids
    )
    root_reference_absent = snapshot.root_reference_id not in adapter._roots
    scheduler_requests = getattr(view.scheduler, "requests", None)
    if not isinstance(scheduler_requests, dict):
        raise V11OwnershipProofError("vLLM scheduler request table disappeared after release")
    old_runtime_ids = {runtime_id for _session_id, runtime_id in snapshot.runtime_request_ids}
    old_runtime_requests_absent = not (old_runtime_ids & set(scheduler_requests))
    live_referrers = _live_referrers_by_block(adapter)
    source_block_ids = {record.block_id for record in snapshot.blocks}
    live_source_referrers = {
        block_id: owners
        for block_id, owners in live_referrers.items()
        if block_id in source_block_ids and owners
    }
    no_live_reference = not live_source_referrers and old_runtime_requests_absent
    surviving_native_request_tables = old_runtime_ids & _native_request_table_keys(adapter)
    native_request_tables_absent = not surviving_native_request_tables

    issued_records = snapshot._allocator_epoch_proof.records
    _require_allocator_epochs_released(epoch_source, issued_records)
    allocator_epochs_tombstoned = True
    memory_after = adapter.inspect_gpu_memory_state(timeout_s=_remaining_seconds(deadline))
    after_pool, after_assigned, after_unassigned = _memory_tuple(memory_after)
    allocator_hbm_agreement = (
        after_pool == snapshot.kv_pool_reserved_bytes
        and after_assigned == 0
        and snapshot.kv_assigned_bytes - after_assigned == snapshot.source_physical_bytes
        and after_unassigned - snapshot.kv_unassigned_bytes == snapshot.source_physical_bytes
    )
    full_free_pool = (
        int(release_evidence.pool_free_block_count) == int(release_evidence.pool_usable_block_count)
        and after_unassigned == after_pool
    )

    transitions: list[V11BlockOwnershipTransition] = []
    for before in snapshot.blocks:
        after = observed[before.runtime_block_id]
        if (
            int(after.block_index) != before.block_id
            or int(after.allocation_epoch) != before.allocation_epoch
        ):
            raise V11OwnershipProofError(
                f"post-free block identity changed for source block {before.block_id}"
            )
        transitions.append(
            V11BlockOwnershipTransition(
                runtime_block_id=before.runtime_block_id,
                block_id=before.block_id,
                allocation_epoch=before.allocation_epoch,
                pre_release_owner=before.pre_release_owner,
                pre_release_runtime_owner=before.pre_release_runtime_owner,
                pre_release_refcount=before.pre_release_refcount,
                release_operation=(
                    "LLMEngine.abort_request->KVCacheManager.free->BlockPool.reset_prefix_cache"
                ),
                post_release_owner=(),
                post_release_refcount=int(after.native_refcount),
                allocator_available=bool(after.allocator_available) and not bool(after.is_null),
                block_hash_present=bool(after.block_hash_present),
                no_live_branch_reference=no_live_reference,
                old_session_inaccessible=old_layout_inaccessible and old_runtime_requests_absent,
                allocator_epoch_tombstoned=allocator_epochs_tombstoned,
                native_request_tables_absent=native_request_tables_absent,
                physical_bytes=before.physical_bytes,
            )
        )
    passed = (
        exact_coverage
        and all_destroyed
        and root_reference_absent
        and old_layout_inaccessible
        and no_live_reference
        and native_request_tables_absent
        and allocator_epochs_tombstoned
        and allocator_hbm_agreement
        and full_free_pool
        and all(record.passed for record in transitions)
    )
    if not passed:
        raise V11OwnershipProofError("v11 source release failed the post-free ownership gate")
    require_production_gate_v11(
        admission_gate,
        view.scheduler,
        view.manager,
        operation="EXPORT_CAPTURE",
    )
    return V11PostFreeOwnershipProof(
        schema_version=V11_OWNERSHIP_PROOF_SCHEMA_VERSION,
        runtime="vllm",
        runtime_version=VLLM_LIVE_RUNTIME_VERSION,
        device=snapshot.device,
        root_session_id=snapshot.root_session_id,
        root_reference_id=snapshot.root_reference_id,
        branch_session_ids=snapshot.branch_session_ids,
        release_started_at_monotonic_ns=release_started_ns,
        release_completed_at_monotonic_ns=release_completed_ns,
        records=tuple(transitions),
        source_block_count=len(transitions),
        source_physical_bytes=snapshot.source_physical_bytes,
        pool_free_block_count=int(release_evidence.pool_free_block_count),
        pool_usable_block_count=int(release_evidence.pool_usable_block_count),
        kv_pool_reserved_bytes_before=snapshot.kv_pool_reserved_bytes,
        kv_pool_reserved_bytes_after=after_pool,
        kv_assigned_bytes_before=snapshot.kv_assigned_bytes,
        kv_assigned_bytes_after=after_assigned,
        kv_unassigned_bytes_before=snapshot.kv_unassigned_bytes,
        kv_unassigned_bytes_after=after_unassigned,
        exact_source_coverage=exact_coverage,
        all_old_sessions_destroyed=all_destroyed,
        root_reference_released=root_reference_absent,
        old_session_layout_inaccessible=old_layout_inaccessible,
        no_live_branch_reference=no_live_reference,
        native_manager_request_tables_absent=native_request_tables_absent,
        all_allocator_epochs_tombstoned=allocator_epochs_tombstoned,
        allocator_hbm_agreement=allocator_hbm_agreement,
        full_free_pool_recovered=full_free_pool,
        passed=passed,
        _proof_seal=_V11_POST_FREE_OWNERSHIP_PROOF_SEAL,
    )


__all__ = [
    "V11_ALLOCATOR_QUIESCENCE_SCHEMA_VERSION",
    "V11_OWNERSHIP_PROOF_SCHEMA_VERSION",
    "V11_SOURCE_CAPTURE_COMMIT_SCHEMA_VERSION",
    "SourceCaptureCommit",
    "V11AllocatorQuiescenceEvidence",
    "V11BlockOwnershipTransition",
    "V11OwnershipProofError",
    "V11PostFreeOwnershipProof",
    "V11PreReleaseBlockOwnership",
    "V11SourceCaptureBranch",
    "V11SourceIdentityValidation",
    "V11SourceOwnershipSnapshot",
    "V11SourceSemanticAllocation",
    "capture_source_ownership_v11",
    "create_source_capture_commit_v11",
    "expected_owner_set",
    "expected_refcount",
    "prove_allocator_quiescence_v11",
    "release_source_and_prove_v11",
    "validate_source_capture_commit_v11",
]
