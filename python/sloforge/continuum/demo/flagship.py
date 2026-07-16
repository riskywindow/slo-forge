"""Artifact-backed deterministic CPU demonstration of Continuum's core thesis."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.continuum.adapters import (
    AdapterUnavailableError,
    CapturedState,
    DirtyDelta,
    ModelContract,
    ReferenceHeadMajorAdapter,
    ReferenceTokenMajorAdapter,
)
from sloforge.continuum.adapters import (
    TokenEvent as RuntimeTokenEvent,
)
from sloforge.continuum.capture import publish_capture
from sloforge.continuum.compatibility import (
    CompatibilityDecision,
    CompatibilityRequest,
    ExactnessClass,
    ModelSemantics,
    RuntimeCapabilities,
    StateDependencyEvidence,
    analyze_compatibility,
)
from sloforge.continuum.faults import FaultActivation, FaultKind, fault_definition
from sloforge.continuum.ir import canonical_hash
from sloforge.continuum.migration import (
    MigrationWallObservation,
    PrecopyMigrationRequest,
    PrecopyMigrationResult,
    migrate_precopy,
)
from sloforge.continuum.operations import (
    RecomputeEvidence,
    checkpoint_full,
    clone_checkpoint,
    recompute_from_token_history,
)
from sloforge.continuum.reference.runtime import DeterministicHybridRuntimeAdapter
from sloforge.continuum.storage import ChunkRef, MemoryContentStore, StoredManifest
from sloforge.continuum.transaction import (
    CutoverPhase,
    DurableCoordinator,
    GatewayCommitLedger,
    SessionLease,
)
from sloforge.continuum.transaction import TokenEvent as GatewayTokenEvent
from sloforge.continuum.transport import (
    DeterministicSimulatedTransport,
    StateTransport,
    TransferReceipt,
)


class DemoModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class FlagshipDemoRequest(DemoModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    work_dir: Path
    session_id: str = Field(min_length=1, max_length=256)
    tenant_id: str = Field(min_length=1, max_length=256)
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    initial_output_tokens: Annotated[int, Field(ge=8, le=128)] = 24
    successful_delta_rounds: tuple[Annotated[int, Field(ge=1, le=64)], ...] = (4, 3)
    resumed_tokens: Annotated[int, Field(ge=2, le=64)] = 5
    capture_timestamp: str = "2026-08-02T00:00:00Z"
    git_commit: str = Field(min_length=7, max_length=64)
    continuum_version: str = "0.1.0"


class TimelineEvent(DemoModel):
    sequence: Annotated[int, Field(ge=0)]
    category: Literal[
        "runtime",
        "token",
        "transaction",
        "transfer",
        "fault",
        "fork",
        "compatibility",
    ]
    label: str = Field(min_length=1, max_length=512)
    transaction_id: str | None = None
    phase: str | None = None
    session_id: str | None = None
    owner_epoch: int | None = None
    token_index: int | None = None
    byte_count: int | None = None
    evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeStateEvidence(DemoModel):
    runtime_name: str
    adapter_version: str
    layout: str
    tensor_parallel_degree: Annotated[int, Field(ge=1)]
    page_size_tokens: Annotated[int, Field(ge=1)]
    simulated_devices: tuple[str, ...]


class FailedMigrationEvidence(DemoModel):
    transaction_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    capsule_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    fault: FaultActivation
    phase_history: tuple[str, ...]
    transfer_receipts: tuple[TransferReceipt, ...]
    source_epoch_before: Annotated[int, Field(ge=1)]
    source_epoch_after: Annotated[int, Field(ge=1)]
    source_lifecycle_after: str
    gateway_watermark_before: int
    gateway_watermark_after: int
    accepted_token_indices: tuple[Annotated[int, Field(ge=0)], ...]
    attention_segment_count: Annotated[int, Field(ge=1)]
    recurrent_present: bool
    sampler_present: bool
    guided_decoding_present: bool
    failure_code: str


class ForkBranchEvidence(DemoModel):
    session_id: str
    owner_id: str
    owner_epoch: Annotated[int, Field(ge=1)]
    runtime: RuntimeStateEvidence
    initial_next_token: Annotated[int, Field(ge=0)]
    emitted_token_indices: tuple[Annotated[int, Field(ge=0)], ...]
    emitted_token_ids: tuple[Annotated[int, Field(ge=0)], ...]
    incremental_manifest_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    copy_on_write_new_chunks: Annotated[int, Field(ge=1)]
    copy_on_write_new_bytes: Annotated[int, Field(ge=1)]


class ForkEvidence(DemoModel):
    parent_session_id: str
    parent_owner_epoch: Annotated[int, Field(ge=1)]
    parent_manifest_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    shared_checkpoint_manifest_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    shared_chunk_count: Annotated[int, Field(ge=1)]
    full_copy_baseline_bytes: Annotated[int, Field(ge=1)]
    content_addressed_checkpoint_bytes: Annotated[int, Field(ge=1)]
    checkpoint_bytes_deduplicated: Annotated[int, Field(ge=1)]
    divergence_unique_bytes: Annotated[int, Field(ge=1)]
    branches: tuple[ForkBranchEvidence, ForkBranchEvidence]


class CompatibilityCaseEvidence(DemoModel):
    source_model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    changed_model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    shapes_match: bool
    direct_reuse: CompatibilityDecision
    recomputation_assisted: CompatibilityDecision
    recomputation_source: Literal["token_history"]
    recomputation_token_count: Annotated[int, Field(ge=1)]
    recomputation_execution: RecomputationExecutionEvidence


class RecomputationExecutionEvidence(DemoModel):
    transaction_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_capsule_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    recomputed_session_id: str = Field(min_length=1, max_length=256)
    source_owner_epoch: Annotated[int, Field(ge=1)]
    destination_owner_epoch: Annotated[int, Field(ge=2)]
    recomputation: RecomputeEvidence
    phase_history: tuple[str, ...]
    imported_structurally_valid: bool
    imported_continuation_valid: bool
    imported_logical_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dry_run_next_token: Annotated[int, Field(ge=0)]
    resumed_token_indices: tuple[Annotated[int, Field(ge=0)], ...]
    resumed_token_ids: tuple[Annotated[int, Field(ge=0)], ...]
    coordinator_scope: Literal["ephemeral_local_closed"]
    evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def verify_execution_seal(self) -> RecomputationExecutionEvidence:
        payload = self.model_dump(mode="python", exclude={"evidence_digest"})
        if canonical_hash(payload) != self.evidence_digest:
            raise ValueError("recomputation execution evidence digest is invalid")
        if not self.imported_structurally_valid or not self.imported_continuation_valid:
            raise ValueError("recomputed destination import did not validate")
        if self.destination_owner_epoch != self.source_owner_epoch + 1:
            raise ValueError("recomputation activation did not advance exactly one owner epoch")
        expected_start = self.recomputation.token_count - len(self.recomputation.first_run_tokens)
        del expected_start
        if self.resumed_token_ids != self.recomputation.first_run_tokens:
            raise ValueError("activated recomputation diverged from bounded replay evidence")
        if len(self.resumed_token_indices) != self.recomputation.continuation_horizon:
            raise ValueError("activated recomputation trace has the wrong horizon")
        if self.phase_history[-1:] != (CutoverPhase.COMPLETED.value,):
            raise ValueError("recomputation transaction did not complete")
        return self


class FlagshipInvariants(DemoModel):
    rollback_preserved_source_epoch: bool
    rollback_preserved_gateway_watermark: bool
    coordinator_restart_recovered_rollback: bool
    cross_adapter: bool
    cross_layout: bool
    tensor_parallel_changed: bool
    page_size_changed: bool
    no_gateway_duplicate: bool
    no_gateway_gap: bool
    stale_source_rejected: bool
    fork_sessions_distinct: bool
    fork_owners_distinct: bool
    unsafe_weight_reuse_rejected: bool
    recomputation_plan_generated: bool
    recomputation_executed: bool


class FlagshipDemoResult(DemoModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    source: RuntimeStateEvidence
    destination: RuntimeStateEvidence
    failed_migration: FailedMigrationEvidence
    successful_migration: PrecopyMigrationResult
    fork: ForkEvidence
    compatibility_case: CompatibilityCaseEvidence
    invariants: FlagshipInvariants
    accepted_token_indices: tuple[Annotated[int, Field(ge=0)], ...]
    timeline: tuple[TimelineEvent, ...]


class _Timeline:
    def __init__(self) -> None:
        self.events: list[TimelineEvent] = []

    def emit(
        self,
        *,
        category: Literal[
            "runtime",
            "token",
            "transaction",
            "transfer",
            "fault",
            "fork",
            "compatibility",
        ],
        label: str,
        transaction_id: str | None = None,
        phase: str | None = None,
        session_id: str | None = None,
        owner_epoch: int | None = None,
        token_index: int | None = None,
        byte_count: int | None = None,
    ) -> int:
        sequence = len(self.events)
        material = {
            "sequence": sequence,
            "category": category,
            "label": label,
            "transaction_id": transaction_id,
            "phase": phase,
            "session_id": session_id,
            "owner_epoch": owner_epoch,
            "token_index": token_index,
            "byte_count": byte_count,
        }
        digest = hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.events.append(
            TimelineEvent(
                sequence=sequence,
                category=category,
                label=label,
                transaction_id=transaction_id,
                phase=phase,
                session_id=session_id,
                owner_epoch=owner_epoch,
                token_index=token_index,
                byte_count=byte_count,
                evidence_digest=digest,
            )
        )
        return sequence


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _gateway_event(event: RuntimeTokenEvent) -> GatewayTokenEvent:
    return GatewayTokenEvent(
        session_id=event.session_id,
        owner_epoch=event.owner_epoch,
        token_index=event.token_index,
        token_id=event.token_id,
        state_commit_version=event.state_commit_version,
        transaction_id=event.transaction_id,
    )


def _runtime_evidence(runtime: DeterministicHybridRuntimeAdapter) -> RuntimeStateEvidence:
    layout = runtime.capabilities.layouts[0]
    return RuntimeStateEvidence(
        runtime_name=runtime.identity.runtime_name,
        adapter_version=runtime.identity.adapter_version,
        layout=layout.kind.value,
        tensor_parallel_degree=layout.tensor_parallel_degree,
        page_size_tokens=layout.page_size_tokens,
        simulated_devices=layout.simulated_devices,
    )


def _layout_fingerprint(runtime: DeterministicHybridRuntimeAdapter) -> str:
    layout = runtime.capabilities.layouts[0]
    document = {
        "alignment_bytes": layout.alignment_bytes,
        "kind": layout.kind.value,
        "kv_packing": layout.kv_packing,
        "ordering": layout.ordering,
        "page_size_tokens": layout.page_size_tokens,
        "tensor_parallel_degree": layout.tensor_parallel_degree,
    }
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _model_semantics(runtime: DeterministicHybridRuntimeAdapter) -> ModelSemantics:
    model = runtime.config.model
    return ModelSemantics(
        model_id=model.model_id,
        architecture="continuum_hybrid_decoder",
        weights_hash=model.model_hash,
        state_producing_weights_hash=model.state_producer_hash,
        output_head_hash=model.model_hash,
        tokenizer_hash=model.tokenizer_hash,
        special_tokens_hash=model.tokenizer_hash,
        positional_encoding="absolute_position",
        rope_fingerprint=model.positional_encoding_hash,
        attention_mask_semantics="dense_causal_full_context",
        sliding_window=None,
        layer_count=runtime.config.layers,
        head_count=runtime.config.kv_heads,
        kv_head_count=runtime.config.kv_heads,
        head_dim=runtime.config.head_dimension,
        recurrent_update_fingerprint=model.recurrent_update_hash,
        adapter_hash=model.adapter_hash,
        state_dtype="int32",
        quantization="none",
        sampler_algorithm="continuum-counter-v1",
    )


def _runtime_capabilities(runtime: DeterministicHybridRuntimeAdapter) -> RuntimeCapabilities:
    return RuntimeCapabilities(
        runtime_name=runtime.identity.runtime_name,
        runtime_version=runtime.identity.runtime_version,
        adapter_version=runtime.identity.adapter_version,
        supported_state_types=(
            "attention.kv",
            "recurrent",
            "sampler",
            "guided_decoding",
            "token_history",
            "client_delivery",
        ),
        supported_dtypes=("int32",),
        supported_quantizations=("none",),
        can_recompute_from_token_history=True,
    )


def _transition(
    coordinator: DurableCoordinator,
    timeline: _Timeline,
    transaction_id: str,
    expected: CutoverPhase,
    target: CutoverPhase,
    logical_clock: int,
    label: str,
) -> int:
    logical_clock += 1
    coordinator.transition(
        transaction_id,
        expected=expected,
        target=target,
        event_id=label,
        at_ms=logical_clock,
        payload_hash=_hash(label),
    )
    timeline.emit(
        category="transaction",
        label=label,
        transaction_id=transaction_id,
        phase=target.value,
    )
    return logical_clock


def _copy_capture(
    captured: CapturedState,
    *,
    source_store: MemoryContentStore,
    destination_store: MemoryContentStore,
    transport: StateTransport,
    seed: int,
) -> tuple[CapturedState, TransferReceipt]:
    references = tuple(
        source_store.put(captured.logical.tenant_id, segment.payload)
        for segment in captured.segments
    )
    receipt = transport.transfer(
        source=source_store,
        destination=destination_store,
        tenant_id=captured.logical.tenant_id,
        references=references,
        deadline_us=60_000_000,
        seed=seed,
    )
    by_digest = {reference.digest: reference for reference in receipt.destination_refs}
    transferred = replace(
        captured,
        segments=tuple(
            replace(
                segment,
                payload=destination_store.read(
                    captured.logical.tenant_id,
                    by_digest[segment.descriptor.checksum],
                ),
            )
            for segment in captured.segments
        ),
    )
    transferred.verify()
    return transferred, receipt


def _copy_delta(
    delta: DirtyDelta,
    *,
    source_store: MemoryContentStore,
    destination_store: MemoryContentStore,
    transport: StateTransport,
    seed: int,
) -> tuple[DirtyDelta, TransferReceipt]:
    references = tuple(
        source_store.put(delta.logical.tenant_id, segment.payload)
        for segment in delta.changed_segments
    )
    receipt = transport.transfer(
        source=source_store,
        destination=destination_store,
        tenant_id=delta.logical.tenant_id,
        references=references,
        deadline_us=60_000_000,
        seed=seed,
    )
    by_digest = {reference.digest: reference for reference in receipt.destination_refs}
    transferred = replace(
        delta,
        changed_segments=tuple(
            replace(
                segment,
                payload=destination_store.read(
                    delta.logical.tenant_id,
                    by_digest[segment.descriptor.checksum],
                ),
            )
            for segment in delta.changed_segments
        ),
    )
    return transferred, receipt


def _run_failed_precopy(
    request: FlagshipDemoRequest,
    *,
    source: DeterministicHybridRuntimeAdapter,
    destination: DeterministicHybridRuntimeAdapter,
    coordinator: DurableCoordinator,
    gateway: GatewayCommitLedger,
    source_store: MemoryContentStore,
    destination_store: MemoryContentStore,
    transport: StateTransport,
    timeline: _Timeline,
) -> FailedMigrationEvidence:
    source_metadata = source.inspect_session(request.session_id)
    watermark_before = gateway.watermark(request.session_id)
    lease = coordinator.assert_owner(
        session_id=request.session_id,
        owner_runtime=source.identity.runtime_name,
        owner_epoch=source_metadata.owner_epoch,
        fencing_token=coordinator.lease(request.session_id).fencing_token,
        now_ms=0,
    )
    transaction = coordinator.begin_transaction(
        session_id=request.session_id,
        destination_candidate=destination.identity.runtime_name,
        migration_plan_hash=_hash(f"flagship-failed-plan:{request.seed}"),
        seed=request.seed,
        now_ms=0,
        timeout_ms=60_000,
    )
    timeline.emit(
        category="transaction",
        label="failed migration proposed",
        transaction_id=transaction.transaction_id,
        phase=CutoverPhase.PROPOSED.value,
    )
    clock = 0
    compatibility = analyze_compatibility(
        CompatibilityRequest(
            source=_model_semantics(source),
            destination=_model_semantics(destination),
            source_runtime=_runtime_capabilities(source),
            destination_runtime=_runtime_capabilities(destination),
            source_layout_fingerprint=_layout_fingerprint(source),
            destination_layout_fingerprint=_layout_fingerprint(destination),
            required_state_types=(
                "attention.kv",
                "recurrent",
                "sampler",
                "guided_decoding",
                "token_history",
                "client_delivery",
            ),
            required_exactness=ExactnessClass.EXACT_SEMANTIC,
        )
    )
    if not compatibility.safe:
        raise RuntimeError("flagship source and destination unexpectedly failed compatibility")
    clock = _transition(
        coordinator,
        timeline,
        transaction.transaction_id,
        CutoverPhase.PROPOSED,
        CutoverPhase.COMPATIBILITY_VALIDATED,
        clock,
        "failed-attempt-compatibility-validated",
    )
    tracking = source.start_dirty_tracking(request.session_id)
    captured = source.capture_consistent(request.session_id)
    published = publish_capture(
        captured,
        store=source_store,
        lease=lease,
        transaction=transaction,
        journal=coordinator.journal(transaction.transaction_id),
        published_at_ms=0,
        capture_timestamp=request.capture_timestamp,
        git_commit=request.git_commit,
        continuum_version=request.continuum_version,
    )
    transferred, initial_receipt = _copy_capture(
        captured,
        source_store=source_store,
        destination_store=destination_store,
        transport=transport,
        seed=request.seed,
    )
    timeline.emit(
        category="transfer",
        label="failed attempt initial snapshot transferred",
        transaction_id=transaction.transaction_id,
        phase=CutoverPhase.DESTINATION_PREPARING.value,
        byte_count=initial_receipt.bytes_on_wire,
    )
    clock = _transition(
        coordinator,
        timeline,
        transaction.transaction_id,
        CutoverPhase.COMPATIBILITY_VALIDATED,
        CutoverPhase.DESTINATION_PREPARING,
        clock,
        "failed-attempt-destination-preparing",
    )
    destination.prepare_destination_session(
        transferred,
        destination_session_id=request.session_id,
        proposed_owner_epoch=transaction.proposed_destination_epoch,
    )
    destination.import_captured_state(request.session_id, transferred)
    clock = _transition(
        coordinator,
        timeline,
        transaction.transaction_id,
        CutoverPhase.DESTINATION_PREPARING,
        CutoverPhase.PRECOPYING,
        clock,
        "failed-attempt-initial-imported",
    )
    clock = _transition(
        coordinator,
        timeline,
        transaction.transaction_id,
        CutoverPhase.PRECOPYING,
        CutoverPhase.DELTA_SYNCING,
        clock,
        "failed-attempt-delta-syncing",
    )
    delta = source.obtain_dirty_delta(tracking)
    transferred_delta, delta_receipt = _copy_delta(
        delta,
        source_store=source_store,
        destination_store=destination_store,
        transport=transport,
        seed=request.seed + 1,
    )
    destination.apply_dirty_delta(request.session_id, transferred_delta)
    clock = _transition(
        coordinator,
        timeline,
        transaction.transaction_id,
        CutoverPhase.DELTA_SYNCING,
        CutoverPhase.CUTOVER_REQUESTED,
        clock,
        "failed-attempt-cutover-requested",
    )
    clock = _transition(
        coordinator,
        timeline,
        transaction.transaction_id,
        CutoverPhase.CUTOVER_REQUESTED,
        CutoverPhase.SOURCE_QUIESCING,
        clock,
        "failed-attempt-source-quiescing",
    )
    source.quiesce_at_token_boundary(request.session_id)
    clock = _transition(
        coordinator,
        timeline,
        transaction.transaction_id,
        CutoverPhase.SOURCE_QUIESCING,
        CutoverPhase.SOURCE_FROZEN,
        clock,
        "failed-attempt-source-frozen",
    )
    final_delta = source.export_final_delta(tracking)
    clock = _transition(
        coordinator,
        timeline,
        transaction.transaction_id,
        CutoverPhase.SOURCE_FROZEN,
        CutoverPhase.FINAL_DELTA_TRANSFERRING,
        clock,
        "failed-attempt-final-delta-transferring",
    )
    transferred_final, final_receipt = _copy_delta(
        final_delta,
        source_store=source_store,
        destination_store=destination_store,
        transport=transport,
        seed=request.seed + 2,
    )
    destination.apply_dirty_delta(request.session_id, transferred_final)
    clock = _transition(
        coordinator,
        timeline,
        transaction.transaction_id,
        CutoverPhase.FINAL_DELTA_TRANSFERRING,
        CutoverPhase.DESTINATION_IMPORTING,
        clock,
        "failed-attempt-final-delta-imported",
    )
    clock = _transition(
        coordinator,
        timeline,
        transaction.transaction_id,
        CutoverPhase.DESTINATION_IMPORTING,
        CutoverPhase.DESTINATION_VALIDATING,
        clock,
        "failed-attempt-destination-validating",
    )
    activation_sequence = timeline.emit(
        category="fault",
        label=FaultKind.DESTINATION_CRASH_DURING_VALIDATION.value,
        transaction_id=transaction.transaction_id,
        phase=CutoverPhase.DESTINATION_VALIDATING.value,
        session_id=request.session_id,
    )
    destination.crash()
    failure_code = ""
    try:
        destination.validate_imported_state(request.session_id)
    except AdapterUnavailableError as error:
        failure_code = error.code
    else:
        raise RuntimeError("injected destination crash did not stop validation")
    clock = _transition(
        coordinator,
        timeline,
        transaction.transaction_id,
        CutoverPhase.DESTINATION_VALIDATING,
        CutoverPhase.ABORTING,
        clock,
        "destination-crash-aborting",
    )
    clock = _transition(
        coordinator,
        timeline,
        transaction.transaction_id,
        CutoverPhase.ABORTING,
        CutoverPhase.ROLLED_BACK,
        clock,
        "destination-crash-rolled-back",
    )
    destination.restart()
    destination.abort_destination(request.session_id)
    source.resume_session(request.session_id, expected_owner_epoch=transaction.source_epoch)
    source.stop_dirty_tracking(tracking)
    clear_sequence = timeline.emit(
        category="runtime",
        label="source resumed after durable pre-commit rollback",
        transaction_id=transaction.transaction_id,
        phase=CutoverPhase.ROLLED_BACK.value,
        session_id=request.session_id,
        owner_epoch=source.inspect_session(request.session_id).owner_epoch,
    )
    definition = fault_definition(FaultKind.DESTINATION_CRASH_DURING_VALIDATION)
    activation = FaultActivation(
        definition=definition,
        transaction_id=transaction.transaction_id,
        activation_sequence=activation_sequence,
        clear_sequence=clear_sequence,
        observed_protocol_response=(
            "destination staging discarded; transaction persisted ROLLED_BACK; "
            "source resumed at unchanged epoch"
        ),
        injected=True,
    )
    after = source.inspect_session(request.session_id)
    accepted = gateway.accepted_tokens(request.session_id)
    indices = tuple(event.token_index for event in accepted)
    if indices != tuple(range(len(indices))):
        raise RuntimeError("failed attempt corrupted the gateway token sequence")
    return FailedMigrationEvidence(
        transaction_id=transaction.transaction_id,
        capsule_id=published.capsule.identity.capsule_id,
        fault=activation,
        phase_history=tuple(
            entry.to_phase.value for entry in coordinator.journal(transaction.transaction_id)
        ),
        transfer_receipts=(initial_receipt, delta_receipt, final_receipt),
        source_epoch_before=source_metadata.owner_epoch,
        source_epoch_after=after.owner_epoch,
        source_lifecycle_after=after.lifecycle.value,
        gateway_watermark_before=watermark_before,
        gateway_watermark_after=gateway.watermark(request.session_id),
        accepted_token_indices=indices,
        attention_segment_count=sum(
            1
            for segment in captured.segments
            if segment.descriptor.state_kind.value.startswith("attention")
        ),
        recurrent_present=bool(captured.logical.recurrent_state),
        sampler_present=captured.logical.sampler.counter >= 0,
        guided_decoding_present=bool(captured.logical.guided_decoding.automaton_id),
        failure_code=failure_code,
    )


def _unique_references(
    store: MemoryContentStore,
    tenant_id: str,
    payloads: tuple[bytes, ...],
) -> tuple[ChunkRef, ...]:
    by_digest: dict[str, ChunkRef] = {}
    for payload in payloads:
        reference = store.put(tenant_id, payload)
        by_digest.setdefault(reference.digest, reference)
    return tuple(by_digest[digest] for digest in sorted(by_digest))


def _incremental_manifest(
    store: MemoryContentStore,
    *,
    tenant_id: str,
    parent: StoredManifest,
    captured: CapturedState,
    published_at_ms: int,
) -> tuple[StoredManifest, tuple[ChunkRef, ...]]:
    parent_digests = {reference.digest for reference in parent.chunks}
    all_references = _unique_references(
        store,
        tenant_id,
        tuple(segment.payload for segment in captured.segments),
    )
    changed = tuple(
        reference for reference in all_references if reference.digest not in parent_digests
    )
    if not changed:
        raise RuntimeError("fork branch did not produce copy-on-write state")
    manifest = store.publish(
        tenant_id=tenant_id,
        kind="incremental",
        chunks=changed,
        parent_manifest_id=parent.manifest_id,
        published_at_ms=published_at_ms,
    )
    return manifest, changed


def _fork_migrated_state(
    request: FlagshipDemoRequest,
    *,
    migrated: DeterministicHybridRuntimeAdapter,
    gateway: GatewayCommitLedger,
    timeline: _Timeline,
) -> ForkEvidence:
    captured = migrated.capture_consistent(request.session_id)
    store = MemoryContentStore()
    references = _unique_references(
        store,
        request.tenant_id,
        tuple(segment.payload for segment in captured.segments),
    )
    parent = store.publish(
        tenant_id=request.tenant_id,
        kind="complete",
        chunks=references,
        published_at_ms=100,
    )
    branch_checkpoint_a = store.fork(
        tenant_id=request.tenant_id,
        parent_manifest_id=parent.manifest_id,
        published_at_ms=101,
    )
    branch_checkpoint_b = store.fork(
        tenant_id=request.tenant_id,
        parent_manifest_id=parent.manifest_id,
        published_at_ms=102,
    )
    if branch_checkpoint_a.chunks != parent.chunks or branch_checkpoint_b.chunks != parent.chunks:
        raise RuntimeError("fork did not retain the immutable content-addressed checkpoint")

    branch_a_runtime = ReferenceTokenMajorAdapter(page_size_tokens=4)
    branch_b_runtime = ReferenceHeadMajorAdapter(page_size_tokens=7)
    branch_specs = (
        ("branch-a", 3, branch_a_runtime, 1, branch_checkpoint_a, 103),
        ("branch-b", 4, branch_b_runtime, 2, branch_checkpoint_b, 104),
    )
    branch_evidence: list[ForkBranchEvidence] = []
    divergence_references: dict[str, ChunkRef] = {}
    base_next_index = len(captured.logical.committed_output_token_ids)
    for branch_id, epoch, runtime, token_count, checkpoint, published_at in branch_specs:
        runtime.prepare_destination_session(
            captured,
            destination_session_id=branch_id,
            proposed_owner_epoch=epoch,
        )
        runtime.import_captured_state(branch_id, captured)
        validation = runtime.validate_imported_state(branch_id)
        runtime.activate_destination(
            branch_id,
            committed_owner_epoch=epoch,
            fencing_token=_hash(f"{branch_id}:{epoch}:{request.seed}"),
        )
        gateway.register(
            session_id=branch_id,
            owner_epoch=epoch,
            next_token_index=base_next_index,
        )
        emitted = runtime.stream_tokens(branch_id, count=token_count)
        for event in emitted:
            gateway.accept(_gateway_event(event))
            runtime.acknowledge_gateway(
                branch_id,
                token_index=event.token_index,
                owner_epoch=event.owner_epoch,
            )
            timeline.emit(
                category="token",
                label="fork branch token accepted",
                session_id=branch_id,
                owner_epoch=event.owner_epoch,
                token_index=event.token_index,
            )
        branch_capture = runtime.capture_consistent(branch_id)
        incremental, changed = _incremental_manifest(
            store,
            tenant_id=request.tenant_id,
            parent=checkpoint,
            captured=branch_capture,
            published_at_ms=published_at,
        )
        for reference in changed:
            divergence_references.setdefault(reference.digest, reference)
        timeline.emit(
            category="fork",
            label="copy-on-write branch published",
            session_id=branch_id,
            owner_epoch=epoch,
            byte_count=sum(reference.size_bytes for reference in changed),
        )
        branch_evidence.append(
            ForkBranchEvidence(
                session_id=branch_id,
                owner_id=f"{runtime.identity.runtime_name}:epoch-{epoch}",
                owner_epoch=epoch,
                runtime=_runtime_evidence(runtime),
                initial_next_token=validation.dry_run_next_token,
                emitted_token_indices=tuple(event.token_index for event in emitted),
                emitted_token_ids=tuple(event.token_id for event in emitted),
                incremental_manifest_id=incremental.manifest_id,
                copy_on_write_new_chunks=len(changed),
                copy_on_write_new_bytes=sum(reference.size_bytes for reference in changed),
            )
        )
    parent_bytes = sum(reference.size_bytes for reference in references)
    if branch_evidence[0].session_id == branch_evidence[1].session_id:
        raise RuntimeError("fork descendants reused a session identity")
    if branch_evidence[0].owner_id == branch_evidence[1].owner_id:
        raise RuntimeError("fork descendants reused an output owner identity")
    return ForkEvidence(
        parent_session_id=request.session_id,
        parent_owner_epoch=captured.logical.owner_epoch,
        parent_manifest_id=parent.manifest_id,
        shared_checkpoint_manifest_id=branch_checkpoint_a.manifest_id,
        shared_chunk_count=len(references),
        full_copy_baseline_bytes=parent_bytes * 2,
        content_addressed_checkpoint_bytes=parent_bytes,
        checkpoint_bytes_deduplicated=parent_bytes,
        divergence_unique_bytes=sum(
            reference.size_bytes for reference in divergence_references.values()
        ),
        branches=(branch_evidence[0], branch_evidence[1]),
    )


def _compatibility_case(
    request: FlagshipDemoRequest,
    *,
    source: DeterministicHybridRuntimeAdapter,
    timeline: _Timeline,
) -> CompatibilityCaseEvidence:
    base = source.config.model
    changed = ModelContract(
        model_id=f"{base.model_id}-attention-revision",
        model_hash=_hash(f"changed-model:{request.seed}"),
        tokenizer_hash=base.tokenizer_hash,
        adapter_hash=base.adapter_hash,
        state_producer_hash=_hash(f"changed-attention-state-producer:{request.seed}"),
        recurrent_update_hash=base.recurrent_update_hash,
        positional_encoding_hash=base.positional_encoding_hash,
        vocabulary_size=base.vocabulary_size,
    )
    changed_runtime = ReferenceHeadMajorAdapter(
        page_size_tokens=source.config.layout.page_size_tokens,
        model=changed,
    )
    evidence = StateDependencyEvidence(
        dependency_graph_hash=_hash("continuum-hybrid-attention-dependency-graph-v1"),
        changed_components=("attention",),
        state_producing_components=("attention", "recurrent_update"),
        affected_state_components=("attention.kv",),
        recomputable_state_components=("attention.kv",),
        output_head_is_state_sink=True,
        token_history_available=True,
    )
    source_model = _model_semantics(source)
    destination_model = _model_semantics(changed_runtime)
    source_capabilities = _runtime_capabilities(source)
    destination_capabilities = _runtime_capabilities(changed_runtime)
    source_layout = _layout_fingerprint(source)
    destination_layout = _layout_fingerprint(changed_runtime)
    required_state_types = (
        "attention.kv",
        "recurrent",
        "sampler",
        "guided_decoding",
        "token_history",
        "client_delivery",
    )
    direct = analyze_compatibility(
        CompatibilityRequest(
            source=source_model,
            destination=destination_model,
            source_runtime=source_capabilities,
            destination_runtime=destination_capabilities,
            source_layout_fingerprint=source_layout,
            destination_layout_fingerprint=destination_layout,
            required_state_types=required_state_types,
            dependency_evidence=evidence,
            required_exactness=ExactnessClass.EXACT_SEMANTIC,
            allow_recomputation=False,
        )
    )
    recomputation = analyze_compatibility(
        CompatibilityRequest(
            source=source_model,
            destination=destination_model,
            source_runtime=source_capabilities,
            destination_runtime=destination_capabilities,
            source_layout_fingerprint=source_layout,
            destination_layout_fingerprint=destination_layout,
            required_state_types=required_state_types,
            dependency_evidence=evidence,
            required_exactness=ExactnessClass.RECOMPUTATION_ASSISTED,
            allow_recomputation=True,
        )
    )
    if direct.safe or direct.compatibility_class is not ExactnessClass.INCOMPATIBLE:
        raise RuntimeError("changed state-producing weights were incorrectly reusable")
    if (
        not recomputation.safe
        or recomputation.compatibility_class is not ExactnessClass.RECOMPUTATION_ASSISTED
    ):
        raise RuntimeError("verified token-history recomputation plan was not generated")
    manifest = source.capture_consistent(request.session_id).logical
    execution = _execute_recomputation(
        request,
        source=source,
        changed_runtime=changed_runtime,
        timeline=timeline,
    )
    return CompatibilityCaseEvidence(
        source_model_hash=base.model_hash,
        changed_model_hash=changed.model_hash,
        shapes_match=(
            source.config.layers == changed_runtime.config.layers
            and source.config.kv_heads == changed_runtime.config.kv_heads
            and source.config.head_dimension == changed_runtime.config.head_dimension
        ),
        direct_reuse=direct,
        recomputation_assisted=recomputation,
        recomputation_source="token_history",
        recomputation_token_count=(
            len(manifest.input_token_ids) + len(manifest.committed_output_token_ids)
        ),
        recomputation_execution=execution,
    )


def _execute_recomputation(
    request: FlagshipDemoRequest,
    *,
    source: DeterministicHybridRuntimeAdapter,
    changed_runtime: DeterministicHybridRuntimeAdapter,
    timeline: _Timeline,
) -> RecomputationExecutionEvidence:
    """Execute changed-weight replay, transactional activation, and bounded output."""

    source_store = MemoryContentStore()
    clone_store = MemoryContentStore()
    metadata = source.inspect_session(request.session_id)
    source_lease = SessionLease(
        session_id=request.session_id,
        owner_runtime=source.identity.runtime_name,
        owner_epoch=metadata.owner_epoch,
        fencing_token=metadata.owner_epoch,
        expiration_ms=120_000,
        coordinator_version=metadata.owner_epoch,
        last_committed_state_version=metadata.state_version,
        last_committed_token_index=metadata.client_visible_index,
    )
    checkpoint = checkpoint_full(
        source,
        request.session_id,
        store=source_store,
        lease=source_lease,
        published_at_ms=200,
        capture_timestamp=request.capture_timestamp,
        git_commit=request.git_commit,
        continuum_version=request.continuum_version,
    )
    recomputed_session_id = f"{request.session_id}-recomputed"
    clone_lease = SessionLease(
        session_id=recomputed_session_id,
        owner_runtime=source.identity.runtime_name,
        owner_epoch=1,
        fencing_token=1,
        expiration_ms=120_000,
        coordinator_version=1,
        last_committed_state_version=metadata.state_version,
        last_committed_token_index=metadata.client_visible_index,
    )
    clone = clone_checkpoint(
        checkpoint,
        source_store=source_store,
        destination_store=clone_store,
        expected_tenant_id=request.tenant_id,
        expected_model=source.config.model,
        clone_lease=clone_lease,
        seed=request.seed + 500,
        published_at_ms=201,
        capture_timestamp=request.capture_timestamp,
        git_commit=request.git_commit,
        continuum_version=request.continuum_version,
    ).clone
    recomputed = recompute_from_token_history(
        clone,
        store=clone_store,
        destination=changed_runtime,
        expected_tenant_id=request.tenant_id,
        seed=request.seed + 501,
        continuation_horizon=request.resumed_tokens,
    )
    precommit = (
        CutoverPhase.COMPATIBILITY_VALIDATED,
        CutoverPhase.DESTINATION_PREPARING,
        CutoverPhase.PRECOPYING,
        CutoverPhase.DELTA_SYNCING,
        CutoverPhase.CUTOVER_REQUESTED,
        CutoverPhase.SOURCE_QUIESCING,
        CutoverPhase.SOURCE_FROZEN,
        CutoverPhase.FINAL_DELTA_TRANSFERRING,
        CutoverPhase.DESTINATION_IMPORTING,
        CutoverPhase.DESTINATION_VALIDATING,
    )
    resumed_indices: list[int] = []
    resumed_ids: list[int] = []
    with DurableCoordinator(":memory:") as coordinator, GatewayCommitLedger(":memory:") as gateway:
        coordinator.create_lease(
            session_id=recomputed_session_id,
            owner_runtime=source.identity.runtime_name,
            expiration_ms=120_000,
            initial_token_index=metadata.client_visible_index,
        )
        gateway.register(
            session_id=recomputed_session_id,
            owner_epoch=1,
            next_token_index=metadata.client_visible_index + 1,
        )
        transaction = coordinator.begin_transaction(
            session_id=recomputed_session_id,
            destination_candidate=changed_runtime.identity.runtime_name,
            migration_plan_hash=canonical_hash(
                {
                    "schema": "sloforge.continuum.recomputation-activation-plan/v1",
                    "source_capsule_id": clone.capsule.identity.capsule_id,
                    "destination_model_hash": changed_runtime.config.model.model_hash,
                    "recomputation_evidence": recomputed.evidence.evidence_digest,
                }
            ),
            seed=request.seed + 502,
            now_ms=0,
            timeout_ms=60_000,
        )
        changed_runtime.prepare_destination_session(
            recomputed.captured,
            destination_session_id=recomputed_session_id,
            proposed_owner_epoch=transaction.proposed_destination_epoch,
        )
        current = CutoverPhase.PROPOSED
        logical_clock = 0
        for target in precommit:
            logical_clock += 1
            event_id = f"recompute-{target.value.lower()}"
            transaction = coordinator.transition(
                transaction.transaction_id,
                expected=current,
                target=target,
                event_id=event_id,
                at_ms=logical_clock,
                payload_hash=_hash(f"{event_id}:{clone.capsule.identity.capsule_id}"),
            )
            timeline.emit(
                category="transaction",
                label=event_id,
                transaction_id=transaction.transaction_id,
                phase=target.value,
                session_id=recomputed_session_id,
            )
            current = target
            if target is CutoverPhase.DESTINATION_IMPORTING:
                changed_runtime.import_captured_state(recomputed_session_id, recomputed.captured)
        validation = changed_runtime.validate_imported_state(recomputed_session_id)
        logical_clock += 1
        transaction = coordinator.transition(
            transaction.transaction_id,
            expected=CutoverPhase.DESTINATION_VALIDATING,
            target=CutoverPhase.COMMIT_INTENT_RECORDED,
            event_id="recompute-commit-intent",
            at_ms=logical_clock,
            payload_hash=_hash(f"recompute-commit:{recomputed.evidence.evidence_digest}"),
            state_hashes=(validation.imported_logical_hash,),
            commit_watermark=metadata.client_visible_index,
            rollback_watermark=metadata.client_visible_index,
        )
        logical_clock += 1
        transaction, committed_lease = coordinator.commit_ownership(
            transaction.transaction_id,
            event_id="recompute-ownership-committed",
            at_ms=logical_clock,
            state_version=recomputed.captured.logical.state_version,
        )
        logical_clock += 1
        transaction = coordinator.transition(
            transaction.transaction_id,
            expected=CutoverPhase.OWNERSHIP_COMMITTED,
            target=CutoverPhase.GATEWAY_SWITCHING,
            event_id="recompute-gateway-switching",
            at_ms=logical_clock,
            payload_hash=_hash("recompute-gateway-switching"),
        )
        gateway.switch_owner(
            session_id=recomputed_session_id,
            expected_epoch=transaction.source_epoch,
            destination_epoch=committed_lease.owner_epoch,
            expected_watermark=metadata.client_visible_index,
        )
        changed_runtime.activate_destination(
            recomputed_session_id,
            committed_owner_epoch=committed_lease.owner_epoch,
            fencing_token=str(committed_lease.fencing_token),
        )
        logical_clock += 1
        transaction = coordinator.transition(
            transaction.transaction_id,
            expected=CutoverPhase.GATEWAY_SWITCHING,
            target=CutoverPhase.DESTINATION_ACTIVE,
            event_id="recompute-destination-active",
            at_ms=logical_clock,
            payload_hash=_hash("recompute-destination-active"),
        )
        for event in changed_runtime.stream_tokens(
            recomputed_session_id,
            count=request.resumed_tokens,
            transaction_id=transaction.transaction_id,
        ):
            gateway.accept(_gateway_event(event))
            changed_runtime.acknowledge_gateway(
                recomputed_session_id,
                token_index=event.token_index,
                owner_epoch=event.owner_epoch,
            )
            resumed_indices.append(event.token_index)
            resumed_ids.append(event.token_id)
            timeline.emit(
                category="token",
                label="recomputed changed-model token accepted",
                transaction_id=transaction.transaction_id,
                phase=CutoverPhase.DESTINATION_ACTIVE.value,
                session_id=recomputed_session_id,
                owner_epoch=event.owner_epoch,
                token_index=event.token_index,
            )
        for target in (CutoverPhase.SOURCE_DRAINING, CutoverPhase.COMPLETED):
            logical_clock += 1
            transaction = coordinator.transition(
                transaction.transaction_id,
                expected=transaction.phase,
                target=target,
                event_id=f"recompute-{target.value.lower()}",
                at_ms=logical_clock,
                payload_hash=_hash(f"recompute-{target.value.lower()}"),
            )
        phase_history = tuple(
            entry.to_phase.value for entry in coordinator.journal(transaction.transaction_id)
        )
    payload = {
        "transaction_id": transaction.transaction_id,
        "checkpoint_capsule_id": clone.capsule.identity.capsule_id,
        "recomputed_session_id": recomputed_session_id,
        "source_owner_epoch": 1,
        "destination_owner_epoch": committed_lease.owner_epoch,
        "recomputation": recomputed.evidence.model_dump(mode="python"),
        "phase_history": phase_history,
        "imported_structurally_valid": validation.structurally_valid,
        "imported_continuation_valid": validation.continuation_valid,
        "imported_logical_hash": validation.imported_logical_hash,
        "dry_run_next_token": validation.dry_run_next_token,
        "resumed_token_indices": tuple(resumed_indices),
        "resumed_token_ids": tuple(resumed_ids),
        "coordinator_scope": "ephemeral_local_closed",
    }
    return RecomputationExecutionEvidence(
        transaction_id=transaction.transaction_id,
        checkpoint_capsule_id=clone.capsule.identity.capsule_id,
        recomputed_session_id=recomputed_session_id,
        source_owner_epoch=1,
        destination_owner_epoch=committed_lease.owner_epoch,
        recomputation=recomputed.evidence,
        phase_history=phase_history,
        imported_structurally_valid=validation.structurally_valid,
        imported_continuation_valid=validation.continuation_valid,
        imported_logical_hash=validation.imported_logical_hash,
        dry_run_next_token=validation.dry_run_next_token,
        resumed_token_indices=tuple(resumed_indices),
        resumed_token_ids=tuple(resumed_ids),
        coordinator_scope="ephemeral_local_closed",
        evidence_digest=canonical_hash(payload),
    )


def _record_successful_timeline(
    timeline: _Timeline,
    *,
    result: PrecopyMigrationResult,
    coordinator: DurableCoordinator,
    gateway: GatewayCommitLedger,
) -> None:
    if result.stale_source_rejected:
        timeline.emit(
            category="transaction",
            label="stale source generation rejected after ownership commit",
            transaction_id=result.transaction_id,
            phase=CutoverPhase.OWNERSHIP_COMMITTED.value,
            session_id=result.capsule.identity.session_id,
            owner_epoch=result.source_owner_epoch,
        )
    source_tokens = tuple(
        event
        for event in gateway.accepted_tokens(result.capsule.identity.session_id)
        if event.transaction_id == result.transaction_id
        and event.owner_epoch == result.source_owner_epoch
    )
    destination_tokens = tuple(
        event
        for event in gateway.accepted_tokens(result.capsule.identity.session_id)
        if event.transaction_id == result.transaction_id
        and event.owner_epoch == result.destination_owner_epoch
    )
    receipts = iter(result.transfer_receipts)
    for entry in coordinator.journal(result.transaction_id):
        timeline.emit(
            category="transaction",
            label=entry.event_id,
            transaction_id=result.transaction_id,
            phase=entry.to_phase.value,
        )
        if entry.to_phase is CutoverPhase.DESTINATION_PREPARING:
            receipt = next(receipts, None)
            if receipt is not None:
                timeline.emit(
                    category="transfer",
                    label="successful initial state transfer",
                    transaction_id=result.transaction_id,
                    phase=entry.to_phase.value,
                    byte_count=receipt.bytes_on_wire,
                )
        if entry.to_phase is CutoverPhase.DELTA_SYNCING:
            for event in source_tokens:
                timeline.emit(
                    category="token",
                    label="source token accepted while pre-copy continued",
                    transaction_id=result.transaction_id,
                    phase=entry.to_phase.value,
                    session_id=event.session_id,
                    owner_epoch=event.owner_epoch,
                    token_index=event.token_index,
                )
        if entry.to_phase is CutoverPhase.FINAL_DELTA_TRANSFERRING:
            for receipt in receipts:
                timeline.emit(
                    category="transfer",
                    label="successful delta transfer",
                    transaction_id=result.transaction_id,
                    phase=entry.to_phase.value,
                    byte_count=receipt.bytes_on_wire,
                )
        if entry.to_phase is CutoverPhase.DESTINATION_ACTIVE:
            for event in destination_tokens:
                timeline.emit(
                    category="token",
                    label="destination token accepted after owner switch",
                    transaction_id=result.transaction_id,
                    phase=entry.to_phase.value,
                    session_id=event.session_id,
                    owner_epoch=event.owner_epoch,
                    token_index=event.token_index,
                )


def run_flagship_demo(
    request: FlagshipDemoRequest,
    *,
    wall_observation: MigrationWallObservation | None = None,
) -> FlagshipDemoResult:
    """Run the CPU flagship and return only evidence derived from executed operations."""

    request.work_dir.mkdir(parents=True, exist_ok=True)
    coordinator_path = request.work_dir / "coordinator.sqlite"
    gateway_path = request.work_dir / "gateway.sqlite"
    if coordinator_path.exists() or gateway_path.exists():
        raise FileExistsError("flagship work directory must not contain an earlier coordinator")
    source = ReferenceTokenMajorAdapter(page_size_tokens=3)
    destination = ReferenceHeadMajorAdapter(page_size_tokens=5)
    source_store = MemoryContentStore()
    destination_store = MemoryContentStore()
    transport = DeterministicSimulatedTransport(
        bandwidth_bytes_per_second=8_000_000,
        latency_us=40,
        maximum_attempts=4,
        loss_probability=0.0,
        duplicate_probability=0.03,
        corruption_probability=0.0,
    )
    timeline = _Timeline()
    prompt = tuple((request.seed + index * 7) % 256 for index in range(12))
    source.create_session(
        session_id=request.session_id,
        request_id=f"request-{request.session_id}",
        tenant_id=request.tenant_id,
        input_token_ids=prompt,
        seed=request.seed,
    )
    timeline.emit(
        category="runtime",
        label="long-running hybrid session created",
        session_id=request.session_id,
        owner_epoch=1,
    )
    with GatewayCommitLedger(gateway_path) as gateway:
        gateway.register(session_id=request.session_id, owner_epoch=1)
        for event in source.stream_tokens(
            request.session_id,
            count=request.initial_output_tokens,
        ):
            gateway.accept(_gateway_event(event))
            source.acknowledge_gateway(
                request.session_id,
                token_index=event.token_index,
                owner_epoch=event.owner_epoch,
            )
            timeline.emit(
                category="token",
                label="source token accepted before migration",
                session_id=request.session_id,
                owner_epoch=event.owner_epoch,
                token_index=event.token_index,
            )
        with DurableCoordinator(coordinator_path) as coordinator:
            coordinator.create_lease(
                session_id=request.session_id,
                owner_runtime=source.identity.runtime_name,
                expiration_ms=120_000,
                initial_token_index=gateway.watermark(request.session_id),
            )
            failed = _run_failed_precopy(
                request,
                source=source,
                destination=destination,
                coordinator=coordinator,
                gateway=gateway,
                source_store=source_store,
                destination_store=destination_store,
                transport=transport,
                timeline=timeline,
            )

    with DurableCoordinator(coordinator_path) as recovered_coordinator:
        recovered_transaction = recovered_coordinator.transaction(failed.transaction_id)
        recovered_lease = recovered_coordinator.lease(request.session_id)
        restart_recovered = (
            recovered_transaction.phase is CutoverPhase.ROLLED_BACK
            and recovered_lease.owner_epoch == failed.source_epoch_before
        )
        if not restart_recovered:
            raise RuntimeError("durable coordinator restart did not recover rollback state")
        with GatewayCommitLedger(gateway_path) as recovered_gateway:
            successful = migrate_precopy(
                PrecopyMigrationRequest(
                    session_id=request.session_id,
                    seed=request.seed + 100,
                    plan_hash=_hash(f"flagship-success-plan:{request.seed}"),
                    delta_round_token_counts=request.successful_delta_rounds,
                    resume_token_count=request.resumed_tokens,
                    capture_timestamp=request.capture_timestamp,
                    git_commit=request.git_commit,
                    continuum_version=request.continuum_version,
                ),
                source=source,
                destination=destination,
                coordinator=recovered_coordinator,
                gateway=recovered_gateway,
                source_store=source_store,
                destination_store=destination_store,
                transport=transport,
                wall_observation=wall_observation,
            )
            _record_successful_timeline(
                timeline,
                result=successful,
                coordinator=recovered_coordinator,
                gateway=recovered_gateway,
            )
            fork = _fork_migrated_state(
                request,
                migrated=destination,
                gateway=recovered_gateway,
                timeline=timeline,
            )
            compatibility_case = _compatibility_case(
                request,
                source=destination,
                timeline=timeline,
            )
            timeline.emit(
                category="compatibility",
                label="changed state-producing weights rejected for direct reuse",
                session_id=request.session_id,
            )
            timeline.emit(
                category="compatibility",
                label="token-history recomputation transaction executed and verified",
                session_id=request.session_id,
            )
            accepted = recovered_gateway.accepted_tokens(request.session_id)
            indices = tuple(event.token_index for event in accepted)

    source_evidence = _runtime_evidence(source)
    destination_evidence = _runtime_evidence(destination)
    no_duplicate = len(indices) == len(set(indices))
    no_gap = indices == tuple(range(len(indices)))
    branch_a, branch_b = fork.branches
    invariants = FlagshipInvariants(
        rollback_preserved_source_epoch=(
            failed.source_epoch_before == failed.source_epoch_after == 1
        ),
        rollback_preserved_gateway_watermark=(
            failed.gateway_watermark_before == failed.gateway_watermark_after
        ),
        coordinator_restart_recovered_rollback=restart_recovered,
        cross_adapter=source_evidence.adapter_version != destination_evidence.adapter_version,
        cross_layout=source_evidence.layout != destination_evidence.layout,
        tensor_parallel_changed=(
            source_evidence.tensor_parallel_degree != destination_evidence.tensor_parallel_degree
        ),
        page_size_changed=(
            source_evidence.page_size_tokens != destination_evidence.page_size_tokens
        ),
        no_gateway_duplicate=no_duplicate,
        no_gateway_gap=no_gap,
        stale_source_rejected=successful.stale_source_rejected,
        fork_sessions_distinct=branch_a.session_id != branch_b.session_id,
        fork_owners_distinct=branch_a.owner_id != branch_b.owner_id,
        unsafe_weight_reuse_rejected=(not compatibility_case.direct_reuse.safe),
        recomputation_plan_generated=compatibility_case.recomputation_assisted.safe,
        recomputation_executed=(
            compatibility_case.recomputation_execution.phase_history[-1]
            == CutoverPhase.COMPLETED.value
        ),
    )
    if not all(invariants.model_dump(mode="python").values()):
        raise RuntimeError("flagship evidence violated a declared invariant")
    run_id = _hash(
        f"continuum-flagship:{request.session_id}:{request.seed}:"
        f"{successful.transaction_id}:{fork.parent_manifest_id}"
    )
    return FlagshipDemoResult(
        run_id=run_id,
        seed=request.seed,
        source=source_evidence,
        destination=destination_evidence,
        failed_migration=failed,
        successful_migration=successful,
        fork=fork,
        compatibility_case=compatibility_case,
        invariants=invariants,
        accepted_token_indices=indices,
        timeline=tuple(timeline.events),
    )


def write_flagship_artifact(result: FlagshipDemoResult, output: Path) -> None:
    """Atomically publish the bounded JSON artifact without logging state payloads."""

    encoded = result.model_dump_json(indent=2).encode("utf-8")
    if len(encoded) > 32 * 1024 * 1024:
        raise ValueError("flagship artifact exceeds the 32 MiB report bound")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".continuum-demo-", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
