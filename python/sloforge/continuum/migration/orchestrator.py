"""End-to-end pre-copy orchestration over the deterministic reference adapters."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from sloforge.continuum.adapters import CapturedState, DirtyDelta, SessionLifecycle
from sloforge.continuum.adapters import TokenEvent as RuntimeTokenEvent
from sloforge.continuum.capture import publish_capture
from sloforge.continuum.compatibility import (
    CompatibilityRequest,
    ExactnessClass,
    ModelSemantics,
    RuntimeCapabilities,
    analyze_compatibility,
    to_canonical_report,
)
from sloforge.continuum.conversion import LiveConversionEvidence, direct_convert_capture
from sloforge.continuum.ir import (
    CompatibilityReport,
    Digest,
    ExecutionStateCapsule,
    RuntimeIdentity,
)
from sloforge.continuum.reference.runtime import DeterministicHybridRuntimeAdapter
from sloforge.continuum.storage import ContentStore
from sloforge.continuum.transaction import (
    CutoverPhase,
    DurableCoordinator,
    GatewayCommitLedger,
)
from sloforge.continuum.transaction import (
    TokenEvent as GatewayTokenEvent,
)
from sloforge.continuum.transport import StateTransport, TransferReceipt


class MigrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


@dataclass(slots=True)
class MigrationWallObservation:
    """Non-wire host timing sink; never contributes to deterministic protocol artifacts."""

    cutover_wall_ns: int = 0
    total_wall_ns: int = 0


class MigrationExecutionError(RuntimeError):
    """Typed failure raised only after durable migration recovery is attempted."""

    code = "migration_execution_failed"

    def __init__(
        self,
        message: str,
        *,
        transaction_id: str,
        terminal_phase: CutoverPhase,
        ownership_committed: bool,
        failed_operation: str,
    ) -> None:
        super().__init__(message)
        self.transaction_id = transaction_id
        self.terminal_phase = terminal_phase
        self.ownership_committed = ownership_committed
        self.failed_operation = failed_operation


class PrecopyMigrationRequest(MigrationModel):
    session_id: str = Field(min_length=1, max_length=256)
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    delta_round_token_counts: tuple[Annotated[int, Field(ge=0, le=1024)], ...]
    resume_token_count: Annotated[int, Field(ge=1, le=1024)]
    timeout_ms: Annotated[int, Field(ge=1, le=86_400_000)] = 60_000
    transfer_deadline_us: Annotated[int, Field(ge=1, le=3_600_000_000)] = 60_000_000
    published_at_ms: Annotated[int, Field(ge=0)] = 0
    capture_timestamp: str = Field(min_length=1)
    git_commit: str = Field(min_length=1)
    continuum_version: str = Field(min_length=1)


class PrecopyMigrationResult(MigrationModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    strategy: Literal["pre_copy"] = "pre_copy"
    transaction_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    capsule: ExecutionStateCapsule
    compatibility: CompatibilityReport
    live_conversion_evidence: LiveConversionEvidence
    transfer_receipts: tuple[TransferReceipt, ...]
    source_runtime: str
    destination_runtime: str
    source_layout: str
    destination_layout: str
    source_owner_epoch: Annotated[int, Field(ge=1)]
    destination_owner_epoch: Annotated[int, Field(ge=2)]
    cutover_token_index: int
    accepted_token_indices: tuple[Annotated[int, Field(ge=0)], ...]
    source_next_token: Annotated[int, Field(ge=0)]
    destination_dry_run_token: Annotated[int, Field(ge=0)]
    stale_source_rejected: bool
    phase_history: tuple[str, ...]


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
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _model(runtime: DeterministicHybridRuntimeAdapter) -> ModelSemantics:
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


def _capabilities(runtime: DeterministicHybridRuntimeAdapter) -> RuntimeCapabilities:
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


def _runtime_identity(runtime: DeterministicHybridRuntimeAdapter) -> RuntimeIdentity:
    identity = runtime.identity
    return RuntimeIdentity(
        runtime_name=identity.runtime_name,
        runtime_version=identity.runtime_version,
        adapter_version=identity.adapter_version,
        build_hash=Digest(value=identity.build_hash),
        dependency_versions=tuple(
            f"{name}={version}" for name, version in identity.dependency_versions
        ),
        target_hardware=(identity.target_hardware,),
    )


def _event_hash(label: str) -> str:
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


def _transferred_capture(
    captured: CapturedState,
    *,
    destination_store: ContentStore,
    tenant_id: str,
    receipt: TransferReceipt,
) -> CapturedState:
    by_digest = {reference.digest: reference for reference in receipt.destination_refs}
    segments = tuple(
        replace(
            segment,
            payload=destination_store.read(
                tenant_id,
                by_digest[segment.descriptor.checksum],
            ),
        )
        for segment in captured.segments
    )
    transferred = replace(captured, segments=segments)
    transferred.verify()
    return transferred


def _transferred_delta(
    delta: DirtyDelta,
    *,
    source_store: ContentStore,
    destination_store: ContentStore,
    transport: StateTransport,
    tenant_id: str,
    deadline_us: int,
    seed: int,
) -> tuple[DirtyDelta, TransferReceipt]:
    references = tuple(
        source_store.put(tenant_id, segment.payload, compression="none")
        for segment in delta.changed_segments
    )
    receipt = transport.transfer(
        source=source_store,
        destination=destination_store,
        tenant_id=tenant_id,
        references=references,
        deadline_us=deadline_us,
        seed=seed,
    )
    by_digest = {reference.digest: reference for reference in receipt.destination_refs}
    changed = tuple(
        replace(
            segment,
            payload=destination_store.read(
                tenant_id,
                by_digest[segment.descriptor.checksum],
            ),
        )
        for segment in delta.changed_segments
    )
    return replace(delta, changed_segments=changed), receipt


def _merge_source_delta(captured: CapturedState, delta: DirtyDelta) -> CapturedState:
    """Apply transported source-layout bytes to a verified local conversion mirror."""

    if delta.session_id != captured.logical.session_id:
        raise ValueError("dirty delta belongs to another captured session")
    if delta.from_epoch != captured.logical.dirty_epoch:
        raise ValueError("dirty delta leaves an epoch gap in the conversion mirror")
    if delta.source_layout != captured.layout:
        raise ValueError("dirty delta source layout changed during migration")
    merged = {segment.descriptor.segment_id: segment for segment in captured.segments}
    for segment in delta.changed_segments:
        merged[segment.descriptor.segment_id] = segment
    live_paged_ids = {segment_id for page in delta.page_table for segment_id in page.segment_ids}
    segments = tuple(
        sorted(
            (
                segment
                for segment in merged.values()
                if segment.descriptor.page_id is None
                or segment.descriptor.segment_id in live_paged_ids
            ),
            key=lambda segment: segment.descriptor.segment_id,
        )
    )
    updated = CapturedState(
        handle=replace(
            captured.handle,
            owner_epoch=delta.owner_epoch,
            state_version=delta.to_epoch,
            dirty_epoch=delta.to_epoch,
            segment_count=len(segments),
        ),
        runtime=captured.runtime,
        layout=delta.source_layout,
        logical=delta.logical,
        segments=segments,
        page_table=delta.page_table,
    )
    updated.verify()
    return updated


def _converted_delta(
    source_mirror: CapturedState,
    *,
    prior_converted: CapturedState,
    source_delta: DirtyDelta,
    destination: DeterministicHybridRuntimeAdapter,
    maximum_temporary_bytes: int,
) -> tuple[CapturedState, DirtyDelta]:
    """Convert a complete verified mirror and emit only changed destination segments."""

    converted, _evidence = direct_convert_capture(
        source_mirror,
        destination=destination,
        maximum_temporary_bytes=maximum_temporary_bytes,
    )
    prior = {segment.descriptor.segment_id: segment for segment in prior_converted.segments}
    changed = tuple(
        segment
        for segment in converted.segments
        if prior.get(segment.descriptor.segment_id) != segment
    )
    if source_delta.to_epoch > source_delta.from_epoch and not changed:
        raise ValueError("state version advanced without a converted destination delta")
    if any(segment.descriptor.dirty_epoch <= source_delta.from_epoch for segment in changed):
        raise ValueError("converted delta contains a stale destination segment")
    destination_delta = DirtyDelta(
        tracking_id=source_delta.tracking_id,
        session_id=source_delta.session_id,
        from_epoch=source_delta.from_epoch,
        to_epoch=source_delta.to_epoch,
        owner_epoch=source_delta.owner_epoch,
        source_layout=converted.layout,
        logical=converted.logical,
        changed_segments=changed,
        page_table=converted.page_table,
        final=source_delta.final,
    )
    return converted, destination_delta


def migrate_precopy(
    request: PrecopyMigrationRequest,
    *,
    source: DeterministicHybridRuntimeAdapter,
    destination: DeterministicHybridRuntimeAdapter,
    coordinator: DurableCoordinator,
    gateway: GatewayCommitLedger,
    source_store: ContentStore,
    destination_store: ContentStore,
    transport: StateTransport,
    wall_observation: MigrationWallObservation | None = None,
) -> PrecopyMigrationResult:
    """Migrate a live reference session with real chunks and transactional cutover."""

    migration_started_ns = time.perf_counter_ns()
    source_metadata = source.inspect_session(request.session_id)
    lease = coordinator.assert_owner(
        session_id=request.session_id,
        owner_runtime=source.identity.runtime_name,
        owner_epoch=source_metadata.owner_epoch,
        fencing_token=coordinator.lease(request.session_id).fencing_token,
        now_ms=request.published_at_ms,
    )
    if gateway.watermark(request.session_id) != source_metadata.client_visible_index:
        raise ValueError("gateway and source commit watermarks differ before migration")
    transaction = coordinator.begin_transaction(
        session_id=request.session_id,
        destination_candidate=destination.identity.runtime_name,
        migration_plan_hash=request.plan_hash,
        seed=request.seed,
        now_ms=request.published_at_ms,
        timeout_ms=request.timeout_ms,
    )
    logical_clock = request.published_at_ms

    def advance(
        expected: CutoverPhase,
        target: CutoverPhase,
        label: str,
        *,
        commit_watermark: int | None = None,
        rollback_watermark: int | None = None,
        state_hashes: tuple[str, ...] | None = None,
    ) -> None:
        nonlocal logical_clock
        logical_clock += 1
        coordinator.transition(
            transaction.transaction_id,
            expected=expected,
            target=target,
            event_id=label,
            at_ms=logical_clock,
            payload_hash=_event_hash(label),
            commit_watermark=commit_watermark,
            rollback_watermark=rollback_watermark,
            state_hashes=state_hashes,
        )

    decision = analyze_compatibility(
        CompatibilityRequest(
            source=_model(source),
            destination=_model(destination),
            source_runtime=_capabilities(source),
            destination_runtime=_capabilities(destination),
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
    if not decision.safe:
        advance(CutoverPhase.PROPOSED, CutoverPhase.REJECTED, "compatibility-rejected")
        raise ValueError("semantic compatibility rejected the migration")
    advance(
        CutoverPhase.PROPOSED,
        CutoverPhase.COMPATIBILITY_VALIDATED,
        "compatibility-validated",
    )

    tracking = None
    failed_operation = "start_dirty_tracking"
    try:
        tracking = source.start_dirty_tracking(request.session_id)
        failed_operation = "initial_capture"
        initial_capture = source.capture_consistent(request.session_id)
        published = publish_capture(
            initial_capture,
            store=source_store,
            lease=lease,
            transaction=transaction,
            journal=coordinator.journal(transaction.transaction_id),
            published_at_ms=request.published_at_ms,
            capture_timestamp=request.capture_timestamp,
            git_commit=request.git_commit,
            continuum_version=request.continuum_version,
        )
        failed_operation = "initial_transfer"
        initial_receipt = transport.transfer(
            source=source_store,
            destination=destination_store,
            tenant_id=initial_capture.logical.tenant_id,
            references=published.chunk_references,
            deadline_us=request.transfer_deadline_us,
            seed=request.seed,
        )
        transferred_initial = _transferred_capture(
            initial_capture,
            destination_store=destination_store,
            tenant_id=initial_capture.logical.tenant_id,
            receipt=initial_receipt,
        )
        failed_operation = "initial_conversion"
        converted_current, live_conversion_evidence = direct_convert_capture(
            transferred_initial,
            destination=destination,
            maximum_temporary_bytes=512,
        )
        source_mirror = transferred_initial
        compatibility = to_canonical_report(
            decision,
            source_capsule_id=published.capsule.identity.capsule_id,
            destination_runtime=_runtime_identity(destination),
            destination_physical_plan=Digest(value=_layout_fingerprint(destination)),
        )
        advance(
            CutoverPhase.COMPATIBILITY_VALIDATED,
            CutoverPhase.DESTINATION_PREPARING,
            "destination-preparing",
        )
        failed_operation = "destination_prepare"
        destination.prepare_destination_session(
            converted_current,
            destination_session_id=request.session_id,
            proposed_owner_epoch=transaction.proposed_destination_epoch,
        )
        failed_operation = "destination_import"
        destination.import_captured_state(request.session_id, converted_current)
        advance(
            CutoverPhase.DESTINATION_PREPARING,
            CutoverPhase.PRECOPYING,
            "initial-snapshot-imported",
        )
        advance(CutoverPhase.PRECOPYING, CutoverPhase.DELTA_SYNCING, "delta-syncing")

        receipts = [initial_receipt]
        for round_index, count in enumerate(request.delta_round_token_counts):
            failed_operation = f"precopy_round_{round_index}_generation"
            for runtime_event in source.stream_tokens(
                request.session_id,
                count=count,
                transaction_id=transaction.transaction_id,
            ):
                gateway.accept(_gateway_event(runtime_event))
                source.acknowledge_gateway(
                    request.session_id,
                    token_index=runtime_event.token_index,
                    owner_epoch=runtime_event.owner_epoch,
                )
            failed_operation = f"precopy_round_{round_index}_transfer"
            source_delta = source.obtain_dirty_delta(tracking)
            transferred_delta, receipt = _transferred_delta(
                source_delta,
                source_store=source_store,
                destination_store=destination_store,
                transport=transport,
                tenant_id=initial_capture.logical.tenant_id,
                deadline_us=request.transfer_deadline_us,
                seed=request.seed + round_index + 1,
            )
            source_mirror = _merge_source_delta(source_mirror, transferred_delta)
            failed_operation = f"precopy_round_{round_index}_conversion"
            next_converted, destination_delta = _converted_delta(
                source_mirror,
                prior_converted=converted_current,
                source_delta=transferred_delta,
                destination=destination,
                maximum_temporary_bytes=512,
            )
            failed_operation = f"precopy_round_{round_index}_import"
            destination.apply_dirty_delta(request.session_id, destination_delta)
            converted_current = next_converted
            receipts.append(receipt)

        # Make the cutover delta real: this token is generated and gateway-committed
        # after the last iterative delta, then transferred only in the frozen round.
        failed_operation = "cutover_boundary_generation"
        boundary_event = source.generate_token(
            request.session_id,
            transaction_id=transaction.transaction_id,
        )
        gateway.accept(_gateway_event(boundary_event))
        source.acknowledge_gateway(
            request.session_id,
            token_index=boundary_event.token_index,
            owner_epoch=boundary_event.owner_epoch,
        )
        advance(
            CutoverPhase.DELTA_SYNCING,
            CutoverPhase.CUTOVER_REQUESTED,
            "cutover-requested",
        )
        advance(
            CutoverPhase.CUTOVER_REQUESTED,
            CutoverPhase.SOURCE_QUIESCING,
            "source-quiescing",
        )
        cutover_started_ns = time.perf_counter_ns()
        failed_operation = "source_quiesce"
        source.quiesce_at_token_boundary(request.session_id)
        advance(
            CutoverPhase.SOURCE_QUIESCING,
            CutoverPhase.SOURCE_FROZEN,
            "source-frozen",
        )
        source_next_token = source.dry_run_next_token(request.session_id)
        final_delta = source.export_final_delta(tracking)
        if not final_delta.changed_segments or final_delta.to_epoch <= final_delta.from_epoch:
            raise RuntimeError("final quiesced delta must contain cutover-boundary state")
        advance(
            CutoverPhase.SOURCE_FROZEN,
            CutoverPhase.FINAL_DELTA_TRANSFERRING,
            "final-delta-transferring",
        )
        failed_operation = "final_delta_transfer"
        transferred_final, final_receipt = _transferred_delta(
            final_delta,
            source_store=source_store,
            destination_store=destination_store,
            transport=transport,
            tenant_id=initial_capture.logical.tenant_id,
            deadline_us=request.transfer_deadline_us,
            seed=request.seed + len(request.delta_round_token_counts) + 1,
        )
        source_mirror = _merge_source_delta(source_mirror, transferred_final)
        failed_operation = "final_delta_conversion"
        converted_final, destination_final = _converted_delta(
            source_mirror,
            prior_converted=converted_current,
            source_delta=transferred_final,
            destination=destination,
            maximum_temporary_bytes=512,
        )
        failed_operation = "final_delta_import"
        destination.apply_dirty_delta(request.session_id, destination_final)
        receipts.append(final_receipt)
        advance(
            CutoverPhase.FINAL_DELTA_TRANSFERRING,
            CutoverPhase.DESTINATION_IMPORTING,
            "destination-imported-final-delta",
        )
        advance(
            CutoverPhase.DESTINATION_IMPORTING,
            CutoverPhase.DESTINATION_VALIDATING,
            "destination-validating",
        )
        failed_operation = "destination_validation"
        validation = destination.validate_imported_state(request.session_id)
        if validation.dry_run_next_token != source_next_token:
            raise ValueError("source and destination continuation validation disagree")
        cutover = gateway.watermark(request.session_id)
        failed_operation = "source_fencing"
        source.fence_source_writer(
            request.session_id,
            expected_owner_epoch=transaction.source_epoch,
        )
        advance(
            CutoverPhase.DESTINATION_VALIDATING,
            CutoverPhase.COMMIT_INTENT_RECORDED,
            "commit-intent-recorded",
            commit_watermark=cutover,
            rollback_watermark=cutover,
            state_hashes=(validation.imported_logical_hash,),
        )
        logical_clock += 1
        failed_operation = "ownership_commit"
        _record, committed_lease = coordinator.commit_ownership(
            transaction.transaction_id,
            event_id="ownership-committed",
            at_ms=logical_clock,
            state_version=converted_final.logical.state_version,
        )
        advance(
            CutoverPhase.OWNERSHIP_COMMITTED,
            CutoverPhase.GATEWAY_SWITCHING,
            "gateway-switching",
        )
        failed_operation = "gateway_switch"
        gateway.switch_owner(
            session_id=request.session_id,
            expected_epoch=transaction.source_epoch,
            destination_epoch=transaction.proposed_destination_epoch,
            expected_watermark=cutover,
        )
        failed_operation = "destination_activate"
        destination.activate_destination(
            request.session_id,
            committed_owner_epoch=committed_lease.owner_epoch,
            fencing_token=str(committed_lease.fencing_token),
        )
        cutover_finished_ns = time.perf_counter_ns()
        advance(
            CutoverPhase.GATEWAY_SWITCHING,
            CutoverPhase.DESTINATION_ACTIVE,
            "destination-active",
        )
        stale_source_rejected = False
        try:
            source.generate_token(request.session_id, transaction_id=transaction.transaction_id)
        except Exception as error:
            stale_source_rejected = getattr(error, "code", None) == "stale_owner_epoch"
        if not stale_source_rejected:
            raise RuntimeError("fenced source was not rejected after ownership commit")
        failed_operation = "destination_resume"
        for runtime_event in destination.stream_tokens(
            request.session_id,
            count=request.resume_token_count,
            transaction_id=transaction.transaction_id,
        ):
            gateway.accept(_gateway_event(runtime_event))
            destination.acknowledge_gateway(
                request.session_id,
                token_index=runtime_event.token_index,
                owner_epoch=runtime_event.owner_epoch,
            )
        advance(
            CutoverPhase.DESTINATION_ACTIVE,
            CutoverPhase.SOURCE_DRAINING,
            "source-draining",
        )
        accepted = gateway.accepted_tokens(request.session_id)
        indices = tuple(event.token_index for event in accepted)
        if indices != tuple(range(len(indices))):
            raise RuntimeError("gateway accepted a duplicate token or token gap")
        failed_operation = "destination_final_capture"
        destination_capture = destination.capture_consistent(request.session_id)
        source.stop_dirty_tracking(tracking)
        tracking = None
        advance(CutoverPhase.SOURCE_DRAINING, CutoverPhase.COMPLETED, "migration-completed")
        if wall_observation is not None:
            wall_observation.cutover_wall_ns = max(1, cutover_finished_ns - cutover_started_ns)
            wall_observation.total_wall_ns = max(1, time.perf_counter_ns() - migration_started_ns)
        return PrecopyMigrationResult(
            transaction_id=transaction.transaction_id,
            capsule=published.capsule,
            compatibility=compatibility,
            live_conversion_evidence=live_conversion_evidence,
            transfer_receipts=tuple(receipts),
            source_runtime=source.identity.runtime_name,
            destination_runtime=destination.identity.runtime_name,
            source_layout=initial_capture.layout.kind.value,
            destination_layout=destination_capture.layout.kind.value,
            source_owner_epoch=transaction.source_epoch,
            destination_owner_epoch=committed_lease.owner_epoch,
            cutover_token_index=cutover,
            accepted_token_indices=indices,
            source_next_token=source_next_token,
            destination_dry_run_token=validation.dry_run_next_token,
            stale_source_rejected=stale_source_rejected,
            phase_history=tuple(
                entry.to_phase.value for entry in coordinator.journal(transaction.transaction_id)
            ),
        )
    except Exception as error:
        failure_code = str(getattr(error, "code", type(error).__name__)).lower()
        failure_code = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in failure_code
        )[:64]
        cleanup_errors: list[str] = []
        latest = coordinator.transaction(transaction.transaction_id)
        current_lease = coordinator.lease(request.session_id)
        ownership_committed = (
            current_lease.owner_epoch == transaction.proposed_destination_epoch
            and current_lease.owner_runtime == destination.identity.runtime_name
        )

        if ownership_committed:
            if tracking is not None:
                try:
                    source.stop_dirty_tracking(tracking)
                except Exception as cleanup_error:
                    cleanup_errors.append(type(cleanup_error).__name__)
            try:
                source_state = source.inspect_session(request.session_id)
                if source_state.lifecycle is not SessionLifecycle.FENCED:
                    source.fence_source_writer(
                        request.session_id,
                        expected_owner_epoch=transaction.source_epoch,
                    )
            except Exception as cleanup_error:
                cleanup_errors.append(type(cleanup_error).__name__)
            target = (
                CutoverPhase.OPERATOR_REQUIRED
                if failed_operation in {"gateway_switch", "destination_activate"} or cleanup_errors
                else CutoverPhase.FAILED_AFTER_COMMIT
            )
            reason = f"{failed_operation}:{failure_code}"
            if cleanup_errors:
                reason += ":cleanup=" + ",".join(cleanup_errors)
            logical_clock += 1
            coordinator.transition(
                transaction.transaction_id,
                expected=latest.phase,
                target=target,
                event_id=f"postcommit-failure-{failure_code}",
                at_ms=logical_clock,
                payload_hash=_event_hash(reason),
                failure_reason=reason,
            )
            terminal = target
        else:
            logical_clock += 1
            coordinator.transition(
                transaction.transaction_id,
                expected=latest.phase,
                target=CutoverPhase.ABORTING,
                event_id=f"migration-aborting-{failure_code}",
                at_ms=logical_clock,
                payload_hash=_event_hash(f"{failed_operation}:{failure_code}"),
            )
            if tracking is not None:
                try:
                    source.stop_dirty_tracking(tracking)
                except Exception as cleanup_error:
                    cleanup_errors.append(type(cleanup_error).__name__)
            try:
                destination.abort_destination(request.session_id)
            except Exception as cleanup_error:
                cleanup_errors.append(type(cleanup_error).__name__)
            try:
                source_state = source.inspect_session(request.session_id)
                if source_state.lifecycle is SessionLifecycle.FENCED:
                    source_state = source.release_source_fence(
                        request.session_id,
                        expected_owner_epoch=transaction.source_epoch,
                        coordinator_owner_epoch=current_lease.owner_epoch,
                        ownership_committed=False,
                    )
                if source_state.lifecycle is SessionLifecycle.PAUSED:
                    source_state = source.resume_session(
                        request.session_id,
                        expected_owner_epoch=transaction.source_epoch,
                    )
                if source_state.lifecycle is not SessionLifecycle.ACTIVE:
                    raise RuntimeError("source did not return to the active rollback state")
                if gateway.watermark(request.session_id) != source_state.client_visible_index:
                    raise RuntimeError("source and gateway rollback watermarks differ")
                source.dry_run_next_token(request.session_id)
            except Exception as cleanup_error:
                cleanup_errors.append(type(cleanup_error).__name__)
            logical_clock += 1
            if cleanup_errors:
                reason = f"{failed_operation}:{failure_code}:cleanup=" + ",".join(cleanup_errors)
                coordinator.transition(
                    transaction.transaction_id,
                    expected=CutoverPhase.ABORTING,
                    target=CutoverPhase.FAILED_BEFORE_COMMIT,
                    event_id=f"migration-recovery-failed-{failure_code}",
                    at_ms=logical_clock,
                    payload_hash=_event_hash(reason),
                    failure_reason=reason,
                )
                terminal = CutoverPhase.FAILED_BEFORE_COMMIT
            else:
                coordinator.transition(
                    transaction.transaction_id,
                    expected=CutoverPhase.ABORTING,
                    target=CutoverPhase.ROLLED_BACK,
                    event_id=f"migration-rolled-back-{failure_code}",
                    at_ms=logical_clock,
                    payload_hash=_event_hash(f"{failed_operation}:{failure_code}:recovered"),
                )
                terminal = CutoverPhase.ROLLED_BACK
        raise MigrationExecutionError(
            f"migration failed during {failed_operation}; recovery ended in {terminal.value}",
            transaction_id=transaction.transaction_id,
            terminal_phase=terminal,
            ownership_committed=ownership_committed,
            failed_operation=failed_operation,
        ) from error
