"""Semantics-preserving Continuum state-lifecycle characterization harness."""

from __future__ import annotations

import hashlib
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from sloforge.continuum.adapters import (
    CapturedState,
    ReferenceHeadMajorAdapter,
    ReferenceTokenMajorAdapter,
)
from sloforge.continuum.conversion import direct_convert_capture
from sloforge.continuum.migration import (
    MigrationWallObservation,
    PrecopyMigrationRequest,
    migrate_precopy,
)
from sloforge.continuum.operations import (
    CheckpointArtifact,
    checkpoint_full,
    checkpoint_incremental,
    fork_checkpoint,
    restore_reference_capture,
)
from sloforge.continuum.storage import ChunkRef, ContentStore, MemoryContentStore
from sloforge.continuum.transaction import (
    DurableCoordinator,
    GatewayCommitLedger,
    SessionLease,
    TokenEvent,
)
from sloforge.continuum.transport import (
    DeterministicSimulatedTransport,
    InProcessTransport,
    StateTransport,
    TransferReceipt,
)

from .models import (
    CharacterizationResult,
    InstrumentationOverhead,
    ListRecorder,
    MeasurementKind,
    OverheadSample,
    SharingAnalysis,
    StateOperationObservation,
    StateOperationRecorder,
    TraceLevel,
)

_T = TypeVar("_T")
_STAMP = "2026-08-09T00:00:00Z"
_COMMIT = "0" * 40


def _payload_bytes(captured: CapturedState) -> int:
    return sum(segment.descriptor.payload_bytes for segment in captured.segments)


def _page_size_bytes(captured: CapturedState) -> int:
    by_segment = {
        segment.descriptor.segment_id: segment.descriptor.payload_bytes
        for segment in captured.segments
    }
    return max(
        (
            sum(by_segment[segment_id] for segment_id in page.segment_ids)
            for page in captured.page_table
        ),
        default=0,
    )


def _changed_payload_bytes(before: CapturedState, after: CapturedState) -> int:
    before_checksums = {
        segment.descriptor.segment_id: segment.descriptor.checksum for segment in before.segments
    }
    return sum(
        segment.descriptor.payload_bytes
        for segment in after.segments
        if before_checksums.get(segment.descriptor.segment_id) != segment.descriptor.checksum
    )


def _layout(captured: CapturedState) -> str:
    # Reference layouts name simulated GPUs, but all bytes in this harness reside
    # in host memory.  Do not report those logical devices as GPU measurements.
    return f"host_dram/{captured.layout.kind.value}/tp{captured.layout.tensor_parallel_degree}"


def _lease(
    adapter: ReferenceTokenMajorAdapter, session_id: str, *, owner_epoch: int
) -> SessionLease:
    metadata = adapter.inspect_session(session_id)
    return SessionLease(
        session_id=session_id,
        owner_runtime=adapter.identity.runtime_name,
        owner_epoch=owner_epoch,
        fencing_token=owner_epoch,
        expiration_ms=120_000,
        coordinator_version=owner_epoch,
        last_committed_state_version=metadata.state_version,
        last_committed_token_index=metadata.committed_output_index,
    )


@dataclass(slots=True)
class _Emitter:
    recorder: StateOperationRecorder
    level: TraceLevel
    seed: int
    sequence: int = 0

    def measure(
        self,
        operation: str,
        function: Callable[[], _T],
        *,
        logical_state_id: str,
        branch_id: str,
        tenant: str,
        source: str,
        destination: str,
        bytes_count: int,
        alignment: int,
        page_size: int,
        chunk_size: int = 0,
        fanout: int = 1,
        dependencies: tuple[int, ...] = (),
        state_epoch: int = 0,
        measurement_kind: MeasurementKind = MeasurementKind.HARDWARE_BACKED_REAL,
        transfer_time_ns: int = 0,
        evidence: tuple[str, ...],
        metadata: dict[str, Any] | None = None,
        result_bytes: Callable[[_T], int] | None = None,
        result_page_size: Callable[[_T], int] | None = None,
        result_metadata: Callable[[_T], dict[str, Any]] | None = None,
    ) -> _T:
        if self.level is TraceLevel.DISABLED:
            return function()
        started = time.perf_counter_ns()
        cpu_started = time.process_time_ns()
        try:
            result = function()
        except Exception as error:
            ended = time.perf_counter_ns()
            cpu_ended = time.process_time_ns()
            self._emit(
                operation=operation,
                started=started,
                duration=max(1, ended - started),
                cpu_time=max(1, cpu_ended - cpu_started),
                logical_state_id=logical_state_id,
                branch_id=branch_id,
                tenant=tenant,
                source=source,
                destination=destination,
                bytes_count=bytes_count,
                alignment=alignment,
                page_size=page_size,
                chunk_size=chunk_size,
                fanout=fanout,
                dependencies=dependencies,
                state_epoch=state_epoch,
                measurement_kind=measurement_kind,
                transfer_time_ns=transfer_time_ns,
                result="failure",
                failure=type(error).__name__,
                evidence=evidence,
                metadata=metadata,
            )
            raise
        ended = time.perf_counter_ns()
        cpu_ended = time.process_time_ns()
        if result_bytes is not None:
            bytes_count = result_bytes(result)
        if result_page_size is not None:
            page_size = result_page_size(result)
        if result_metadata is not None:
            metadata = {**(metadata or {}), **result_metadata(result)}
        self._emit(
            operation=operation,
            started=started,
            duration=max(1, ended - started),
            cpu_time=max(1, cpu_ended - cpu_started),
            logical_state_id=logical_state_id,
            branch_id=branch_id,
            tenant=tenant,
            source=source,
            destination=destination,
            bytes_count=bytes_count,
            alignment=alignment,
            page_size=page_size,
            chunk_size=chunk_size,
            fanout=fanout,
            dependencies=dependencies,
            state_epoch=state_epoch,
            measurement_kind=measurement_kind,
            transfer_time_ns=transfer_time_ns,
            result="success",
            failure=None,
            evidence=evidence,
            metadata=metadata,
        )
        return result

    def observed(
        self,
        operation: str,
        *,
        started: int,
        duration: int,
        cpu_time: int,
        logical_state_id: str,
        branch_id: str,
        tenant: str,
        source: str,
        destination: str,
        bytes_count: int,
        alignment: int,
        page_size: int,
        state_epoch: int,
        evidence: tuple[str, ...],
        metadata: dict[str, Any],
    ) -> None:
        if self.level is TraceLevel.DISABLED:
            return
        self._emit(
            operation=operation,
            started=started,
            duration=duration,
            cpu_time=cpu_time,
            logical_state_id=logical_state_id,
            branch_id=branch_id,
            tenant=tenant,
            source=source,
            destination=destination,
            bytes_count=bytes_count,
            alignment=alignment,
            page_size=page_size,
            chunk_size=0,
            fanout=1,
            dependencies=(),
            state_epoch=state_epoch,
            measurement_kind=MeasurementKind.HARDWARE_BACKED_REAL,
            transfer_time_ns=0,
            result="success",
            failure=None,
            evidence=evidence,
            metadata=metadata,
        )

    def _emit(self, **values: Any) -> None:
        if self.level is TraceLevel.DISABLED:
            return
        metadata = dict(values.pop("metadata") or {})
        if self.level is TraceLevel.MINIMAL:
            metadata = {
                key: value
                for key, value in metadata.items()
                if key
                in {
                    "timing_scope",
                    "transport",
                    "changed_segment_count",
                    "modeled_transport",
                }
            }
        observation = StateOperationObservation(
            sequence=self.sequence,
            monotonic_start_ns=values.pop("started"),
            duration_ns=values.pop("duration"),
            cpu_time_ns=values.pop("cpu_time"),
            bytes=values.pop("bytes_count"),
            tenant_security_domain=values.pop("tenant"),
            source_physical_representation=values.pop("source"),
            destination_physical_representation=values.pop("destination"),
            queue_delay_ns=0,
            concurrency=1,
            seed=self.seed,
            workload_evidence_class=MeasurementKind.SYNTHETIC,
            metadata=metadata,
            evidence_references=values.pop("evidence"),
            **values,
        )
        self.sequence += 1
        self.recorder.record(observation)


class _RecordingTransport:
    """Times a real transport call and emits separate send/receive observations."""

    def __init__(
        self,
        wrapped: StateTransport,
        *,
        emitter: _Emitter,
        branch_id: str,
        logical_state_id: str,
        measurement_kind: MeasurementKind,
    ) -> None:
        self.wrapped = wrapped
        self.capabilities = wrapped.capabilities
        self.emitter = emitter
        self.branch_id = branch_id
        self.logical_state_id = logical_state_id
        self.measurement_kind = measurement_kind

    def transfer(
        self,
        *,
        source: ContentStore,
        destination: ContentStore,
        tenant_id: str,
        references: Sequence[ChunkRef],
        deadline_us: int,
        seed: int,
        cancelled: bool = False,
    ) -> TransferReceipt:
        if self.emitter.level is TraceLevel.DISABLED:
            return self.wrapped.transfer(
                source=source,
                destination=destination,
                tenant_id=tenant_id,
                references=references,
                deadline_us=deadline_us,
                seed=seed,
                cancelled=cancelled,
            )
        started = time.perf_counter_ns()
        cpu_started = time.process_time_ns()
        receipt = self.wrapped.transfer(
            source=source,
            destination=destination,
            tenant_id=tenant_id,
            references=references,
            deadline_us=deadline_us,
            seed=seed,
            cancelled=cancelled,
        )
        duration = max(1, time.perf_counter_ns() - started)
        cpu_time = max(1, time.process_time_ns() - cpu_started)
        common = dict(
            logical_state_id=self.logical_state_id,
            branch_id=self.branch_id,
            tenant=tenant_id,
            bytes_count=receipt.bytes_on_wire,
            alignment=1,
            page_size=0,
            chunk_size=max((reference.stored_bytes for reference in references), default=0),
            state_epoch=0,
            measurement_kind=self.measurement_kind,
            transfer_time_ns=receipt.elapsed_us * 1_000,
            evidence=(
                "python/sloforge/continuum/transport/core.py:StateTransport.transfer",
                "python/sloforge/continuum/transport/core.py:TransferReceipt",
            ),
            metadata={
                "transport": receipt.transport,
                "modeled_transport": self.measurement_kind is MeasurementKind.SIMULATED_HARDWARE,
                "host_call_duration_ns": duration,
                "unique_plaintext_bytes": receipt.unique_plaintext_bytes,
                "bytes_on_wire": receipt.bytes_on_wire,
                "source_chunks": receipt.source_chunks,
                "retransmissions": receipt.retransmissions,
            },
        )
        self.emitter._emit(
            operation="STATE_SEND",
            started=started,
            duration=duration,
            cpu_time=cpu_time,
            source="host_dram",
            destination=f"transport/{receipt.transport}",
            fanout=1,
            dependencies=(),
            result="success",
            failure=None,
            **common,
        )
        self.emitter._emit(
            operation="STATE_RECEIVE",
            started=started,
            duration=duration,
            cpu_time=cpu_time,
            source=f"transport/{receipt.transport}",
            destination="host_dram",
            fanout=1,
            dependencies=(self.emitter.sequence - 1,),
            result="success",
            failure=None,
            **common,
        )
        return receipt


class _RecordingCoordinator(DurableCoordinator):
    """Records only the durable ownership CAS, leaving coordinator logic intact."""

    def __init__(self, emitter: _Emitter) -> None:
        super().__init__(":memory:")
        self._characterization_emitter = emitter

    def commit_ownership(
        self, identifier: str, *, event_id: str, at_ms: int, state_version: int
    ) -> tuple[Any, SessionLease]:
        before = self.transaction(identifier)
        return self._characterization_emitter.measure(
            "STATE_COMMIT",
            lambda: super(_RecordingCoordinator, self).commit_ownership(
                identifier,
                event_id=event_id,
                at_ms=at_ms,
                state_version=state_version,
            ),
            logical_state_id=identifier,
            branch_id=before.session_id,
            tenant="tenant-a",
            source="host_dram/sqlite-transaction",
            destination="host_dram/sqlite-lease",
            bytes_count=0,
            alignment=1,
            page_size=0,
            state_epoch=state_version,
            evidence=(
                "python/sloforge/continuum/transaction/coordinator.py:DurableCoordinator.commit_ownership",
            ),
            metadata={"timing_scope": "durable_ownership_compare_and_swap"},
        )


def _gateway_event(event: Any) -> TokenEvent:
    return TokenEvent(
        session_id=event.session_id,
        owner_epoch=event.owner_epoch,
        token_index=event.token_index,
        token_id=event.token_id,
        state_commit_version=event.state_commit_version,
        transaction_id=event.transaction_id,
    )


def _sharing_analysis(
    branches: tuple[CheckpointArtifact, ...],
    *,
    shared_digests: tuple[str, ...],
    before: CapturedState,
    after: CapturedState,
) -> SharingAnalysis:
    sizes: dict[str, int] = {}
    branch_digests: list[set[str]] = []
    metadata_sizes: list[int] = []
    for artifact in branches:
        branch_digests.append({reference.digest for reference in artifact.chunk_references})
        sizes.update(
            {reference.digest: reference.size_bytes for reference in artifact.chunk_references}
        )
        metadata_sizes.append(len(artifact.capsule.model_dump_json().encode("utf-8")))
    shared = set(shared_digests)
    shared_bytes = sum(sizes[digest] for digest in shared)
    private = tuple(sum(sizes[digest] for digest in digests - shared) for digests in branch_digests)
    naive = sum(sum(sizes[digest] for digest in digests) for digests in branch_digests)
    unique_digests = set().union(*branch_digests)
    unique_bytes = sum(sizes[digest] for digest in unique_digests)

    before_segments = {
        segment.descriptor.segment_id: segment.descriptor.checksum for segment in before.segments
    }
    changed_segments = {
        segment.descriptor.segment_id
        for segment in after.segments
        if before_segments.get(segment.descriptor.segment_id) != segment.descriptor.checksum
    }
    changed_pages = tuple(
        page for page in after.page_table if changed_segments.intersection(page.segment_ids)
    )
    changed_page_segments = {
        segment_id for page in changed_pages for segment_id in page.segment_ids
    }
    cow_bytes = sum(
        segment.descriptor.payload_bytes
        for segment in after.segments
        if segment.descriptor.segment_id in changed_page_segments
    )
    source_refcount = max((page.copy_on_write_refs for page in before.page_table), default=1)
    return SharingAnalysis(
        branch_count=len(branches),
        shared_logical_bytes=shared_bytes,
        physical_allocated_bytes=unique_bytes,
        logical_unique_bytes=unique_bytes,
        naive_independent_allocation_bytes=naive,
        private_bytes_per_branch=private,
        metadata_bytes_per_branch=tuple(metadata_sizes),
        cow_bytes=cow_bytes,
        cow_page_count=len(changed_pages),
        source_page_refcount=source_refcount,
        physical_amplification=unique_bytes / unique_bytes if unique_bytes else 1.0,
        sharing_efficiency=1.0 - unique_bytes / naive if naive else 0.0,
        evidence_references=(
            "python/sloforge/continuum/operations/branching.py:fork_checkpoint",
            "python/sloforge/continuum/storage/content_store.py:MemoryContentStore.put",
            "python/sloforge/continuum/adapters/sdk.py:PageTableEntry.copy_on_write_refs",
        ),
    )


def run_continuum_characterization(
    *,
    seed: int,
    trace_level: TraceLevel = TraceLevel.FULL,
    recorder: StateOperationRecorder | None = None,
) -> CharacterizationResult:
    """Exercise fork, first write, delta, checkpoint, conversion, transfer, and migration.

    State bytes and semantic results come from real Continuum artifacts.  Reference
    runtime computation is measured on the host CPU.  The migration network model
    is always marked ``SIMULATED_HARDWARE`` even though it moves and verifies real
    fixture bytes.
    """

    if not 0 <= seed <= 2**64 - 1:
        raise ValueError("seed must fit uint64")
    sink = recorder or ListRecorder()
    emitter = _Emitter(recorder=sink, level=trace_level, seed=seed)
    source = ReferenceTokenMajorAdapter(page_size_tokens=3)
    session_id = "characterization-root"
    source.create_session(
        session_id=session_id,
        request_id="continuum-characterization",
        tenant_id="tenant-a",
        input_token_ids=(2, 3, 5, 7),
        seed=seed,
    )
    for event in source.stream_tokens(session_id, count=6):
        source.acknowledge_gateway(
            session_id, token_index=event.token_index, owner_epoch=event.owner_epoch
        )

    initial = emitter.measure(
        "STATE_SNAPSHOT",
        lambda: source.capture_consistent(session_id),
        logical_state_id=session_id,
        branch_id=session_id,
        tenant="tenant-a",
        source="host_dram/runtime",
        destination="host_dram/snapshot",
        bytes_count=0,
        alignment=source.config.layout.alignment_bytes,
        page_size=0,
        state_epoch=source.inspect_session(session_id).state_version,
        evidence=(
            "python/sloforge/continuum/adapters/sdk.py:ContinuumRuntimeAdapter.capture_consistent",
        ),
        metadata={
            "timing_scope": "consistent_snapshot_capture",
            "logical_page_size_tokens": source.config.layout.page_size_tokens,
        },
        result_bytes=_payload_bytes,
        result_page_size=_page_size_bytes,
        result_metadata=lambda captured: {
            "segment_count": len(captured.segments),
            "page_count": len(captured.page_table),
        },
    )
    store = MemoryContentStore()
    root = emitter.measure(
        "STATE_PUBLISH",
        lambda: checkpoint_full(
            source,
            session_id,
            store=store,
            lease=_lease(source, session_id, owner_epoch=1),
            published_at_ms=1,
            capture_timestamp=_STAMP,
            git_commit=_COMMIT,
            continuum_version="0.1.0",
        ),
        logical_state_id=initial.logical.continuation_hash,
        branch_id=session_id,
        tenant="tenant-a",
        source=_layout(initial),
        destination="host_dram/content-addressed-store",
        bytes_count=_payload_bytes(initial),
        alignment=initial.layout.alignment_bytes,
        page_size=_page_size_bytes(initial),
        state_epoch=initial.logical.state_version,
        evidence=("python/sloforge/continuum/operations/checkpoint.py:checkpoint_full",),
        metadata={"timing_scope": "capture_hash_store_and_manifest_publish"},
    )
    watermark = source.inspect_session(session_id).committed_output_index
    branch_ids = ("characterization-branch-a", "characterization-branch-b")
    leases = tuple(
        SessionLease(
            session_id=branch_id,
            owner_runtime=source.identity.runtime_name,
            owner_epoch=1,
            fencing_token=1,
            expiration_ms=120_000,
            coordinator_version=1,
            last_committed_state_version=initial.logical.state_version,
            last_committed_token_index=watermark,
        )
        for branch_id in branch_ids
    )
    forked = emitter.measure(
        "STATE_FORK",
        lambda: fork_checkpoint(
            root,
            store=store,
            expected_tenant_id="tenant-a",
            expected_model=source.config.model,
            branch_leases=leases,
            seed=seed,
            published_at_ms=10,
            capture_timestamp=_STAMP,
            git_commit=_COMMIT,
            continuum_version="0.1.0",
        ),
        logical_state_id=initial.logical.continuation_hash,
        branch_id=session_id,
        tenant="tenant-a",
        source="host_dram/content-addressed-store",
        destination="host_dram/content-addressed-store",
        bytes_count=sum(reference.size_bytes for reference in root.chunk_references),
        alignment=initial.layout.alignment_bytes,
        page_size=_page_size_bytes(initial),
        fanout=len(branch_ids),
        state_epoch=initial.logical.state_version,
        evidence=("python/sloforge/continuum/operations/branching.py:fork_checkpoint",),
        metadata={"timing_scope": "restore_encode_and_publish_all_descendants"},
    )

    branch_artifact = forked.branches[0]
    branch_capture = restore_reference_capture(
        branch_artifact,
        store=store,
        expected_tenant_id="tenant-a",
        expected_model=source.config.model,
    )
    branch_runtime = ReferenceTokenMajorAdapter(page_size_tokens=3)
    branch_runtime.prepare_destination_session(
        branch_capture,
        destination_session_id=branch_ids[0],
        proposed_owner_epoch=2,
    )
    branch_runtime.import_captured_state(branch_ids[0], branch_capture)
    branch_runtime.validate_imported_state(branch_ids[0])
    branch_runtime.activate_destination(
        branch_ids[0], committed_owner_epoch=2, fencing_token="characterization-branch"
    )
    tracking = branch_runtime.start_dirty_tracking(branch_ids[0])
    if trace_level is TraceLevel.DISABLED:
        cow_started = cow_cpu_started = cow_duration = cow_cpu_time = 0
        generated = branch_runtime.generate_token(branch_ids[0])
        branch_runtime.acknowledge_gateway(
            branch_ids[0], token_index=generated.token_index, owner_epoch=generated.owner_epoch
        )
    else:
        cow_started = time.perf_counter_ns()
        cow_cpu_started = time.process_time_ns()
        generated = branch_runtime.generate_token(branch_ids[0])
        branch_runtime.acknowledge_gateway(
            branch_ids[0], token_index=generated.token_index, owner_epoch=generated.owner_epoch
        )
        cow_duration = max(1, time.perf_counter_ns() - cow_started)
        cow_cpu_time = max(1, time.process_time_ns() - cow_cpu_started)
    divergent = branch_runtime.capture_consistent(branch_ids[0])
    sharing = _sharing_analysis(
        forked.branches,
        shared_digests=forked.shared_immutable_digests,
        before=branch_capture,
        after=divergent,
    )
    emitter.observed(
        "STATE_COW",
        started=cow_started,
        duration=cow_duration,
        cpu_time=cow_cpu_time,
        logical_state_id=branch_capture.logical.continuation_hash,
        branch_id=branch_ids[0],
        tenant="tenant-a",
        source=_layout(branch_capture),
        destination=_layout(divergent),
        bytes_count=sharing.cow_bytes,
        alignment=divergent.layout.alignment_bytes,
        page_size=_page_size_bytes(divergent),
        state_epoch=divergent.logical.state_version,
        evidence=(
            "python/sloforge/continuum/adapters/sdk.py:PageTableEntry.copy_on_write_refs",
            "python/sloforge/continuum/reference/runtime.py:DeterministicHybridRuntimeAdapter.generate_token",
        ),
        metadata={
            "timing_scope": "inclusive_first_private_generate_and_gateway_acknowledgment",
            "cow_implementation": ("logical_page_refcount_attribution_not_an_os_or_gpu_page_fault"),
            "source_refcount": sharing.source_page_refcount,
            "cow_pages": sharing.cow_page_count,
        },
    )
    emitter.observed(
        "STATE_APPEND",
        started=cow_started,
        duration=cow_duration,
        cpu_time=cow_cpu_time,
        logical_state_id=divergent.logical.continuation_hash,
        branch_id=branch_ids[0],
        tenant="tenant-a",
        source=_layout(branch_capture),
        destination=_layout(divergent),
        bytes_count=_changed_payload_bytes(branch_capture, divergent),
        alignment=divergent.layout.alignment_bytes,
        page_size=_page_size_bytes(divergent),
        state_epoch=divergent.logical.state_version,
        evidence=(
            "python/sloforge/continuum/reference/runtime.py:DeterministicHybridRuntimeAdapter.generate_token",
        ),
        metadata={
            "timing_scope": "same_inclusive_first_private_write_as_STATE_COW",
            "logical_growth_bytes": max(
                0, _payload_bytes(divergent) - _payload_bytes(branch_capture)
            ),
            "physical_dirty_page_bytes": sharing.cow_bytes,
        },
    )
    emitter.measure(
        "STATE_DELTA",
        lambda: branch_runtime.obtain_dirty_delta(tracking),
        logical_state_id=divergent.logical.continuation_hash,
        branch_id=branch_ids[0],
        tenant="tenant-a",
        source=_layout(divergent),
        destination="host_dram/dirty-delta",
        bytes_count=0,
        alignment=divergent.layout.alignment_bytes,
        page_size=_page_size_bytes(divergent),
        state_epoch=divergent.logical.state_version,
        evidence=(
            "python/sloforge/continuum/reference/runtime.py:DeterministicHybridRuntimeAdapter.obtain_dirty_delta",
        ),
        metadata={"timing_scope": "dirty_delta_extraction"},
        result_bytes=lambda result: sum(
            segment.descriptor.payload_bytes for segment in result.changed_segments
        ),
        result_metadata=lambda result: {
            "changed_segment_count": len(result.changed_segments),
            "from_epoch": result.from_epoch,
            "to_epoch": result.to_epoch,
            "final": result.final,
        },
    )
    branch_runtime.stop_dirty_tracking(tracking)

    branch_checkpoint = emitter.measure(
        "STATE_SNAPSHOT",
        lambda: checkpoint_incremental(
            branch_runtime,
            branch_ids[0],
            store=store,
            lease=_lease(branch_runtime, branch_ids[0], owner_epoch=2),
            parent=branch_artifact,
            published_at_ms=20,
            capture_timestamp=_STAMP,
            git_commit=_COMMIT,
            continuum_version="0.1.0",
        ),
        logical_state_id=divergent.logical.continuation_hash,
        branch_id=branch_ids[0],
        tenant="tenant-a",
        source=_layout(divergent),
        destination="host_dram/content-addressed-store",
        bytes_count=_payload_bytes(divergent),
        alignment=divergent.layout.alignment_bytes,
        page_size=_page_size_bytes(divergent),
        state_epoch=divergent.logical.state_version,
        evidence=("python/sloforge/continuum/operations/checkpoint.py:checkpoint_incremental",),
        metadata={"timing_scope": "incremental_capture_hash_store_and_publish"},
    )

    conversion_destination = ReferenceHeadMajorAdapter(page_size_tokens=5)
    converted, conversion_evidence = emitter.measure(
        "STATE_RESHARD",
        lambda: direct_convert_capture(
            divergent, destination=conversion_destination, maximum_temporary_bytes=512
        ),
        logical_state_id=divergent.logical.continuation_hash,
        branch_id=branch_ids[0],
        tenant="tenant-a",
        source=_layout(divergent),
        destination="host_dram/paged_head_major_packed_kv/tp2",
        bytes_count=_payload_bytes(divergent),
        alignment=divergent.layout.alignment_bytes,
        page_size=_page_size_bytes(divergent),
        state_epoch=divergent.logical.state_version,
        evidence=("python/sloforge/continuum/conversion/adapter_bridge.py:direct_convert_capture",),
        metadata={
            "timing_scope": "fused_reshard_repack_and_independent_verification",
            "operation_chain": ("STATE_RESHARD", "STATE_REPACK", "STATE_CHECKSUM"),
        },
    )
    if not conversion_evidence.canonical_attention_match:
        raise RuntimeError("characterization conversion did not preserve attention state")

    destination_store = MemoryContentStore()
    real_transport = _RecordingTransport(
        InProcessTransport(),
        emitter=emitter,
        branch_id=branch_ids[0],
        logical_state_id=converted.logical.continuation_hash,
        measurement_kind=MeasurementKind.HARDWARE_BACKED_REAL,
    )
    real_transport.transfer(
        source=store,
        destination=destination_store,
        tenant_id="tenant-a",
        references=branch_checkpoint.chunk_references,
        deadline_us=60_000_000,
        seed=seed,
    )

    # A separate real Continuum pre-copy transaction supplies lifecycle coverage.
    # Only its network timing is modeled; real fixture bytes are moved and all host
    # execution/ownership CAS timings are measured.
    migration_source = ReferenceTokenMajorAdapter(page_size_tokens=3)
    migration_destination = ReferenceHeadMajorAdapter(page_size_tokens=5)
    migration_id = "characterization-migration"
    migration_source.create_session(
        session_id=migration_id,
        request_id="continuum-characterization-migration",
        tenant_id="tenant-a",
        input_token_ids=(11, 13, 17),
        seed=seed + 1,
    )
    migration_gateway = GatewayCommitLedger(":memory:")
    migration_gateway.register(session_id=migration_id, owner_epoch=1)
    for runtime_event in migration_source.stream_tokens(migration_id, count=3):
        migration_gateway.accept(_gateway_event(runtime_event))
        migration_source.acknowledge_gateway(
            migration_id,
            token_index=runtime_event.token_index,
            owner_epoch=runtime_event.owner_epoch,
        )
    migration_store = MemoryContentStore()
    migration_destination_store = MemoryContentStore()
    coordinator = _RecordingCoordinator(emitter)
    coordinator.create_lease(
        session_id=migration_id,
        owner_runtime=migration_source.identity.runtime_name,
        expiration_ms=120_000,
        initial_token_index=2,
    )
    simulated_transport = _RecordingTransport(
        DeterministicSimulatedTransport(
            bandwidth_bytes_per_second=10_000_000,
            latency_us=25,
        ),
        emitter=emitter,
        branch_id=migration_id,
        logical_state_id=migration_id,
        measurement_kind=MeasurementKind.SIMULATED_HARDWARE,
    )
    wall = MigrationWallObservation()
    try:
        with coordinator, migration_gateway:
            migration = migrate_precopy(
                PrecopyMigrationRequest(
                    session_id=migration_id,
                    seed=seed + 1,
                    plan_hash=hashlib.sha256(b"continuum-characterization-precopy").hexdigest(),
                    delta_round_token_counts=(2,),
                    resume_token_count=2,
                    capture_timestamp=_STAMP,
                    git_commit=_COMMIT,
                    continuum_version="0.1.0",
                ),
                source=migration_source,
                destination=migration_destination,
                coordinator=coordinator,
                gateway=migration_gateway,
                source_store=migration_store,
                destination_store=migration_destination_store,
                transport=simulated_transport,
                wall_observation=wall,
            )
    finally:
        # Context managers are idempotent, but the explicit finally makes resource
        # lifetime evident if a future migration failure path changes.
        coordinator.close()

    reclaim_bytes = sum(reference.size_bytes for reference in branch_checkpoint.chunk_references)
    emitter.measure(
        "STATE_RECLAIM",
        lambda: store.delete_manifest(
            branch_checkpoint.store_manifest.tenant_id,
            branch_checkpoint.store_manifest.manifest_id,
        ),
        logical_state_id=divergent.logical.continuation_hash,
        branch_id=branch_ids[0],
        tenant="tenant-a",
        source="host_dram/content-addressed-store",
        destination="host_dram/unreferenced-or-shared-chunks",
        bytes_count=reclaim_bytes,
        alignment=1,
        page_size=0,
        state_epoch=divergent.logical.state_version,
        evidence=(
            "python/sloforge/continuum/storage/content_store.py:MemoryContentStore.delete_manifest",
        ),
        metadata={
            "timing_scope": "manifest_reference_release",
            "physical_bytes_freed": 0,
            "reason": "parent_and_sibling_manifests_retain_shared_chunks",
        },
    )

    events = tuple(sink.events) if isinstance(sink, ListRecorder) else ()
    dropped = sink.dropped_events if isinstance(sink, ListRecorder) else 0
    return CharacterizationResult(
        seed=seed,
        workload_evidence_class=MeasurementKind.SYNTHETIC,
        trace_level=trace_level,
        events=events,
        dropped_events=dropped,
        sharing=sharing,
        source_continuation_hash=initial.logical.continuation_hash,
        divergent_continuation_hash=divergent.logical.continuation_hash,
        migration_transaction_id=migration.transaction_id,
        migration_transport_kind=MeasurementKind.SIMULATED_HARDWARE,
    )


def measure_instrumentation_overhead(*, seed: int, repetitions: int = 3) -> InstrumentationOverhead:
    """Measure disabled/minimal/full modes in deterministic randomized order."""

    if not 1 <= repetitions <= 100:
        raise ValueError("repetitions must be within 1..100")
    order = [level for _ in range(repetitions) for level in TraceLevel]
    random.Random(seed).shuffle(order)
    counts = {level: 0 for level in TraceLevel}
    samples: list[OverheadSample] = []
    for level in order:
        repetition = counts[level]
        counts[level] += 1
        recorder = ListRecorder()
        started = time.perf_counter_ns()
        cpu_started = time.process_time_ns()
        result = run_continuum_characterization(
            seed=seed + repetition, trace_level=level, recorder=recorder
        )
        samples.append(
            OverheadSample(
                trace_level=level,
                repetition=repetition,
                seed=seed + repetition,
                duration_ns=max(1, time.perf_counter_ns() - started),
                cpu_time_ns=max(1, time.process_time_ns() - cpu_started),
                event_count=len(result.events),
                dropped_events=result.dropped_events,
            )
        )
    medians = {
        level.value: int(
            statistics.median(
                sample.duration_ns for sample in samples if sample.trace_level is level
            )
        )
        for level in TraceLevel
    }
    baseline = medians[TraceLevel.DISABLED.value]
    overhead = {level.value: (medians[level.value] - baseline) / baseline for level in TraceLevel}
    return InstrumentationOverhead(
        seed=seed,
        repetitions=repetitions,
        randomized_order=tuple(level.value for level in order),
        raw_samples=tuple(samples),
        median_duration_ns=medians,
        duration_overhead_fraction=overhead,
    )
