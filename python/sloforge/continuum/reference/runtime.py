"""Deterministic, stateful CPU runtime implementing the Continuum adapter SDK."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock

from sloforge.continuum.adapters.sdk import (
    AdapterUnavailableError,
    CapabilityMatrix,
    CapturedState,
    ContinuumRuntimeAdapter,
    DirtyDelta,
    DirtyLogOverflowError,
    DirtyTrackingHandle,
    DirtyTrackingStrategy,
    FailurePoint,
    FailureRule,
    ImportValidation,
    ImportValidationError,
    InjectedFailureError,
    LogicalStateManifest,
    PageTableEntry,
    ResourceLimitError,
    RuntimeCapability,
    RuntimeIdentity,
    RuntimeLayout,
    SegmentDescriptor,
    SegmentIntegrityError,
    SessionLifecycle,
    SessionMetadata,
    SessionNotFoundError,
    SessionStateError,
    SnapshotConsistencyError,
    SnapshotHandle,
    StaleDeltaError,
    StaleOwnerEpochError,
    StateKind,
    StateSegment,
    TokenEvent,
)
from sloforge.continuum.reference.codec import EncodedState, decode_segments, encode_state
from sloforge.continuum.reference.models import HybridDecoderConfig, HybridDecoderState


@dataclass(slots=True)
class _SnapshotRecord:
    handle: SnapshotHandle
    encoded: EncodedState


@dataclass(slots=True)
class _TrackingRecord:
    tracking_id: str
    session_id: str
    baseline_epoch: int
    last_exported_epoch: int


@dataclass(slots=True)
class _StagingRecord:
    destination_session_id: str
    source_session_id: str
    proposed_owner_epoch: int
    source_runtime: RuntimeIdentity
    source_layout: RuntimeLayout
    source_segments: dict[str, StateSegment]
    source_page_table: tuple[PageTableEntry, ...]
    source_logical: LogicalStateManifest
    imported_state: HybridDecoderState | None
    expected_continuation_hash: str | None
    source_dirty_epoch: int
    validated: bool


@dataclass(slots=True)
class _FailureState:
    rule: FailureRule
    calls: int = 0


class DeterministicHybridRuntimeAdapter(ContinuumRuntimeAdapter):
    """In-process runtime with explicit state, bounded resources, and no fallback path."""

    def __init__(
        self,
        *,
        identity: RuntimeIdentity,
        config: HybridDecoderConfig,
        max_sessions: int = 32,
        max_open_snapshots: int = 16,
        max_failure_rules: int = 32,
        max_stream_batch_tokens: int = 256,
    ) -> None:
        if (
            min(
                max_sessions,
                max_open_snapshots,
                max_failure_rules,
                max_stream_batch_tokens,
            )
            <= 0
        ):
            raise ValueError("all runtime resource bounds must be positive")
        self._config = config
        self._capabilities = CapabilityMatrix(
            runtime=identity,
            operations=frozenset(RuntimeCapability),
            state_types=frozenset(StateKind),
            layouts=(config.layout,),
            dirty_tracking_strategies=frozenset(
                {
                    DirtyTrackingStrategy.EXPLICIT_SEGMENT_VERSIONING,
                    DirtyTrackingStrategy.APPEND_LOG,
                }
            ),
            max_sessions=max_sessions,
            max_open_snapshots=max_open_snapshots,
            max_dirty_events=config.max_dirty_events,
            max_stream_buffer_tokens=max_stream_batch_tokens,
        )
        self._max_failure_rules = max_failure_rules
        self._lock = RLock()
        self._sessions: dict[str, HybridDecoderState] = {}
        self._snapshots: dict[str, _SnapshotRecord] = {}
        self._tracking: dict[str, _TrackingRecord] = {}
        self._staging: dict[str, _StagingRecord] = {}
        self._dirty_history: dict[str, deque[int]] = {}
        self._overflow_floor: dict[str, int] = {}
        self._fencing_token_hashes: dict[str, str] = {}
        self._failure_rules: list[_FailureState] = []
        self._id_counter = 0
        self._available = True

    @property
    def capabilities(self) -> CapabilityMatrix:
        return self._capabilities

    @property
    def config(self) -> HybridDecoderConfig:
        return self._config

    def _ensure_available(self, operation: str, session_id: str | None = None) -> None:
        if not self._available:
            raise AdapterUnavailableError(
                "reference runtime is unavailable after a simulated crash",
                operation=operation,
                session_id=session_id,
            )

    def _next_id(self, label: str, session_id: str) -> str:
        self._id_counter += 1
        material = (
            f"{self.identity.runtime_name}|{self.identity.adapter_version}|{label}|"
            f"{session_id}|{self._id_counter}"
        )
        return sha256(material.encode("utf-8")).hexdigest()

    def _hit_failure(self, point: FailurePoint, session_id: str | None) -> None:
        for failure in tuple(self._failure_rules):
            rule = failure.rule
            if rule.point is not point:
                continue
            if rule.session_id is not None and rule.session_id != session_id:
                continue
            failure.calls += 1
            if failure.calls < rule.trigger_on_call:
                continue
            if not rule.repeat:
                self._failure_rules.remove(failure)
            raise InjectedFailureError(point, session_id=session_id)

    def _session(self, session_id: str, operation: str) -> HybridDecoderState:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise SessionNotFoundError(
                "session does not exist",
                operation=operation,
                session_id=session_id,
            ) from error

    def _metadata(self, state: HybridDecoderState) -> SessionMetadata:
        return SessionMetadata(
            session_id=state.session_id,
            request_id=state.request_id,
            tenant_id=state.tenant_id,
            lifecycle=state.lifecycle,
            owner_epoch=state.owner_epoch,
            state_version=state.state_version,
            committed_output_index=state.gateway_committed_index,
            client_visible_index=state.gateway_committed_index,
            layout=self._config.layout,
            model=self._config.model,
        )

    def _record_dirty(self, session_id: str, epoch: int) -> None:
        history = self._dirty_history[session_id]
        if len(history) == history.maxlen:
            discarded = history.popleft()
            self._overflow_floor[session_id] = max(self._overflow_floor[session_id], discarded)
        history.append(epoch)

    def create_session(
        self,
        *,
        session_id: str,
        request_id: str,
        tenant_id: str,
        input_token_ids: tuple[int, ...],
        seed: int,
        owner_epoch: int = 1,
    ) -> SessionMetadata:
        with self._lock:
            self._ensure_available(FailurePoint.CREATE_SESSION.value, session_id)
            self._hit_failure(FailurePoint.CREATE_SESSION, session_id)
            if session_id in self._sessions or session_id in self._staging:
                raise SessionStateError(
                    "session identifier is already in use",
                    operation=FailurePoint.CREATE_SESSION.value,
                    session_id=session_id,
                )
            if len(self._sessions) + len(self._staging) >= self.capabilities.max_sessions:
                raise ResourceLimitError(
                    "runtime session bound reached",
                    operation=FailurePoint.CREATE_SESSION.value,
                    session_id=session_id,
                )
            try:
                state = HybridDecoderState.create(
                    self._config,
                    session_id=session_id,
                    request_id=request_id,
                    tenant_id=tenant_id,
                    seed=seed,
                    owner_epoch=owner_epoch,
                    input_token_ids=input_token_ids,
                )
            except ValueError as error:
                raise SessionStateError(
                    str(error),
                    operation=FailurePoint.CREATE_SESSION.value,
                    session_id=session_id,
                ) from error
            self._sessions[session_id] = state
            self._dirty_history[session_id] = deque(maxlen=self.capabilities.max_dirty_events)
            self._overflow_floor[session_id] = 0
            return self._metadata(state)

    def inspect_session(self, session_id: str) -> SessionMetadata:
        with self._lock:
            self._ensure_available(RuntimeCapability.INSPECT.value, session_id)
            return self._metadata(self._session(session_id, RuntimeCapability.INSPECT.value))

    def list_sessions(self) -> tuple[SessionMetadata, ...]:
        with self._lock:
            self._ensure_available(RuntimeCapability.INSPECT.value)
            return tuple(self._metadata(self._sessions[key]) for key in sorted(self._sessions))

    def generate_token(self, session_id: str, *, transaction_id: str | None = None) -> TokenEvent:
        with self._lock:
            self._ensure_available(FailurePoint.GENERATE.value, session_id)
            self._hit_failure(FailurePoint.GENERATE, session_id)
            state = self._session(session_id, FailurePoint.GENERATE.value)
            if state.lifecycle is SessionLifecycle.FENCED:
                raise StaleOwnerEpochError(
                    "fenced source cannot generate or emit output",
                    operation=FailurePoint.GENERATE.value,
                    session_id=session_id,
                )
            try:
                event = state.generate(self._config, transaction_id=transaction_id)
            except ValueError as error:
                raise SessionStateError(
                    str(error),
                    operation=FailurePoint.GENERATE.value,
                    session_id=session_id,
                ) from error
            self._record_dirty(session_id, state.state_version)
            return event

    def dry_run_next_token(self, session_id: str) -> int:
        with self._lock:
            self._ensure_available(RuntimeCapability.DRY_RUN_VALIDATION.value, session_id)
            state = self._session(session_id, RuntimeCapability.DRY_RUN_VALIDATION.value)
            if state.lifecycle is SessionLifecycle.FENCED:
                raise StaleOwnerEpochError(
                    "fenced source cannot execute continuation validation",
                    operation=RuntimeCapability.DRY_RUN_VALIDATION.value,
                    session_id=session_id,
                )
            return state.peek_next_token(self._config)

    def acknowledge_gateway(
        self,
        session_id: str,
        *,
        token_index: int,
        owner_epoch: int,
    ) -> SessionMetadata:
        with self._lock:
            self._ensure_available("acknowledge_gateway", session_id)
            state = self._session(session_id, "acknowledge_gateway")
            if state.lifecycle is SessionLifecycle.FENCED:
                raise StaleOwnerEpochError(
                    "gateway cannot accept output from a fenced source",
                    operation="acknowledge_gateway",
                    session_id=session_id,
                )
            try:
                state.acknowledge_gateway(token_index=token_index, owner_epoch=owner_epoch)
            except ValueError as error:
                if owner_epoch != state.owner_epoch:
                    raise StaleOwnerEpochError(
                        str(error),
                        operation="acknowledge_gateway",
                        session_id=session_id,
                    ) from error
                raise SessionStateError(
                    str(error),
                    operation="acknowledge_gateway",
                    session_id=session_id,
                ) from error
            self._record_dirty(session_id, state.state_version)
            return self._metadata(state)

    def stream_tokens(
        self,
        session_id: str,
        *,
        count: int,
        transaction_id: str | None = None,
    ) -> tuple[TokenEvent, ...]:
        if not 0 <= count <= self.capabilities.max_stream_buffer_tokens:
            raise ResourceLimitError(
                "requested stream batch exceeds the published buffer bound",
                operation=RuntimeCapability.STREAMING.value,
                session_id=session_id,
            )
        return tuple(
            self.generate_token(session_id, transaction_id=transaction_id) for _ in range(count)
        )

    def pause_session(self, session_id: str) -> SessionMetadata:
        with self._lock:
            self._ensure_available(RuntimeCapability.PAUSE_RESUME.value, session_id)
            state = self._session(session_id, RuntimeCapability.PAUSE_RESUME.value)
            if state.lifecycle is not SessionLifecycle.ACTIVE:
                raise SessionStateError(
                    "only an active session may be paused",
                    operation=RuntimeCapability.PAUSE_RESUME.value,
                    session_id=session_id,
                )
            state.lifecycle = SessionLifecycle.PAUSED
            return self._metadata(state)

    def resume_session(self, session_id: str, *, expected_owner_epoch: int) -> SessionMetadata:
        with self._lock:
            self._ensure_available(RuntimeCapability.PAUSE_RESUME.value, session_id)
            state = self._session(session_id, RuntimeCapability.PAUSE_RESUME.value)
            if state.owner_epoch != expected_owner_epoch:
                raise StaleOwnerEpochError(
                    "resume owner epoch is stale",
                    operation=RuntimeCapability.PAUSE_RESUME.value,
                    session_id=session_id,
                )
            if state.lifecycle is SessionLifecycle.FENCED:
                raise StaleOwnerEpochError(
                    "a fenced writer cannot be resumed",
                    operation=RuntimeCapability.PAUSE_RESUME.value,
                    session_id=session_id,
                )
            if state.lifecycle is not SessionLifecycle.PAUSED:
                raise SessionStateError(
                    "only a paused session may be resumed",
                    operation=RuntimeCapability.PAUSE_RESUME.value,
                    session_id=session_id,
                )
            state.lifecycle = SessionLifecycle.ACTIVE
            return self._metadata(state)

    def cancel_session(self, session_id: str) -> SessionMetadata:
        with self._lock:
            self._ensure_available(RuntimeCapability.CANCELLATION.value, session_id)
            state = self._session(session_id, RuntimeCapability.CANCELLATION.value)
            if state.lifecycle in {SessionLifecycle.FENCED, SessionLifecycle.TERMINAL}:
                raise SessionStateError(
                    "session cannot be cancelled in its current lifecycle",
                    operation=RuntimeCapability.CANCELLATION.value,
                    session_id=session_id,
                )
            state.lifecycle = SessionLifecycle.CANCELLED
            state.state_version += 1
            self._record_dirty(session_id, state.state_version)
            return self._metadata(state)

    def acknowledge_client(self, session_id: str, *, token_index: int) -> SessionMetadata:
        with self._lock:
            self._ensure_available("acknowledge_client", session_id)
            state = self._session(session_id, "acknowledge_client")
            if not state.client_acknowledged_index <= token_index <= state.gateway_committed_index:
                raise SessionStateError(
                    "client acknowledgment is non-monotonic or beyond the gateway watermark",
                    operation="acknowledge_client",
                    session_id=session_id,
                )
            if token_index != state.client_acknowledged_index:
                state.client_acknowledged_index = token_index
                state.state_version += 1
                self._record_dirty(session_id, state.state_version)
            return self._metadata(state)

    def begin_consistent_snapshot(self, session_id: str) -> SnapshotHandle:
        with self._lock:
            self._ensure_available(FailurePoint.BEGIN_SNAPSHOT.value, session_id)
            self._hit_failure(FailurePoint.BEGIN_SNAPSHOT, session_id)
            if len(self._snapshots) >= self.capabilities.max_open_snapshots:
                raise ResourceLimitError(
                    "open snapshot bound reached",
                    operation=FailurePoint.BEGIN_SNAPSHOT.value,
                    session_id=session_id,
                )
            state = self._session(session_id, FailurePoint.BEGIN_SNAPSHOT.value)
            try:
                encoded = encode_state(state.clone(), self._config)
            except ValueError as error:
                raise SnapshotConsistencyError(
                    str(error),
                    operation=FailurePoint.BEGIN_SNAPSHOT.value,
                    session_id=session_id,
                ) from error
            snapshot_id = self._next_id("snapshot", session_id)
            handle = SnapshotHandle(
                snapshot_id=snapshot_id,
                session_id=session_id,
                owner_epoch=state.owner_epoch,
                state_version=state.state_version,
                dirty_epoch=state.state_version,
                segment_count=len(encoded.segments),
            )
            self._snapshots[snapshot_id] = _SnapshotRecord(handle=handle, encoded=encoded)
            return handle

    def _snapshot(self, handle: SnapshotHandle) -> _SnapshotRecord:
        try:
            record = self._snapshots[handle.snapshot_id]
        except KeyError as error:
            raise SnapshotConsistencyError(
                "snapshot handle is unknown or has already been closed",
                operation=RuntimeCapability.CONSISTENT_SNAPSHOT.value,
                session_id=handle.session_id,
            ) from error
        if record.handle != handle:
            raise SnapshotConsistencyError(
                "snapshot handle metadata was altered",
                operation=RuntimeCapability.CONSISTENT_SNAPSHOT.value,
                session_id=handle.session_id,
            )
        return record

    def enumerate_state_segments(self, handle: SnapshotHandle) -> tuple[SegmentDescriptor, ...]:
        with self._lock:
            self._ensure_available(RuntimeCapability.CONSISTENT_SNAPSHOT.value, handle.session_id)
            return tuple(segment.descriptor for segment in self._snapshot(handle).encoded.segments)

    def read_state_segment(self, handle: SnapshotHandle, segment_id: str) -> StateSegment:
        with self._lock:
            self._ensure_available(FailurePoint.READ_SEGMENT.value, handle.session_id)
            self._hit_failure(FailurePoint.READ_SEGMENT, handle.session_id)
            for segment in self._snapshot(handle).encoded.segments:
                if segment.descriptor.segment_id == segment_id:
                    return segment
            raise SnapshotConsistencyError(
                "snapshot segment identifier does not exist",
                operation=FailurePoint.READ_SEGMENT.value,
                session_id=handle.session_id,
            )

    def read_page_table(self, handle: SnapshotHandle) -> tuple[PageTableEntry, ...]:
        with self._lock:
            self._ensure_available(RuntimeCapability.CONSISTENT_SNAPSHOT.value, handle.session_id)
            return self._snapshot(handle).encoded.page_table

    def read_logical_state(self, handle: SnapshotHandle) -> LogicalStateManifest:
        with self._lock:
            self._ensure_available(RuntimeCapability.CONSISTENT_SNAPSHOT.value, handle.session_id)
            return self._snapshot(handle).encoded.logical

    def end_consistent_snapshot(self, handle: SnapshotHandle) -> None:
        with self._lock:
            self._snapshots.pop(handle.snapshot_id, None)

    def start_dirty_tracking(self, session_id: str) -> DirtyTrackingHandle:
        with self._lock:
            self._ensure_available(FailurePoint.START_DIRTY_TRACKING.value, session_id)
            self._hit_failure(FailurePoint.START_DIRTY_TRACKING, session_id)
            state = self._session(session_id, FailurePoint.START_DIRTY_TRACKING.value)
            tracking_id = self._next_id("dirty", session_id)
            record = _TrackingRecord(
                tracking_id=tracking_id,
                session_id=session_id,
                baseline_epoch=state.state_version,
                last_exported_epoch=state.state_version,
            )
            self._tracking[tracking_id] = record
            return DirtyTrackingHandle(
                tracking_id=tracking_id,
                session_id=session_id,
                baseline_epoch=state.state_version,
                last_exported_epoch=state.state_version,
            )

    def _tracking_record(self, handle: DirtyTrackingHandle) -> _TrackingRecord:
        try:
            record = self._tracking[handle.tracking_id]
        except KeyError as error:
            raise StaleDeltaError(
                "dirty tracking handle is unknown or closed",
                operation=RuntimeCapability.DIRTY_TRACKING.value,
                session_id=handle.session_id,
            ) from error
        if record.session_id != handle.session_id or record.baseline_epoch != handle.baseline_epoch:
            raise StaleDeltaError(
                "dirty tracking handle metadata was altered",
                operation=RuntimeCapability.DIRTY_TRACKING.value,
                session_id=handle.session_id,
            )
        return record

    def _delta(self, handle: DirtyTrackingHandle, *, final: bool) -> DirtyDelta:
        record = self._tracking_record(handle)
        state = self._session(record.session_id, RuntimeCapability.DIRTY_TRACKING.value)
        if final and state.lifecycle is not SessionLifecycle.PAUSED:
            raise SessionStateError(
                "final delta requires a source quiesced at a token boundary",
                operation=RuntimeCapability.FINAL_DELTA.value,
                session_id=record.session_id,
            )
        overflow_floor = self._overflow_floor[record.session_id]
        if record.last_exported_epoch < overflow_floor:
            raise DirtyLogOverflowError(
                "dirty history no longer covers the requested delta; restart from a full snapshot",
                operation=RuntimeCapability.DIRTY_TRACKING.value,
                session_id=record.session_id,
            )
        try:
            encoded = encode_state(state.clone(), self._config)
        except ValueError as error:
            raise SnapshotConsistencyError(
                str(error),
                operation=RuntimeCapability.DIRTY_TRACKING.value,
                session_id=record.session_id,
            ) from error
        from_epoch = record.last_exported_epoch
        changed = tuple(
            segment for segment in encoded.segments if segment.descriptor.dirty_epoch > from_epoch
        )
        delta = DirtyDelta(
            tracking_id=record.tracking_id,
            session_id=record.session_id,
            from_epoch=from_epoch,
            to_epoch=state.state_version,
            owner_epoch=state.owner_epoch,
            source_layout=self._config.layout,
            logical=encoded.logical,
            changed_segments=changed,
            page_table=encoded.page_table,
            final=final,
        )
        record.last_exported_epoch = state.state_version
        return delta

    def obtain_dirty_delta(self, handle: DirtyTrackingHandle) -> DirtyDelta:
        with self._lock:
            self._ensure_available(FailurePoint.READ_DELTA.value, handle.session_id)
            self._hit_failure(FailurePoint.READ_DELTA, handle.session_id)
            return self._delta(handle, final=False)

    def quiesce_at_token_boundary(self, session_id: str) -> SessionMetadata:
        with self._lock:
            self._ensure_available(FailurePoint.QUIESCE.value, session_id)
            self._hit_failure(FailurePoint.QUIESCE, session_id)
            return self.pause_session(session_id)

    def export_final_delta(self, handle: DirtyTrackingHandle) -> DirtyDelta:
        with self._lock:
            self._ensure_available(RuntimeCapability.FINAL_DELTA.value, handle.session_id)
            return self._delta(handle, final=True)

    def stop_dirty_tracking(self, handle: DirtyTrackingHandle) -> None:
        with self._lock:
            self._tracking.pop(handle.tracking_id, None)

    def fence_source_writer(self, session_id: str, *, expected_owner_epoch: int) -> SessionMetadata:
        with self._lock:
            self._ensure_available(RuntimeCapability.FENCING.value, session_id)
            state = self._session(session_id, RuntimeCapability.FENCING.value)
            if state.owner_epoch != expected_owner_epoch:
                raise StaleOwnerEpochError(
                    "source fencing owner epoch is stale",
                    operation=RuntimeCapability.FENCING.value,
                    session_id=session_id,
                )
            if state.lifecycle not in {SessionLifecycle.ACTIVE, SessionLifecycle.PAUSED}:
                raise SessionStateError(
                    "source cannot be fenced in its current lifecycle",
                    operation=RuntimeCapability.FENCING.value,
                    session_id=session_id,
                )
            state.lifecycle = SessionLifecycle.FENCED
            return self._metadata(state)

    def release_source_fence(
        self,
        session_id: str,
        *,
        expected_owner_epoch: int,
        coordinator_owner_epoch: int,
        ownership_committed: bool,
    ) -> SessionMetadata:
        with self._lock:
            self._ensure_available(RuntimeCapability.FENCING.value, session_id)
            state = self._session(session_id, RuntimeCapability.FENCING.value)
            if ownership_committed:
                raise StaleOwnerEpochError(
                    "a source fence cannot be released after ownership commit",
                    operation="release_source_fence",
                    session_id=session_id,
                )
            if not (state.owner_epoch == expected_owner_epoch == coordinator_owner_epoch):
                raise StaleOwnerEpochError(
                    "coordinator no longer proves the source epoch current",
                    operation="release_source_fence",
                    session_id=session_id,
                )
            if state.lifecycle is not SessionLifecycle.FENCED:
                raise SessionStateError(
                    "only a fenced source may enter pre-commit rollback",
                    operation="release_source_fence",
                    session_id=session_id,
                )
            state.lifecycle = SessionLifecycle.PAUSED
            return self._metadata(state)

    def prepare_destination_session(
        self,
        captured: CapturedState,
        *,
        destination_session_id: str,
        proposed_owner_epoch: int,
    ) -> SessionMetadata:
        with self._lock:
            self._ensure_available(FailurePoint.PREPARE_IMPORT.value, destination_session_id)
            self._hit_failure(FailurePoint.PREPARE_IMPORT, destination_session_id)
            captured.verify()
            if proposed_owner_epoch <= captured.logical.owner_epoch:
                raise StaleOwnerEpochError(
                    "destination owner epoch must increase",
                    operation=FailurePoint.PREPARE_IMPORT.value,
                    session_id=destination_session_id,
                )
            if destination_session_id in self._sessions or destination_session_id in self._staging:
                raise SessionStateError(
                    "destination session identifier is already in use",
                    operation=FailurePoint.PREPARE_IMPORT.value,
                    session_id=destination_session_id,
                )
            if len(self._sessions) + len(self._staging) >= self.capabilities.max_sessions:
                raise ResourceLimitError(
                    "runtime session bound reached",
                    operation=FailurePoint.PREPARE_IMPORT.value,
                    session_id=destination_session_id,
                )
            self._staging[destination_session_id] = _StagingRecord(
                destination_session_id=destination_session_id,
                source_session_id=captured.logical.session_id,
                proposed_owner_epoch=proposed_owner_epoch,
                source_runtime=captured.runtime,
                source_layout=captured.layout,
                source_segments={},
                source_page_table=(),
                source_logical=captured.logical,
                imported_state=None,
                expected_continuation_hash=None,
                source_dirty_epoch=captured.logical.dirty_epoch,
                validated=False,
            )
            return SessionMetadata(
                session_id=destination_session_id,
                request_id=captured.logical.request_id,
                tenant_id=captured.logical.tenant_id,
                lifecycle=SessionLifecycle.PREPARED,
                owner_epoch=proposed_owner_epoch,
                state_version=captured.logical.state_version,
                committed_output_index=(len(captured.logical.committed_output_token_ids) - 1),
                client_visible_index=(
                    captured.logical.client_delivery.last_gateway_committed_token_index
                ),
                layout=self._config.layout,
                model=self._config.model,
            )

    def _staged(self, destination_session_id: str, operation: str) -> _StagingRecord:
        try:
            return self._staging[destination_session_id]
        except KeyError as error:
            raise SessionNotFoundError(
                "prepared destination session does not exist",
                operation=operation,
                session_id=destination_session_id,
            ) from error

    def import_captured_state(self, destination_session_id: str, captured: CapturedState) -> None:
        with self._lock:
            self._ensure_available(FailurePoint.IMPORT.value, destination_session_id)
            self._hit_failure(FailurePoint.IMPORT, destination_session_id)
            staged = self._staged(destination_session_id, FailurePoint.IMPORT.value)
            if captured.logical.session_id != staged.source_session_id:
                raise ImportValidationError(
                    "captured source session does not match destination preparation",
                    operation=FailurePoint.IMPORT.value,
                    session_id=destination_session_id,
                )
            try:
                captured.verify()
                state = decode_segments(
                    source_layout=captured.layout,
                    source_segments=captured.segments,
                    manifest=captured.logical,
                    destination_config=self._config,
                    destination_session_id=destination_session_id,
                )
            except (ValueError, SnapshotConsistencyError, SegmentIntegrityError) as error:
                raise ImportValidationError(
                    str(error),
                    operation=FailurePoint.IMPORT.value,
                    session_id=destination_session_id,
                ) from error
            staged.source_layout = captured.layout
            staged.source_segments = {
                segment.descriptor.segment_id: segment for segment in captured.segments
            }
            staged.source_page_table = captured.page_table
            staged.source_logical = captured.logical
            staged.imported_state = state
            imported_hash = state.continuation_hash(self._config)
            if imported_hash != captured.logical.continuation_hash:
                raise ImportValidationError(
                    "imported continuation digest differs from source-declared evidence",
                    operation=FailurePoint.IMPORT.value,
                    session_id=destination_session_id,
                )
            staged.expected_continuation_hash = captured.logical.continuation_hash
            staged.source_dirty_epoch = captured.logical.dirty_epoch
            staged.validated = False

    def apply_dirty_delta(self, destination_session_id: str, delta: DirtyDelta) -> None:
        with self._lock:
            self._ensure_available(FailurePoint.APPLY_DELTA.value, destination_session_id)
            self._hit_failure(FailurePoint.APPLY_DELTA, destination_session_id)
            staged = self._staged(destination_session_id, FailurePoint.APPLY_DELTA.value)
            if staged.imported_state is None:
                raise SessionStateError(
                    "an initial snapshot must be imported before deltas",
                    operation=FailurePoint.APPLY_DELTA.value,
                    session_id=destination_session_id,
                )
            if delta.session_id != staged.source_session_id:
                raise StaleDeltaError(
                    "delta belongs to another source session",
                    operation=FailurePoint.APPLY_DELTA.value,
                    session_id=destination_session_id,
                )
            if delta.from_epoch != staged.source_dirty_epoch:
                raise StaleDeltaError(
                    "delta epoch is stale or leaves a gap",
                    operation=FailurePoint.APPLY_DELTA.value,
                    session_id=destination_session_id,
                )
            merged = dict(staged.source_segments)
            for segment in delta.changed_segments:
                merged[segment.descriptor.segment_id] = segment
            try:
                state = decode_segments(
                    source_layout=delta.source_layout,
                    source_segments=tuple(merged[key] for key in sorted(merged)),
                    manifest=delta.logical,
                    destination_config=self._config,
                    destination_session_id=destination_session_id,
                )
            except (ValueError, SnapshotConsistencyError, SegmentIntegrityError) as error:
                raise ImportValidationError(
                    str(error),
                    operation=FailurePoint.APPLY_DELTA.value,
                    session_id=destination_session_id,
                ) from error
            staged.source_layout = delta.source_layout
            staged.source_segments = merged
            staged.source_page_table = delta.page_table
            staged.source_logical = delta.logical
            staged.imported_state = state
            imported_hash = state.continuation_hash(self._config)
            if imported_hash != delta.logical.continuation_hash:
                raise ImportValidationError(
                    "delta continuation digest differs from source-declared evidence",
                    operation=FailurePoint.APPLY_DELTA.value,
                    session_id=destination_session_id,
                )
            staged.expected_continuation_hash = delta.logical.continuation_hash
            staged.source_dirty_epoch = delta.to_epoch
            staged.validated = False

    def validate_imported_state(self, destination_session_id: str) -> ImportValidation:
        with self._lock:
            self._ensure_available(FailurePoint.VALIDATE_IMPORT.value, destination_session_id)
            self._hit_failure(FailurePoint.VALIDATE_IMPORT, destination_session_id)
            staged = self._staged(destination_session_id, FailurePoint.VALIDATE_IMPORT.value)
            state = staged.imported_state
            expected_hash = staged.expected_continuation_hash
            if state is None or expected_hash is None:
                raise SessionStateError(
                    "destination has no imported state to validate",
                    operation=FailurePoint.VALIDATE_IMPORT.value,
                    session_id=destination_session_id,
                )
            encoded = encode_state(state.clone(), self._config)
            try:
                round_trip = decode_segments(
                    source_layout=self._config.layout,
                    source_segments=encoded.segments,
                    manifest=encoded.logical,
                    destination_config=self._config,
                    destination_session_id=destination_session_id,
                )
            except ValueError as error:
                raise ImportValidationError(
                    str(error),
                    operation=FailurePoint.VALIDATE_IMPORT.value,
                    session_id=destination_session_id,
                ) from error
            imported_hash = round_trip.continuation_hash(self._config)
            continuation_valid = imported_hash == expected_hash
            if not continuation_valid:
                raise ImportValidationError(
                    "destination continuation hash differs after native-layout round trip",
                    operation=FailurePoint.VALIDATE_IMPORT.value,
                    session_id=destination_session_id,
                )
            staged.validated = True
            return ImportValidation(
                session_id=destination_session_id,
                source_logical_hash=expected_hash,
                imported_logical_hash=imported_hash,
                dry_run_next_token=round_trip.peek_next_token(self._config),
                segment_count=len(encoded.segments),
                structurally_valid=True,
                continuation_valid=True,
            )

    def activate_destination(
        self,
        destination_session_id: str,
        *,
        committed_owner_epoch: int,
        fencing_token: str,
    ) -> SessionMetadata:
        with self._lock:
            self._ensure_available(FailurePoint.ACTIVATE.value, destination_session_id)
            self._hit_failure(FailurePoint.ACTIVATE, destination_session_id)
            staged = self._staged(destination_session_id, FailurePoint.ACTIVATE.value)
            if not staged.validated or staged.imported_state is None:
                raise ImportValidationError(
                    "destination must validate before activation",
                    operation=FailurePoint.ACTIVATE.value,
                    session_id=destination_session_id,
                )
            if committed_owner_epoch != staged.proposed_owner_epoch:
                raise StaleOwnerEpochError(
                    "committed owner epoch differs from the prepared epoch",
                    operation=FailurePoint.ACTIVATE.value,
                    session_id=destination_session_id,
                )
            if not fencing_token or len(fencing_token) > 4096:
                raise StaleOwnerEpochError(
                    "a bounded non-empty fencing token is required",
                    operation=FailurePoint.ACTIVATE.value,
                    session_id=destination_session_id,
                )
            state = staged.imported_state
            state.owner_epoch = committed_owner_epoch
            state.lifecycle = SessionLifecycle.ACTIVE
            self._sessions[destination_session_id] = state
            self._dirty_history[destination_session_id] = deque(
                maxlen=self.capabilities.max_dirty_events
            )
            self._overflow_floor[destination_session_id] = state.state_version
            self._fencing_token_hashes[destination_session_id] = sha256(
                fencing_token.encode("utf-8")
            ).hexdigest()
            del self._staging[destination_session_id]
            return self._metadata(state)

    def abort_destination(self, destination_session_id: str) -> None:
        with self._lock:
            self._staging.pop(destination_session_id, None)

    def inject_failure(self, rule: FailureRule) -> None:
        with self._lock:
            if len(self._failure_rules) >= self._max_failure_rules:
                raise ResourceLimitError(
                    "failure rule bound reached",
                    operation="inject_failure",
                    session_id=rule.session_id,
                )
            self._failure_rules.append(_FailureState(rule=rule))

    def clear_failures(self) -> None:
        with self._lock:
            self._failure_rules.clear()

    def crash(self) -> None:
        with self._lock:
            self._available = False
            self._snapshots.clear()
            self._tracking.clear()
            self._staging.clear()

    def restart(self) -> None:
        with self._lock:
            self._available = True

    @property
    def open_snapshot_count(self) -> int:
        with self._lock:
            return len(self._snapshots)

    @property
    def prepared_session_count(self) -> int:
        with self._lock:
            return len(self._staging)
