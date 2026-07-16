from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pytest

from sloforge.continuum.adapters import (
    CapturedState,
    DirtyDelta,
    DirtyTrackingHandle,
    FailurePoint,
    FailureRule,
    LayoutKind,
    ReferenceHeadMajorAdapter,
    ReferenceTokenMajorAdapter,
    SessionLifecycle,
    StateKind,
)
from sloforge.continuum.migration.orchestrator import (
    MigrationExecutionError,
    PrecopyMigrationRequest,
    migrate_precopy,
)
from sloforge.continuum.storage import ChunkRef, ContentStore, MemoryContentStore
from sloforge.continuum.transaction import (
    CutoverPhase,
    DurableCoordinator,
    GatewayCommitLedger,
    TokenEvent,
)
from sloforge.continuum.transport import (
    DeterministicSimulatedTransport,
    TransferFailure,
    TransferReceipt,
)

_SESSION_ID = "migration-recovery"


class TrackingSource(ReferenceTokenMajorAdapter):
    def __init__(self) -> None:
        super().__init__(page_size_tokens=3)
        self.stop_calls = 0

    def stop_dirty_tracking(self, handle: DirtyTrackingHandle) -> None:
        super().stop_dirty_tracking(handle)
        self.stop_calls += 1


class RecordingDestination(ReferenceHeadMajorAdapter):
    def __init__(self) -> None:
        super().__init__(page_size_tokens=5)
        self.imported_captures: list[CapturedState] = []
        self.imported_deltas: list[DirtyDelta] = []

    def import_captured_state(self, destination_session_id: str, captured: CapturedState) -> None:
        self.imported_captures.append(captured)
        super().import_captured_state(destination_session_id, captured)

    def apply_dirty_delta(self, destination_session_id: str, delta: DirtyDelta) -> None:
        self.imported_deltas.append(delta)
        super().apply_dirty_delta(destination_session_id, delta)


class InitialTransferFailure(DeterministicSimulatedTransport):
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
        del source, destination, tenant_id, references, deadline_us, seed, cancelled
        raise TransferFailure("deterministic initial transfer failure")


class GatewaySwitchFailure(GatewayCommitLedger):
    def __init__(self) -> None:
        super().__init__(":memory:")
        self.switch_calls = 0

    def switch_owner(
        self,
        *,
        session_id: str,
        expected_epoch: int,
        destination_epoch: int,
        expected_watermark: int,
    ) -> None:
        del session_id, expected_epoch, destination_epoch, expected_watermark
        self.switch_calls += 1
        raise RuntimeError("deterministic gateway switch failure")


def _request(*, seed: int = 73) -> PrecopyMigrationRequest:
    return PrecopyMigrationRequest(
        session_id=_SESSION_ID,
        seed=seed,
        plan_hash=hashlib.sha256(f"recovery-plan-{seed}".encode()).hexdigest(),
        delta_round_token_counts=(2,),
        resume_token_count=2,
        capture_timestamp="2026-08-02T00:00:00Z",
        git_commit="7e51ea7f7338755d23f889820558a4e046d6c42e",
        continuum_version="0.1.0",
    )


def _prepared_pair() -> tuple[TrackingSource, RecordingDestination]:
    source = TrackingSource()
    destination = RecordingDestination()
    source.create_session(
        session_id=_SESSION_ID,
        request_id="recovery-request",
        tenant_id="tenant-recovery",
        input_token_ids=(2, 3, 5, 7),
        seed=73,
    )
    return source, destination


def _gateway_with_history(source: TrackingSource, gateway: GatewayCommitLedger) -> None:
    gateway.register(session_id=_SESSION_ID, owner_epoch=1)
    for runtime_event in source.stream_tokens(_SESSION_ID, count=4):
        gateway.accept(
            TokenEvent(
                session_id=runtime_event.session_id,
                owner_epoch=runtime_event.owner_epoch,
                token_index=runtime_event.token_index,
                token_id=runtime_event.token_id,
                state_commit_version=runtime_event.state_commit_version,
            )
        )
        source.acknowledge_gateway(
            _SESSION_ID,
            token_index=runtime_event.token_index,
            owner_epoch=runtime_event.owner_epoch,
        )


def _lease(coordinator: DurableCoordinator, source: TrackingSource) -> None:
    coordinator.create_lease(
        session_id=_SESSION_ID,
        owner_runtime=source.identity.runtime_name,
        expiration_ms=120_000,
        initial_token_index=3,
    )


def _assert_precommit_rollback(
    *,
    error: MigrationExecutionError,
    source: TrackingSource,
    destination: RecordingDestination,
    coordinator: DurableCoordinator,
    gateway: GatewayCommitLedger,
) -> None:
    assert error.terminal_phase is CutoverPhase.ROLLED_BACK
    assert not error.ownership_committed
    assert coordinator.transaction(error.transaction_id).phase is CutoverPhase.ROLLED_BACK
    assert tuple(entry.to_phase for entry in coordinator.journal(error.transaction_id))[-2:] == (
        CutoverPhase.ABORTING,
        CutoverPhase.ROLLED_BACK,
    )
    assert coordinator.lease(_SESSION_ID).owner_epoch == 1
    source_state = source.inspect_session(_SESSION_ID)
    assert source_state.lifecycle is SessionLifecycle.ACTIVE
    assert source_state.client_visible_index == gateway.watermark(_SESSION_ID)
    assert destination.prepared_session_count == 0
    assert source.stop_calls == 1


def test_initial_and_delta_imports_use_explicit_destination_layout_conversion() -> None:
    source, destination = _prepared_pair()
    with DurableCoordinator(":memory:") as coordinator, GatewayCommitLedger(":memory:") as gateway:
        _gateway_with_history(source, gateway)
        _lease(coordinator, source)
        result = migrate_precopy(
            _request(),
            source=source,
            destination=destination,
            coordinator=coordinator,
            gateway=gateway,
            source_store=MemoryContentStore(),
            destination_store=MemoryContentStore(),
            transport=DeterministicSimulatedTransport(
                bandwidth_bytes_per_second=8_000_000,
                latency_us=11,
            ),
        )

    assert len(destination.imported_captures) == 1
    imported = destination.imported_captures[0]
    assert imported.layout.kind is LayoutKind.PAGED_HEAD_MAJOR_PACKED_KV
    attention_kinds = {
        segment.descriptor.state_kind
        for segment in imported.segments
        if segment.descriptor.layer is not None
    }
    assert attention_kinds == {StateKind.ATTENTION_PACKED_KV}
    assert len(destination.imported_deltas) == 2
    assert all(
        delta.source_layout.kind is LayoutKind.PAGED_HEAD_MAJOR_PACKED_KV
        for delta in destination.imported_deltas
    )
    assert destination.imported_deltas[-1].final
    assert destination.imported_deltas[-1].changed_segments
    assert result.transfer_receipts[-1].source_chunks > 0
    assert result.live_conversion_evidence.canonical_attention_match


def test_transfer_failure_durably_rolls_back_and_cleans_runtime_state() -> None:
    source, destination = _prepared_pair()
    with DurableCoordinator(":memory:") as coordinator, GatewayCommitLedger(":memory:") as gateway:
        _gateway_with_history(source, gateway)
        _lease(coordinator, source)
        with pytest.raises(MigrationExecutionError) as captured:
            migrate_precopy(
                _request(seed=74),
                source=source,
                destination=destination,
                coordinator=coordinator,
                gateway=gateway,
                source_store=MemoryContentStore(),
                destination_store=MemoryContentStore(),
                transport=InitialTransferFailure(
                    bandwidth_bytes_per_second=1,
                    latency_us=0,
                ),
            )
        assert captured.value.failed_operation == "initial_transfer"
        _assert_precommit_rollback(
            error=captured.value,
            source=source,
            destination=destination,
            coordinator=coordinator,
            gateway=gateway,
        )


@pytest.mark.parametrize(
    ("failure_point", "expected_operation"),
    (
        (FailurePoint.IMPORT, "destination_import"),
        (FailurePoint.VALIDATE_IMPORT, "destination_validation"),
    ),
)
def test_import_and_validation_failures_restore_the_valid_source(
    failure_point: FailurePoint, expected_operation: str
) -> None:
    source, destination = _prepared_pair()
    destination.inject_failure(
        FailureRule(point=failure_point, trigger_on_call=1, session_id=_SESSION_ID)
    )
    with DurableCoordinator(":memory:") as coordinator, GatewayCommitLedger(":memory:") as gateway:
        _gateway_with_history(source, gateway)
        _lease(coordinator, source)
        with pytest.raises(MigrationExecutionError) as captured:
            migrate_precopy(
                _request(seed=75 + list(FailurePoint).index(failure_point)),
                source=source,
                destination=destination,
                coordinator=coordinator,
                gateway=gateway,
                source_store=MemoryContentStore(),
                destination_store=MemoryContentStore(),
                transport=DeterministicSimulatedTransport(
                    bandwidth_bytes_per_second=8_000_000,
                    latency_us=11,
                ),
            )
        assert captured.value.failed_operation == expected_operation
        _assert_precommit_rollback(
            error=captured.value,
            source=source,
            destination=destination,
            coordinator=coordinator,
            gateway=gateway,
        )


def test_gateway_switch_failure_is_postcommit_and_never_unfences_source() -> None:
    source, destination = _prepared_pair()
    with DurableCoordinator(":memory:") as coordinator, GatewaySwitchFailure() as gateway:
        _gateway_with_history(source, gateway)
        _lease(coordinator, source)
        with pytest.raises(MigrationExecutionError) as captured:
            migrate_precopy(
                _request(seed=91),
                source=source,
                destination=destination,
                coordinator=coordinator,
                gateway=gateway,
                source_store=MemoryContentStore(),
                destination_store=MemoryContentStore(),
                transport=DeterministicSimulatedTransport(
                    bandwidth_bytes_per_second=8_000_000,
                    latency_us=11,
                ),
            )
        error = captured.value
        assert error.failed_operation == "gateway_switch"
        assert error.ownership_committed
        assert error.terminal_phase is CutoverPhase.OPERATOR_REQUIRED
        assert coordinator.transaction(error.transaction_id).phase is (
            CutoverPhase.OPERATOR_REQUIRED
        )
        assert coordinator.lease(_SESSION_ID).owner_epoch == 2
        assert coordinator.lease(_SESSION_ID).owner_runtime == destination.identity.runtime_name
        assert source.inspect_session(_SESSION_ID).lifecycle is SessionLifecycle.FENCED
        assert destination.prepared_session_count == 1
        assert source.stop_calls == 1
        assert gateway.switch_calls == 1
