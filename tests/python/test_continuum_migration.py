from __future__ import annotations

import hashlib

from sloforge.continuum.adapters import ReferenceHeadMajorAdapter, ReferenceTokenMajorAdapter
from sloforge.continuum.migration import PrecopyMigrationRequest, migrate_precopy
from sloforge.continuum.storage import MemoryContentStore
from sloforge.continuum.transaction import DurableCoordinator, GatewayCommitLedger, TokenEvent
from sloforge.continuum.transport import DeterministicSimulatedTransport


def test_live_precopy_cross_adapter_transactional_cutover_has_no_duplicate_or_gap() -> None:
    source = ReferenceTokenMajorAdapter(page_size_tokens=3)
    destination = ReferenceHeadMajorAdapter(page_size_tokens=5)
    source.create_session(
        session_id="session-vertical",
        request_id="request-vertical",
        tenant_id="tenant-a",
        input_token_ids=(2, 3, 5, 7),
        seed=101,
    )
    gateway = GatewayCommitLedger(":memory:")
    gateway.register(session_id="session-vertical", owner_epoch=1)
    for runtime_event in source.stream_tokens("session-vertical", count=8):
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
            "session-vertical",
            token_index=runtime_event.token_index,
            owner_epoch=runtime_event.owner_epoch,
        )
    with DurableCoordinator(":memory:") as coordinator, gateway:
        coordinator.create_lease(
            session_id="session-vertical",
            owner_runtime=source.identity.runtime_name,
            expiration_ms=120_000,
            initial_token_index=7,
        )
        result = migrate_precopy(
            PrecopyMigrationRequest(
                session_id="session-vertical",
                seed=101,
                plan_hash=hashlib.sha256(b"vertical-precopy-plan").hexdigest(),
                delta_round_token_counts=(3, 2),
                resume_token_count=4,
                capture_timestamp="2026-08-02T00:00:00Z",
                git_commit="7e51ea7f7338755d23f889820558a4e046d6c42e",
                continuum_version="0.1.0",
            ),
            source=source,
            destination=destination,
            coordinator=coordinator,
            gateway=gateway,
            source_store=MemoryContentStore(),
            destination_store=MemoryContentStore(),
            transport=DeterministicSimulatedTransport(
                bandwidth_bytes_per_second=10_000_000,
                latency_us=25,
            ),
        )

        assert result.source_runtime != result.destination_runtime
        assert result.source_layout != result.destination_layout
        assert (result.source_owner_epoch, result.destination_owner_epoch) == (1, 2)
        assert result.source_next_token == result.destination_dry_run_token
        assert result.stale_source_rejected
        assert result.accepted_token_indices == tuple(range(18))
        assert result.phase_history[-1] == "COMPLETED"
        assert result.compatibility.compatibility_class.value == "exact_semantic"
        assert result.live_conversion_evidence.canonical_attention_match
        assert result.live_conversion_evidence.compared_attention_bytes > 0
        assert result.capsule.logical_state.attention is not None
        assert result.capsule.logical_state.recurrent
        assert len(result.transfer_receipts) == 4
        assert result.transfer_receipts[-1].source_chunks > 0
        assert result.transfer_receipts[-1].unique_plaintext_bytes > 0
