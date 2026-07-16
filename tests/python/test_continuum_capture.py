from __future__ import annotations

import hashlib

from sloforge.continuum.adapters import ReferenceTokenMajorAdapter, StateKind
from sloforge.continuum.capture import publish_capture
from sloforge.continuum.ir import validate_capsule
from sloforge.continuum.storage import MemoryContentStore
from sloforge.continuum.transaction import DurableCoordinator


def test_live_capture_publishes_exact_chunks_and_seals_capsule() -> None:
    runtime = ReferenceTokenMajorAdapter()
    runtime.create_session(
        session_id="session-capture",
        request_id="request-capture",
        tenant_id="tenant-a",
        input_token_ids=(3, 5, 8),
        seed=41,
    )
    for event in runtime.stream_tokens("session-capture", count=7):
        runtime.acknowledge_gateway(
            "session-capture",
            token_index=event.token_index,
            owner_epoch=event.owner_epoch,
        )
    captured = runtime.capture_consistent("session-capture")
    assert {segment.descriptor.state_kind for segment in captured.segments} >= {
        StateKind.ATTENTION_KEY,
        StateKind.ATTENTION_VALUE,
        StateKind.RECURRENT,
        StateKind.SAMPLER,
        StateKind.GUIDED_DECODING,
    }

    store = MemoryContentStore()
    with DurableCoordinator(":memory:") as coordinator:
        lease = coordinator.create_lease(
            session_id="session-capture",
            owner_runtime=runtime.identity.runtime_name,
            expiration_ms=60_000,
            initial_token_index=6,
        )
        transaction = coordinator.begin_transaction(
            session_id="session-capture",
            destination_candidate="continuum-reference-head-major",
            migration_plan_hash=hashlib.sha256(b"capture-plan").hexdigest(),
            seed=41,
            now_ms=0,
            timeout_ms=10_000,
        )
        published = publish_capture(
            captured,
            store=store,
            lease=lease,
            transaction=transaction,
            journal=coordinator.journal(transaction.transaction_id),
            published_at_ms=1,
            capture_timestamp="2026-08-02T00:00:00Z",
            git_commit="7e51ea7f7338755d23f889820558a4e046d6c42e",
            continuum_version="0.1.0",
        )

    validate_capsule(published.capsule)
    assert published.capsule.logical_state.attention is not None
    assert published.capsule.logical_state.recurrent
    assert published.capsule.logical_state.guided_decoding is not None
    assert published.capsule.identity.owner_epoch == 1
    assert published.capsule.transaction.commit_watermark == 6
    assert len(published.chunk_references) == len(captured.segments)
    for reference in published.chunk_references:
        assert store.read("tenant-a", reference)
