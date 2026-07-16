from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.continuum.storage import ChunkRef, MemoryContentStore
from sloforge.continuum.transport import (
    TCPFaultProfile,
    TCPReplayRejected,
    TCPStateTransport,
    TCPTransportListener,
    TransferFailure,
)

_AUTHORIZATION_KEY = bytes(range(32))
_PAYLOAD_ENCRYPTION_KEY = bytes(reversed(range(32)))


def _references(source: MemoryContentStore, tenant_id: str = "tenant-a") -> tuple[ChunkRef, ...]:
    return (
        source.put(tenant_id, b"attention-state" * 128),
        source.put(tenant_id, b"recurrent-state" * 73, compression="zlib"),
    )


def test_tcp_transport_moves_real_chunks_with_checksum_acknowledgments(tmp_path: Path) -> None:
    source = MemoryContentStore()
    destination = MemoryContentStore()
    references = _references(source)
    with TCPTransportListener(
        destination=destination,
        allowed_tenant_id="tenant-a",
        replay_journal_path=tmp_path / "replay.db",
        authorization_key=_AUTHORIZATION_KEY,
    ) as listener:
        host, port = listener.address
        transport = TCPStateTransport(host, port, authorization_key=_AUTHORIZATION_KEY)
        receipt = transport.transfer(
            source=source,
            destination=destination,
            tenant_id="tenant-a",
            references=references,
            deadline_us=2_000_000,
            seed=17,
        )

    assert receipt.acknowledged_chunks == 2
    assert receipt.retransmissions == 0
    assert [destination.read("tenant-a", item) for item in receipt.destination_refs] == [
        source.read("tenant-a", item) for item in references
    ]
    assert listener.errors == ()


def test_tcp_transport_aes_gcm_encrypts_payloads_and_fails_closed_on_wrong_key() -> None:
    source = MemoryContentStore()
    destination = MemoryContentStore()
    references = _references(source)
    with TCPTransportListener(
        destination=destination,
        allowed_tenant_id="tenant-a",
        authorization_key=_AUTHORIZATION_KEY,
        payload_encryption_key=_PAYLOAD_ENCRYPTION_KEY,
    ) as listener:
        host, port = listener.address
        transport = TCPStateTransport(
            host,
            port,
            authorization_key=_AUTHORIZATION_KEY,
            payload_encryption_key=_PAYLOAD_ENCRYPTION_KEY,
        )
        assert transport.capabilities.supports_encryption
        receipt = transport.transfer(
            source=source,
            destination=destination,
            tenant_id="tenant-a",
            references=references,
            deadline_us=2_000_000,
            seed=8202,
        )

    assert receipt.transport == "tcp_v1_aes256gcm"
    assert receipt.bytes_on_wire > receipt.unique_plaintext_bytes
    assert [destination.read("tenant-a", item) for item in receipt.destination_refs] == [
        source.read("tenant-a", item) for item in references
    ]

    rejected_destination = MemoryContentStore()
    with TCPTransportListener(
        destination=rejected_destination,
        allowed_tenant_id="tenant-a",
        authorization_key=_AUTHORIZATION_KEY,
        payload_encryption_key=_PAYLOAD_ENCRYPTION_KEY,
    ) as listener:
        host, port = listener.address
        with pytest.raises(TransferFailure, match="exhausted"):
            TCPStateTransport(
                host,
                port,
                maximum_attempts=1,
                authorization_key=_AUTHORIZATION_KEY,
                payload_encryption_key=b"x" * 32,
            ).transfer(
                source=source,
                destination=rejected_destination,
                tenant_id="tenant-a",
                references=references,
                deadline_us=2_000_000,
                seed=8203,
            )


@pytest.mark.parametrize(
    ("fault_profile", "minimum_retransmissions"),
    [
        (TCPFaultProfile(corrupt_first_attempt_probability=1.0), 1),
        (TCPFaultProfile(truncate_first_connection_probability=1.0), 1),
    ],
)
def test_tcp_transport_recovers_from_seeded_corruption_and_truncation(
    fault_profile: TCPFaultProfile,
    minimum_retransmissions: int,
) -> None:
    source = MemoryContentStore()
    destination = MemoryContentStore()
    references = _references(source)
    with TCPTransportListener(
        destination=destination,
        allowed_tenant_id="tenant-a",
        maximum_io_timeout_seconds=0.5,
        authorization_key=_AUTHORIZATION_KEY,
    ) as listener:
        host, port = listener.address
        receipt = TCPStateTransport(
            host,
            port,
            maximum_attempts=3,
            maximum_io_timeout_seconds=0.5,
            fault_profile=fault_profile,
            authorization_key=_AUTHORIZATION_KEY,
        ).transfer(
            source=source,
            destination=destination,
            tenant_id="tenant-a",
            references=references,
            deadline_us=3_000_000,
            seed=29,
        )

    assert receipt.acknowledged_chunks == 2
    assert receipt.retransmissions >= minimum_retransmissions
    assert all(
        destination.read("tenant-a", received) == source.read("tenant-a", original)
        for received, original in zip(receipt.destination_refs, references, strict=True)
    )


def test_tcp_transport_cancellation_fails_before_sending_state() -> None:
    source = MemoryContentStore()
    destination = MemoryContentStore()
    references = _references(source)
    with TCPTransportListener(
        destination=destination,
        allowed_tenant_id="tenant-a",
        authorization_key=_AUTHORIZATION_KEY,
    ) as listener:
        host, port = listener.address
        with pytest.raises(TransferFailure, match="cancelled before connection"):
            TCPStateTransport(host, port, authorization_key=_AUTHORIZATION_KEY).transfer(
                source=source,
                destination=destination,
                tenant_id="tenant-a",
                references=references,
                deadline_us=2_000_000,
                seed=37,
                cancelled=True,
            )
    assert listener.errors == ()


def test_tcp_listener_rejects_wrong_tenant_and_completed_transfer_replay(tmp_path: Path) -> None:
    source = MemoryContentStore()
    destination = MemoryContentStore()
    references = _references(source)
    fixed_transfer_id = "migration-tx-0001"
    replay_path = tmp_path / "durable-replay.db"
    with TCPTransportListener(
        destination=destination,
        allowed_tenant_id="tenant-a",
        replay_journal_path=replay_path,
        authorization_key=_AUTHORIZATION_KEY,
    ) as listener:
        host, port = listener.address
        transport = TCPStateTransport(
            host,
            port,
            maximum_attempts=1,
            transfer_id_factory=lambda: fixed_transfer_id,
            authorization_key=_AUTHORIZATION_KEY,
        )
        first_receipt = transport.transfer(
            source=source,
            destination=destination,
            tenant_id="tenant-a",
            references=references,
            deadline_us=2_000_000,
            seed=41,
        )

    with TCPTransportListener(
        destination=destination,
        allowed_tenant_id="tenant-a",
        replay_journal_path=replay_path,
        authorization_key=_AUTHORIZATION_KEY,
    ) as restarted_listener:
        host, port = restarted_listener.address
        transport = TCPStateTransport(
            host,
            port,
            maximum_attempts=1,
            transfer_id_factory=lambda: fixed_transfer_id,
            authorization_key=_AUTHORIZATION_KEY,
        )
        with pytest.raises(TCPReplayRejected, match="completed transfer replay"):
            transport.transfer(
                source=source,
                destination=destination,
                tenant_id="tenant-a",
                references=references,
                deadline_us=2_000_000,
                seed=41,
            )

        wrong_source = MemoryContentStore()
        wrong_references = _references(wrong_source, tenant_id="tenant-b")
        with pytest.raises(TransferFailure, match="exhausted"):
            TCPStateTransport(
                host,
                port,
                maximum_attempts=1,
                authorization_key=_AUTHORIZATION_KEY,
            ).transfer(
                source=wrong_source,
                destination=destination,
                tenant_id="tenant-b",
                references=wrong_references,
                deadline_us=2_000_000,
                seed=43,
            )

        with pytest.raises(TransferFailure, match="exhausted"):
            TCPStateTransport(
                host,
                port,
                maximum_attempts=1,
                authorization_key=b"unauthorized-transport-key-value!",
            ).transfer(
                source=source,
                destination=destination,
                tenant_id="tenant-a",
                references=references,
                deadline_us=2_000_000,
                seed=47,
            )

    assert all(
        destination.read("tenant-a", received) == source.read("tenant-a", original)
        for received, original in zip(first_receipt.destination_refs, references, strict=True)
    )
