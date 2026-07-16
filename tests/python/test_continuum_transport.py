from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.continuum.storage import ChunkRef, MemoryContentStore
from sloforge.continuum.transport import (
    DeterministicSimulatedTransport,
    InProcessTransport,
    LocalFileTransport,
    TransferFailure,
)


def _source() -> tuple[MemoryContentStore, tuple[ChunkRef, ...]]:
    store = MemoryContentStore()
    references = tuple(
        store.put("tenant-a", bytes([index]) * (512 + index), compression="zlib")
        for index in range(4)
    )
    return store, references


def test_in_process_moves_real_chunks_and_preserves_hashes() -> None:
    source, references = _source()
    destination = MemoryContentStore()
    receipt = InProcessTransport().transfer(
        source=source,
        destination=destination,
        tenant_id="tenant-a",
        references=references,
        deadline_us=10_000,
        seed=19,
    )
    assert receipt.acknowledged_chunks == 4
    assert receipt.retransmissions == 0
    assert [reference.digest for reference in receipt.destination_refs] == [
        reference.digest for reference in references
    ]


def test_simulated_transfer_is_seeded_and_retransmits_faults() -> None:
    source, references = _source()
    transport = DeterministicSimulatedTransport(
        bandwidth_bytes_per_second=1_000_000,
        latency_us=50,
        maximum_attempts=8,
        loss_probability=0.4,
        duplicate_probability=1.0,
        corruption_probability=0.2,
    )
    first = transport.transfer(
        source=source,
        destination=MemoryContentStore(),
        tenant_id="tenant-a",
        references=references,
        deadline_us=1_000_000,
        seed=77,
    )
    second = transport.transfer(
        source=source,
        destination=MemoryContentStore(),
        tenant_id="tenant-a",
        references=references,
        deadline_us=1_000_000,
        seed=77,
    )
    assert first == second
    assert first.retransmissions > 0
    assert first.duplicates_detected > 0


def test_simulated_transport_fails_closed_on_deadline_and_retry_exhaustion() -> None:
    source, references = _source()
    deadline = DeterministicSimulatedTransport(
        bandwidth_bytes_per_second=1, latency_us=1, maximum_attempts=1
    )
    with pytest.raises(TransferFailure, match="deadline"):
        deadline.transfer(
            source=source,
            destination=MemoryContentStore(),
            tenant_id="tenant-a",
            references=references,
            deadline_us=10,
            seed=1,
        )
    loss = DeterministicSimulatedTransport(
        bandwidth_bytes_per_second=1_000_000,
        latency_us=1,
        maximum_attempts=2,
        loss_probability=1.0,
    )
    with pytest.raises(TransferFailure, match="exhausted"):
        loss.transfer(
            source=source,
            destination=MemoryContentStore(),
            tenant_id="tenant-a",
            references=references,
            deadline_us=1_000_000,
            seed=1,
        )


def test_local_file_transport_atomically_spools_and_cleans_up(tmp_path: Path) -> None:
    source, references = _source()
    destination = MemoryContentStore()
    spool = tmp_path / "spool"
    receipt = LocalFileTransport(spool).transfer(
        source=source,
        destination=destination,
        tenant_id="tenant-a",
        references=references,
        deadline_us=100_000,
        seed=23,
    )
    assert receipt.transport == "local_file"
    assert receipt.acknowledged_chunks == len(references)
    assert tuple(spool.iterdir()) == ()
    for source_reference, destination_reference in zip(
        references, receipt.destination_refs, strict=True
    ):
        assert destination.read("tenant-a", destination_reference) == source.read(
            "tenant-a", source_reference
        )
