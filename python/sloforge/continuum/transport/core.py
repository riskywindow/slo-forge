"""Bounded in-process and deterministic simulated state transports."""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.continuum.storage.content_store import ChunkRef, ContentStore

_MAX_TRANSFER_CHUNKS = 4096
_MAX_TRANSFER_BYTES = 1024 * 1024 * 1024


class TransportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class TransportCapabilities(TransportModel):
    name: str
    memory_types: tuple[Literal["host", "gpu", "registered_host"], ...]
    supports_streaming: bool
    supports_retransmission: bool
    supports_cancellation: bool
    supports_encryption: bool
    supports_compression: bool
    supports_zero_copy: bool
    supports_gpudirect: bool
    maximum_chunk_bytes: Annotated[int, Field(ge=1)]
    maximum_in_flight_chunks: Annotated[int, Field(ge=1, le=1024)]


class TransferEvent(TransportModel):
    sequence: Annotated[int, Field(ge=0)]
    chunk_digest: str
    attempt: Annotated[int, Field(ge=1)]
    disposition: Literal[
        "sent",
        "lost",
        "corrupted",
        "duplicate",
        "acknowledged",
        "cancelled",
        "deadline_exceeded",
    ]
    at_us: Annotated[int, Field(ge=0)]
    bytes_on_wire: Annotated[int, Field(ge=0)]


class TransferReceipt(TransportModel):
    transport: str
    source_chunks: Annotated[int, Field(ge=0)]
    acknowledged_chunks: Annotated[int, Field(ge=0)]
    unique_plaintext_bytes: Annotated[int, Field(ge=0)]
    bytes_on_wire: Annotated[int, Field(ge=0)]
    retransmissions: Annotated[int, Field(ge=0)]
    duplicates_detected: Annotated[int, Field(ge=0)]
    elapsed_us: Annotated[int, Field(ge=0)]
    destination_refs: tuple[ChunkRef, ...]
    events: tuple[TransferEvent, ...]

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if self.acknowledged_chunks != len(self.destination_refs):
            raise ValueError("acknowledgment count differs from destination references")
        if self.acknowledged_chunks > self.source_chunks:
            raise ValueError("acknowledgments cannot exceed source chunk count")
        if len(self.events) > _MAX_TRANSFER_CHUNKS * 16:
            raise ValueError("transfer event log exceeds its bound")
        return self


class TransferFailure(RuntimeError):
    """A bounded transfer exhausted its retry, cancellation, or deadline policy."""


class StateTransport(Protocol):
    capabilities: TransportCapabilities

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
    ) -> TransferReceipt: ...


def _validate_request(references: Sequence[ChunkRef], deadline_us: int) -> None:
    if len(references) > _MAX_TRANSFER_CHUNKS:
        raise ValueError(f"transfer exceeds {_MAX_TRANSFER_CHUNKS} chunks")
    if len({reference.digest for reference in references}) != len(references):
        raise ValueError("transfer contains duplicate source chunk references")
    if sum(reference.size_bytes for reference in references) > _MAX_TRANSFER_BYTES:
        raise ValueError(f"transfer exceeds {_MAX_TRANSFER_BYTES} plaintext bytes")
    if deadline_us <= 0:
        raise ValueError("transfer deadline must be positive")


class InProcessTransport:
    capabilities = TransportCapabilities(
        name="in_process",
        memory_types=("host",),
        supports_streaming=True,
        supports_retransmission=False,
        supports_cancellation=True,
        supports_encryption=False,
        supports_compression=True,
        supports_zero_copy=False,
        supports_gpudirect=False,
        maximum_chunk_bytes=64 * 1024 * 1024,
        maximum_in_flight_chunks=1,
    )

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
        del seed
        _validate_request(references, deadline_us)
        if cancelled:
            raise TransferFailure("transfer cancelled before the first chunk")
        elapsed = 0
        events: list[TransferEvent] = []
        destination_refs: list[ChunkRef] = []
        for sequence, reference in enumerate(references):
            data = source.read(tenant_id, reference)
            elapsed += max(1, len(data) // 4096)
            if elapsed > deadline_us:
                raise TransferFailure("in-process transfer deadline exceeded")
            copied = destination.put(
                tenant_id,
                data,
                compression=reference.compression,
            )
            if copied.digest != reference.digest:
                raise TransferFailure("destination changed a state chunk digest")
            destination_refs.append(copied)
            events.extend(
                (
                    TransferEvent(
                        sequence=sequence * 2,
                        chunk_digest=reference.digest,
                        attempt=1,
                        disposition="sent",
                        at_us=elapsed,
                        bytes_on_wire=reference.stored_bytes,
                    ),
                    TransferEvent(
                        sequence=sequence * 2 + 1,
                        chunk_digest=reference.digest,
                        attempt=1,
                        disposition="acknowledged",
                        at_us=elapsed,
                        bytes_on_wire=0,
                    ),
                )
            )
        total = sum(reference.size_bytes for reference in references)
        wire = sum(reference.stored_bytes for reference in references)
        return TransferReceipt(
            transport=self.capabilities.name,
            source_chunks=len(references),
            acknowledged_chunks=len(destination_refs),
            unique_plaintext_bytes=total,
            bytes_on_wire=wire,
            retransmissions=0,
            duplicates_detected=0,
            elapsed_us=elapsed,
            destination_refs=tuple(destination_refs),
            events=tuple(events),
        )


class LocalFileTransport:
    """Portable file-spool transport with atomic publish and verified reads."""

    capabilities = TransportCapabilities(
        name="local_file",
        memory_types=("host",),
        supports_streaming=False,
        supports_retransmission=False,
        supports_cancellation=True,
        supports_encryption=False,
        supports_compression=True,
        supports_zero_copy=False,
        supports_gpudirect=False,
        maximum_chunk_bytes=64 * 1024 * 1024,
        maximum_in_flight_chunks=1,
    )

    def __init__(self, spool_root: Path) -> None:
        self.spool_root = spool_root
        self.spool_root.mkdir(parents=True, exist_ok=True)

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
        del seed
        _validate_request(references, deadline_us)
        if cancelled:
            raise TransferFailure("file transfer cancelled before the first chunk")
        elapsed = 0
        events: list[TransferEvent] = []
        destination_refs: list[ChunkRef] = []
        with tempfile.TemporaryDirectory(prefix="continuum-transfer-", dir=self.spool_root) as raw:
            transfer_root = Path(raw)
            for sequence, reference in enumerate(references):
                plaintext = source.read(tenant_id, reference)
                elapsed += max(1, len(plaintext) // 2048)
                if elapsed > deadline_us:
                    raise TransferFailure("file transfer deadline exceeded")
                final_path = transfer_root / reference.digest
                descriptor, temporary_name = tempfile.mkstemp(prefix=".chunk-", dir=transfer_root)
                try:
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(plaintext)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary_name, final_path)
                finally:
                    Path(temporary_name).unlink(missing_ok=True)
                received = final_path.read_bytes()
                if hashlib.sha256(received).hexdigest() != reference.digest:
                    raise TransferFailure("file-spooled state chunk failed integrity validation")
                copied = destination.put(
                    tenant_id,
                    received,
                    compression=reference.compression,
                )
                if copied.digest != reference.digest:
                    raise TransferFailure("destination changed a file-spooled chunk digest")
                destination_refs.append(copied)
                events.extend(
                    (
                        TransferEvent(
                            sequence=sequence * 2,
                            chunk_digest=reference.digest,
                            attempt=1,
                            disposition="sent",
                            at_us=elapsed,
                            bytes_on_wire=reference.size_bytes,
                        ),
                        TransferEvent(
                            sequence=sequence * 2 + 1,
                            chunk_digest=reference.digest,
                            attempt=1,
                            disposition="acknowledged",
                            at_us=elapsed,
                            bytes_on_wire=0,
                        ),
                    )
                )
        return TransferReceipt(
            transport=self.capabilities.name,
            source_chunks=len(references),
            acknowledged_chunks=len(destination_refs),
            unique_plaintext_bytes=sum(item.size_bytes for item in references),
            bytes_on_wire=sum(item.size_bytes for item in references),
            retransmissions=0,
            duplicates_detected=0,
            elapsed_us=elapsed,
            destination_refs=tuple(destination_refs),
            events=tuple(events),
        )


class DeterministicSimulatedTransport:
    """Measured-parameter simulation that moves and verifies real fixture bytes."""

    def __init__(
        self,
        *,
        bandwidth_bytes_per_second: int,
        latency_us: int,
        maximum_attempts: int = 4,
        loss_probability: float = 0.0,
        duplicate_probability: float = 0.0,
        corruption_probability: float = 0.0,
    ) -> None:
        if bandwidth_bytes_per_second <= 0 or latency_us < 0:
            raise ValueError("bandwidth must be positive and latency non-negative")
        if not 1 <= maximum_attempts <= 16:
            raise ValueError("maximum attempts must be within 1..16")
        for value in (loss_probability, duplicate_probability, corruption_probability):
            if not 0.0 <= value <= 1.0:
                raise ValueError("fault probabilities must be within zero and one")
        self.bandwidth_bytes_per_second = bandwidth_bytes_per_second
        self.latency_us = latency_us
        self.maximum_attempts = maximum_attempts
        self.loss_probability = loss_probability
        self.duplicate_probability = duplicate_probability
        self.corruption_probability = corruption_probability
        self.capabilities = TransportCapabilities(
            name="deterministic_simulated",
            memory_types=("host", "gpu", "registered_host"),
            supports_streaming=True,
            supports_retransmission=True,
            supports_cancellation=True,
            supports_encryption=False,
            supports_compression=True,
            supports_zero_copy=False,
            supports_gpudirect=False,
            maximum_chunk_bytes=64 * 1024 * 1024,
            maximum_in_flight_chunks=8,
        )

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
        _validate_request(references, deadline_us)
        if cancelled:
            raise TransferFailure("simulated transfer cancelled before the first chunk")
        random_source = random.Random(seed)
        elapsed = 0
        retransmissions = 0
        duplicates = 0
        bytes_on_wire = 0
        events: list[TransferEvent] = []
        destination_refs: list[ChunkRef] = []
        event_sequence = 0
        for reference in references:
            plaintext = source.read(tenant_id, reference)
            acknowledged = False
            for attempt in range(1, self.maximum_attempts + 1):
                duration = self.latency_us + max(
                    1,
                    (reference.stored_bytes * 1_000_000 + self.bandwidth_bytes_per_second - 1)
                    // self.bandwidth_bytes_per_second,
                )
                elapsed += duration
                bytes_on_wire += reference.stored_bytes
                if elapsed > deadline_us:
                    events.append(
                        TransferEvent(
                            sequence=event_sequence,
                            chunk_digest=reference.digest,
                            attempt=attempt,
                            disposition="deadline_exceeded",
                            at_us=elapsed,
                            bytes_on_wire=reference.stored_bytes,
                        )
                    )
                    raise TransferFailure("simulated transfer deadline exceeded")
                disposition: Literal["sent", "lost", "corrupted"] = "sent"
                if random_source.random() < self.loss_probability:
                    disposition = "lost"
                elif random_source.random() < self.corruption_probability:
                    disposition = "corrupted"
                events.append(
                    TransferEvent(
                        sequence=event_sequence,
                        chunk_digest=reference.digest,
                        attempt=attempt,
                        disposition=disposition,
                        at_us=elapsed,
                        bytes_on_wire=reference.stored_bytes,
                    )
                )
                event_sequence += 1
                if disposition != "sent":
                    retransmissions += 1
                    continue
                copied = destination.put(
                    tenant_id,
                    plaintext,
                    compression=reference.compression,
                )
                if (
                    hashlib.sha256(destination.read(tenant_id, copied)).hexdigest()
                    != reference.digest
                ):
                    retransmissions += 1
                    continue
                if random_source.random() < self.duplicate_probability:
                    duplicates += 1
                    destination.put(
                        tenant_id,
                        plaintext,
                        compression=reference.compression,
                    )
                    events.append(
                        TransferEvent(
                            sequence=event_sequence,
                            chunk_digest=reference.digest,
                            attempt=attempt,
                            disposition="duplicate",
                            at_us=elapsed,
                            bytes_on_wire=reference.stored_bytes,
                        )
                    )
                    event_sequence += 1
                    bytes_on_wire += reference.stored_bytes
                events.append(
                    TransferEvent(
                        sequence=event_sequence,
                        chunk_digest=reference.digest,
                        attempt=attempt,
                        disposition="acknowledged",
                        at_us=elapsed,
                        bytes_on_wire=0,
                    )
                )
                event_sequence += 1
                destination_refs.append(copied)
                acknowledged = True
                break
            if not acknowledged:
                raise TransferFailure(
                    f"state chunk {reference.digest} exhausted {self.maximum_attempts} attempts"
                )
        return TransferReceipt(
            transport=self.capabilities.name,
            source_chunks=len(references),
            acknowledged_chunks=len(destination_refs),
            unique_plaintext_bytes=sum(reference.size_bytes for reference in references),
            bytes_on_wire=bytes_on_wire,
            retransmissions=retransmissions,
            duplicates_detected=duplicates,
            elapsed_us=elapsed,
            destination_refs=tuple(destination_refs),
            events=tuple(events),
        )
