"""Bounded, versioned TCP transport for portable state chunks.

This module deliberately implements byte movement only.  A listener owns the
destination store and authorizes a single tenant; the client reads plaintext from
the source store, sends checksummed frames, and requires an explicit receiver
acknowledgment.  Semantic compatibility and ownership transfer remain above this
layer.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import random
import secrets
import socket
import sqlite3
import struct
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from sloforge.continuum.storage.content_store import ChunkRef, ContentStore
from sloforge.continuum.transport.core import (
    TransferEvent,
    TransferFailure,
    TransferReceipt,
    TransportCapabilities,
)

_PROTOCOL = "sloforge.continuum.transport.tcp/v1"
_MAX_HEADER_BYTES = 64 * 1024
_MAX_CHUNK_BYTES = 64 * 1024 * 1024
_MAX_TRANSFER_CHUNKS = 4096
_MAX_TRANSFER_BYTES = 1024 * 1024 * 1024
_MAX_TRANSFER_ID_BYTES = 128
_LENGTH = struct.Struct("!I")
_AUTHORIZATION_TAG = "authorization_tag"
_AUTHENTICATION_DOMAIN = b"sloforge.continuum.transport.tcp/v1\x00"
_PAYLOAD_ENCRYPTION = "aes256gcm"
_PAYLOAD_PLAINTEXT = "none"
_AES_GCM_TAG_BYTES = 16


class TCPFaultProfile(BaseModel):
    """Seeded, first-attempt-only faults for deterministic protocol validation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)

    corrupt_first_attempt_probability: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    truncate_first_connection_probability: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0


class TCPProtocolError(TransferFailure):
    """The remote peer violated the bounded TCP state-transfer protocol."""


class TCPReplayRejected(TCPProtocolError):
    """A previously completed transfer identifier was presented again."""


def _canonical_json(value: dict[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _authenticated_message(
    value: dict[str, object],
    authorization_key: bytes,
    *,
    direction: Literal["client", "server"],
) -> dict[str, object]:
    """Bind every protocol message to its direction and authorization domain."""

    if _AUTHORIZATION_TAG in value:
        raise ValueError("protocol message is already authenticated")
    signed = dict(value)
    signed[_AUTHORIZATION_TAG] = hmac.new(
        authorization_key,
        _AUTHENTICATION_DOMAIN + direction.encode("ascii") + b"\x00" + _canonical_json(value),
        hashlib.sha256,
    ).hexdigest()
    return signed


def _verify_authenticated_message(
    value: dict[str, object],
    authorization_key: bytes,
    *,
    direction: Literal["client", "server"],
) -> None:
    supplied_tag = _require_string(value, _AUTHORIZATION_TAG, 64)
    authenticated = dict(value)
    del authenticated[_AUTHORIZATION_TAG]
    expected_tag = _authenticated_message(
        authenticated,
        authorization_key,
        direction=direction,
    )[_AUTHORIZATION_TAG]
    if not isinstance(expected_tag, str) or not hmac.compare_digest(supplied_tag, expected_tag):
        raise TCPProtocolError("TCP protocol message authentication failed")


def _send_message(connection: socket.socket, value: dict[str, object]) -> int:
    payload = _canonical_json(value)
    if len(payload) > _MAX_HEADER_BYTES:
        raise TCPProtocolError("protocol metadata exceeds the bounded header size")
    connection.sendall(_LENGTH.pack(len(payload)))
    connection.sendall(payload)
    return _LENGTH.size + len(payload)


def _remaining(deadline: float, maximum_io_timeout_seconds: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TransferFailure("TCP state transfer deadline exceeded")
    return min(maximum_io_timeout_seconds, remaining)


def _receive_exact(
    connection: socket.socket,
    count: int,
    *,
    deadline: float,
    maximum_io_timeout_seconds: float,
) -> bytes:
    if count < 0:
        raise TCPProtocolError("negative protocol frame length")
    received = bytearray()
    while len(received) < count:
        connection.settimeout(_remaining(deadline, maximum_io_timeout_seconds))
        try:
            block = connection.recv(count - len(received))
        except TimeoutError as exc:
            raise TransferFailure("TCP state transfer receive timed out") from exc
        if not block:
            raise TCPProtocolError("TCP peer closed a truncated protocol frame")
        received.extend(block)
    return bytes(received)


def _receive_message(
    connection: socket.socket,
    *,
    deadline: float,
    maximum_io_timeout_seconds: float,
) -> dict[str, object]:
    raw_length = _receive_exact(
        connection,
        _LENGTH.size,
        deadline=deadline,
        maximum_io_timeout_seconds=maximum_io_timeout_seconds,
    )
    (length,) = _LENGTH.unpack(raw_length)
    if length == 0 or length > _MAX_HEADER_BYTES:
        raise TCPProtocolError("invalid bounded protocol header length")
    payload = _receive_exact(
        connection,
        length,
        deadline=deadline,
        maximum_io_timeout_seconds=maximum_io_timeout_seconds,
    )
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TCPProtocolError("protocol metadata is not canonical JSON") from exc
    if not isinstance(decoded, dict) or any(not isinstance(key, str) for key in decoded):
        raise TCPProtocolError("protocol metadata must be a JSON object")
    return cast(dict[str, object], decoded)


def _require_string(message: dict[str, object], name: str, maximum: int) -> str:
    value = message.get(name)
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise TCPProtocolError(f"protocol field {name!r} is invalid")
    return value


def _require_int(message: dict[str, object], name: str, maximum: int) -> int:
    value = message.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= maximum:
        raise TCPProtocolError(f"protocol field {name!r} is invalid")
    return value


def _aes_gcm(key: bytes) -> Any:
    if len(key) != 32:
        raise ValueError("TCP payload encryption requires exactly 32 key bytes")
    try:
        module = importlib.import_module("cryptography.hazmat.primitives.ciphers.aead")
    except (ImportError, OSError) as exc:
        raise TCPProtocolError("AES-GCM payload encryption support is unavailable") from exc
    return module.AESGCM(key)


def _payload_nonce(key: bytes, transfer_id: str, sequence: int, attempt: int) -> bytes:
    return hmac.new(
        key,
        b"sloforge.continuum.transport.payload-nonce/v1\x00"
        + transfer_id.encode("utf-8")
        + b"\x00"
        + sequence.to_bytes(4, "big")
        + attempt.to_bytes(2, "big"),
        hashlib.sha256,
    ).digest()[:12]


def _payload_aad(
    transfer_id: str,
    sequence: int,
    attempt: int,
    digest: str,
    plaintext_size: int,
) -> bytes:
    return _canonical_json(
        {
            "digest": digest,
            "attempt": attempt,
            "plaintext_size": plaintext_size,
            "protocol": _PROTOCOL,
            "sequence": sequence,
            "transfer_id": transfer_id,
        }
    )


class _ReplayJournal:
    """Bounded durable completed-transfer replay journal."""

    def __init__(self, path: Path | None, maximum_entries: int) -> None:
        if not 1 <= maximum_entries <= 1_000_000:
            raise ValueError("replay journal bound must be within 1..1000000")
        database = ":memory:" if path is None else str(path)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            database,
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS completed_transfers ("
            "transfer_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, completed_ns INTEGER NOT NULL)"
        )
        self._maximum_entries = maximum_entries
        self._lock = threading.Lock()

    def contains(self, transfer_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM completed_transfers WHERE transfer_id=?", (transfer_id,)
            ).fetchone()
        return row is not None

    def complete(self, transfer_id: str, tenant_id: str) -> None:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    "INSERT INTO completed_transfers VALUES(?,?,?)",
                    (transfer_id, tenant_id, time.time_ns()),
                )
                self._connection.execute(
                    "DELETE FROM completed_transfers WHERE transfer_id IN ("
                    "SELECT transfer_id FROM completed_transfers "
                    "ORDER BY completed_ns DESC,transfer_id DESC LIMIT -1 OFFSET ?)",
                    (self._maximum_entries,),
                )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class TCPTransportListener:
    """Receiver-owned state listener with bounded frames and tenant authorization."""

    def __init__(
        self,
        *,
        destination: ContentStore,
        allowed_tenant_id: str,
        host: str = "127.0.0.1",
        port: int = 0,
        maximum_io_timeout_seconds: float = 2.0,
        maximum_transfer_seconds: float = 30.0,
        replay_journal_path: Path | None = None,
        maximum_replay_entries: int = 65_536,
        staging_ttl_seconds: int = 3600,
        authorization_key: bytes,
        payload_encryption_key: bytes | None = None,
    ) -> None:
        if not allowed_tenant_id or len(allowed_tenant_id.encode("utf-8")) > 256:
            raise ValueError("listener requires a bounded tenant identity")
        if not 0 <= port <= 65_535:
            raise ValueError("TCP listener port is invalid")
        if maximum_io_timeout_seconds <= 0 or maximum_transfer_seconds <= 0:
            raise ValueError("TCP listener timeouts must be positive")
        if len(authorization_key) < 32:
            raise ValueError("TCP tenant authorization keys must contain at least 32 bytes")
        if payload_encryption_key is not None:
            _aes_gcm(payload_encryption_key)
        if not 1 <= staging_ttl_seconds <= 7 * 24 * 3600:
            raise ValueError("TCP staging TTL must be within one second and seven days")
        self._destination = destination
        self._allowed_tenant_id = allowed_tenant_id
        self._host = host
        self._requested_port = port
        self._maximum_io_timeout_seconds = maximum_io_timeout_seconds
        self._maximum_transfer_seconds = maximum_transfer_seconds
        self._authorization_key = bytes(authorization_key)
        self._payload_encryption_key = (
            bytes(payload_encryption_key) if payload_encryption_key is not None else None
        )
        self._staging_ttl_seconds = staging_ttl_seconds
        self._journal = _ReplayJournal(replay_journal_path, maximum_replay_entries)
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._active: set[str] = set()
        self._active_lock = threading.Lock()
        self._errors: list[str] = []
        self._errors_lock = threading.Lock()

    @property
    def address(self) -> tuple[str, int]:
        listener = self._socket
        if listener is None:
            raise RuntimeError("TCP transport listener has not started")
        address = listener.getsockname()
        return str(address[0]), int(address[1])

    @property
    def errors(self) -> tuple[str, ...]:
        with self._errors_lock:
            return tuple(self._errors)

    def start(self) -> TCPTransportListener:
        if self._socket is not None:
            raise RuntimeError("TCP transport listener is already started")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self._host, self._requested_port))
        listener.listen(16)
        listener.settimeout(0.1)
        self._socket = listener
        self._thread = threading.Thread(
            target=self._serve,
            name="continuum-tcp-transport",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=2.0):
            self.close()
            raise RuntimeError("TCP transport listener failed to start")
        return self

    def _serve(self) -> None:
        self._ready.set()
        listener = self._socket
        if listener is None:
            return
        while not self._stop.is_set():
            try:
                connection, _peer = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            with connection:
                try:
                    self._receive_transfer(connection)
                except (OSError, TransferFailure, ValueError) as exc:
                    # Errors are deliberately payload-free and bounded for diagnostics.
                    with self._errors_lock:
                        if len(self._errors) < 1024:
                            self._errors.append(f"{type(exc).__name__}: receiver failed closed")

    def _receive_transfer(self, connection: socket.socket) -> None:
        deadline = time.monotonic() + self._maximum_transfer_seconds
        begin = _receive_message(
            connection,
            deadline=deadline,
            maximum_io_timeout_seconds=self._maximum_io_timeout_seconds,
        )
        if begin.get("protocol") != _PROTOCOL or begin.get("kind") != "begin":
            self._reject(connection, "unsupported_protocol")
            return
        try:
            _verify_authenticated_message(
                begin,
                self._authorization_key,
                direction="client",
            )
        except TCPProtocolError:
            self._reject(connection, "tenant_unauthorized")
            return
        tenant_id = _require_string(begin, "tenant_id", 256)
        transfer_id = _require_string(begin, "transfer_id", _MAX_TRANSFER_ID_BYTES)
        chunk_count = _require_int(begin, "chunk_count", _MAX_TRANSFER_CHUNKS)
        total_bytes = _require_int(begin, "total_bytes", _MAX_TRANSFER_BYTES)
        payload_encryption_value = begin.get("payload_encryption", _PAYLOAD_PLAINTEXT)
        if not isinstance(payload_encryption_value, str) or not payload_encryption_value:
            self._reject(connection, "payload_encryption_mismatch")
            return
        payload_encryption = payload_encryption_value
        expected_encryption = (
            _PAYLOAD_ENCRYPTION if self._payload_encryption_key is not None else _PAYLOAD_PLAINTEXT
        )
        if payload_encryption != expected_encryption:
            self._reject(connection, "payload_encryption_mismatch")
            return
        if tenant_id != self._allowed_tenant_id:
            self._reject(connection, "tenant_unauthorized")
            return
        if self._journal.contains(transfer_id):
            self._reject(connection, "transfer_replay")
            return
        with self._active_lock:
            if transfer_id in self._active:
                self._reject(connection, "transfer_already_active")
                return
            self._active.add(transfer_id)
        try:
            self._send_authenticated(
                connection,
                {"kind": "ready", "protocol": _PROTOCOL, "transfer_id": transfer_id},
            )
            acknowledged: dict[int, ChunkRef] = {}
            observed_bytes = 0
            while len(acknowledged) < chunk_count:
                header = _receive_message(
                    connection,
                    deadline=deadline,
                    maximum_io_timeout_seconds=self._maximum_io_timeout_seconds,
                )
                if header.get("protocol") != _PROTOCOL or header.get("kind") != "chunk":
                    raise TCPProtocolError("expected a versioned chunk frame")
                _verify_authenticated_message(
                    header,
                    self._authorization_key,
                    direction="client",
                )
                if _require_string(header, "transfer_id", _MAX_TRANSFER_ID_BYTES) != transfer_id:
                    raise TCPProtocolError("chunk frame changed transfer identity")
                sequence = _require_int(header, "sequence", chunk_count - 1)
                size = _require_int(header, "size", _MAX_CHUNK_BYTES + _AES_GCM_TAG_BYTES)
                plaintext_size_value = header.get("plaintext_size", size)
                if (
                    not isinstance(plaintext_size_value, int)
                    or isinstance(plaintext_size_value, bool)
                    or not 0 <= plaintext_size_value <= _MAX_CHUNK_BYTES
                ):
                    raise TCPProtocolError("chunk plaintext size is invalid")
                plaintext_size = plaintext_size_value
                attempt_value = header.get("attempt", 1)
                if (
                    not isinstance(attempt_value, int)
                    or isinstance(attempt_value, bool)
                    or not 1 <= attempt_value <= 16
                ):
                    raise TCPProtocolError("chunk attempt is invalid")
                attempt = attempt_value
                digest = _require_string(header, "digest", 64)
                compression = _require_string(header, "compression", 16)
                frame_encryption_value = header.get("payload_encryption", _PAYLOAD_PLAINTEXT)
                if not isinstance(frame_encryption_value, str) or not frame_encryption_value:
                    raise TCPProtocolError("chunk payload encryption mode is invalid")
                frame_encryption = frame_encryption_value
                if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
                    raise TCPProtocolError("chunk digest is not lowercase SHA-256")
                if compression not in {"none", "zlib"}:
                    raise TCPProtocolError("unsupported content-store compression")
                if frame_encryption != expected_encryption:
                    raise TCPProtocolError("chunk frame changed payload encryption mode")
                if observed_bytes + plaintext_size > total_bytes:
                    raise TCPProtocolError("chunk exceeds the declared transfer byte bound")
                wire_payload = _receive_exact(
                    connection,
                    size,
                    deadline=deadline,
                    maximum_io_timeout_seconds=self._maximum_io_timeout_seconds,
                )
                if frame_encryption == _PAYLOAD_ENCRYPTION:
                    nonce_hex = _require_string(header, "nonce", 24)
                    try:
                        nonce = bytes.fromhex(nonce_hex)
                    except ValueError as exc:
                        raise TCPProtocolError("chunk encryption nonce is not hexadecimal") from exc
                    if len(nonce) != 12 or self._payload_encryption_key is None:
                        raise TCPProtocolError("chunk encryption nonce or key is invalid")
                    try:
                        payload = _aes_gcm(self._payload_encryption_key).decrypt(
                            nonce,
                            wire_payload,
                            _payload_aad(
                                transfer_id,
                                sequence,
                                attempt,
                                digest,
                                plaintext_size,
                            ),
                        )
                    except Exception:
                        self._send_authenticated(
                            connection,
                            {
                                "kind": "nack",
                                "protocol": _PROTOCOL,
                                "reason": "payload_authentication_failed",
                                "sequence": sequence,
                                "transfer_id": transfer_id,
                            },
                        )
                        continue
                else:
                    if size != plaintext_size or "nonce" in header:
                        raise TCPProtocolError("plaintext chunk has inconsistent wire metadata")
                    payload = wire_payload
                if len(payload) != plaintext_size or hashlib.sha256(payload).hexdigest() != digest:
                    self._send_authenticated(
                        connection,
                        {
                            "kind": "nack",
                            "protocol": _PROTOCOL,
                            "reason": "checksum_mismatch",
                            "sequence": sequence,
                            "transfer_id": transfer_id,
                        },
                    )
                    continue
                previous = acknowledged.get(sequence)
                if previous is not None:
                    if (
                        previous.digest != digest
                        or previous.size_bytes != plaintext_size
                        or previous.compression != compression
                    ):
                        raise TCPProtocolError("sequence replay changed chunk identity")
                    self._send_authenticated(
                        connection,
                        {
                            "chunk_ref": previous.model_dump(mode="json"),
                            "duplicate": True,
                            "kind": "ack",
                            "protocol": _PROTOCOL,
                            "sequence": sequence,
                            "transfer_id": transfer_id,
                        },
                    )
                    continue
                reference = self._destination.put(
                    tenant_id,
                    payload,
                    compression=cast(Literal["none", "zlib"], compression),
                    expires_at_ms=(int(time.time() * 1000) + self._staging_ttl_seconds * 1000),
                )
                if reference.digest != digest:
                    raise TCPProtocolError("destination changed a state chunk digest")
                acknowledged[sequence] = reference
                observed_bytes += plaintext_size
                if observed_bytes > total_bytes:
                    raise TCPProtocolError("received bytes exceed the declared transfer bound")
                self._send_authenticated(
                    connection,
                    {
                        "chunk_ref": reference.model_dump(mode="json"),
                        "duplicate": False,
                        "kind": "ack",
                        "protocol": _PROTOCOL,
                        "sequence": sequence,
                        "transfer_id": transfer_id,
                    },
                )
            finish = _receive_message(
                connection,
                deadline=deadline,
                maximum_io_timeout_seconds=self._maximum_io_timeout_seconds,
            )
            if finish.get("protocol") != _PROTOCOL or finish.get("kind") != "finish":
                raise TCPProtocolError("transfer did not end with a versioned finish frame")
            _verify_authenticated_message(
                finish,
                self._authorization_key,
                direction="client",
            )
            if _require_string(finish, "transfer_id", _MAX_TRANSFER_ID_BYTES) != transfer_id:
                raise TCPProtocolError("finish frame changed transfer identity")
            if observed_bytes != total_bytes:
                raise TCPProtocolError("received bytes differ from the declared transfer size")
            self._journal.complete(transfer_id, tenant_id)
            self._send_authenticated(
                connection,
                {
                    "acknowledged_chunks": len(acknowledged),
                    "kind": "complete",
                    "protocol": _PROTOCOL,
                    "transfer_id": transfer_id,
                },
            )
        finally:
            with self._active_lock:
                self._active.discard(transfer_id)

    def _send_authenticated(self, connection: socket.socket, value: dict[str, object]) -> int:
        return _send_message(
            connection,
            _authenticated_message(value, self._authorization_key, direction="server"),
        )

    def _reject(self, connection: socket.socket, reason: str) -> None:
        self._send_authenticated(
            connection,
            {"kind": "reject", "protocol": _PROTOCOL, "reason": reason},
        )

    def close(self) -> None:
        self._stop.set()
        listener = self._socket
        self._socket = None
        if listener is not None:
            with suppress(OSError):
                listener.shutdown(socket.SHUT_RDWR)
            listener.close()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError("TCP transport listener did not stop within its bound")
        self._journal.close()

    def __enter__(self) -> TCPTransportListener:
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.close()


class TCPStateTransport:
    """Client side of the portable TCP chunk protocol."""

    capabilities = TransportCapabilities(
        name="tcp_v1",
        memory_types=("host",),
        supports_streaming=True,
        supports_retransmission=True,
        supports_cancellation=True,
        supports_encryption=False,
        supports_compression=True,
        supports_zero_copy=False,
        supports_gpudirect=False,
        maximum_chunk_bytes=_MAX_CHUNK_BYTES,
        maximum_in_flight_chunks=1,
    )

    def __init__(
        self,
        host: str,
        port: int,
        *,
        maximum_attempts: int = 3,
        maximum_io_timeout_seconds: float = 2.0,
        fault_profile: TCPFaultProfile | None = None,
        transfer_id_factory: Callable[[], str] | None = None,
        authorization_key: bytes,
        payload_encryption_key: bytes | None = None,
    ) -> None:
        if not host or not 1 <= port <= 65_535:
            raise ValueError("TCP transport endpoint is invalid")
        if not 1 <= maximum_attempts <= 16:
            raise ValueError("TCP maximum attempts must be within 1..16")
        if maximum_io_timeout_seconds <= 0:
            raise ValueError("TCP I/O timeout must be positive")
        if len(authorization_key) < 32:
            raise ValueError("TCP tenant authorization keys must contain at least 32 bytes")
        if payload_encryption_key is not None:
            _aes_gcm(payload_encryption_key)
        self._host = host
        self._port = port
        self._maximum_attempts = maximum_attempts
        self._maximum_io_timeout_seconds = maximum_io_timeout_seconds
        self._fault_profile = fault_profile or TCPFaultProfile()
        self._transfer_id_factory = transfer_id_factory or (lambda: secrets.token_hex(16))
        self._authorization_key = bytes(authorization_key)
        self._payload_encryption_key = (
            bytes(payload_encryption_key) if payload_encryption_key is not None else None
        )
        self.capabilities = self.capabilities.model_copy(
            update={
                "name": (
                    "tcp_v1_aes256gcm" if self._payload_encryption_key is not None else "tcp_v1"
                ),
                "supports_encryption": self._payload_encryption_key is not None,
            }
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
        # The receiver owns ``destination``.  It remains in this signature for the
        # common StateTransport protocol and must be the store configured there.
        del destination
        self._validate_request(tenant_id, references, deadline_us)
        if cancelled:
            raise TransferFailure("TCP state transfer cancelled before connection")
        transfer_id = self._transfer_id_factory()
        if not transfer_id or len(transfer_id.encode("utf-8")) > _MAX_TRANSFER_ID_BYTES:
            raise ValueError("generated TCP transfer identifier is invalid")
        random_source = random.Random(seed)
        inject_corruption = (
            random_source.random() < self._fault_profile.corrupt_first_attempt_probability
        )
        inject_truncation = (
            random_source.random() < self._fault_profile.truncate_first_connection_probability
        )
        start = time.monotonic()
        deadline = start + deadline_us / 1_000_000
        events: list[TransferEvent] = []
        retransmissions = 0
        wire_counter = [0]
        destination_refs: tuple[ChunkRef, ...] = ()
        connection_attempt = 0
        while connection_attempt < self._maximum_attempts:
            connection_attempt += 1
            try:
                destination_refs, _wire, retry_count = self._transfer_once(
                    source=source,
                    tenant_id=tenant_id,
                    references=references,
                    transfer_id=transfer_id,
                    start=start,
                    deadline=deadline,
                    events=events,
                    wire_counter=wire_counter,
                    corrupt_first_chunk=inject_corruption and connection_attempt == 1,
                    truncate_connection=inject_truncation and connection_attempt == 1,
                )
                retransmissions += retry_count
                break
            except TCPReplayRejected:
                raise
            except (OSError, TransferFailure) as exc:
                if connection_attempt >= self._maximum_attempts or time.monotonic() >= deadline:
                    raise TransferFailure(
                        f"TCP state transfer exhausted {connection_attempt} bounded attempts"
                    ) from exc
                retransmissions += 1
        if len(destination_refs) != len(references):
            raise TransferFailure("TCP receiver did not acknowledge every source chunk")
        elapsed_us = max(1, int((time.monotonic() - start) * 1_000_000))
        return TransferReceipt(
            transport=self.capabilities.name,
            source_chunks=len(references),
            acknowledged_chunks=len(destination_refs),
            unique_plaintext_bytes=sum(reference.size_bytes for reference in references),
            bytes_on_wire=wire_counter[0],
            retransmissions=retransmissions,
            duplicates_detected=0,
            elapsed_us=elapsed_us,
            destination_refs=destination_refs,
            events=tuple(events),
        )

    def _transfer_once(
        self,
        *,
        source: ContentStore,
        tenant_id: str,
        references: Sequence[ChunkRef],
        transfer_id: str,
        start: float,
        deadline: float,
        events: list[TransferEvent],
        wire_counter: list[int],
        corrupt_first_chunk: bool,
        truncate_connection: bool,
    ) -> tuple[tuple[ChunkRef, ...], int, int]:
        timeout = _remaining(deadline, self._maximum_io_timeout_seconds)
        with socket.create_connection((self._host, self._port), timeout=timeout) as connection:
            begin: dict[str, object] = {
                "chunk_count": len(references),
                "kind": "begin",
                "protocol": _PROTOCOL,
                "tenant_id": tenant_id,
                "total_bytes": sum(reference.size_bytes for reference in references),
                "transfer_id": transfer_id,
                "payload_encryption": (
                    _PAYLOAD_ENCRYPTION
                    if self._payload_encryption_key is not None
                    else _PAYLOAD_PLAINTEXT
                ),
            }
            begin = _authenticated_message(
                begin,
                self._authorization_key,
                direction="client",
            )
            wire = _send_message(connection, begin)
            wire_counter[0] += wire
            ready = _receive_message(
                connection,
                deadline=deadline,
                maximum_io_timeout_seconds=self._maximum_io_timeout_seconds,
            )
            _verify_authenticated_message(
                ready,
                self._authorization_key,
                direction="server",
            )
            if ready.get("kind") == "reject":
                reason = str(ready.get("reason", "unspecified"))
                if reason == "transfer_replay":
                    raise TCPReplayRejected("TCP receiver rejected a completed transfer replay")
                raise TCPProtocolError(f"TCP receiver rejected transfer: {reason}")
            if ready.get("protocol") != _PROTOCOL or ready.get("kind") != "ready":
                raise TCPProtocolError("TCP receiver did not acknowledge transfer setup")
            if _require_string(ready, "transfer_id", _MAX_TRANSFER_ID_BYTES) != transfer_id:
                raise TCPProtocolError("TCP receiver changed transfer identity")
            received: list[ChunkRef] = []
            retry_count = 0
            event_sequence = len(events)
            for sequence, reference in enumerate(references):
                plaintext = source.read(tenant_id, reference)
                attempt = 0
                while attempt < self._maximum_attempts:
                    attempt += 1
                    payload_encryption = (
                        _PAYLOAD_ENCRYPTION
                        if self._payload_encryption_key is not None
                        else _PAYLOAD_PLAINTEXT
                    )
                    nonce: bytes | None = None
                    if self._payload_encryption_key is None:
                        payload = plaintext
                    else:
                        nonce = _payload_nonce(
                            self._payload_encryption_key,
                            transfer_id,
                            sequence,
                            attempt,
                        )
                        payload = _aes_gcm(self._payload_encryption_key).encrypt(
                            nonce,
                            plaintext,
                            _payload_aad(
                                transfer_id,
                                sequence,
                                attempt,
                                reference.digest,
                                len(plaintext),
                            ),
                        )
                    disposition: Literal["sent", "corrupted"] = "sent"
                    if corrupt_first_chunk and sequence == 0 and attempt == 1 and payload:
                        mutated = bytearray(payload)
                        mutated[0] ^= 1
                        payload = bytes(mutated)
                        disposition = "corrupted"
                    header = {
                        "compression": reference.compression,
                        "digest": reference.digest,
                        "attempt": attempt,
                        "kind": "chunk",
                        "payload_encryption": payload_encryption,
                        "plaintext_size": len(plaintext),
                        "protocol": _PROTOCOL,
                        "sequence": sequence,
                        "size": len(payload),
                        "transfer_id": transfer_id,
                    }
                    if nonce is not None:
                        header["nonce"] = nonce.hex()
                    header = _authenticated_message(
                        header,
                        self._authorization_key,
                        direction="client",
                    )
                    wire += _send_message(connection, header)
                    wire_counter[0] += _LENGTH.size + len(_canonical_json(header))
                    if truncate_connection and sequence == 0 and attempt == 1:
                        partial = payload[: max(0, len(payload) // 2)]
                        connection.sendall(partial)
                        wire_counter[0] += len(partial)
                        connection.shutdown(socket.SHUT_RDWR)
                        raise TCPProtocolError("injected deterministic truncated TCP frame")
                    connection.sendall(payload)
                    wire += len(payload)
                    wire_counter[0] += len(payload)
                    at_us = max(1, int((time.monotonic() - start) * 1_000_000))
                    events.append(
                        TransferEvent(
                            sequence=event_sequence,
                            chunk_digest=reference.digest,
                            attempt=attempt,
                            disposition=disposition,
                            at_us=at_us,
                            bytes_on_wire=len(payload),
                        )
                    )
                    event_sequence += 1
                    response = _receive_message(
                        connection,
                        deadline=deadline,
                        maximum_io_timeout_seconds=self._maximum_io_timeout_seconds,
                    )
                    _verify_authenticated_message(
                        response,
                        self._authorization_key,
                        direction="server",
                    )
                    if (
                        _require_string(response, "transfer_id", _MAX_TRANSFER_ID_BYTES)
                        != transfer_id
                    ):
                        raise TCPProtocolError("TCP response changed transfer identity")
                    if response.get("kind") == "nack":
                        retry_count += 1
                        continue
                    if response.get("protocol") != _PROTOCOL or response.get("kind") != "ack":
                        raise TCPProtocolError("TCP receiver returned an invalid chunk response")
                    if _require_int(response, "sequence", _MAX_TRANSFER_CHUNKS - 1) != sequence:
                        raise TCPProtocolError("TCP receiver acknowledged the wrong sequence")
                    raw_reference = response.get("chunk_ref")
                    if not isinstance(raw_reference, dict):
                        raise TCPProtocolError("TCP acknowledgment omitted its chunk reference")
                    copied = ChunkRef.model_validate(raw_reference, strict=True)
                    if copied.tenant_id != tenant_id or copied.digest != reference.digest:
                        raise TCPProtocolError("TCP acknowledgment substituted chunk identity")
                    received.append(copied)
                    events.append(
                        TransferEvent(
                            sequence=event_sequence,
                            chunk_digest=reference.digest,
                            attempt=attempt,
                            disposition="acknowledged",
                            at_us=at_us,
                            bytes_on_wire=0,
                        )
                    )
                    event_sequence += 1
                    break
                else:
                    raise TransferFailure(
                        f"TCP chunk exhausted {self._maximum_attempts} checksum attempts"
                    )
            finish = _authenticated_message(
                {"kind": "finish", "protocol": _PROTOCOL, "transfer_id": transfer_id},
                self._authorization_key,
                direction="client",
            )
            wire += _send_message(connection, finish)
            wire_counter[0] += _LENGTH.size + len(_canonical_json(finish))
            complete = _receive_message(
                connection,
                deadline=deadline,
                maximum_io_timeout_seconds=self._maximum_io_timeout_seconds,
            )
            _verify_authenticated_message(
                complete,
                self._authorization_key,
                direction="server",
            )
            if complete.get("protocol") != _PROTOCOL or complete.get("kind") != "complete":
                raise TCPProtocolError("TCP receiver did not commit transfer completion")
            if _require_string(complete, "transfer_id", _MAX_TRANSFER_ID_BYTES) != transfer_id:
                raise TCPProtocolError("TCP completion changed transfer identity")
            if _require_int(complete, "acknowledged_chunks", _MAX_TRANSFER_CHUNKS) != len(
                references
            ):
                raise TCPProtocolError("TCP completion count differs from the request")
            return tuple(received), wire, retry_count

    @staticmethod
    def _validate_request(tenant_id: str, references: Sequence[ChunkRef], deadline_us: int) -> None:
        if not tenant_id or len(tenant_id.encode("utf-8")) > 256:
            raise ValueError("TCP transfer requires a bounded tenant identity")
        if not 0 <= len(references) <= _MAX_TRANSFER_CHUNKS:
            raise ValueError(f"TCP transfer exceeds {_MAX_TRANSFER_CHUNKS} chunks")
        if any(reference.tenant_id != tenant_id for reference in references):
            raise PermissionError("TCP source chunk crosses a tenant boundary")
        if len({reference.digest for reference in references}) != len(references):
            raise ValueError("TCP transfer contains duplicate chunk references")
        if sum(reference.size_bytes for reference in references) > _MAX_TRANSFER_BYTES:
            raise ValueError(f"TCP transfer exceeds {_MAX_TRANSFER_BYTES} bytes")
        if deadline_us <= 0:
            raise ValueError("TCP transfer deadline must be positive")
