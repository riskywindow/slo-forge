from __future__ import annotations

import hashlib
import socket
import sqlite3
import stat
import time
import zlib
from contextlib import suppress
from pathlib import Path

import pytest

from sloforge.continuum.storage import ChunkCorrupt, ChunkMissing, FileContentStore
from sloforge.continuum.storage import content_store as content_store_module
from sloforge.continuum.transport import TCPTransportListener
from sloforge.continuum.transport import tcp as tcp_module

_AUTHORIZATION_KEY = bytes(range(32))


def test_file_store_rejects_metadata_path_escape_and_uses_private_modes(tmp_path: Path) -> None:
    root = tmp_path / "sensitive-store"
    outside = tmp_path / "outside-state"
    outside.write_bytes(b"not authorized state")

    with FileContentStore(root) as store:
        reference = store.put("tenant-a", b"authorized state", expires_at_ms=10)
        database = sqlite3.connect(root / "metadata.db")
        try:
            database.execute(
                "UPDATE chunks SET path=? WHERE tenant_id=? AND digest=?",
                ("../outside-state", "tenant-a", reference.digest),
            )
            database.commit()
        finally:
            database.close()

        with pytest.raises(ChunkCorrupt, match="non-canonical storage path"):
            store.read("tenant-a", reference)

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "chunks").stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "metadata.db").stat().st_mode) == 0o600


def test_file_store_uses_shortest_duplicate_retention_policy(tmp_path: Path) -> None:
    with FileContentStore(tmp_path / "retention-store") as store:
        first = store.put("tenant-a", b"expiring-state", expires_at_ms=100)
        second = store.put("tenant-a", b"expiring-state", expires_at_ms=20)
        assert first == second
        assert store.gc(now_ms=19) == ()
        assert store.gc(now_ms=20) == (first.digest,)


def test_compressed_chunk_expansion_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(content_store_module, "_MAX_CHUNK_BYTES", 64)
    compressed_bomb = zlib.compress(b"x" * 65, level=9)
    with pytest.raises(ChunkCorrupt, match="plaintext bound"):
        content_store_module._decode(compressed_bomb, "zlib")


def test_tcp_rejects_mitm_chunk_substitution_without_logging_secrets() -> None:
    destination = content_store_module.MemoryContentStore()
    secret_marker = "state-secret-must-not-enter-diagnostics"
    expected = b"expected attention state"
    substitute = b"substituted recurrent state"
    transfer_id = f"tx-{secret_marker}"

    with TCPTransportListener(
        destination=destination,
        allowed_tenant_id="tenant-a",
        authorization_key=_AUTHORIZATION_KEY,
    ) as listener:
        deadline = time.monotonic() + 2.0
        with socket.create_connection(listener.address, timeout=1.0) as connection:
            begin = tcp_module._authenticated_message(
                {
                    "chunk_count": 1,
                    "kind": "begin",
                    "protocol": tcp_module._PROTOCOL,
                    "tenant_id": "tenant-a",
                    "total_bytes": len(expected),
                    "transfer_id": transfer_id,
                },
                _AUTHORIZATION_KEY,
                direction="client",
            )
            tcp_module._send_message(connection, begin)
            ready = tcp_module._receive_message(
                connection,
                deadline=deadline,
                maximum_io_timeout_seconds=1.0,
            )
            tcp_module._verify_authenticated_message(
                ready,
                _AUTHORIZATION_KEY,
                direction="server",
            )

            signed_header = tcp_module._authenticated_message(
                {
                    "compression": "none",
                    "digest": hashlib.sha256(expected).hexdigest(),
                    "kind": "chunk",
                    "protocol": tcp_module._PROTOCOL,
                    "sequence": 0,
                    "size": len(expected),
                    "transfer_id": transfer_id,
                },
                _AUTHORIZATION_KEY,
                direction="client",
            )
            # A network attacker changes both the content identity and bytes while
            # retaining the captured authentication tag.
            signed_header["digest"] = hashlib.sha256(substitute).hexdigest()
            signed_header["size"] = len(substitute)
            tcp_module._send_message(connection, signed_header)
            with suppress(BrokenPipeError, ConnectionResetError, OSError):
                connection.sendall(substitute)

    substituted_reference = content_store_module.ChunkRef(
        tenant_id="tenant-a",
        digest=hashlib.sha256(substitute).hexdigest(),
        size_bytes=len(substitute),
        stored_bytes=len(substitute),
    )
    with pytest.raises(ChunkMissing):
        destination.read("tenant-a", substituted_reference)
    assert listener.errors
    assert all(secret_marker not in error for error in listener.errors)
    assert all(substitute.decode() not in error for error in listener.errors)


def test_tcp_message_authentication_is_direction_and_transfer_scoped() -> None:
    chunk = {
        "compression": "none",
        "digest": hashlib.sha256(b"state").hexdigest(),
        "kind": "chunk",
        "protocol": tcp_module._PROTOCOL,
        "sequence": 0,
        "size": 5,
        "transfer_id": "transaction-a",
    }
    signed = tcp_module._authenticated_message(
        chunk,
        _AUTHORIZATION_KEY,
        direction="client",
    )
    tcp_module._verify_authenticated_message(
        signed,
        _AUTHORIZATION_KEY,
        direction="client",
    )

    replayed = dict(signed)
    replayed["transfer_id"] = "transaction-b"
    with pytest.raises(tcp_module.TCPProtocolError, match="authentication failed"):
        tcp_module._verify_authenticated_message(
            replayed,
            _AUTHORIZATION_KEY,
            direction="client",
        )
    with pytest.raises(tcp_module.TCPProtocolError, match="authentication failed"):
        tcp_module._verify_authenticated_message(
            signed,
            _AUTHORIZATION_KEY,
            direction="server",
        )
