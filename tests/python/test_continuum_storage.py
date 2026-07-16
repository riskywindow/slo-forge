from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sloforge.continuum.storage import ChunkCorrupt, FileContentStore, MemoryContentStore


@pytest.mark.parametrize("compression", ["none", "zlib"])
def test_memory_store_integrity_partial_reads_and_tenant_isolation(compression: str) -> None:
    store = MemoryContentStore()
    data = b"portable-state" * 128
    reference = store.put("tenant-a", data, compression=compression, expires_at_ms=10)  # type: ignore[arg-type]
    assert store.read("tenant-a", reference, offset=9, length=5) == data[9:14]
    with pytest.raises(PermissionError):
        store.read("tenant-b", reference)
    other = store.put("tenant-b", data, expires_at_ms=10)
    assert other.digest == reference.digest
    assert other.tenant_id != reference.tenant_id


def test_memory_store_copy_on_write_reference_lifecycle_and_corruption() -> None:
    store = MemoryContentStore()
    reference = store.put("tenant-a", b"shared immutable prefix", expires_at_ms=10)
    parent = store.publish(
        tenant_id="tenant-a",
        kind="complete",
        chunks=(reference,),
        published_at_ms=0,
    )
    branch = store.fork(
        tenant_id="tenant-a", parent_manifest_id=parent.manifest_id, published_at_ms=1
    )
    assert branch.chunks == parent.chunks
    store.delete_manifest("tenant-a", parent.manifest_id)
    assert store.gc(now_ms=10) == ()
    store.delete_manifest("tenant-a", branch.manifest_id)
    assert store.gc(now_ms=10) == (reference.digest,)

    corrupt = store.put("tenant-a", b"integrity", expires_at_ms=20)
    store.corrupt_for_test("tenant-a", corrupt.digest, b"changed")
    with pytest.raises(ChunkCorrupt):
        store.read("tenant-a", corrupt)


def test_file_store_transactional_publish_restart_cow_gc_and_corruption(tmp_path: Path) -> None:
    root = tmp_path / "store"
    with FileContentStore(root) as store:
        first = store.put("tenant-a", b"a" * 4096, compression="zlib", expires_at_ms=50)
        second = store.put("tenant-a", b"b" * 2048, expires_at_ms=50)
        parent = store.publish(
            tenant_id="tenant-a",
            kind="complete",
            chunks=(first, second),
            published_at_ms=0,
        )
        fork = store.fork(
            tenant_id="tenant-a", parent_manifest_id=parent.manifest_id, published_at_ms=1
        )
        assert store.read("tenant-a", first, offset=100, length=10) == b"a" * 10

    with FileContentStore(root) as reopened:
        assert reopened.manifest("tenant-a", parent.manifest_id) == parent
        reopened.delete_manifest("tenant-a", parent.manifest_id)
        assert reopened.gc(now_ms=50) == ()
        reopened.delete_manifest("tenant-a", fork.manifest_id)
        assert set(reopened.gc(now_ms=50)) == {first.digest, second.digest}

        corrupt = reopened.put("tenant-a", b"authenticated", expires_at_ms=100)
        reopened.corrupt_for_test("tenant-a", corrupt.digest, b"tampered")
        with pytest.raises(ChunkCorrupt):
            reopened.read("tenant-a", corrupt)


def test_file_store_serializes_concurrent_put_publish_read_and_delete(tmp_path: Path) -> None:
    workers = 8
    barrier = threading.Barrier(workers)
    with FileContentStore(tmp_path / "concurrent-store") as store:

        def publish(index: int) -> tuple[str, str, str]:
            barrier.wait(timeout=5.0)
            shared = store.put("tenant-a", b"shared-state", expires_at_ms=10)
            unique = store.put("tenant-a", f"state-{index}".encode(), expires_at_ms=10)
            manifest = store.publish(
                tenant_id="tenant-a",
                kind="complete",
                chunks=(shared, unique),
                published_at_ms=index,
            )
            assert store.read("tenant-a", unique) == f"state-{index}".encode()
            return manifest.manifest_id, shared.digest, unique.digest

        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = tuple(executor.map(publish, range(workers)))

        assert len({manifest_id for manifest_id, _shared, _unique in results}) == workers
        assert len({shared for _manifest, shared, _unique in results}) == 1

        with ThreadPoolExecutor(max_workers=workers) as executor:
            tuple(
                executor.map(
                    lambda manifest_id: store.delete_manifest("tenant-a", manifest_id),
                    (manifest_id for manifest_id, _shared, _unique in results),
                )
            )

        expected = {results[0][1]} | {unique for _manifest, _shared, unique in results}
        assert set(store.gc(now_ms=10)) == expected
