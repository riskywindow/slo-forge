from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from sloforge.genesis.artifacts import ArtifactStoreError, ContentAddressedArtifactStore
from sloforge.genesis.capsule import Digest


def test_content_addressed_store_deduplicates_and_verifies(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "cas")
    first = store.put_bytes(b"proof carrying runtime")
    second = store.put_stream(io.BytesIO(b"proof carrying runtime"))

    assert first == second
    assert first.object_path.name == first.digest.value
    assert store.verify(first.digest)
    assert store.read_bytes(first.digest) == b"proof carrying runtime"
    assert first.object_path.stat().st_mode & 0o222 == 0


def test_content_addressed_store_detects_corruption(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "cas")
    artifact = store.put_bytes(b"trusted evidence")
    os.chmod(artifact.object_path, 0o644)
    artifact.object_path.write_bytes(b"modified evidence")

    assert not store.verify(artifact.digest)
    with pytest.raises(ArtifactStoreError, match="integrity"):
        store.put_bytes(b"trusted evidence")


def test_content_addressed_store_bounds_and_materialization(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "cas", maximum_object_bytes=8)
    with pytest.raises(ArtifactStoreError, match="maximum_object_bytes"):
        store.put_bytes(b"123456789")

    store = ContentAddressedArtifactStore(tmp_path / "cas-large", maximum_object_bytes=32)
    artifact = store.put_json({"b": 2, "a": 1})
    destination = tmp_path / "capsule" / "evidence.json"
    store.materialize(artifact.digest, destination)
    assert destination.read_bytes() == b'{"a":1,"b":2}'
    with pytest.raises(ArtifactStoreError, match="already exists"):
        store.materialize(artifact.digest, destination)


def test_content_addressed_store_rejects_symlink_source(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "cas")
    source = tmp_path / "source"
    source.write_bytes(b"source")
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(ArtifactStoreError, match="non-symlink"):
        store.put_file(link)


def test_content_addressed_store_rejects_symlinked_internal_directory(tmp_path: Path) -> None:
    root = tmp_path / "cas"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "objects").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactStoreError, match="symlink"):
        ContentAddressedArtifactStore(root)


def test_content_addressed_store_rejects_symlinked_materialization_parent(
    tmp_path: Path,
) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "cas")
    artifact = store.put_bytes(b"trusted evidence")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "capsule"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactStoreError, match="symlink component"):
        store.materialize(artifact.digest, linked_parent / "evidence.json")

    assert not (outside / "evidence.json").exists()


def test_content_addressed_store_rehashes_during_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "cas")
    artifact = store.put_bytes(b"trusted evidence")
    original_verify = store._verify_existing

    def swap_after_verify(path: Path, digest: Digest, size_bytes: int | None) -> int:
        size = original_verify(path, digest, size_bytes)
        os.chmod(path, 0o644)
        path.write_bytes(b"hostile replacement")
        return size

    monkeypatch.setattr(store, "_verify_existing", swap_after_verify)
    destination = tmp_path / "capsule" / "evidence.json"
    with pytest.raises(ArtifactStoreError, match="changed during materialization"):
        store.materialize(artifact.digest, destination)
    assert not destination.exists()
