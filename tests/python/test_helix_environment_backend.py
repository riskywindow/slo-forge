from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

import pytest

from sloforge.helix.environments import (
    REDACTED,
    CapsuleRetiredError,
    CaptureDisabledError,
    ContentCorruptionError,
    EnvironmentBackend,
    PathSafetyError,
    RepositorySecurityError,
    ResourceBounds,
    ResourceLimitError,
    ServiceDescriptor,
)
from sloforge.helix.security import build_production_capture_grant


def _fixture(root: Path) -> None:
    (root / "app").mkdir(parents=True)
    (root / "app" / "config.txt").write_text("version=1\n")
    (root / "app" / "config.txt").chmod(0o640)
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (root / "uv.lock").write_text("version = 1\n")
    (root / ".env").write_text("API_TOKEN=top-secret\n")
    (root / "current").symlink_to("app/config.txt")
    connection = sqlite3.connect(root / "service.sqlite")
    connection.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("INSERT INTO events(value) VALUES ('captured')")
    connection.commit()
    connection.close()


def test_capture_restore_is_deterministic_redacted_and_reconstructs_sqlite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _fixture(source)
    backend = EnvironmentBackend(tmp_path / "store", tenant_id="tenant-a")
    service = ServiceDescriptor(
        service_id="fixture-db",
        kind="sqlite",
        state_paths=("service.sqlite",),
        environment={"DATABASE_PASSWORD": "top-secret", "MODE": "test"},
        metadata={"token": "top-secret", "owner": "fixture"},
    )

    first = backend.capture(
        source,
        seed=73,
        services=(service,),
        tool_state={"api_token": "top-secret", "version": "1"},
        cache_state={"key": "prefix-top-secret-suffix"},
        fault_state={"scheduled": ["disk-delay"]},
        virtual_time_ns=1234,
        rng_counter=3,
        event_watermark=9,
        secret_values=("top-secret",),
    )
    second = backend.capture(
        source,
        seed=73,
        services=(service,),
        tool_state={"api_token": "top-secret", "version": "1"},
        cache_state={"key": "prefix-top-secret-suffix"},
        fault_state={"scheduled": ["disk-delay"]},
        virtual_time_ns=1234,
        rng_counter=3,
        event_watermark=9,
        secret_values=("top-secret",),
    )

    assert first.capsule_id == second.capsule_id
    assert first.runtime.rng_state_hash == second.runtime.rng_state_hash
    assert backend.restore_rng(first).random() == backend.restore_rng(second).random()
    assert backend.artifact_watermark(first).watermark == 9
    assert first.dependency_locks[0].path == "uv.lock"
    assert "top-secret" not in first.to_json()
    assert first.tool_state["api_token"] == REDACTED
    assert first.services[0].environment["DATABASE_PASSWORD"] == REDACTED
    assert any(entry.path == ".env" and entry.kind.value == "redacted" for entry in first.files)

    restored = backend.restore(first, tmp_path / "restored")
    assert (restored / "app" / "config.txt").read_text() == "version=1\n"
    assert stat.S_IMODE((restored / "app" / "config.txt").stat().st_mode) == 0o640
    assert (restored / "current").is_symlink()
    assert os.readlink(restored / "current") == "app/config.txt"
    assert (restored / ".env").read_text() == REDACTED
    reconstructed = backend.reconstruct_services(first, restored)
    assert reconstructed[0].status == "ready"
    with sqlite3.connect(restored / "service.sqlite") as connection:
        assert connection.execute("SELECT value FROM events").fetchone() == ("captured",)
    assert (
        backend.accounting().object_count
        < sum(entry.kind.value == "file" for entry in first.files) * 2
    )


def test_cow_branches_are_isolated_checkpointed_compared_and_cleaned(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _fixture(source)
    backend = EnvironmentBackend(tmp_path / "store", tenant_id="tenant-a")
    base = backend.capture(source, seed=11)
    left = backend.fork(base, branch_id="left", bounds=ResourceBounds(max_bytes=1_000_000))
    right = backend.fork(base, branch_id="right", bounds=ResourceBounds(max_bytes=1_000_000))

    left.write_text("app/config.txt", "version=left\n")
    left.write_text("app/added.txt", "only-left\n")
    right.write_text("app/config.txt", "version=right\n")
    assert left.read_text("app/config.txt") == "version=left\n"
    assert right.read_text("app/config.txt") == "version=right\n"
    assert (source / "app" / "config.txt").read_text() == "version=1\n"
    comparison = left.compare(right)
    assert {difference.path for difference in comparison.differences} == {
        "app/added.txt",
        "app/config.txt",
    }
    assert {item.path for item in left.overlay()} == {"app/added.txt", "app/config.txt"}
    checkpoint = left.checkpoint()
    assert checkpoint.parent_capsule_id == base.capsule_id
    assert checkpoint.capsule_id != base.capsule_id
    checkpoint_config = next(entry for entry in checkpoint.files if entry.path == "app/config.txt")
    assert (
        backend.store.read_digest(
            "tenant-a",
            checkpoint_config.content_hash or "",
            expected_size=checkpoint_config.size_bytes,
        )
        == b"version=left\n"
    )
    assert left.log[-1].operation == "checkpoint"

    left_workspace = left.workspace
    left.cleanup()
    assert not left_workspace.exists()
    assert right.workspace.exists()
    right.cleanup()
    assert backend.accounting().branch_workspace_bytes == 0


def test_path_safety_resource_bounds_and_production_default(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.txt").write_text("safe")
    (source / "escape").symlink_to("../outside")
    backend = EnvironmentBackend(tmp_path / "store")
    with pytest.raises(PathSafetyError):
        backend.capture(source, seed=1)
    (source / "escape").unlink()
    with pytest.raises(CaptureDisabledError):
        backend.capture(source, seed=1, production=True)
    capsule = backend.capture(source, seed=1)
    branch = backend.fork(
        capsule,
        branch_id="bounded",
        bounds=ResourceBounds(max_bytes=5, max_files=10, max_log_entries=10),
    )
    with pytest.raises(ResourceLimitError):
        branch.write_text("more.txt", "too much")
    with pytest.raises(PathSafetyError):
        branch.write_text("../escape", "bad")
    branch.cleanup()


def test_corruption_is_detected_before_restore(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "state.bin").write_bytes(b"authenticated-state")
    backend = EnvironmentBackend(tmp_path / "store", tenant_id="tenant-a")
    capsule = backend.capture(source, seed=5)
    file_entry = next(entry for entry in capsule.files if entry.path == "state.bin")
    object_path = backend.store.object_path("tenant-a", file_entry.content_hash or "")
    object_path.write_bytes(b"corrupt")
    with pytest.raises(ContentCorruptionError):
        backend.restore(capsule, tmp_path / "restore")


def test_capsule_retirement_is_idempotent_tenant_scoped_and_preserves_shared_cas(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "shared.txt").write_text("shared bytes\n")
    backend = EnvironmentBackend(tmp_path / "store", tenant_id="tenant-a")
    retired = backend.capture(source, seed=1)
    survivor = backend.capture(source, seed=2)
    shared = next(entry for entry in retired.files if entry.path == "shared.txt")
    object_path = backend.store.object_path("tenant-a", shared.content_hash or "")

    receipt = backend.retire_capsule(
        retired.capsule_id,
        retired_at_ms=10,
        reason="retention window elapsed",
    )
    assert (
        backend.retire_capsule(
            retired.capsule_id,
            retired_at_ms=10,
            reason="retention window elapsed",
        )
        == receipt
    )
    assert receipt.cas_objects_deleted == ()
    assert object_path.is_file()
    with pytest.raises(CapsuleRetiredError):
        backend.load_capsule(retired.capsule_id)
    restored = backend.restore(survivor, tmp_path / "survivor")
    assert (restored / "shared.txt").read_text() == "shared bytes\n"
    with pytest.raises(CapsuleRetiredError, match="cannot be resurrected"):
        backend.capture(source, seed=1)


def test_production_capture_requires_scoped_trusted_grant(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "state.txt").write_text("authorized\n")
    approval = "a" * 64
    grant = build_production_capture_grant(
        grant_id="grant-1",
        tenant_id="tenant-a",
        request_id="capture-1",
        approver_id="privacy-reviewer",
        issued_at_ms=10,
        expires_at_ms=20,
        approval_artifact_sha256=approval,
    )
    backend = EnvironmentBackend(
        tmp_path / "store",
        tenant_id="tenant-a",
        allow_production_capture=True,
        trusted_production_approval_digests={approval},
    )
    with pytest.raises(CaptureDisabledError, match="PRODUCTION_GRANT_REQUIRED"):
        backend.capture(
            source,
            seed=1,
            production=True,
            capture_request_id="capture-1",
            authorization_checked_at_ms=15,
        )
    with pytest.raises(CaptureDisabledError, match="PRODUCTION_REQUEST_ID_REQUIRED"):
        backend.capture(
            source,
            seed=1,
            production=True,
            production_grant=grant,
            authorization_checked_at_ms=15,
        )
    capsule = backend.capture(
        source,
        seed=1,
        production=True,
        capture_request_id="capture-1",
        production_grant=grant,
        authorization_checked_at_ms=15,
    )
    assert capsule.policies.production_capture_enabled
    assert f"production-grant:{grant.grant_digest}" in capsule.policies.security_labels


def test_hostile_git_configuration_is_rejected_without_invoking_git(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / ".git").mkdir(parents=True)
    (source / ".git" / "config").write_text("[core]\nfsmonitor = hostile-helper\n")
    (source / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (source / "state.txt").write_text("safe\n")
    backend = EnvironmentBackend(tmp_path / "store")
    with pytest.raises(RepositorySecurityError, match="REPOSITORY_GIT_CONFIG_EXECUTION"):
        backend.capture(source, seed=1)


def test_git_identity_is_read_passively_without_process_execution(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / ".git" / "refs" / "heads").mkdir(parents=True)
    commit = "b" * 40
    (source / ".git" / "config").write_text("[core]\nrepositoryformatversion = 0\n")
    (source / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (source / ".git" / "refs" / "heads" / "main").write_text(f"{commit}\n")
    (source / "state.txt").write_text("safe\n")
    capsule = EnvironmentBackend(tmp_path / "store").capture(source, seed=1)
    assert capsule.git.present
    assert capsule.git.branch == "main"
    assert capsule.git.head == commit
