"""Deterministic local capture, restore, and copy-on-write environment branches."""

from __future__ import annotations

import json
import os
import random
import shutil
import sqlite3
import stat
import tempfile
import threading
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, cast

from sloforge.helix.security import (
    CaptureAccessContext,
    ProductionCaptureGrant,
    RepositoryScanPolicy,
    scan_repository,
    validate_capture_access,
)

if TYPE_CHECKING:
    from sloforge.helix.capture import ArtifactWatermark

from .models import (
    BranchComparison,
    BranchInfo,
    BranchLogEntry,
    CapsuleRetirementReceipt,
    DependencyLock,
    EntryKind,
    EnvironmentPolicies,
    EnvironmentStateCapsule,
    FileDifference,
    FileEntry,
    GitState,
    ResourceBounds,
    RuntimeState,
    ServiceDescriptor,
    StorageAccounting,
    canonical_json,
    content_digest,
    service_from_ir,
)
from .security import (
    REDACTED,
    PathSafetyError,
    is_redacted_path,
    normalize_relative_path,
    redact_mapping,
    safe_destination,
    validate_symlink_target,
)
from .store import ContentCorruptionError, LocalContentStore, validate_identifier

_LOCK_NAMES = frozenset(
    {
        "cargo.lock",
        "composer.lock",
        "gemfile.lock",
        "package-lock.json",
        "pnpm-lock.yaml",
        "poetry.lock",
        "pdm.lock",
        "pipfile.lock",
        "uv.lock",
        "yarn.lock",
    }
)


class CaptureDisabledError(PermissionError):
    """A production capture was requested without explicit local authorization."""


class RepositorySecurityError(PathSafetyError):
    """A hostile repository failed the tool-free preflight scan."""


class ResourceLimitError(RuntimeError):
    """A bounded capture or branch resource limit was exceeded."""


class BranchNotFoundError(KeyError):
    """The requested branch is not live in this backend."""


class CapsuleRetiredError(FileNotFoundError):
    """The tenant explicitly retired this capsule identity."""


@dataclass(frozen=True, slots=True)
class ReconstructedService:
    service_id: str
    kind: str
    status: str
    state_paths: tuple[str, ...]
    detail: str


@dataclass(slots=True)
class _BranchState:
    info: BranchInfo
    base: EnvironmentStateCapsule
    log: list[BranchLogEntry]


def _mapping_get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _rng_state_hash(seed: int, counter: int) -> str:
    generator = random.Random(seed)
    for _ in range(counter):
        generator.random()
    return content_digest(repr(generator.getstate()).encode())


def _capture_git(root: Path) -> GitState:
    """Read bounded inert Git identity metadata without invoking Git or repository helpers."""

    git_directory = root / ".git"
    if not git_directory.is_dir() or git_directory.is_symlink():
        return GitState()
    head_path = git_directory / "HEAD"
    if not head_path.is_file() or head_path.is_symlink() or head_path.stat().st_size > 4096:
        return GitState(present=True)
    head_value = head_path.read_text(encoding="utf-8", errors="replace").strip()
    branch: str | None = None
    head: str | None = None
    if head_value.startswith("ref: "):
        reference = head_value[5:]
        try:
            normalized = normalize_relative_path(reference)
        except PathSafetyError:
            return GitState(present=True)
        if normalized.startswith("refs/heads/"):
            branch = normalized[len("refs/heads/") :]
        reference_path = safe_destination(git_directory, normalized)
        if (
            reference_path.is_file()
            and not reference_path.is_symlink()
            and reference_path.stat().st_size <= 4096
        ):
            candidate = reference_path.read_text(encoding="ascii", errors="ignore").strip()
            if len(candidate) in {40, 64} and all(char in "0123456789abcdef" for char in candidate):
                head = candidate
    elif len(head_value) in {40, 64} and all(char in "0123456789abcdef" for char in head_value):
        head = head_value
    return GitState(
        present=True,
        head=head,
        branch=branch,
        tracked_paths=(),
        untracked_paths=(),
        dirty_paths=(),
    )


class EnvironmentBackend:
    """Local-only backend with authenticated snapshots and isolated branch namespaces."""

    def __init__(
        self,
        storage_root: Path | str,
        *,
        tenant_id: str = "default",
        allow_production_capture: bool = False,
        trusted_production_approval_digests: Collection[str] = (),
        max_capture_bytes: int = 512 * 1024 * 1024,
        max_capture_files: int = 100_000,
    ) -> None:
        self.storage_root = Path(storage_root).resolve()
        self.tenant_id = validate_identifier(tenant_id, label="tenant id")
        self.allow_production_capture = allow_production_capture
        self.trusted_production_approval_digests = frozenset(trusted_production_approval_digests)
        if min(max_capture_bytes, max_capture_files) < 1:
            raise ValueError("capture resource bounds must be positive")
        self.max_capture_bytes = max_capture_bytes
        self.max_capture_files = max_capture_files
        self.store = LocalContentStore(self.storage_root, max_object_bytes=max_capture_bytes)
        self._lock = threading.RLock()
        self._capsules: dict[str, EnvironmentStateCapsule] = {}
        self._branches: dict[str, _BranchState] = {}

    def capture(
        self,
        source_root: Path | str,
        *,
        seed: int,
        parent: EnvironmentStateCapsule | None = None,
        services: Sequence[ServiceDescriptor | object] = (),
        working_directories: Sequence[str] = (".",),
        tool_state: Mapping[str, object] | None = None,
        cache_state: Mapping[str, object] | None = None,
        fault_state: Mapping[str, object] | None = None,
        rng_counter: int = 0,
        virtual_time_ns: int = 0,
        event_watermark: int = -1,
        production: bool = False,
        capture_request_id: str | None = None,
        actor_tenant_id: str | None = None,
        production_grant: ProductionCaptureGrant | None = None,
        authorization_checked_at_ms: int | None = None,
        secret_values: Sequence[str] = (),
        redacted_paths: Sequence[str] = (),
        allowed_tools: Sequence[str] = (),
        security_labels: Sequence[str] = (),
    ) -> EnvironmentStateCapsule:
        if production and not self.allow_production_capture:
            raise CaptureDisabledError("production environment capture is disabled by default")
        if production and capture_request_id is None:
            raise CaptureDisabledError(
                "production capture authorization failed: PRODUCTION_REQUEST_ID_REQUIRED"
            )
        if production and authorization_checked_at_ms is None:
            raise CaptureDisabledError(
                "production capture authorization failed: PRODUCTION_AUTHORIZATION_TIME_REQUIRED"
            )
        request_id = capture_request_id or "local-capture"
        access = CaptureAccessContext(
            request_id=request_id,
            actor_tenant_id=actor_tenant_id or self.tenant_id,
            resource_tenant_id=self.tenant_id,
            production=production,
            explicit_production_opt_in=production and production_grant is not None,
            production_grant=production_grant,
            reuse_source_tenant_id=parent.tenant_id if parent is not None else None,
            cross_tenant_reuse_enabled=False,
        )
        access_report = validate_capture_access(
            access,
            checked_at_ms=authorization_checked_at_ms or 0,
            trusted_approval_digests=self.trusted_production_approval_digests,
        )
        if not access_report.passed:
            codes = ", ".join(item.code for item in access_report.violations)
            if production:
                raise CaptureDisabledError(f"production capture authorization failed: {codes}")
            raise PermissionError(f"environment capture authorization failed: {codes}")
        if virtual_time_ns < 0 or not 0 <= rng_counter <= 1_000_000 or event_watermark < -1:
            raise ValueError("runtime counters and watermarks are outside their valid range")
        requested_source = Path(source_root).absolute()
        if requested_source.is_symlink():
            raise ValueError("environment source must be a non-symlink directory")
        source = requested_source.resolve()
        if not source.is_dir():
            raise ValueError("environment source must be a non-symlink directory")
        repository_scan = scan_repository(
            source,
            policy=RepositoryScanPolicy(
                max_entries=self.max_capture_files,
                max_total_bytes=self.max_capture_bytes,
                max_file_bytes=self.max_capture_bytes,
                max_depth=128,
                allow_internal_symlinks=True,
            ),
        )
        if not repository_scan.passed:
            codes = ", ".join(sorted({item.code for item in repository_scan.findings}))
            raise RepositorySecurityError(f"repository security preflight failed: {codes}")
        if parent is not None:
            parent.verify_identity()
            if parent.tenant_id != self.tenant_id:
                raise PermissionError("incremental parents cannot cross tenant boundaries")
        normalized_working = tuple(
            "." if item == "." else normalize_relative_path(item) for item in working_directories
        )
        for item in normalized_working:
            candidate = source if item == "." else safe_destination(source, item)
            if not candidate.is_dir() or candidate.is_symlink():
                raise ValueError(f"declared working directory is unavailable: {item}")
        normalized_services = tuple(service_from_ir(item) for item in services)
        normalized_services = tuple(
            self._redact_service(item, secrets=secret_values) for item in normalized_services
        )
        for service in normalized_services:
            if service.working_directory != ".":
                normalize_relative_path(service.working_directory)
            for state_path in service.state_paths:
                normalize_relative_path(state_path)
                state_source = safe_destination(source, state_path)
                if not state_source.exists() and not state_source.is_symlink():
                    raise ValueError(
                        f"service {service.service_id!r} state path is unavailable: {state_path}"
                    )
        entries = self._capture_files(
            source, redacted_paths=tuple(redacted_paths), secret_values=tuple(secret_values)
        )
        dependency_locks = tuple(
            DependencyLock(path=entry.path, content_hash=cast(str, entry.content_hash))
            for entry in entries
            if entry.kind is EntryKind.FILE
            and PurePosixPath(entry.path).name.lower() in _LOCK_NAMES
        )
        policies = EnvironmentPolicies(
            tenant_id=self.tenant_id,
            production_capture_enabled=production,
            external_side_effects_enabled=False,
            cross_tenant_sharing_enabled=False,
            network_enabled=False,
            root_privileges_enabled=False,
            secret_redaction_enabled=True,
            redacted_paths=tuple(sorted(redacted_paths)),
            allowed_tools=tuple(sorted(set(allowed_tools))),
            security_labels=tuple(
                sorted(
                    {
                        *security_labels,
                        *(
                            (f"production-grant:{production_grant.grant_digest}",)
                            if production_grant is not None
                            else ()
                        ),
                    }
                )
            ),
        )
        capsule = EnvironmentStateCapsule.build(
            tenant_id=self.tenant_id,
            seed=seed,
            files=entries,
            git=_capture_git(source),
            dependency_locks=dependency_locks,
            working_directories=normalized_working,
            services=normalized_services,
            tool_state=redact_mapping(tool_state or {}, secrets=secret_values),
            cache_state=redact_mapping(cache_state or {}, secrets=secret_values),
            runtime=RuntimeState(
                seed=seed,
                rng_state_hash=_rng_state_hash(seed, rng_counter),
                rng_counter=rng_counter,
                virtual_time_ns=virtual_time_ns,
                fault_state=redact_mapping(fault_state or {}, secrets=secret_values),
            ),
            policies=policies,
            parent_capsule_id=parent.capsule_id if parent is not None else None,
            event_watermark=event_watermark,
        )
        capsule.verify_identity()
        self._persist_capsule(capsule)
        self._capsules[capsule.capsule_id] = capsule
        return capsule

    def capture_from_ir(
        self,
        environment: object,
        *,
        seed: int,
        source_root: Path | str | None = None,
        parent: EnvironmentStateCapsule | None = None,
    ) -> EnvironmentStateCapsule:
        """Capture a duck-typed Helix IR environment without importing its package."""

        root_value = (
            source_root
            or _mapping_get(environment, "source_root")
            or _mapping_get(environment, "root")
        )
        if root_value is None:
            raise ValueError("IR environment does not declare a source root")
        return self.capture(
            cast(Path | str, root_value),
            seed=seed,
            parent=parent,
            services=cast(Sequence[object], _mapping_get(environment, "services", ())),
            working_directories=cast(
                Sequence[str], _mapping_get(environment, "working_directories", (".",))
            ),
            tool_state=cast(Mapping[str, object], _mapping_get(environment, "tool_state", {})),
            cache_state=cast(Mapping[str, object], _mapping_get(environment, "cache_state", {})),
            fault_state=cast(Mapping[str, object], _mapping_get(environment, "fault_state", {})),
            rng_counter=int(cast(int | str, _mapping_get(environment, "rng_counter", 0))),
            virtual_time_ns=int(
                cast(int | str | bytes, _mapping_get(environment, "virtual_time_ns", 0))
            ),
            event_watermark=int(cast(int | str, _mapping_get(environment, "event_watermark", -1))),
            production=bool(_mapping_get(environment, "production", False)),
            redacted_paths=cast(Sequence[str], _mapping_get(environment, "redacted_paths", ())),
            allowed_tools=cast(Sequence[str], _mapping_get(environment, "allowed_tools", ())),
            security_labels=cast(Sequence[str], _mapping_get(environment, "security_labels", ())),
        )

    @staticmethod
    def _redact_service(service: ServiceDescriptor, *, secrets: Sequence[str]) -> ServiceDescriptor:
        environment = {
            key: str(value)
            for key, value in redact_mapping(service.environment, secrets=secrets).items()
        }
        return replace(
            service,
            environment=environment,
            metadata=redact_mapping(service.metadata, secrets=secrets),
        )

    def _capture_files(
        self,
        root: Path,
        *,
        redacted_paths: tuple[str, ...],
        secret_values: tuple[str, ...],
    ) -> tuple[FileEntry, ...]:
        del secret_values  # file paths are redacted as units; secret bytes never enter the CAS
        entries: list[FileEntry] = []
        total_bytes = 0
        pending = [root]
        # Exclude the CAS only when it is nested inside the requested source.
        # Branch workspaces intentionally live below the backend storage root;
        # treating every descendant of the CAS as excluded made branch
        # checkpoints silently publish empty, identical capsules.
        exclude_storage = self.storage_root == root or self.storage_root.is_relative_to(root)
        while pending:
            directory = pending.pop()
            children = sorted(os.scandir(directory), key=lambda item: item.name, reverse=True)
            for child in children:
                path = Path(child.path)
                if exclude_storage and (
                    path == self.storage_root or self.storage_root in path.parents
                ):
                    continue
                relative = path.relative_to(root).as_posix()
                if relative == ".git" or relative.startswith(".git/"):
                    continue
                stat_result = child.stat(follow_symlinks=False)
                mode = stat.S_IMODE(stat_result.st_mode)
                if child.is_symlink():
                    target = os.readlink(child.path)
                    validate_symlink_target(relative, target)
                    entries.append(
                        FileEntry(
                            path=relative,
                            kind=EntryKind.SYMLINK,
                            mode=mode,
                            symlink_target=target,
                        )
                    )
                elif child.is_dir(follow_symlinks=False):
                    entries.append(FileEntry(path=relative, kind=EntryKind.DIRECTORY, mode=mode))
                    pending.append(path)
                elif child.is_file(follow_symlinks=False):
                    if stat_result.st_size > self.max_capture_bytes:
                        raise ResourceLimitError(f"file exceeds capture bound: {relative}")
                    if is_redacted_path(relative, redacted_paths):
                        entries.append(
                            FileEntry(
                                path=relative,
                                kind=EntryKind.REDACTED,
                                mode=mode,
                                content_hash=content_digest(REDACTED.encode()),
                                size_bytes=len(REDACTED),
                            )
                        )
                    else:
                        data = path.read_bytes()
                        if len(data) != stat_result.st_size:
                            raise OSError(f"file changed while being captured: {relative}")
                        reference = self.store.put(self.tenant_id, data)
                        entries.append(
                            FileEntry(
                                path=relative,
                                kind=EntryKind.FILE,
                                mode=mode,
                                content_hash=reference.digest,
                                size_bytes=reference.size_bytes,
                            )
                        )
                        total_bytes += len(data)
                else:
                    raise ValueError(f"unsupported special filesystem entry: {relative}")
                if len(entries) > self.max_capture_files:
                    raise ResourceLimitError("environment capture file count bound exceeded")
                if total_bytes > self.max_capture_bytes:
                    raise ResourceLimitError("environment capture byte bound exceeded")
        return tuple(sorted(entries, key=lambda item: item.path))

    def _persist_capsule(self, capsule: EnvironmentStateCapsule) -> None:
        with self._lock:
            directory = self.storage_root / "tenants" / self.tenant_id / "capsules"
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            target = directory / f"{capsule.capsule_id}.json"
            if self._retirement_exists(capsule.capsule_id):
                raise CapsuleRetiredError(
                    "a retired environment capsule identity cannot be resurrected"
                )
            if target.exists():
                existing = EnvironmentStateCapsule.from_dict(json.loads(target.read_text()))
                if existing != capsule:
                    raise ContentCorruptionError("capsule identifier collision")
                return
            self._atomic_write(target, capsule.to_json().encode(), prefix=".capsule-")

    @staticmethod
    def _atomic_write(target: Path, payload: bytes, *, prefix: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary_name = tempfile.mkstemp(prefix=prefix, dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def _tenant_storage(self) -> Path:
        return self.storage_root / "tenants" / self.tenant_id

    def _retirement_paths(self, capsule_id: str) -> tuple[Path, Path, Path]:
        root = self._tenant_storage()
        return (
            root / "retirement-intents" / f"{capsule_id}.json",
            root / "retirement-receipts" / f"{capsule_id}.json",
            root / "retired-capsules" / f"{capsule_id}.json",
        )

    def _retirement_exists(self, capsule_id: str) -> bool:
        return any(path.exists() for path in self._retirement_paths(capsule_id))

    def retire_capsule(
        self,
        capsule_id: str,
        *,
        retired_at_ms: int,
        reason: str,
    ) -> CapsuleRetirementReceipt:
        """Retire one exact tenant capsule while retaining all possibly shared CAS bytes."""

        capsule_id = validate_identifier(capsule_id, label="capsule id")
        with self._lock:
            intent, completed, retired_manifest = self._retirement_paths(capsule_id)
            receipt_path = completed if completed.is_file() else intent
            if receipt_path.is_file() and not receipt_path.is_symlink():
                receipt = CapsuleRetirementReceipt.from_dict(json.loads(receipt_path.read_text()))
                if (
                    receipt.tenant_id != self.tenant_id
                    or receipt.capsule_id != capsule_id
                    or receipt.retired_at_ms != retired_at_ms
                    or receipt.reason != reason
                ):
                    raise ValueError("capsule retirement retry changed its immutable request")
            else:
                live = [
                    state.info.branch_id
                    for state in self._branches.values()
                    if state.base.capsule_id == capsule_id
                ]
                if live:
                    raise ResourceLimitError(
                        "capsule retirement is blocked by live branches: " + ", ".join(sorted(live))
                    )
                manifest = self._tenant_storage() / "capsules" / f"{capsule_id}.json"
                if not manifest.is_file() or manifest.is_symlink():
                    raise FileNotFoundError(capsule_id)
                capsule = EnvironmentStateCapsule.from_dict(json.loads(manifest.read_text()))
                if capsule.tenant_id != self.tenant_id:
                    raise PermissionError("cross-tenant capsule retirement is disabled")
                receipt = CapsuleRetirementReceipt.build(
                    capsule_id=capsule_id,
                    tenant_id=self.tenant_id,
                    retired_at_ms=retired_at_ms,
                    reason=reason,
                    cas_content_digests=tuple(
                        entry.content_hash
                        for entry in capsule.files
                        if entry.kind is EntryKind.FILE and entry.content_hash is not None
                    ),
                )
                self._atomic_write(
                    intent,
                    canonical_json(receipt.to_dict()),
                    prefix=".retirement-intent-",
                )
            manifest = self._tenant_storage() / "capsules" / f"{capsule_id}.json"
            if manifest.exists():
                retired_manifest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                manifest.replace(retired_manifest)
            elif not retired_manifest.is_file():
                raise ContentCorruptionError("retired capsule manifest is missing")
            self._atomic_write(
                completed,
                canonical_json(receipt.to_dict()),
                prefix=".retirement-receipt-",
            )
            intent.unlink(missing_ok=True)
            self._capsules.pop(capsule_id, None)
            return receipt

    def load_capsule(self, capsule_id: str) -> EnvironmentStateCapsule:
        validate_identifier(capsule_id, label="capsule id")
        with self._lock:
            if self._retirement_exists(capsule_id):
                raise CapsuleRetiredError(capsule_id)
            cached = self._capsules.get(capsule_id)
            if cached is not None:
                cached.verify_identity()
                return cached
            path = (
                self.storage_root / "tenants" / self.tenant_id / "capsules" / f"{capsule_id}.json"
            )
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(capsule_id)
            capsule = EnvironmentStateCapsule.from_dict(json.loads(path.read_text()))
            if capsule.tenant_id != self.tenant_id:
                raise PermissionError("cross-tenant capsule access is disabled")
            self._capsules[capsule.capsule_id] = capsule
            return capsule

    def restore(
        self,
        capsule: EnvironmentStateCapsule,
        destination: Path | str,
        *,
        allow_existing_empty: bool = True,
    ) -> Path:
        capsule.verify_identity()
        if capsule.tenant_id != self.tenant_id:
            raise PermissionError("cross-tenant capsule restore is disabled")
        root = Path(destination).absolute()
        if root.is_symlink():
            raise PathSafetyError("restore destination cannot be a symlink")
        if root.exists():
            if not root.is_dir():
                raise ValueError("restore destination is not a directory")
            if not allow_existing_empty or any(root.iterdir()):
                raise FileExistsError("restore destination must be empty")
        else:
            root.mkdir(parents=True, mode=0o700)
        directories: list[tuple[Path, int]] = []
        for entry in capsule.files:
            target = safe_destination(root, entry.path)
            if entry.kind is EntryKind.DIRECTORY:
                target.mkdir(parents=True, exist_ok=False, mode=0o700)
                directories.append((target, entry.mode))
                continue
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if entry.kind is EntryKind.SYMLINK:
                target_value = cast(str, entry.symlink_target)
                validate_symlink_target(entry.path, target_value)
                target.symlink_to(target_value)
            elif entry.kind is EntryKind.REDACTED:
                target.write_bytes(REDACTED.encode())
                target.chmod(entry.mode)
            else:
                digest = cast(str, entry.content_hash)
                target.write_bytes(
                    self.store.read_digest(self.tenant_id, digest, expected_size=entry.size_bytes)
                )
                target.chmod(entry.mode)
        for directory, mode in sorted(
            directories, key=lambda item: len(item[0].parts), reverse=True
        ):
            directory.chmod(mode)
        return root

    @staticmethod
    def restore_rng(capsule: EnvironmentStateCapsule) -> random.Random:
        """Reconstruct the declared deterministic RNG position from seed and draw counter."""

        capsule.verify_identity()
        generator = random.Random(capsule.seed)
        for _ in range(capsule.runtime.rng_counter):
            generator.random()
        if content_digest(repr(generator.getstate()).encode()) != capsule.runtime.rng_state_hash:
            raise ContentCorruptionError("reconstructed RNG state differs from the capsule")
        return generator

    @staticmethod
    def artifact_watermark(capsule: EnvironmentStateCapsule) -> ArtifactWatermark:
        """Late-bind a coordinated-capture reference when that package is available."""

        capsule.verify_identity()
        from sloforge.helix.capture import ArtifactWatermark

        return ArtifactWatermark(
            artifact_id=capsule.capsule_id,
            watermark=capsule.event_watermark,
            digest=capsule.capsule_id,
        )

    @staticmethod
    def artifact_payload(capsule: EnvironmentStateCapsule) -> bytes:
        """Return the exact manifest bytes authenticated by ``artifact_watermark``."""

        capsule.verify_identity()
        return canonical_json(capsule.manifest_dict())

    def reconstruct_services(
        self,
        capsule: EnvironmentStateCapsule,
        destination: Path | str,
        *,
        start_processes: bool = False,
    ) -> tuple[ReconstructedService, ...]:
        """Validate reconstructed service state; process launch is intentionally disabled."""

        if start_processes:
            raise PermissionError("external service process launch is disabled by default")
        root = Path(destination).absolute()
        results: list[ReconstructedService] = []
        for service in capsule.services:
            paths = tuple(normalize_relative_path(item) for item in service.state_paths)
            for item in paths:
                target = safe_destination(root, item)
                if not target.exists() or target.is_symlink():
                    raise FileNotFoundError(
                        f"reconstructed service state is unavailable for {service.service_id}: {item}"
                    )
            detail = "recipe and declared state restored; process not started"
            status = "ready"
            if service.kind.lower() in {"sqlite", "sqlite3", "database"}:
                for item in paths:
                    database = safe_destination(root, item)
                    try:
                        connection = sqlite3.connect(
                            f"file:{database.as_posix()}?mode=ro", uri=True, timeout=2.0
                        )
                        try:
                            row = connection.execute("PRAGMA integrity_check").fetchone()
                        finally:
                            connection.close()
                    except sqlite3.DatabaseError as exc:
                        raise ContentCorruptionError(
                            f"reconstructed SQLite state is corrupt: {item}"
                        ) from exc
                    if row is None or row[0] != "ok":
                        raise ContentCorruptionError(
                            f"reconstructed SQLite state failed integrity check: {item}"
                        )
                detail = "SQLite state restored and integrity checked; no server process required"
            results.append(
                ReconstructedService(
                    service_id=service.service_id,
                    kind=service.kind,
                    status=status if service.reconstructible else "metadata-only",
                    state_paths=paths,
                    detail=detail,
                )
            )
        return tuple(results)

    def _fork(
        self,
        capsule: EnvironmentStateCapsule,
        *,
        branch_id: str,
        seed: int | None = None,
        bounds: ResourceBounds | None = None,
    ) -> EnvironmentBranch:
        capsule.verify_identity()
        if capsule.tenant_id != self.tenant_id:
            raise PermissionError("branches cannot cross tenant boundaries")
        branch_id = validate_identifier(branch_id, label="branch id")
        if branch_id in self._branches:
            raise ValueError(f"branch {branch_id!r} already exists")
        chosen_seed = capsule.seed if seed is None else seed
        branch_bounds = bounds or ResourceBounds()
        namespace = content_digest(
            canonical_json(
                {
                    "base": capsule.capsule_id,
                    "branch": branch_id,
                    "seed": chosen_seed,
                    "tenant": self.tenant_id,
                }
            )
        )[:24]
        workspace = (
            self.storage_root
            / "tenants"
            / self.tenant_id
            / "branches"
            / branch_id
            / namespace
            / "root"
        )
        if workspace.parent.parent.exists():
            raise FileExistsError("branch namespace already exists on disk")
        workspace.parent.mkdir(parents=True, exist_ok=False, mode=0o700)
        try:
            self.restore(capsule, workspace)
            bytes_used, files_used = _workspace_usage(workspace)
            if bytes_used > branch_bounds.max_bytes or files_used > branch_bounds.max_files:
                raise ResourceLimitError("restored branch exceeds its resource bounds")
        except BaseException:
            shutil.rmtree(workspace.parent.parent, ignore_errors=True)
            raise
        info = BranchInfo(
            branch_id=branch_id,
            tenant_id=self.tenant_id,
            base_capsule_id=capsule.capsule_id,
            namespace=namespace,
            workspace=os.fspath(workspace),
            seed=chosen_seed,
            bounds=branch_bounds,
        )
        self._branches[branch_id] = _BranchState(info=info, base=capsule, log=[])
        return EnvironmentBranch(self, branch_id)

    def fork(
        self,
        capsule: EnvironmentStateCapsule,
        *,
        branch_id: str,
        seed: int | None = None,
        bounds: ResourceBounds | None = None,
    ) -> EnvironmentBranch:
        with self._lock:
            return self._fork(capsule, branch_id=branch_id, seed=seed, bounds=bounds)

    create_branch = fork
    fork_branch = fork

    def branch(self, branch_id: str) -> EnvironmentBranch:
        self._branch_state(branch_id)
        return EnvironmentBranch(self, branch_id)

    def _branch_state(self, branch_id: str) -> _BranchState:
        try:
            return self._branches[branch_id]
        except KeyError as exc:
            raise BranchNotFoundError(branch_id) from exc

    def _record_branch(
        self, branch_id: str, operation: str, path: str | None, detail: object
    ) -> None:
        with self._lock:
            state = self._branch_state(branch_id)
            if len(state.log) >= state.info.bounds.max_log_entries:
                raise ResourceLimitError("branch operation log bound exceeded")
            state.log.append(
                BranchLogEntry(
                    sequence=len(state.log),
                    operation=operation,
                    path=path,
                    detail_hash=content_digest(canonical_json(detail)),
                )
            )

    def _checkpoint(self, branch_id: str) -> EnvironmentStateCapsule:
        state = self._branch_state(branch_id)
        if len(state.log) >= state.info.bounds.max_log_entries:
            raise ResourceLimitError("branch operation log bound exceeded")
        capsule = self.capture(
            state.info.workspace,
            seed=state.info.seed,
            parent=state.base,
            services=state.base.services,
            working_directories=state.base.working_directories,
            tool_state=state.base.tool_state,
            cache_state=state.base.cache_state,
            fault_state=state.base.runtime.fault_state,
            rng_counter=state.base.runtime.rng_counter,
            virtual_time_ns=state.base.runtime.virtual_time_ns,
            event_watermark=max(state.base.event_watermark, len(state.log) - 1),
            redacted_paths=state.base.policies.redacted_paths,
            allowed_tools=state.base.policies.allowed_tools,
            security_labels=state.base.policies.security_labels,
        )
        self._record_branch(branch_id, "checkpoint", None, {"capsule": capsule.capsule_id})
        return capsule

    def checkpoint(self, branch_id: str) -> EnvironmentStateCapsule:
        with self._lock:
            return self._checkpoint(branch_id)

    def _compare(
        self,
        left: EnvironmentStateCapsule | EnvironmentBranch | str,
        right: EnvironmentStateCapsule | EnvironmentBranch | str,
    ) -> BranchComparison:
        left_id, left_files = self._comparison_files(left)
        right_id, right_files = self._comparison_files(right)
        differences: list[FileDifference] = []
        for path in sorted(set(left_files) | set(right_files)):
            left_hash = left_files.get(path)
            right_hash = right_files.get(path)
            if left_hash == right_hash:
                continue
            status = "modified"
            if left_hash is None:
                status = "added"
            elif right_hash is None:
                status = "removed"
            differences.append(
                FileDifference(path=path, left_hash=left_hash, right_hash=right_hash, status=status)
            )
        return BranchComparison(left_id=left_id, right_id=right_id, differences=tuple(differences))

    def compare(
        self,
        left: EnvironmentStateCapsule | EnvironmentBranch | str,
        right: EnvironmentStateCapsule | EnvironmentBranch | str,
    ) -> BranchComparison:
        with self._lock:
            return self._compare(left, right)

    def _comparison_files(
        self, value: EnvironmentStateCapsule | EnvironmentBranch | str
    ) -> tuple[str, dict[str, str]]:
        if isinstance(value, EnvironmentStateCapsule):
            return value.capsule_id, {
                entry.path: _entry_comparison_hash(entry) for entry in value.files
            }
        branch_id = value.branch_id if isinstance(value, EnvironmentBranch) else value
        state = self._branch_state(branch_id)
        return branch_id, _hash_workspace(Path(state.info.workspace))

    def _cleanup_branch(self, branch_id: str) -> None:
        expected_parent = self.storage_root / "tenants" / self.tenant_id / "branches"
        branch_id = validate_identifier(branch_id, label="branch id")
        state = self._branches.get(branch_id)
        if state is None:
            # Exact, explicit cleanup also recovers a namespace left by a prior
            # backend process crash. Missing is an idempotent success.
            branch_root = expected_parent / branch_id
            if not branch_root.exists() and not branch_root.is_symlink():
                return
        else:
            workspace = Path(state.info.workspace)
            branch_root = workspace.parent.parent
        if branch_root.parent != expected_parent or branch_root.name != branch_id:
            raise PathSafetyError("refusing cleanup outside the branch namespace")
        if branch_root.is_symlink():
            raise PathSafetyError("refusing cleanup of a symlink branch namespace")
        shutil.rmtree(branch_root)
        self._branches.pop(branch_id, None)

    def cleanup_branch(self, branch_id: str) -> None:
        with self._lock:
            self._cleanup_branch(branch_id)

    cleanup = cleanup_branch

    def accounting(self) -> StorageAccounting:
        capsule_bytes = sum(
            entry.size_bytes
            for capsule in self._capsules.values()
            for entry in capsule.files
            if entry.kind in {EntryKind.FILE, EntryKind.REDACTED}
        )
        branch_bytes = sum(
            _workspace_usage(Path(state.info.workspace))[0] for state in self._branches.values()
        )
        return self.store.accounting(
            self.tenant_id,
            capsule_logical_bytes=capsule_bytes,
            branch_workspace_bytes=branch_bytes,
        )

    storage_accounting = accounting
    compare_branches = compare
    checkpoint_branch = checkpoint

    def branch_overlay(self, branch: EnvironmentBranch | str) -> tuple[FileDifference, ...]:
        """Return the branch-local changes relative to its immutable base capsule."""

        with self._lock:
            branch_id = branch.branch_id if isinstance(branch, EnvironmentBranch) else branch
            state = self._branch_state(branch_id)
            return self.compare(state.base, branch_id).differences


class EnvironmentBranch:
    """A small handle exposing mutations inside one isolated branch namespace."""

    def __init__(self, backend: EnvironmentBackend, branch_id: str) -> None:
        self._backend = backend
        self.branch_id = branch_id

    @property
    def info(self) -> BranchInfo:
        with self._backend._lock:
            state = self._backend._branch_state(self.branch_id)
            return replace(state.info, log=tuple(state.log))

    @property
    def workspace(self) -> Path:
        return Path(self.info.workspace)

    @property
    def log(self) -> tuple[BranchLogEntry, ...]:
        return self.info.log

    def _read_bytes(self, path: str) -> bytes:
        normalized = normalize_relative_path(path)
        target = safe_destination(self.workspace, normalized)
        if not target.is_file() or target.is_symlink():
            raise FileNotFoundError(normalized)
        data = target.read_bytes()
        self._backend._record_branch(
            self.branch_id, "read", normalized, {"hash": content_digest(data)}
        )
        return data

    def read_bytes(self, path: str) -> bytes:
        with self._backend._lock:
            return self._read_bytes(path)

    def read_text(self, path: str, *, encoding: str = "utf-8") -> str:
        return self.read_bytes(path).decode(encoding)

    def _write_bytes(self, path: str, data: bytes, *, mode: int | None = None) -> None:
        normalized = normalize_relative_path(path)
        target = safe_destination(self.workspace, normalized)
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise PathSafetyError("branch writes cannot replace symlinks or directories")
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        bytes_used, files_used = _workspace_usage(self.workspace)
        old_size = target.stat().st_size if target.exists() else 0
        next_files = files_used if target.exists() else files_used + 1
        bounds = self.info.bounds
        if bytes_used - old_size + len(data) > bounds.max_bytes or next_files > bounds.max_files:
            raise ResourceLimitError("branch mutation exceeds its resource bounds")
        if len(self.log) >= bounds.max_log_entries:
            raise ResourceLimitError("branch operation log bound exceeded")
        descriptor, temporary_name = tempfile.mkstemp(prefix=".helix-write-", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if mode is None and target.exists():
                mode = stat.S_IMODE(target.stat().st_mode)
            os.chmod(temporary, 0o600 if mode is None else mode)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        self._backend._record_branch(
            self.branch_id,
            "write",
            normalized,
            {"hash": content_digest(data), "size": len(data)},
        )

    def write_bytes(self, path: str, data: bytes, *, mode: int | None = None) -> None:
        with self._backend._lock:
            self._write_bytes(path, data, mode=mode)

    def write_text(
        self, path: str, value: str, *, encoding: str = "utf-8", mode: int | None = None
    ) -> None:
        self.write_bytes(path, value.encode(encoding), mode=mode)

    def _delete(self, path: str) -> None:
        normalized = normalize_relative_path(path)
        target = safe_destination(self.workspace, normalized)
        if not target.exists() and not target.is_symlink():
            raise FileNotFoundError(normalized)
        if len(self.log) >= self.info.bounds.max_log_entries:
            raise ResourceLimitError("branch operation log bound exceeded")
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
        self._backend._record_branch(self.branch_id, "delete", normalized, {})

    def delete(self, path: str) -> None:
        with self._backend._lock:
            self._delete(path)

    def checkpoint(self) -> EnvironmentStateCapsule:
        return self._backend.checkpoint(self.branch_id)

    def compare(self, other: EnvironmentStateCapsule | EnvironmentBranch | str) -> BranchComparison:
        return self._backend.compare(self, other)

    def overlay(self) -> tuple[FileDifference, ...]:
        return self._backend.branch_overlay(self)

    def cleanup(self) -> None:
        self._backend.cleanup_branch(self.branch_id)


def _entry_comparison_hash(entry: FileEntry) -> str:
    if entry.kind in {EntryKind.FILE, EntryKind.REDACTED}:
        return cast(str, entry.content_hash)
    if entry.kind is EntryKind.DIRECTORY:
        return content_digest(canonical_json({"kind": "directory", "mode": entry.mode}))
    return content_digest(
        canonical_json(
            {
                "kind": "symlink",
                "mode": entry.mode,
                "target": entry.symlink_target,
            }
        )
    )


def _hash_workspace(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        for child in sorted(os.scandir(directory), key=lambda item: item.name):
            path = Path(child.path)
            relative = path.relative_to(root).as_posix()
            mode = stat.S_IMODE(child.stat(follow_symlinks=False).st_mode)
            if child.is_symlink():
                result[relative] = content_digest(
                    canonical_json({"kind": "symlink", "mode": mode, "target": os.readlink(path)})
                )
            elif child.is_dir(follow_symlinks=False):
                result[relative] = content_digest(
                    canonical_json({"kind": "directory", "mode": mode})
                )
                pending.append(path)
            elif child.is_file(follow_symlinks=False):
                result[relative] = content_digest(path.read_bytes())
    return result


def _workspace_usage(root: Path) -> tuple[int, int]:
    bytes_used = 0
    files_used = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        for child in os.scandir(directory):
            stat_result = child.stat(follow_symlinks=False)
            files_used += 1
            if child.is_dir(follow_symlinks=False):
                pending.append(Path(child.path))
            elif child.is_file(follow_symlinks=False):
                bytes_used += stat_result.st_size
    return bytes_used, files_used
