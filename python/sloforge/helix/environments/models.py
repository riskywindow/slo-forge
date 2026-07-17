"""Typed, deterministic descriptions of a reconstructible Helix environment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, TypeAlias, cast

JsonValue: TypeAlias = bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None


def canonical_json(value: object) -> bytes:
    """Encode metadata without host- or insertion-order-dependent variation."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def content_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class EntryKind(StrEnum):
    DIRECTORY = "directory"
    FILE = "file"
    SYMLINK = "symlink"
    REDACTED = "redacted"


@dataclass(frozen=True, slots=True)
class FileEntry:
    path: str
    kind: EntryKind
    mode: int
    content_hash: str | None = None
    size_bytes: int = 0
    symlink_target: str | None = None

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or self.path in {"", "."}:
            raise ValueError(f"unsafe capsule path {self.path!r}")
        if self.mode < 0 or self.mode > 0o7777:
            raise ValueError("file mode is outside the portable permission range")
        if self.kind is EntryKind.FILE and self.content_hash is None:
            raise ValueError("file entry requires a content hash")
        if self.kind is EntryKind.SYMLINK and self.symlink_target is None:
            raise ValueError("symlink entry requires a target")
        if self.size_bytes < 0:
            raise ValueError("file size cannot be negative")


@dataclass(frozen=True, slots=True)
class DependencyLock:
    path: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class GitState:
    present: bool = False
    head: str | None = None
    branch: str | None = None
    tracked_paths: tuple[str, ...] = ()
    untracked_paths: tuple[str, ...] = ()
    dirty_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ServiceDescriptor:
    """Recipe and persisted inputs for a service; never a live-memory claim."""

    service_id: str
    kind: str
    command: tuple[str, ...] = ()
    working_directory: str = "."
    state_paths: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    reconstructible: bool = True


@dataclass(frozen=True, slots=True)
class RuntimeState:
    seed: int
    rng_algorithm: str = "python-mt19937"
    rng_state_hash: str = ""
    rng_counter: int = 0
    virtual_time_ns: int = 0
    fault_state: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.rng_counter <= 1_000_000 or self.virtual_time_ns < 0:
            raise ValueError("RNG counter and virtual time cannot be negative")


@dataclass(frozen=True, slots=True)
class EnvironmentPolicies:
    tenant_id: str
    production_capture_enabled: bool = False
    external_side_effects_enabled: bool = False
    cross_tenant_sharing_enabled: bool = False
    network_enabled: bool = False
    root_privileges_enabled: bool = False
    secret_redaction_enabled: bool = True
    redacted_paths: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    security_labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EnvironmentStateCapsule:
    """Portable filesystem and recipe state sufficient for local reconstruction.

    The capsule intentionally contains no process-memory image. Services are represented
    only by declared recipes and state files.
    """

    capsule_id: str
    tenant_id: str
    seed: int
    files: tuple[FileEntry, ...]
    git: GitState
    dependency_locks: tuple[DependencyLock, ...]
    working_directories: tuple[str, ...]
    services: tuple[ServiceDescriptor, ...]
    tool_state: dict[str, JsonValue]
    cache_state: dict[str, JsonValue]
    runtime: RuntimeState
    policies: EnvironmentPolicies
    parent_capsule_id: str | None = None
    event_watermark: int = -1
    schema: str = "sloforge.helix.environment-capsule/v1"
    limitations: tuple[str, ...] = (
        "declared services are reconstructed from recipes and state files",
        "arbitrary process memory is not captured or portable",
    )

    def __post_init__(self) -> None:
        if not self.tenant_id or self.tenant_id != self.policies.tenant_id:
            raise ValueError("capsule and policy tenant identifiers must match")
        paths = [entry.path for entry in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("capsule file entries must be unique and sorted")
        if self.runtime.seed != self.seed:
            raise ValueError("runtime seed must match capsule seed")
        if self.event_watermark < -1:
            raise ValueError("environment event watermark must be at least -1")

    def manifest_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.pop("capsule_id", None)
        return result

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return canonical_json(self.to_dict()).decode()

    def verify_identity(self) -> None:
        expected = content_digest(canonical_json(self.manifest_dict()))
        if expected != self.capsule_id:
            raise ValueError("environment capsule manifest digest mismatch")

    @classmethod
    def build(
        cls,
        *,
        tenant_id: str,
        seed: int,
        files: tuple[FileEntry, ...],
        git: GitState,
        dependency_locks: tuple[DependencyLock, ...],
        working_directories: tuple[str, ...],
        services: tuple[ServiceDescriptor, ...],
        tool_state: dict[str, JsonValue],
        cache_state: dict[str, JsonValue],
        runtime: RuntimeState,
        policies: EnvironmentPolicies,
        parent_capsule_id: str | None = None,
        event_watermark: int = -1,
    ) -> EnvironmentStateCapsule:
        provisional = cls(
            capsule_id="",
            tenant_id=tenant_id,
            seed=seed,
            files=files,
            git=git,
            dependency_locks=dependency_locks,
            working_directories=working_directories,
            services=services,
            tool_state=tool_state,
            cache_state=cache_state,
            runtime=runtime,
            policies=policies,
            parent_capsule_id=parent_capsule_id,
            event_watermark=event_watermark,
        )
        return replace(
            provisional,
            capsule_id=content_digest(canonical_json(provisional.manifest_dict())),
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EnvironmentStateCapsule:
        """Load the JSON form. This is deliberately independent of a particular IR library."""

        files = tuple(
            FileEntry(
                path=str(item["path"]),
                kind=EntryKind(item["kind"]),
                mode=int(item["mode"]),
                content_hash=cast(str | None, item.get("content_hash")),
                size_bytes=int(item.get("size_bytes", 0)),
                symlink_target=cast(str | None, item.get("symlink_target")),
            )
            for item in raw["files"]
        )
        git_raw = raw["git"]
        runtime_raw = raw["runtime"]
        policy_raw = raw["policies"]
        capsule = cls(
            capsule_id=str(raw["capsule_id"]),
            tenant_id=str(raw["tenant_id"]),
            seed=int(raw["seed"]),
            files=files,
            git=GitState(
                present=bool(git_raw.get("present", False)),
                head=cast(str | None, git_raw.get("head")),
                branch=cast(str | None, git_raw.get("branch")),
                tracked_paths=tuple(str(value) for value in git_raw.get("tracked_paths", ())),
                untracked_paths=tuple(str(value) for value in git_raw.get("untracked_paths", ())),
                dirty_paths=tuple(str(value) for value in git_raw.get("dirty_paths", ())),
            ),
            dependency_locks=tuple(
                DependencyLock(path=str(item["path"]), content_hash=str(item["content_hash"]))
                for item in raw.get("dependency_locks", ())
            ),
            working_directories=tuple(str(value) for value in raw.get("working_directories", ())),
            services=tuple(_service_from_mapping(item) for item in raw.get("services", ())),
            tool_state=cast(dict[str, JsonValue], raw.get("tool_state", {})),
            cache_state=cast(dict[str, JsonValue], raw.get("cache_state", {})),
            runtime=RuntimeState(
                seed=int(runtime_raw["seed"]),
                rng_algorithm=str(runtime_raw.get("rng_algorithm", "python-mt19937")),
                rng_state_hash=str(runtime_raw.get("rng_state_hash", "")),
                rng_counter=int(runtime_raw.get("rng_counter", 0)),
                virtual_time_ns=int(runtime_raw.get("virtual_time_ns", 0)),
                fault_state=cast(dict[str, JsonValue], runtime_raw.get("fault_state", {})),
            ),
            policies=EnvironmentPolicies(
                tenant_id=str(policy_raw["tenant_id"]),
                production_capture_enabled=bool(
                    policy_raw.get("production_capture_enabled", False)
                ),
                external_side_effects_enabled=bool(
                    policy_raw.get("external_side_effects_enabled", False)
                ),
                cross_tenant_sharing_enabled=bool(
                    policy_raw.get("cross_tenant_sharing_enabled", False)
                ),
                network_enabled=bool(policy_raw.get("network_enabled", False)),
                root_privileges_enabled=bool(policy_raw.get("root_privileges_enabled", False)),
                secret_redaction_enabled=bool(policy_raw.get("secret_redaction_enabled", True)),
                redacted_paths=tuple(str(value) for value in policy_raw.get("redacted_paths", ())),
                allowed_tools=tuple(str(value) for value in policy_raw.get("allowed_tools", ())),
                security_labels=tuple(
                    str(value) for value in policy_raw.get("security_labels", ())
                ),
            ),
            parent_capsule_id=cast(str | None, raw.get("parent_capsule_id")),
            event_watermark=int(raw.get("event_watermark", -1)),
            schema=str(raw.get("schema", "sloforge.helix.environment-capsule/v1")),
            limitations=tuple(str(value) for value in raw.get("limitations", ())),
        )
        capsule.verify_identity()
        return capsule


def _service_from_mapping(raw: object) -> ServiceDescriptor:
    if isinstance(raw, ServiceDescriptor):
        return raw
    if not isinstance(raw, dict):
        raise TypeError("service declaration must be a mapping or ServiceDescriptor")
    value = cast(dict[str, Any], raw)
    return ServiceDescriptor(
        service_id=str(value.get("service_id", value.get("name", "service"))),
        kind=str(value.get("kind", "process")),
        command=tuple(str(item) for item in value.get("command", ())),
        working_directory=str(value.get("working_directory", ".")),
        state_paths=tuple(str(item) for item in value.get("state_paths", ())),
        environment={str(key): str(item) for key, item in value.get("environment", {}).items()},
        metadata=cast(dict[str, JsonValue], value.get("metadata", {})),
        reconstructible=bool(value.get("reconstructible", True)),
    )


def service_from_ir(value: object) -> ServiceDescriptor:
    """Duck-type a Helix IR service without importing or pinning its implementation."""

    if isinstance(value, dict | ServiceDescriptor):
        return _service_from_mapping(value)
    return ServiceDescriptor(
        service_id=str(getattr(value, "service_id", getattr(value, "name", "service"))),
        kind=str(getattr(value, "kind", "process")),
        command=tuple(str(item) for item in getattr(value, "command", ())),
        working_directory=str(getattr(value, "working_directory", ".")),
        state_paths=tuple(str(item) for item in getattr(value, "state_paths", ())),
        environment={
            str(key): str(item) for key, item in getattr(value, "environment", {}).items()
        },
        metadata=cast(dict[str, JsonValue], getattr(value, "metadata", {})),
        reconstructible=bool(getattr(value, "reconstructible", True)),
    )


@dataclass(frozen=True, slots=True)
class ResourceBounds:
    max_bytes: int = 256 * 1024 * 1024
    max_files: int = 100_000
    max_log_entries: int = 10_000

    def __post_init__(self) -> None:
        if min(self.max_bytes, self.max_files, self.max_log_entries) < 1:
            raise ValueError("branch resource bounds must be positive")


@dataclass(frozen=True, slots=True)
class BranchLogEntry:
    sequence: int
    operation: str
    path: str | None
    detail_hash: str


@dataclass(frozen=True, slots=True)
class BranchInfo:
    branch_id: str
    tenant_id: str
    base_capsule_id: str
    namespace: str
    workspace: str
    seed: int
    bounds: ResourceBounds
    log: tuple[BranchLogEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class FileDifference:
    path: str
    left_hash: str | None
    right_hash: str | None
    status: str


@dataclass(frozen=True, slots=True)
class BranchComparison:
    left_id: str
    right_id: str
    differences: tuple[FileDifference, ...]


@dataclass(frozen=True, slots=True)
class StorageAccounting:
    tenant_id: str
    object_count: int
    stored_bytes: int
    capsule_logical_bytes: int
    branch_workspace_bytes: int


@dataclass(frozen=True, slots=True)
class CapsuleRetirementReceipt:
    """Tenant-scoped tombstone; CAS bytes are deliberately retained for shared references."""

    retirement_id: str
    capsule_id: str
    tenant_id: str
    retired_at_ms: int
    reason: str
    cas_content_digests: tuple[str, ...]
    manifest_retired: bool = True
    cas_objects_deleted: tuple[str, ...] = ()
    schema: str = "sloforge.helix.environment-capsule-retirement/v1"

    def __post_init__(self) -> None:
        if self.retired_at_ms < 0:
            raise ValueError("capsule retirement time must be non-negative")
        if not self.reason.strip() or len(self.reason) > 2048:
            raise ValueError("capsule retirement requires a bounded non-empty reason")
        if self.cas_objects_deleted:
            raise ValueError("capsule retirement receipts cannot claim CAS object deletion")
        if tuple(sorted(set(self.cas_content_digests))) != self.cas_content_digests:
            raise ValueError("retained CAS content digests must be unique and sorted")
        self.verify_identity()

    def identity_payload(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("retirement_id")
        return value

    def verify_identity(self) -> None:
        if content_digest(canonical_json(self.identity_payload())) != self.retirement_id:
            raise ValueError("capsule retirement receipt digest mismatch")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def build(
        cls,
        *,
        capsule_id: str,
        tenant_id: str,
        retired_at_ms: int,
        reason: str,
        cas_content_digests: tuple[str, ...],
    ) -> CapsuleRetirementReceipt:
        retained = tuple(sorted(set(cas_content_digests)))
        body = {
            "capsule_id": capsule_id,
            "tenant_id": tenant_id,
            "retired_at_ms": retired_at_ms,
            "reason": reason,
            "cas_content_digests": retained,
            "manifest_retired": True,
            "cas_objects_deleted": (),
            "schema": "sloforge.helix.environment-capsule-retirement/v1",
        }
        return cls(
            retirement_id=content_digest(canonical_json(body)),
            capsule_id=capsule_id,
            tenant_id=tenant_id,
            retired_at_ms=retired_at_ms,
            reason=reason,
            cas_content_digests=retained,
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CapsuleRetirementReceipt:
        return cls(
            retirement_id=str(raw["retirement_id"]),
            capsule_id=str(raw["capsule_id"]),
            tenant_id=str(raw["tenant_id"]),
            retired_at_ms=int(raw["retired_at_ms"]),
            reason=str(raw["reason"]),
            cas_content_digests=tuple(str(item) for item in raw["cas_content_digests"]),
            manifest_retired=bool(raw.get("manifest_retired", True)),
            cas_objects_deleted=tuple(str(item) for item in raw.get("cas_objects_deleted", ())),
            schema=str(raw.get("schema", "sloforge.helix.environment-capsule-retirement/v1")),
        )
