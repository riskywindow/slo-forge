"""Bounded environment-state characterization using the real Helix backend.

The workload fixtures in this module are controlled synthetic inputs. The operation
timings, filesystem bytes, content hashes, and CAS accounting are observations from
the local host. Keeping those two evidence axes separate prevents a synthetic fixture
shape from being mistaken for a production workload distribution.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.helix.characterization.matrix import EnvironmentSize, EvidenceClass, TraceLevel
from sloforge.helix.environments.backend import EnvironmentBackend, EnvironmentBranch
from sloforge.helix.environments.models import (
    EntryKind,
    EnvironmentStateCapsule,
    ResourceBounds,
    ServiceDescriptor,
    canonical_json,
    content_digest,
)

MAX_FIXTURE_BYTES = 4 * 1024 * 1024
MAX_FIXTURE_FILES = 512
MAX_REPETITIONS = 100
MAX_WARMUPS = 10
DEFAULT_TRIAL_TIMEOUT_SECONDS = 30.0


class EnvironmentStudyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class EnvironmentOperation(StrEnum):
    CAPTURE = "capture"
    RESTORE = "restore"
    FORK_RESTORE = "fork_restore"
    WHOLE_FILE_BRANCH_WRITE = "whole_file_branch_write"
    CHECKPOINT = "checkpoint"
    COMPARE = "compare"
    SERVICE_RECONSTRUCT = "service_reconstruct"
    TEARDOWN = "teardown"


class StateCategory(StrEnum):
    FILESYSTEM = "filesystem"
    DATABASE = "database"
    SERVICE = "service"


class FixtureSpec(EnvironmentStudyModel):
    workload: EnvironmentSize
    generated_file_count: int = Field(ge=1, le=MAX_FIXTURE_FILES)
    generated_file_size_bytes: int = Field(ge=1, le=MAX_FIXTURE_BYTES)
    dependency_lock_count: int = Field(ge=0, le=16)
    sqlite_row_count: int = Field(ge=0, le=4096)
    include_process_recipe: bool = False

    @model_validator(mode="after")
    def stays_bounded(self) -> FixtureSpec:
        projected = self.generated_file_count * self.generated_file_size_bytes
        if projected > MAX_FIXTURE_BYTES:
            raise ValueError("fixture payload exceeds the 4 MiB characterization bound")
        if self.workload is EnvironmentSize.NONE:
            raise ValueError("the environment study requires an actual environment fixture")
        return self


FIXTURE_SPECS: dict[EnvironmentSize, FixtureSpec] = {
    EnvironmentSize.TINY: FixtureSpec(
        workload=EnvironmentSize.TINY,
        generated_file_count=4,
        generated_file_size_bytes=256,
        dependency_lock_count=0,
        sqlite_row_count=0,
    ),
    EnvironmentSize.MEDIUM: FixtureSpec(
        workload=EnvironmentSize.MEDIUM,
        generated_file_count=48,
        generated_file_size_bytes=2 * 1024,
        dependency_lock_count=0,
        sqlite_row_count=0,
    ),
    EnvironmentSize.LARGE: FixtureSpec(
        workload=EnvironmentSize.LARGE,
        generated_file_count=128,
        generated_file_size_bytes=8 * 1024,
        dependency_lock_count=0,
        sqlite_row_count=0,
    ),
    EnvironmentSize.DEPENDENCY_HEAVY: FixtureSpec(
        workload=EnvironmentSize.DEPENDENCY_HEAVY,
        generated_file_count=96,
        generated_file_size_bytes=1024,
        dependency_lock_count=4,
        sqlite_row_count=0,
    ),
    EnvironmentSize.SERVICE_HEAVY: FixtureSpec(
        workload=EnvironmentSize.SERVICE_HEAVY,
        generated_file_count=24,
        generated_file_size_bytes=1024,
        dependency_lock_count=1,
        sqlite_row_count=128,
        include_process_recipe=True,
    ),
}


class FixtureFileRecord(EnvironmentStudyModel):
    relative_path: str = Field(min_length=1, max_length=1024)
    category: StateCategory
    size_bytes: int = Field(ge=0)
    mode: int = Field(ge=0, le=0o7777)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FixtureManifest(EnvironmentStudyModel):
    schema_version: Literal["sloforge.branchfabric.environment-fixture/v1"]
    workload_evidence_class: Literal[EvidenceClass.SYNTHETIC]
    workload: EnvironmentSize
    seed: int
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[FixtureFileRecord, ...] = Field(min_length=1, max_length=MAX_FIXTURE_FILES)
    services: tuple[dict[str, object], ...]
    mutable_path: str
    database_paths: tuple[str, ...]
    service_state_paths: tuple[str, ...]
    distribution_claim: Literal[False] = False

    @property
    def payload_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)


class CapsuleFileRecord(EnvironmentStudyModel):
    relative_path: str = Field(min_length=1, max_length=1024)
    kind: str
    category: StateCategory
    mode: int = Field(ge=0, le=0o7777)
    logical_bytes: int = Field(ge=0)
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class CasObjectRecord(EnvironmentStudyModel):
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stored_bytes: int = Field(ge=0)
    reference_count: int = Field(ge=1)
    source_paths: tuple[str, ...] = Field(min_length=1)


class EnvironmentStateComposition(EnvironmentStudyModel):
    filesystem_state_bytes: int = Field(ge=0)
    database_state_bytes: int = Field(ge=0)
    service_state_bytes: int = Field(ge=0)
    process_memory_bytes: Literal[0]
    process_reconstruction_metadata_bytes: int = Field(ge=0)
    service_recipe_metadata_bytes: int = Field(ge=0)
    capsule_manifest_bytes: int = Field(ge=0)
    logical_payload_bytes: int = Field(ge=0)
    unique_cas_bytes: int = Field(ge=0)
    unique_cas_objects: int = Field(ge=0)

    @model_validator(mode="after")
    def payload_categories_sum(self) -> EnvironmentStateComposition:
        categorized = (
            self.filesystem_state_bytes + self.database_state_bytes + self.service_state_bytes
        )
        if categorized != self.logical_payload_bytes:
            raise ValueError("environment payload categories do not sum to logical bytes")
        return self


class OperationObservation(EnvironmentStudyModel):
    logical_bytes: int = Field(ge=0)
    file_count: int = Field(ge=0)
    changed_bytes: int = Field(ge=0)
    difference_count: int = Field(ge=0)
    workspace_bytes: int = Field(ge=0)
    cas_stored_bytes: int = Field(ge=0)
    cas_object_count: int = Field(ge=0)
    result_id: str | None = Field(default=None, max_length=256)


class EnvironmentOperationSample(EnvironmentStudyModel):
    schema_version: Literal["sloforge.branchfabric.environment-operation-sample/v1"]
    workload_evidence_class: Literal[EvidenceClass.SYNTHETIC]
    timing_evidence_class: Literal[EvidenceClass.HARDWARE_BACKED_REAL]
    workload: EnvironmentSize
    trace_level: TraceLevel
    seed: int
    repetition: int = Field(ge=0)
    warmup: bool
    operation_sequence: int = Field(ge=0)
    operation: EnvironmentOperation
    monotonic_start_ns: int = Field(ge=0)
    duration_ns: int = Field(ge=0)
    process_cpu_ns: int = Field(ge=0)
    wall_clock: Literal["time.perf_counter_ns"]
    cpu_clock: Literal["time.process_time_ns"]
    observation: OperationObservation | None
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EnvironmentTrialResult(EnvironmentStudyModel):
    schema_version: Literal["sloforge.branchfabric.environment-trial/v1"]
    workload_evidence_class: Literal[EvidenceClass.SYNTHETIC]
    timing_evidence_class: Literal[EvidenceClass.HARDWARE_BACKED_REAL]
    workload: EnvironmentSize
    trace_level: TraceLevel
    seed: int
    repetition: int = Field(ge=0)
    warmup: bool
    fixture: FixtureManifest
    base_capsule_id: str
    checkpoint_capsule_id: str
    semantic_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_composition: EnvironmentStateComposition
    operations: tuple[EnvironmentOperationSample, ...] = Field(min_length=8, max_length=8)
    base_capsule_files: tuple[CapsuleFileRecord, ...]
    base_cas_objects: tuple[CasObjectRecord, ...]
    checkpoint_capsule_files: tuple[CapsuleFileRecord, ...]
    checkpoint_cas_objects: tuple[CasObjectRecord, ...]
    trial_duration_ns: int = Field(ge=0)
    trial_process_cpu_ns: int = Field(ge=0)
    trace_detail_bytes: int = Field(ge=0)
    physical_cow_measured: Literal[False]
    process_startup_measured: Literal[False]

    @model_validator(mode="after")
    def trace_detail_matches_level(self) -> EnvironmentTrialResult:
        if self.trace_level is TraceLevel.DISABLED:
            if any(item.observation is not None for item in self.operations):
                raise ValueError("disabled tracing cannot include operation observations")
            if (
                self.base_capsule_files
                or self.base_cas_objects
                or self.checkpoint_capsule_files
                or self.checkpoint_cas_objects
            ):
                raise ValueError("disabled tracing cannot include exact object detail")
        if self.trace_level is TraceLevel.MINIMAL and (
            self.base_capsule_files
            or self.base_cas_objects
            or self.checkpoint_capsule_files
            or self.checkpoint_cas_objects
        ):
            raise ValueError("minimal tracing cannot include exact object detail")
        if self.trace_level is not TraceLevel.DISABLED and any(
            item.observation is None for item in self.operations
        ):
            raise ValueError("enabled tracing requires operation observations")
        return self


class EnvironmentStudyArtifact(EnvironmentStudyModel):
    schema_version: Literal["sloforge.branchfabric.environment-study/v1"]
    workload_evidence_class: Literal[EvidenceClass.SYNTHETIC]
    timing_evidence_class: Literal[EvidenceClass.HARDWARE_BACKED_REAL]
    seed: int
    repetitions: int = Field(ge=1, le=MAX_REPETITIONS)
    warmups: int = Field(ge=0, le=MAX_WARMUPS)
    trial_timeout_seconds: float = Field(gt=0, le=300)
    run_order_seed: int
    workloads: tuple[EnvironmentSize, ...]
    trace_levels: tuple[TraceLevel, ...]
    trials: tuple[EnvironmentTrialResult, ...]
    methodology: tuple[str, ...]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def trials_cover_matrix_and_preserve_semantics(self) -> EnvironmentStudyArtifact:
        expected = (self.repetitions + self.warmups) * len(self.workloads) * len(self.trace_levels)
        if len(self.trials) != expected:
            raise ValueError("environment trials do not cover the declared study matrix")
        identities = [
            (
                trial.workload,
                trial.trace_level,
                trial.repetition,
                trial.warmup,
            )
            for trial in self.trials
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("environment study contains duplicate trial identities")
        declared_workloads = set(self.workloads)
        declared_levels = set(self.trace_levels)
        if any(
            trial.workload not in declared_workloads or trial.trace_level not in declared_levels
            for trial in self.trials
        ):
            raise ValueError("environment trial falls outside the declared matrix")
        semantic_groups: dict[tuple[EnvironmentSize, int, int, bool], set[str]] = {}
        for trial in self.trials:
            key = (trial.workload, trial.seed, trial.repetition, trial.warmup)
            semantic_groups.setdefault(key, set()).add(trial.semantic_result_sha256)
        if any(len(values) != 1 for values in semantic_groups.values()):
            raise ValueError("trace level changed an environment trial's semantic result")
        return self


def _stable_seed(seed: int, workload: EnvironmentSize) -> int:
    digest = hashlib.sha256(f"{seed}:{workload.value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _category_for_path(
    path: str, *, database_paths: frozenset[str], service_paths: frozenset[str]
) -> StateCategory:
    if path in database_paths:
        return StateCategory.DATABASE
    if path in service_paths:
        return StateCategory.SERVICE
    return StateCategory.FILESYSTEM


def _fixture_hash(files: Sequence[FixtureFileRecord], services: Sequence[ServiceDescriptor]) -> str:
    return content_digest(
        canonical_json(
            {
                "files": [item.model_dump(mode="json") for item in files],
                "services": [asdict(item) for item in services],
            }
        )
    )


def build_environment_fixture(
    root: Path,
    *,
    workload: EnvironmentSize,
    seed: int,
) -> tuple[FixtureManifest, tuple[ServiceDescriptor, ...]]:
    """Materialize one small, deterministic, non-sparse controlled fixture."""

    if workload not in FIXTURE_SPECS:
        raise ValueError(f"unsupported environment fixture: {workload.value}")
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True, mode=0o700)
    spec = FIXTURE_SPECS[workload]
    generator = random.Random(_stable_seed(seed, workload))

    payload_root = root / "payload"
    payload_root.mkdir()
    for index in range(spec.generated_file_count):
        target = payload_root / f"object-{index:04d}.bin"
        target.write_bytes(generator.randbytes(spec.generated_file_size_bytes))

    state_root = root / "state"
    state_root.mkdir()
    mutable_size = max(256, spec.generated_file_size_bytes)
    mutable_path = "state/mutable.bin"
    (root / mutable_path).write_bytes(generator.randbytes(mutable_size))

    lock_names = ("uv.lock", "package-lock.json", "Cargo.lock", "poetry.lock")
    for index, name in enumerate(lock_names[: spec.dependency_lock_count]):
        lock_payload = canonical_json(
            {
                "fixture": workload.value,
                "packages": [
                    {"name": f"dependency-{item:04d}", "version": f"1.{index}.{item}"}
                    for item in range(32)
                ],
                "seed": seed,
            }
        )
        (root / name).write_bytes(lock_payload)

    services: list[ServiceDescriptor] = []
    database_paths: tuple[str, ...] = ()
    service_paths: tuple[str, ...] = ()
    if spec.sqlite_row_count:
        database_path = "state/service.sqlite"
        database = root / database_path
        with sqlite3.connect(database) as connection:
            connection.execute("PRAGMA page_size = 4096")
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute(
                "CREATE TABLE records (key INTEGER PRIMARY KEY, value BLOB NOT NULL)"
            )
            connection.executemany(
                "INSERT INTO records(key, value) VALUES (?, ?)",
                ((index, generator.randbytes(64)) for index in range(spec.sqlite_row_count)),
            )
            connection.commit()
        services.append(
            ServiceDescriptor(
                service_id="fixture-database",
                kind="sqlite",
                state_paths=(database_path,),
                metadata={"row_count": spec.sqlite_row_count},
            )
        )
        database_paths = (database_path,)
    if spec.include_process_recipe:
        worker_state_path = "state/worker-state.bin"
        (root / worker_state_path).write_bytes(generator.randbytes(4096))
        services.append(
            ServiceDescriptor(
                service_id="fixture-worker",
                kind="process",
                command=("python", "-m", "fixture_worker"),
                state_paths=(worker_state_path,),
                environment={"SLOFORGE_FIXTURE": workload.value},
                metadata={"startup_policy": "disabled_for_characterization"},
            )
        )
        service_paths = (worker_state_path,)

    database_set = frozenset(database_paths)
    service_set = frozenset(service_paths)
    files = tuple(
        FixtureFileRecord(
            relative_path=path.relative_to(root).as_posix(),
            category=_category_for_path(
                path.relative_to(root).as_posix(),
                database_paths=database_set,
                service_paths=service_set,
            ),
            size_bytes=path.stat().st_size,
            mode=path.stat().st_mode & 0o7777,
            sha256=content_digest(path.read_bytes()),
        )
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    )
    if len(files) > MAX_FIXTURE_FILES or sum(item.size_bytes for item in files) > MAX_FIXTURE_BYTES:
        raise ValueError("materialized fixture exceeded the characterization bounds")
    return (
        FixtureManifest(
            schema_version="sloforge.branchfabric.environment-fixture/v1",
            workload_evidence_class=EvidenceClass.SYNTHETIC,
            workload=workload,
            seed=seed,
            fixture_sha256=_fixture_hash(files, services),
            files=files,
            services=tuple(asdict(item) for item in services),
            mutable_path=mutable_path,
            database_paths=database_paths,
            service_state_paths=service_paths,
            distribution_claim=False,
        ),
        tuple(services),
    )


def _workspace_usage(root: Path) -> tuple[int, int]:
    files = tuple(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    return sum(path.stat().st_size for path in files), len(files)


def _capsule_records(
    backend: EnvironmentBackend,
    capsule: EnvironmentStateCapsule,
    fixture: FixtureManifest,
) -> tuple[
    tuple[CapsuleFileRecord, ...],
    tuple[CasObjectRecord, ...],
    EnvironmentStateComposition,
]:
    database_paths = frozenset(fixture.database_paths)
    service_paths = frozenset(fixture.service_state_paths)
    files = tuple(
        CapsuleFileRecord(
            relative_path=entry.path,
            kind=entry.kind.value,
            category=_category_for_path(
                entry.path,
                database_paths=database_paths,
                service_paths=service_paths,
            ),
            mode=entry.mode,
            logical_bytes=entry.size_bytes,
            content_sha256=entry.content_hash,
        )
        for entry in capsule.files
    )
    references: dict[str, list[str]] = {}
    object_sizes: dict[str, int] = {}
    for entry in capsule.files:
        if entry.kind is not EntryKind.FILE or entry.content_hash is None:
            continue
        references.setdefault(entry.content_hash, []).append(entry.path)
        object_sizes[entry.content_hash] = entry.size_bytes
    objects: list[CasObjectRecord] = []
    for digest, paths in sorted(references.items()):
        object_path = backend.store.object_path(backend.tenant_id, digest)
        payload = object_path.read_bytes()
        if len(payload) != object_sizes[digest] or content_digest(payload) != digest:
            raise OSError(f"captured environment object failed verification: {digest}")
        objects.append(
            CasObjectRecord(
                content_sha256=digest,
                stored_bytes=len(payload),
                reference_count=len(paths),
                source_paths=tuple(sorted(paths)),
            )
        )
    composition = _capsule_composition(capsule, fixture)
    if composition.unique_cas_bytes != sum(item.stored_bytes for item in objects):
        raise OSError("environment manifest and verified CAS byte accounting differ")
    return files, tuple(objects), composition


def _capsule_composition(
    capsule: EnvironmentStateCapsule,
    fixture: FixtureManifest,
) -> EnvironmentStateComposition:
    """Compute size accounting from authenticated manifest metadata without CAS rereads."""

    database_paths = frozenset(fixture.database_paths)
    service_paths = frozenset(fixture.service_state_paths)
    logical_by_category = {category: 0 for category in StateCategory}
    unique_objects: dict[str, int] = {}
    for entry in capsule.files:
        category = _category_for_path(
            entry.path,
            database_paths=database_paths,
            service_paths=service_paths,
        )
        logical_by_category[category] += entry.size_bytes
        if entry.kind is EntryKind.FILE and entry.content_hash is not None:
            previous = unique_objects.setdefault(entry.content_hash, entry.size_bytes)
            if previous != entry.size_bytes:
                raise ValueError("one environment content digest has inconsistent sizes")
    process_services = [item for item in capsule.services if item.kind.lower() == "process"]
    service_recipe_bytes = (
        len(canonical_json([asdict(item) for item in capsule.services])) if capsule.services else 0
    )
    process_recipe_bytes = (
        len(canonical_json([asdict(item) for item in process_services])) if process_services else 0
    )
    return EnvironmentStateComposition(
        filesystem_state_bytes=logical_by_category[StateCategory.FILESYSTEM],
        database_state_bytes=logical_by_category[StateCategory.DATABASE],
        service_state_bytes=logical_by_category[StateCategory.SERVICE],
        process_memory_bytes=0,
        process_reconstruction_metadata_bytes=process_recipe_bytes,
        service_recipe_metadata_bytes=service_recipe_bytes,
        capsule_manifest_bytes=len(capsule.to_json().encode()),
        logical_payload_bytes=sum(logical_by_category.values()),
        unique_cas_bytes=sum(unique_objects.values()),
        unique_cas_objects=len(unique_objects),
    )


def _observation(
    backend: EnvironmentBackend,
    *,
    logical_bytes: int = 0,
    file_count: int = 0,
    changed_bytes: int = 0,
    difference_count: int = 0,
    workspace: Path | None = None,
    result_id: str | None = None,
) -> OperationObservation:
    accounting = backend.accounting()
    workspace_bytes = _workspace_usage(workspace)[0] if workspace is not None else 0
    return OperationObservation(
        logical_bytes=logical_bytes,
        file_count=file_count,
        changed_bytes=changed_bytes,
        difference_count=difference_count,
        workspace_bytes=workspace_bytes,
        cas_stored_bytes=accounting.stored_bytes,
        cas_object_count=accounting.object_count,
        result_id=result_id,
    )


T = TypeVar("T")


def _measure(
    operation: EnvironmentOperation,
    action: Callable[[], T],
) -> tuple[T, int, int, int]:
    started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    result = action()
    cpu_finished = time.process_time_ns()
    finished = time.perf_counter_ns()
    return result, started, finished - started, cpu_finished - cpu_started


def _sample(
    *,
    workload: EnvironmentSize,
    trace_level: TraceLevel,
    seed: int,
    repetition: int,
    warmup: bool,
    sequence: int,
    operation: EnvironmentOperation,
    started_ns: int,
    duration_ns: int,
    process_cpu_ns: int,
    observation: Callable[[], OperationObservation],
    result: object,
) -> EnvironmentOperationSample:
    encoded_result = canonical_json(result)
    return EnvironmentOperationSample(
        schema_version="sloforge.branchfabric.environment-operation-sample/v1",
        workload_evidence_class=EvidenceClass.SYNTHETIC,
        timing_evidence_class=EvidenceClass.HARDWARE_BACKED_REAL,
        workload=workload,
        trace_level=trace_level,
        seed=seed,
        repetition=repetition,
        warmup=warmup,
        operation_sequence=sequence,
        operation=operation,
        monotonic_start_ns=started_ns,
        duration_ns=duration_ns,
        process_cpu_ns=process_cpu_ns,
        wall_clock="time.perf_counter_ns",
        cpu_clock="time.process_time_ns",
        observation=None if trace_level is TraceLevel.DISABLED else observation(),
        result_sha256=content_digest(encoded_result),
    )


def _check_deadline(deadline_ns: int) -> None:
    if time.perf_counter_ns() > deadline_ns:
        raise TimeoutError("environment characterization trial exceeded its bounded timeout")


def _run_trial(
    trial_root: Path,
    *,
    workload: EnvironmentSize,
    trace_level: TraceLevel,
    seed: int,
    repetition: int,
    warmup: bool,
    timeout_seconds: float,
) -> EnvironmentTrialResult:
    fixture, services = build_environment_fixture(
        trial_root / "source",
        workload=workload,
        seed=seed,
    )
    backend = EnvironmentBackend(
        trial_root / "backend",
        tenant_id="branchfabric-environment-study",
        max_capture_bytes=MAX_FIXTURE_BYTES,
        max_capture_files=MAX_FIXTURE_FILES,
    )
    trial_started = time.perf_counter_ns()
    trial_cpu_started = time.process_time_ns()
    deadline_ns = trial_started + int(timeout_seconds * 1_000_000_000)
    samples: list[EnvironmentOperationSample] = []
    branch: EnvironmentBranch | None = None
    restore_root = trial_root / "restore"

    try:
        capsule, started, elapsed, cpu = _measure(
            EnvironmentOperation.CAPTURE,
            lambda: backend.capture(
                trial_root / "source",
                seed=seed,
                services=services,
                event_watermark=0,
            ),
        )
        composition = _capsule_composition(capsule, fixture)
        if trace_level is TraceLevel.FULL:
            base_capsule_files, base_cas_objects, verified_composition = _capsule_records(
                backend, capsule, fixture
            )
            if verified_composition != composition:
                raise OSError("verified environment composition differs from manifest accounting")
        else:
            base_capsule_files = ()
            base_cas_objects = ()
        samples.append(
            _sample(
                workload=workload,
                trace_level=trace_level,
                seed=seed,
                repetition=repetition,
                warmup=warmup,
                sequence=0,
                operation=EnvironmentOperation.CAPTURE,
                started_ns=started,
                duration_ns=elapsed,
                process_cpu_ns=cpu,
                observation=lambda: _observation(
                    backend,
                    logical_bytes=composition.logical_payload_bytes,
                    file_count=len(capsule.files),
                    result_id=capsule.capsule_id,
                ),
                result={"capsule_id": capsule.capsule_id},
            )
        )
        _check_deadline(deadline_ns)

        restored, started, elapsed, cpu = _measure(
            EnvironmentOperation.RESTORE,
            lambda: backend.restore(capsule, restore_root),
        )
        restored_bytes, restored_files = _workspace_usage(restored)
        samples.append(
            _sample(
                workload=workload,
                trace_level=trace_level,
                seed=seed,
                repetition=repetition,
                warmup=warmup,
                sequence=1,
                operation=EnvironmentOperation.RESTORE,
                started_ns=started,
                duration_ns=elapsed,
                process_cpu_ns=cpu,
                observation=lambda: _observation(
                    backend,
                    logical_bytes=restored_bytes,
                    file_count=restored_files,
                    workspace=restored,
                    result_id=capsule.capsule_id,
                ),
                result={
                    "capsule_id": capsule.capsule_id,
                    "workspace_sha256": fixture.fixture_sha256,
                },
            )
        )
        _check_deadline(deadline_ns)

        live_branch, started, elapsed, cpu = _measure(
            EnvironmentOperation.FORK_RESTORE,
            lambda: backend.fork(
                capsule,
                branch_id=f"branch-{repetition}",
                seed=seed,
                bounds=ResourceBounds(
                    max_bytes=MAX_FIXTURE_BYTES,
                    max_files=MAX_FIXTURE_FILES,
                    max_log_entries=64,
                ),
            ),
        )
        branch = live_branch
        branch_bytes, branch_files = _workspace_usage(live_branch.workspace)
        samples.append(
            _sample(
                workload=workload,
                trace_level=trace_level,
                seed=seed,
                repetition=repetition,
                warmup=warmup,
                sequence=2,
                operation=EnvironmentOperation.FORK_RESTORE,
                started_ns=started,
                duration_ns=elapsed,
                process_cpu_ns=cpu,
                observation=lambda: _observation(
                    backend,
                    logical_bytes=branch_bytes,
                    file_count=branch_files,
                    workspace=live_branch.workspace,
                    result_id=live_branch.info.namespace,
                ),
                result={
                    "base_capsule_id": capsule.capsule_id,
                    "namespace": live_branch.info.namespace,
                },
            )
        )
        _check_deadline(deadline_ns)

        original_mutable = live_branch.read_bytes(fixture.mutable_path)
        mutation_generator = random.Random(_stable_seed(seed + 1, workload))
        mutated_mutable = mutation_generator.randbytes(len(original_mutable))
        _, started, elapsed, cpu = _measure(
            EnvironmentOperation.WHOLE_FILE_BRANCH_WRITE,
            lambda: live_branch.write_bytes(fixture.mutable_path, mutated_mutable),
        )
        samples.append(
            _sample(
                workload=workload,
                trace_level=trace_level,
                seed=seed,
                repetition=repetition,
                warmup=warmup,
                sequence=3,
                operation=EnvironmentOperation.WHOLE_FILE_BRANCH_WRITE,
                started_ns=started,
                duration_ns=elapsed,
                process_cpu_ns=cpu,
                observation=lambda: _observation(
                    backend,
                    logical_bytes=len(mutated_mutable),
                    file_count=1,
                    changed_bytes=len(mutated_mutable),
                    workspace=live_branch.workspace,
                    result_id=content_digest(mutated_mutable),
                ),
                result={
                    "path": fixture.mutable_path,
                    "before_sha256": content_digest(original_mutable),
                    "after_sha256": content_digest(mutated_mutable),
                },
            )
        )
        _check_deadline(deadline_ns)

        checkpoint, started, elapsed, cpu = _measure(
            EnvironmentOperation.CHECKPOINT,
            live_branch.checkpoint,
        )
        if trace_level is TraceLevel.FULL:
            checkpoint_capsule_files, checkpoint_cas_objects, _ = _capsule_records(
                backend, checkpoint, fixture
            )
        else:
            checkpoint_capsule_files = ()
            checkpoint_cas_objects = ()
        samples.append(
            _sample(
                workload=workload,
                trace_level=trace_level,
                seed=seed,
                repetition=repetition,
                warmup=warmup,
                sequence=4,
                operation=EnvironmentOperation.CHECKPOINT,
                started_ns=started,
                duration_ns=elapsed,
                process_cpu_ns=cpu,
                observation=lambda: _observation(
                    backend,
                    logical_bytes=sum(item.size_bytes for item in checkpoint.files),
                    file_count=len(checkpoint.files),
                    workspace=live_branch.workspace,
                    result_id=checkpoint.capsule_id,
                ),
                result={"checkpoint_capsule_id": checkpoint.capsule_id},
            )
        )
        _check_deadline(deadline_ns)

        comparison, started, elapsed, cpu = _measure(
            EnvironmentOperation.COMPARE,
            lambda: backend.compare(capsule, live_branch),
        )
        comparison_payload = [asdict(item) for item in comparison.differences]
        samples.append(
            _sample(
                workload=workload,
                trace_level=trace_level,
                seed=seed,
                repetition=repetition,
                warmup=warmup,
                sequence=5,
                operation=EnvironmentOperation.COMPARE,
                started_ns=started,
                duration_ns=elapsed,
                process_cpu_ns=cpu,
                observation=lambda: _observation(
                    backend,
                    logical_bytes=composition.logical_payload_bytes,
                    file_count=len(capsule.files),
                    difference_count=len(comparison.differences),
                    workspace=live_branch.workspace,
                    result_id=content_digest(canonical_json(comparison_payload)),
                ),
                result=comparison_payload,
            )
        )
        _check_deadline(deadline_ns)

        reconstructed, started, elapsed, cpu = _measure(
            EnvironmentOperation.SERVICE_RECONSTRUCT,
            lambda: backend.reconstruct_services(capsule, restored, start_processes=False),
        )
        reconstructed_payload = [asdict(item) for item in reconstructed]
        samples.append(
            _sample(
                workload=workload,
                trace_level=trace_level,
                seed=seed,
                repetition=repetition,
                warmup=warmup,
                sequence=6,
                operation=EnvironmentOperation.SERVICE_RECONSTRUCT,
                started_ns=started,
                duration_ns=elapsed,
                process_cpu_ns=cpu,
                observation=lambda: _observation(
                    backend,
                    logical_bytes=(
                        composition.database_state_bytes + composition.service_state_bytes
                    ),
                    file_count=len(fixture.database_paths) + len(fixture.service_state_paths),
                    workspace=restored,
                    result_id=content_digest(canonical_json(reconstructed_payload)),
                ),
                result=reconstructed_payload,
            )
        )
        _check_deadline(deadline_ns)

        bytes_before_teardown = backend.accounting().branch_workspace_bytes + restored_bytes

        def teardown() -> None:
            live_branch.cleanup()
            shutil.rmtree(restored)

        _, started, elapsed, cpu = _measure(EnvironmentOperation.TEARDOWN, teardown)
        branch = None
        samples.append(
            _sample(
                workload=workload,
                trace_level=trace_level,
                seed=seed,
                repetition=repetition,
                warmup=warmup,
                sequence=7,
                operation=EnvironmentOperation.TEARDOWN,
                started_ns=started,
                duration_ns=elapsed,
                process_cpu_ns=cpu,
                observation=lambda: _observation(
                    backend,
                    logical_bytes=bytes_before_teardown,
                    changed_bytes=bytes_before_teardown,
                    result_id="environment-workspaces-reclaimed",
                ),
                result={"workspace_bytes_reclaimed": bytes_before_teardown},
            )
        )

        semantic_payload = {
            "base_capsule_id": capsule.capsule_id,
            "checkpoint_capsule_id": checkpoint.capsule_id,
            "comparison": comparison_payload,
            "fixture_sha256": fixture.fixture_sha256,
            "mutated_sha256": content_digest(mutated_mutable),
            "reconstructed_services": reconstructed_payload,
        }
        detail_payload = {
            "operation_observations": [
                item.observation.model_dump(mode="json")
                for item in samples
                if item.observation is not None
            ],
            "base_capsule_files": [item.model_dump(mode="json") for item in base_capsule_files],
            "base_cas_objects": [item.model_dump(mode="json") for item in base_cas_objects],
            "checkpoint_capsule_files": [
                item.model_dump(mode="json") for item in checkpoint_capsule_files
            ],
            "checkpoint_cas_objects": [
                item.model_dump(mode="json") for item in checkpoint_cas_objects
            ],
        }
        trial_finished = time.perf_counter_ns()
        trial_cpu_finished = time.process_time_ns()
        return EnvironmentTrialResult(
            schema_version="sloforge.branchfabric.environment-trial/v1",
            workload_evidence_class=EvidenceClass.SYNTHETIC,
            timing_evidence_class=EvidenceClass.HARDWARE_BACKED_REAL,
            workload=workload,
            trace_level=trace_level,
            seed=seed,
            repetition=repetition,
            warmup=warmup,
            fixture=fixture,
            base_capsule_id=capsule.capsule_id,
            checkpoint_capsule_id=checkpoint.capsule_id,
            semantic_result_sha256=content_digest(canonical_json(semantic_payload)),
            state_composition=composition,
            operations=tuple(samples),
            base_capsule_files=base_capsule_files,
            base_cas_objects=base_cas_objects,
            checkpoint_capsule_files=checkpoint_capsule_files,
            checkpoint_cas_objects=checkpoint_cas_objects,
            trial_duration_ns=trial_finished - trial_started,
            trial_process_cpu_ns=trial_cpu_finished - trial_cpu_started,
            trace_detail_bytes=(
                0 if trace_level is TraceLevel.DISABLED else len(canonical_json(detail_payload))
            ),
            physical_cow_measured=False,
            process_startup_measured=False,
        )
    finally:
        if branch is not None:
            branch.cleanup()
        if restore_root.exists():
            shutil.rmtree(restore_root)


def _ensure_unique(values: Sequence[T], *, label: str) -> tuple[T, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{label} cannot be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must be unique")
    return result


def run_environment_study(
    output_root: Path,
    *,
    seed: int,
    repetitions: int = 3,
    warmups: int = 1,
    workloads: Sequence[EnvironmentSize] = (
        EnvironmentSize.TINY,
        EnvironmentSize.MEDIUM,
        EnvironmentSize.LARGE,
        EnvironmentSize.DEPENDENCY_HEAVY,
        EnvironmentSize.SERVICE_HEAVY,
    ),
    trace_levels: Sequence[TraceLevel] = (
        TraceLevel.DISABLED,
        TraceLevel.MINIMAL,
        TraceLevel.FULL,
    ),
    trial_timeout_seconds: float = DEFAULT_TRIAL_TIMEOUT_SECONDS,
) -> EnvironmentStudyArtifact:
    """Run randomized measured trials and atomically preserve every raw sample."""

    if not 1 <= repetitions <= MAX_REPETITIONS:
        raise ValueError(f"repetitions must be in [1, {MAX_REPETITIONS}]")
    if not 0 <= warmups <= MAX_WARMUPS:
        raise ValueError(f"warmups must be in [0, {MAX_WARMUPS}]")
    if not 0 < trial_timeout_seconds <= 300:
        raise ValueError("trial timeout must be in (0, 300] seconds")
    chosen_workloads = _ensure_unique(workloads, label="workloads")
    chosen_levels = _ensure_unique(trace_levels, label="trace levels")
    unsupported = [item for item in chosen_workloads if item not in FIXTURE_SPECS]
    if unsupported:
        raise ValueError(f"unsupported environment fixtures: {unsupported}")
    if output_root.exists() and (
        not output_root.is_dir() or output_root.is_symlink() or any(output_root.iterdir())
    ):
        raise FileExistsError("environment study output directory must be an empty directory")
    output_root.mkdir(parents=True, exist_ok=True)
    working_root = output_root / ".working"
    working_root.mkdir(mode=0o700)

    cell_order = [(workload, level) for workload in chosen_workloads for level in chosen_levels]
    generator = random.Random(seed)
    generator.shuffle(cell_order)
    planned: list[tuple[EnvironmentSize, TraceLevel, int, bool]] = []
    for repetition in range(warmups):
        planned.extend((workload, level, repetition, True) for workload, level in cell_order)
    measured = [
        (workload, level, repetition, False)
        for repetition in range(repetitions)
        for workload, level in cell_order
    ]
    generator.shuffle(measured)
    planned.extend(measured)

    trials: list[EnvironmentTrialResult] = []
    try:
        for ordinal, (workload, level, repetition, is_warmup) in enumerate(planned):
            trial_seed = seed + repetition
            trial = _run_trial(
                working_root / f"trial-{ordinal:06d}",
                workload=workload,
                trace_level=level,
                seed=trial_seed,
                repetition=repetition,
                warmup=is_warmup,
                timeout_seconds=trial_timeout_seconds,
            )
            trials.append(trial)
    finally:
        shutil.rmtree(working_root, ignore_errors=True)

    artifact = EnvironmentStudyArtifact(
        schema_version="sloforge.branchfabric.environment-study/v1",
        workload_evidence_class=EvidenceClass.SYNTHETIC,
        timing_evidence_class=EvidenceClass.HARDWARE_BACKED_REAL,
        seed=seed,
        repetitions=repetitions,
        warmups=warmups,
        trial_timeout_seconds=trial_timeout_seconds,
        run_order_seed=seed,
        workloads=chosen_workloads,
        trace_levels=chosen_levels,
        trials=tuple(trials),
        methodology=(
            "fixtures are controlled synthetic sweeps and have no production distribution weight",
            "operation wall time uses time.perf_counter_ns on the executing host",
            "operation CPU time uses time.process_time_ns on the executing process",
            "warmup samples are retained and explicitly marked rather than discarded",
            "measured trial order is deterministically shuffled by the explicit run seed",
            "full tracing authenticates every captured file and unique CAS object",
        ),
        limitations=(
            "EnvironmentBackend fork eagerly restores a private workspace; no physical filesystem COW fault is measured",
            "the whole-file write measures isolated rewrite cost after eager fork, not page-level COW amplification",
            "external process launch is disabled; process memory and service startup are not measured",
            "SQLite reconstruction performs an actual integrity check but starts no database server",
            "results describe the local CPU and filesystem only; no GPU or network behavior is inferred",
        ),
    )
    target = output_root / "environment-study.json"
    temporary = output_root / ".environment-study.json.tmp"
    payload = json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True) + os.linesep
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(target)
    return artifact


__all__ = [
    "FIXTURE_SPECS",
    "CapsuleFileRecord",
    "CasObjectRecord",
    "EnvironmentOperation",
    "EnvironmentOperationSample",
    "EnvironmentStateComposition",
    "EnvironmentStudyArtifact",
    "EnvironmentTrialResult",
    "FixtureFileRecord",
    "FixtureManifest",
    "FixtureSpec",
    "OperationObservation",
    "StateCategory",
    "build_environment_fixture",
    "run_environment_study",
]
