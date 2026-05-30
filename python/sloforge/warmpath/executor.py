"""Safe local reference executor for WarmPath plans."""

from __future__ import annotations

import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Literal

from sloforge.util import canonical_json, sha256_bytes, sha256_file, write_json
from sloforge.warmpath.models import (
    WARMPATH_SCHEMA_VERSION,
    ArtifactGraph,
    ExecutionRecord,
    ExecutorArtifactRecord,
    HostEnvironment,
    MaterializationMode,
    StorageKind,
    StorageTierSpec,
    WarmPathPlan,
)

_EXECUTABLE_LOCAL_TIERS = {
    StorageKind.LOCAL_NVME,
    StorageKind.PAGE_CACHE,
    StorageKind.HOST_MEMORY,
}


class LocalWarmPathExecutor:
    """Materialize local artifacts with bounded caches and checksum verification."""

    def __init__(
        self,
        *,
        maximum_operation_seconds: float = 30.0,
        maximum_artifact_bytes: int = 1 << 30,
        read_chunk_bytes: int = 1 << 20,
    ) -> None:
        if maximum_operation_seconds <= 0.0 or maximum_artifact_bytes <= 0:
            raise ValueError("executor timeout and artifact limit must be positive")
        if read_chunk_bytes <= 0:
            raise ValueError("executor read chunk must be positive")
        self._maximum_operation_seconds = maximum_operation_seconds
        self._maximum_artifact_bytes = maximum_artifact_bytes
        self._read_chunk_bytes = read_chunk_bytes
        self._memory: dict[tuple[str, str], bytes] = {}
        self._disk_entries: dict[tuple[str, str], Path] = {}
        self._access: dict[tuple[str, str], int] = {}
        self._clock = 0

    def _touch(self, key: tuple[str, str]) -> None:
        self._clock += 1
        self._access[key] = self._clock

    def _used_bytes(self, tier_id: str) -> int:
        memory = sum(
            len(value) for (entry_tier, _), value in self._memory.items() if entry_tier == tier_id
        )
        disk = sum(
            path.stat().st_size
            for (entry_tier, _), path in self._disk_entries.items()
            if entry_tier == tier_id and path.exists()
        )
        return memory + disk

    def _evict_for(
        self,
        *,
        tier: StorageTierSpec,
        required_bytes: int,
        protected: set[tuple[str, str]],
    ) -> tuple[str, ...]:
        if required_bytes > tier.capacity_bytes:
            raise OSError(
                f"artifact of {required_bytes} bytes exceeds tier {tier.tier_id} capacity"
            )
        evicted: list[str] = []
        while self._used_bytes(tier.tier_id) + required_bytes > tier.capacity_bytes:
            eligible = [
                key
                for key in set(self._memory) | set(self._disk_entries)
                if key[0] == tier.tier_id and key not in protected
            ]
            if not eligible:
                raise OSError(f"tier {tier.tier_id} has no evictable capacity")
            victim = min(eligible, key=lambda key: (self._access.get(key, 0), key))
            self._memory.pop(victim, None)
            disk = self._disk_entries.pop(victim, None)
            if disk is not None and disk.exists():
                disk.unlink()
            self._access.pop(victim, None)
            evicted.append(victim[1])
        return tuple(evicted)

    @staticmethod
    def _safe_path(root: Path, relative: str) -> Path:
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"path escapes root directory: {relative}") from error
        return path

    def _cache_payload(
        self,
        *,
        tier: StorageTierSpec,
        artifact_id: str,
        checksum: str,
        payload: bytes,
    ) -> Path | None:
        key = (tier.tier_id, artifact_id)
        if tier.kind == StorageKind.HOST_MEMORY:
            self._memory[key] = payload
            self._touch(key)
            return None
        if tier.local_path is None:
            raise ValueError(f"local tier {tier.tier_id} has no local path")
        cache_directory = Path(tier.local_path)
        cache_directory.mkdir(parents=True, exist_ok=True)
        cache_path = cache_directory / f"{artifact_id}-{checksum}"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{artifact_id}-", dir=cache_directory, delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, cache_path)
            temporary = None
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self._disk_entries[key] = cache_path
        self._touch(key)
        return cache_path

    @staticmethod
    def _write_output(payload: bytes, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}-", dir=destination.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _read_source(self, source: Path, *, expected_bytes: int, started_ns: int) -> bytes:
        if expected_bytes > self._maximum_artifact_bytes:
            raise OSError(
                f"artifact of {expected_bytes} bytes exceeds executor byte limit "
                f"{self._maximum_artifact_bytes}"
            )
        chunks: list[bytes] = []
        observed = 0
        with source.open("rb") as handle:
            while chunk := handle.read(self._read_chunk_bytes):
                observed += len(chunk)
                if observed > expected_bytes:
                    raise OSError("artifact grew while being read")
                chunks.append(chunk)
                elapsed = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
                if elapsed > self._maximum_operation_seconds:
                    raise TimeoutError(f"reading {source} exceeded executor timeout")
        if observed != expected_bytes:
            raise OSError("artifact size changed while being read")
        return b"".join(chunks)

    def execute(
        self,
        *,
        execution_id: str,
        plan: WarmPathPlan,
        graph: ArtifactGraph,
        host: HostEnvironment,
        tiers: tuple[StorageTierSpec, ...],
        source_directory: Path,
        output_directory: Path,
        seed: int = 17,
        injected_failures: frozenset[str] = frozenset(),
    ) -> ExecutionRecord:
        """Execute a local plan; ``seed`` is recorded in deterministic failure selection."""

        rng = random.Random(seed)
        if host.host_fingerprint != plan.host_fingerprint:
            raise ValueError("execution host fingerprint differs from compiled plan")
        expected_graph_hash = sha256_bytes(canonical_json(graph.model_dump(mode="json")).encode())
        if expected_graph_hash != plan.graph_hash:
            raise ValueError("artifact graph hash differs from compiled plan")
        tier_map = {tier.tier_id: tier for tier in tiers}
        if len(tier_map) != len(tiers):
            raise ValueError("storage tier identifiers must be unique")
        unsupported = [tier.tier_id for tier in tiers if tier.kind not in _EXECUTABLE_LOCAL_TIERS]
        if unsupported:
            raise ValueError(f"local executor cannot use non-local tiers: {unsupported}")
        artifact_map = {artifact.artifact_id: artifact for artifact in graph.artifacts}
        placement_map = {item.artifact_id: item for item in plan.placements}
        if set(placement_map) != set(artifact_map):
            raise ValueError("plan placements do not exactly cover the artifact graph")

        output_directory.mkdir(parents=True, exist_ok=True)
        records: list[ExecutorArtifactRecord] = []
        evictions: list[str] = []
        execution_started = time.perf_counter_ns()
        failure_reason: str | None = None
        protected: set[tuple[str, str]] = set()

        for artifact in graph.topological_order():
            placement = placement_map[artifact.artifact_id]
            tier = tier_map.get(placement.tier_id)
            if tier is None:
                raise ValueError(f"plan references unknown tier {placement.tier_id}")
            started = time.perf_counter_ns()
            source: Path | None = None
            destination: Path | None = None
            try:
                if artifact.artifact_id in injected_failures:
                    raise OSError("deterministic injected restore failure")
                if (
                    placement.mode == MaterializationMode.EAGER_RESTORE
                    and rng.random() < tier.restore_failure_probability
                ):
                    raise OSError("deterministic modeled restore failure")
                if placement.mode == MaterializationMode.KEEP_WARM:
                    status: Literal["restored", "rebuilt", "kept_warm", "deferred", "failed"] = (
                        "kept_warm"
                    )
                    bytes_materialized = 0
                    verified = True
                elif placement.mode == MaterializationMode.LAZY_RESTORE:
                    status = "deferred"
                    bytes_materialized = 0
                    verified = False
                else:
                    source = self._safe_path(source_directory, artifact.source_relative_path)
                    if not source.is_file():
                        raise FileNotFoundError(f"artifact source does not exist: {source}")
                    if source.stat().st_size != artifact.size_bytes:
                        raise OSError(
                            f"artifact size mismatch: expected {artifact.size_bytes}, "
                            f"got {source.stat().st_size}"
                        )
                    evictions.extend(
                        self._evict_for(
                            tier=tier,
                            required_bytes=artifact.size_bytes,
                            protected=protected,
                        )
                    )
                    payload = self._read_source(
                        source,
                        expected_bytes=artifact.size_bytes,
                        started_ns=started,
                    )
                    if sha256_bytes(payload) != artifact.sha256:
                        raise OSError("source artifact checksum does not match graph")
                    self._cache_payload(
                        tier=tier,
                        artifact_id=artifact.artifact_id,
                        checksum=artifact.sha256,
                        payload=payload,
                    )
                    protected.add((tier.tier_id, artifact.artifact_id))
                    destination = self._safe_path(output_directory, artifact.source_relative_path)
                    self._write_output(payload, destination)
                    if sha256_file(destination) != artifact.sha256:
                        raise OSError("materialized artifact checksum verification failed")
                    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000.0
                    if elapsed > self._maximum_operation_seconds:
                        raise TimeoutError(
                            f"artifact {artifact.artifact_id} exceeded operation timeout"
                        )
                    if placement.mode == MaterializationMode.REBUILD:
                        status = "rebuilt"
                    else:
                        status = "restored"
                    bytes_materialized = len(payload)
                    verified = True
                finished = time.perf_counter_ns()
                records.append(
                    ExecutorArtifactRecord(
                        artifact_id=artifact.artifact_id,
                        tier_id=tier.tier_id,
                        mode=placement.mode,
                        status=status,
                        started_ns=started,
                        finished_ns=finished,
                        bytes_materialized=bytes_materialized,
                        checksum_verified=verified,
                        source_path=str(source) if source is not None else None,
                        destination_path=str(destination) if destination is not None else None,
                    )
                )
            except (OSError, ValueError, TimeoutError) as error:
                finished = time.perf_counter_ns()
                failure_reason = f"{type(error).__name__}: {error}"
                records.append(
                    ExecutorArtifactRecord(
                        artifact_id=artifact.artifact_id,
                        tier_id=tier.tier_id,
                        mode=placement.mode,
                        status="failed",
                        started_ns=started,
                        finished_ns=finished,
                        bytes_materialized=0,
                        checksum_verified=False,
                        source_path=str(source) if source is not None else None,
                        destination_path=str(destination) if destination is not None else None,
                        error=failure_reason,
                    )
                )
                break

        ready_time_ms = (time.perf_counter_ns() - execution_started) / 1_000_000.0
        result_body = {
            "schema_version": WARMPATH_SCHEMA_VERSION,
            "execution_id": execution_id,
            "plan_id": plan.plan_id,
            "success": failure_reason is None,
            "records": [record.model_dump(mode="json") for record in records],
            "ready_time_ms": ready_time_ms,
            "output_directory": str(output_directory),
            "cache_evictions": evictions,
            "failure_reason": failure_reason,
        }
        result = ExecutionRecord(
            execution_id=execution_id,
            plan_id=plan.plan_id,
            success=failure_reason is None,
            records=tuple(records),
            ready_time_ms=ready_time_ms,
            output_directory=str(output_directory),
            cache_evictions=tuple(evictions),
            failure_reason=failure_reason,
            artifact_hash=sha256_bytes(canonical_json(result_body).encode()),
        )
        write_json(output_directory / "warmpath-execution.json", result.model_dump(mode="json"))
        return result


def clear_executor_cache(executor: LocalWarmPathExecutor) -> None:
    """Drop in-memory references; on-disk artifacts remain an explicit operator concern."""

    executor._memory.clear()
    executor._disk_entries.clear()
    executor._access.clear()
    executor._clock = 0


def copy_fixture_artifact(source: Path, destination: Path) -> None:
    """Copy a deterministic mock snapshot for fixture preparation."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def create_mock_snapshot_artifact(path: Path, *, seed: int, size_bytes: int) -> str:
    """Create explicit deterministic snapshot bytes for CPU-only integration fixtures."""

    if size_bytes <= 0 or size_bytes > 64 * 1024 * 1024:
        raise ValueError("mock snapshot size must be between 1 byte and 64 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(random.Random(seed).randbytes(size_bytes))
    return sha256_file(path)
