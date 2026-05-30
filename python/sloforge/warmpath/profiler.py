"""Bounded local startup profiler with raw measurement provenance."""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

from sloforge.util import (
    canonical_json,
    environment_manifest,
    sha256_bytes,
    sha256_file,
    write_json,
)
from sloforge.warmpath.models import (
    ArtifactGraph,
    StageMeasurement,
    StartupProfile,
    StartupStage,
    StorageKind,
    StorageTierSpec,
)
from sloforge.warmpath.statistics import robust_summary

_LOCAL_PROFILE_KINDS = {
    StorageKind.LOCAL_NVME,
    StorageKind.PAGE_CACHE,
    StorageKind.HOST_MEMORY,
}


def _timed_read(path: Path, *, timeout_seconds: float, chunk_bytes: int) -> tuple[float, bytes]:
    started = time.perf_counter_ns()
    chunks: list[bytes] = []
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            chunks.append(chunk)
            if (time.perf_counter_ns() - started) / 1_000_000_000.0 > timeout_seconds:
                raise TimeoutError(f"reading {path} exceeded {timeout_seconds:.3f}s")
    finished = time.perf_counter_ns()
    return (finished - started) / 1_000_000.0, b"".join(chunks)


def _timed_verify(payload: bytes, expected_sha256: str) -> float:
    started = time.perf_counter_ns()
    actual = hashlib.sha256(payload).hexdigest()
    finished = time.perf_counter_ns()
    if actual != expected_sha256:
        raise ValueError(f"artifact checksum mismatch: expected {expected_sha256}, got {actual}")
    return (finished - started) / 1_000_000.0


def _timed_memory_copy(payload: bytes) -> tuple[float, bytes]:
    started = time.perf_counter_ns()
    copied = memoryview(payload).tobytes()
    finished = time.perf_counter_ns()
    return (finished - started) / 1_000_000.0, copied


def _measurement(
    *,
    artifact_id: str,
    tier_id: str,
    stage: StartupStage,
    warmups: int,
    samples: tuple[float, ...],
    environment_fingerprint: str,
    invocation: str,
    timeout_seconds: float,
    seed: int,
) -> StageMeasurement:
    median, p95, mad, low, high = robust_summary(samples, seed=seed)
    raw = {
        "artifact_id": artifact_id,
        "tier_id": tier_id,
        "stage": stage,
        "warmups": warmups,
        "samples_ms": samples,
        "environment_fingerprint": environment_fingerprint,
        "invocation": invocation,
    }
    return StageMeasurement(
        artifact_id=artifact_id,
        tier_id=tier_id,
        stage=stage,
        warmup_count=warmups,
        raw_samples_ms=samples,
        median_ms=median,
        p95_ms=p95,
        median_absolute_deviation_ms=mad,
        confidence_level=0.95,
        confidence_interval_low_ms=low,
        confidence_interval_high_ms=high,
        source="measured",
        environment_fingerprint=environment_fingerprint,
        invocation=invocation,
        timeout_seconds=timeout_seconds,
        artifact_hash=sha256_bytes(canonical_json(raw).encode()),
    )


def profile_local_startup(
    *,
    profile_id: str,
    graph: ArtifactGraph,
    host: object,
    tiers: tuple[StorageTierSpec, ...],
    source_directory: Path,
    output_directory: Path,
    warmups: int = 2,
    sample_count: int = 7,
    seed: int = 17,
    timeout_seconds: float = 10.0,
    maximum_artifact_bytes: int = 1 << 30,
    chunk_bytes: int = 1 << 20,
) -> StartupProfile:
    """Measure local fetch and verification stages without device fallback.

    ``host`` is validated as ``HostEnvironment`` at the public boundary to keep
    importing this module inexpensive for probe-only commands.
    """

    from sloforge.warmpath.models import HostEnvironment

    validated_host = HostEnvironment.model_validate(host)
    if warmups < 0 or sample_count < 3:
        raise ValueError("warmups must be non-negative and sample_count must be at least three")
    if timeout_seconds <= 0.0 or maximum_artifact_bytes <= 0 or chunk_bytes <= 0:
        raise ValueError("profiler limits must be positive")
    if not tiers:
        raise ValueError("at least one storage tier is required")
    unsupported = [tier.tier_id for tier in tiers if tier.kind not in _LOCAL_PROFILE_KINDS]
    if unsupported:
        raise ValueError(f"local profiler cannot measure non-local tiers: {unsupported}")

    output_directory.mkdir(parents=True, exist_ok=True)
    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    manifest = environment_manifest(include_packages=True)
    environment_fingerprint = sha256_bytes(canonical_json(manifest).encode())
    write_json(output_directory / "environment.json", manifest)
    rng = random.Random(seed)
    measurements: list[StageMeasurement] = []

    jobs = [(artifact, tier) for artifact in graph.topological_order() for tier in tiers]
    rng.shuffle(jobs)
    for artifact, tier in jobs:
        source = (source_directory / artifact.source_relative_path).resolve()
        try:
            source.relative_to(source_directory.resolve())
        except ValueError as error:
            raise ValueError(f"artifact {artifact.artifact_id} escapes source directory") from error
        if not source.is_file():
            raise FileNotFoundError(f"artifact source does not exist: {source}")
        if source.stat().st_size != artifact.size_bytes:
            raise ValueError(
                f"artifact {artifact.artifact_id} size differs: "
                f"expected {artifact.size_bytes}, got {source.stat().st_size}"
            )
        if artifact.size_bytes > maximum_artifact_bytes:
            raise ValueError(f"artifact {artifact.artifact_id} exceeds profiler byte limit")
        if sha256_file(source) != artifact.sha256:
            raise ValueError(f"artifact {artifact.artifact_id} checksum differs before profiling")

        memory_payload = source.read_bytes() if tier.kind == StorageKind.HOST_MEMORY else None

        fetch_samples: list[float] = []
        verify_samples: list[float] = []
        for index in range(warmups + sample_count):
            if memory_payload is None:
                fetch_ms, payload = _timed_read(
                    source, timeout_seconds=timeout_seconds, chunk_bytes=chunk_bytes
                )
            else:
                fetch_ms, payload = _timed_memory_copy(memory_payload)
            verify_ms = _timed_verify(payload, artifact.sha256)
            if index >= warmups:
                fetch_samples.append(fetch_ms)
                verify_samples.append(verify_ms)
        invocation = (
            f"host-memory-copy:{source.name}"
            if memory_payload is not None
            else f"local-read:{source.name}:chunk={chunk_bytes}"
        )
        for stage, samples in (
            (StartupStage.FETCH, tuple(fetch_samples)),
            (StartupStage.VERIFY, tuple(verify_samples)),
        ):
            measurement = _measurement(
                artifact_id=artifact.artifact_id,
                tier_id=tier.tier_id,
                stage=stage,
                warmups=warmups,
                samples=samples,
                environment_fingerprint=environment_fingerprint,
                invocation=invocation,
                timeout_seconds=timeout_seconds,
                seed=seed + len(measurements),
            )
            measurements.append(measurement)
            write_json(
                raw_directory / f"{artifact.artifact_id}-{tier.tier_id}-{stage.value}.json",
                measurement.model_dump(mode="json"),
            )

    profile = StartupProfile(
        profile_id=profile_id,
        graph_id=graph.graph_id,
        host=validated_host,
        tiers=tiers,
        measurements=tuple(measurements),
        raw_artifact_directory=str(raw_directory),
        environment_manifest_path=str(output_directory / "environment.json"),
    )
    write_json(output_directory / "profile.json", profile.model_dump(mode="json"))
    return profile


def load_profile(path: Path) -> StartupProfile:
    """Load a strict profile artifact."""

    return StartupProfile.model_validate_json(path.read_text(encoding="utf-8"))


def profile_artifact_hash(profile: StartupProfile) -> str:
    """Hash a profile independent of its formatted on-disk representation."""

    return sha256_bytes(json.dumps(profile.model_dump(mode="json"), sort_keys=True).encode())
