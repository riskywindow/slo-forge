"""Compact trace manifests with raw-artifact provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .buffer import TraceBufferStats
from .canonical import canonical_hash, canonical_json
from .models import (
    HardwareManifestV1,
    SamplingConfigurationV1,
    SoftwareManifestV1,
    TraceArtifactV1,
    TraceLevel,
    TraceManifestV1,
    WorkloadProvenance,
)


def artifact_from_file(
    path: Path,
    *,
    format: str,
    event_count: int,
) -> TraceArtifactV1:
    digest = hashlib.sha256()
    byte_length = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            byte_length += len(block)
    return TraceArtifactV1.model_validate(
        {
            "format": format,
            "uri": str(path),
            "byte_length": byte_length,
            "sha256": digest.hexdigest(),
            "event_count": event_count,
        }
    )


def trace_corpus_hash(trace_id: str, artifacts: tuple[TraceArtifactV1, ...]) -> str:
    """Hash stable artifact identities independent of tuple construction order."""

    identities = sorted(
        (
            {
                "format": artifact.format,
                "uri": artifact.uri,
                "byte_length": artifact.byte_length,
                "sha256": artifact.sha256,
                "event_count": artifact.event_count,
            }
            for artifact in artifacts
        ),
        key=lambda item: (item["uri"], item["format"]),
    )
    return canonical_hash({"trace_id": trace_id, "artifacts": identities})


def build_manifest(
    *,
    trace_id: str,
    session_id: str,
    created_at: str,
    seed: int,
    provenance: WorkloadProvenance,
    collection_level: TraceLevel,
    stats: TraceBufferStats,
    sampling: SamplingConfigurationV1,
    hardware: HardwareManifestV1,
    software: SoftwareManifestV1,
    artifacts: tuple[TraceArtifactV1, ...],
) -> TraceManifestV1:
    return TraceManifestV1(
        trace_id=trace_id,
        session_id=session_id,
        created_at=created_at,
        seed=seed,
        provenance=provenance,
        collection_level=collection_level,
        buffer_capacity_events=stats.capacity_events,
        attempted_events=stats.attempted_events,
        accepted_events=stats.accepted_events,
        dropped_events=stats.dropped_events,
        filtered_events=stats.filtered_events,
        highest_event_sequence=stats.highest_event_sequence,
        event_counts=stats.event_counts,
        sampling=sampling,
        hardware=hardware,
        software=software,
        artifacts=artifacts,
        trace_corpus_hash=trace_corpus_hash(trace_id, artifacts),
    )


def write_manifest(path: Path, manifest: TraceManifestV1, *, overwrite: bool = False) -> None:
    with path.open("wb" if overwrite else "xb") as handle:
        handle.write(canonical_json(manifest))
        handle.write(b"\n")


__all__ = [
    "artifact_from_file",
    "build_manifest",
    "trace_corpus_hash",
    "write_manifest",
]
