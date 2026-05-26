"""Canonical Fabric IR conversion for benchmark harness artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal, cast

from pydantic import JsonValue

from sloforge.fabric.ir import (
    BenchmarkInvocation,
    DocumentReference,
    FabricMeasurementSeries,
    FabricRawSample,
)
from sloforge.fabric.ir import (
    FabricProfile as CanonicalFabricProfile,
)
from sloforge.fabric.profiling.models import (
    BenchmarkResult,
    BenchmarkStatus,
    FabricProfile,
    Primitive,
)
from sloforge.ir import ArtifactDigest, Extensions

_PRIMITIVE_CATEGORY: dict[Primitive, str] = {
    Primitive.KERNEL_LAUNCH: "launch",
    Primitive.DEVICE_SYNCHRONIZE: "synchronize",
    Primitive.DEVICE_MEMORY: "memory",
    Primitive.GEMM: "gemm",
    Primitive.PREFILL: "gemm",
    Primitive.DECODE: "gemm",
    Primitive.HOST_MEMCPY: "copy",
    Primitive.H2D_PAGEABLE: "copy",
    Primitive.H2D_PINNED: "copy",
    Primitive.D2H: "copy",
    Primitive.GPU_P2P: "p2p",
    Primitive.ALL_REDUCE: "collective",
    Primitive.ALL_GATHER: "collective",
    Primitive.REDUCE_SCATTER: "collective",
    Primitive.BROADCAST: "collective",
    Primitive.SEND_RECV: "collective",
    Primitive.ALL_TO_ALL: "collective",
    Primitive.EXPERT_DISPATCH: "expert",
    Primitive.EXPERT_COMBINE: "expert",
    Primitive.KV_TRANSFER: "kv_transfer",
    Primitive.STARTUP: "startup",
    Primitive.GROUP_INITIALIZATION: "startup",
}


def _digest(value: object) -> ArtifactDigest:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return ArtifactDigest(algorithm="sha256", value=hashlib.sha256(payload).hexdigest())


def _series(result: BenchmarkResult) -> FabricMeasurementSeries:
    if result.status is not BenchmarkStatus.SUCCESS or result.summary is None:
        raise ValueError(
            f"canonical FabricProfile cannot encode unsuccessful series {result.case.case_id}; "
            "retain it in the raw profiling record"
        )
    environment_digest = _digest([item.model_dump(mode="json") for item in result.environment])
    return FabricMeasurementSeries(
        measurement_id=result.case.case_id,
        primitive=cast(
            Literal[
                "launch",
                "synchronize",
                "memory",
                "gemm",
                "copy",
                "p2p",
                "collective",
                "expert",
                "kv_transfer",
                "startup",
            ],
            _PRIMITIVE_CATEGORY[result.case.primitive],
        ),
        transport=result.case.invocation.adapter,
        rank_count=result.case.rank_count,
        message_bytes=result.case.message_bytes,
        concurrency=result.case.concurrency,
        warmup_count=result.case.warmup_count,
        samples=tuple(
            FabricRawSample(
                duration_us=max(0.001, sample.duration_microseconds),
                bytes_transferred=result.case.message_bytes,
                success=True,
                failure_reason=None,
            )
            for sample in result.raw_samples
        ),
        summary_median_us=max(0.001, result.summary.median_microseconds),
        summary_p95_us=max(0.001, result.summary.p95_microseconds),
        confidence_low_us=result.summary.median_ci_low_microseconds,
        confidence_high_us=max(0.001, result.summary.median_ci_high_microseconds),
        invocation=BenchmarkInvocation(
            argv=result.case.invocation.argv,
            timeout_seconds=result.case.invocation.timeout_seconds,
            process_placement=json.dumps(
                result.case.placement.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ),
            environment_digest=environment_digest,
        ),
        artifact_digest=ArtifactDigest(algorithm="sha256", value=result.artifact_hash),
    )


def to_canonical_profile(profile: FabricProfile) -> CanonicalFabricProfile:
    """Convert successful measurements while retaining raw artifact references."""
    topology_digest = ArtifactDigest(algorithm="sha256", value=profile.topology_fingerprint)
    environment_payload = [item.model_dump(mode="json") for item in profile.environment]
    environment_digest = _digest(environment_payload)
    hardware_reference = DocumentReference(
        kind="TopologyGraph",
        api_version="sloforge.io/fabric/v1",
        uri=f"urn:sloforge:topology:{profile.topology_fingerprint}",
        digest=topology_digest,
    )
    software_reference = DocumentReference(
        kind="SoftwareManifest",
        api_version="sloforge.io/fabric/v1",
        uri=f"urn:sloforge:software:{environment_digest.value}",
        digest=environment_digest,
    )
    raw_artifacts = tuple(
        DocumentReference(
            kind="FabricRawSamples",
            api_version="sloforge.io/fabric/v1",
            uri=result.raw_artifact or f"urn:sloforge:raw:{result.case.case_id}",
            digest=ArtifactDigest(algorithm="sha256", value=result.artifact_hash),
        )
        for result in profile.results
    )
    modes = sorted({result.mode.value for result in profile.results})
    return CanonicalFabricProfile(
        profile_id=profile.profile_id,
        topology_fingerprint=topology_digest,
        created_at=datetime.fromisoformat(profile.captured_at),
        hardware_manifest=hardware_reference,
        software_manifest=software_reference,
        measurements=tuple(_series(result) for result in profile.results),
        raw_artifacts=raw_artifacts,
        extensions=Extensions(
            root={
                "sloforge.io/measurement-modes": cast(JsonValue, modes),
                "sloforge.io/profile-seed": profile.seed,
                "sloforge.io/profile-suite": profile.suite,
                "sloforge.io/raw-profile-hash": profile.profile_hash,
            }
        ),
    )
