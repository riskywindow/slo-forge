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
    NicNode,
    TopologyGraph,
    canonical_hash,
)
from sloforge.fabric.ir import (
    FabricProfile as CanonicalFabricProfile,
)
from sloforge.fabric.ir import (
    TopologyGraph as CanonicalTopologyGraph,
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


def _transport(result: BenchmarkResult, topology: TopologyGraph) -> str:
    if result.case.invocation.adapter == "nccl-tests":
        return "nccl-local"
    edges = {edge.edge_id: edge for edge in topology.edges}
    path = tuple(edges[edge_id] for edge_id in result.case.topology_path if edge_id in edges)
    connections = {edge.connection.value for edge in path}
    if "nic_network" in connections:
        nic_ids = {
            endpoint for edge in path for endpoint in (edge.source_node_id, edge.target_node_id)
        }
        transports = sorted(
            node.transport
            for node in topology.nodes
            if isinstance(node, NicNode) and node.node_id in nic_ids
        )
        return transports[0] if transports else "tcp"
    if connections & {"nvlink", "nvswitch"}:
        return "nvlink"
    if connections & {"pcie", "gpu_nic", "cpu_gpu"}:
        return "pcie"
    if "cpu_memory" in connections:
        return "shared_memory"
    if result.case.primitive in {Primitive.KERNEL_LAUNCH, Primitive.DEVICE_SYNCHRONIZE}:
        return "runtime"
    return "device"


def _series(result: BenchmarkResult, topology: TopologyGraph) -> FabricMeasurementSeries:
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
        transport=_transport(result, topology),
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


def to_canonical_profile(
    profile: FabricProfile, *, topology: CanonicalTopologyGraph
) -> CanonicalFabricProfile:
    """Convert measurements and bind them to an exact canonical topology hash."""
    canonical_topology_hash = canonical_hash(topology)
    discovery_fingerprint = topology.extensions.root.get("sloforge.io/discovery-fingerprint")
    compatible_fingerprints = {canonical_topology_hash}
    if isinstance(discovery_fingerprint, str):
        compatible_fingerprints.add(discovery_fingerprint)
    if profile.topology_fingerprint not in compatible_fingerprints:
        raise ValueError(
            "raw FabricProfile topology does not correspond to the supplied canonical TopologyGraph"
        )
    topology_digest = ArtifactDigest(algorithm="sha256", value=canonical_topology_hash)
    environment_payload = [item.model_dump(mode="json") for item in profile.environment]
    environment_digest = _digest(environment_payload)
    hardware_reference = DocumentReference(
        kind="TopologyGraph",
        api_version="sloforge.io/fabric/v1",
        uri=f"urn:sloforge:topology:{canonical_topology_hash}",
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
        measurements=tuple(_series(result, topology) for result in profile.results),
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
