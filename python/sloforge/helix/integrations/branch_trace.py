"""Export canonical Helix branch workloads and Autopsy-compatible operation evidence."""

from __future__ import annotations

import math

from sloforge.autopsy import (
    AutopsyEvent,
    AutopsyRun,
    EventType,
    EvidenceRef,
    ResourceRef,
    SourceClock,
)
from sloforge.helix.ir import (
    BranchGroup,
    BranchWorkloadRequest,
    BranchWorkloadStatus,
    BranchWorkloadTrace,
    Digest,
    LineageReference,
    LineageRelation,
    TrajectoryCapsule,
    TrajectoryTerminalStatus,
    canonical_hash,
)

from .models import (
    BranchOperationEvidence,
    BranchOperationKind,
    BranchTraceExport,
    EvidenceClaimScope,
    canonical_digest,
)

_ASSUMPTIONS = (
    "Operation offsets preserve the supplied workload schedule and are not measured durations.",
    "Transfer bytes are the declared capsule payload size, not observed physical network traffic.",
    "Divergence means canonical trajectory-capsule identity differs, not causal or semantic divergence.",
)


def _status(trajectory: TrajectoryCapsule) -> BranchWorkloadStatus:
    if trajectory.terminal_status is TrajectoryTerminalStatus.COMPLETED:
        return BranchWorkloadStatus.COMPLETED
    if trajectory.terminal_status is TrajectoryTerminalStatus.CANCELLED:
        return BranchWorkloadStatus.CANCELLED
    return BranchWorkloadStatus.FAILED


def _lineage(
    branch_group: BranchGroup, trajectory_hashes: dict[str, str]
) -> tuple[LineageReference, ...]:
    return (
        LineageReference(
            artifact_id=branch_group.group_id,
            artifact_kind="sloforge.helix/BranchGroup",
            relation=LineageRelation.SOURCE,
            digest=Digest(value=canonical_hash(branch_group)),
        ),
        *(
            LineageReference(
                artifact_id=trajectory.trajectory_id,
                artifact_kind="sloforge.helix/TrajectoryCapsule",
                relation=LineageRelation.DERIVED_FROM,
                digest=Digest(value=trajectory_hashes[trajectory.trajectory_id]),
            )
            for trajectory in branch_group.trajectories
        ),
    )


def _operation(
    *,
    ordinal: int,
    kind: BranchOperationKind,
    source_artifact_id: str,
    target_artifact_id: str,
    logical_offset_ns: int,
    checksum_sha256: str,
    evidence_uri: str,
    evidence_sha256: str,
    claim_scope: EvidenceClaimScope,
    detail: str,
    trajectory_id: str | None = None,
    byte_count: int | None = None,
) -> BranchOperationEvidence:
    unsealed = BranchOperationEvidence.model_construct(
        operation_id="0" * 64,
        ordinal=ordinal,
        kind=kind,
        trajectory_id=trajectory_id,
        source_artifact_id=source_artifact_id,
        target_artifact_id=target_artifact_id,
        logical_offset_ns=logical_offset_ns,
        checksum_sha256=checksum_sha256,
        byte_count=byte_count,
        evidence_uri=evidence_uri,
        evidence_sha256=evidence_sha256,
        claim_scope=claim_scope,
        detail=detail,
    )
    operation_id = canonical_digest(unsealed.model_dump(mode="json", exclude={"operation_id"}))
    return BranchOperationEvidence(
        operation_id=operation_id,
        ordinal=ordinal,
        kind=kind,
        trajectory_id=trajectory_id,
        source_artifact_id=source_artifact_id,
        target_artifact_id=target_artifact_id,
        logical_offset_ns=logical_offset_ns,
        checksum_sha256=checksum_sha256,
        byte_count=byte_count,
        evidence_uri=evidence_uri,
        evidence_sha256=evidence_sha256,
        claim_scope=claim_scope,
        detail=detail,
    )


def _operations(
    branch_group: BranchGroup,
    offsets_ns: tuple[int, ...],
    trajectory_hashes: dict[str, str],
) -> tuple[BranchOperationEvidence, ...]:
    capsule = branch_group.branch_point.environment_state
    operations: list[BranchOperationEvidence] = []

    operations.append(
        _operation(
            ordinal=len(operations),
            kind=BranchOperationKind.STATE,
            source_artifact_id=branch_group.branch_point.branch_point_id,
            target_artifact_id=capsule.capsule_id,
            logical_offset_ns=0,
            checksum_sha256=capsule.state_digest.value,
            byte_count=capsule.payload_byte_length,
            evidence_uri=capsule.payload_uri,
            evidence_sha256=capsule.state_digest.value,
            claim_scope=EvidenceClaimScope.DECLARED_ARTIFACT,
            detail="captured environment state declared by the branch point",
        )
    )
    operations.append(
        _operation(
            ordinal=len(operations),
            kind=BranchOperationKind.STORAGE,
            source_artifact_id=capsule.capsule_id,
            target_artifact_id=capsule.payload_uri,
            logical_offset_ns=0,
            checksum_sha256=capsule.state_digest.value,
            byte_count=capsule.payload_byte_length,
            evidence_uri=capsule.payload_uri,
            evidence_sha256=capsule.state_digest.value,
            claim_scope=EvidenceClaimScope.DECLARED_ARTIFACT,
            detail="declared capsule payload available for content-addressed storage read",
        )
    )
    for trajectory, offset_ns in zip(branch_group.trajectories, offsets_ns, strict=True):
        trajectory_hash = trajectory_hashes[trajectory.trajectory_id]
        operations.append(
            _operation(
                ordinal=len(operations),
                kind=BranchOperationKind.FORK,
                trajectory_id=trajectory.trajectory_id,
                source_artifact_id=capsule.capsule_id,
                target_artifact_id=trajectory.trajectory_id,
                logical_offset_ns=offset_ns,
                checksum_sha256=trajectory_hash,
                evidence_uri=trajectory.trace_evidence.uri,
                evidence_sha256=trajectory.trace_evidence.digest.value,
                claim_scope=EvidenceClaimScope.DERIVED_CONTENT_IDENTITY,
                detail="trajectory declares descent from the captured state capsule",
            )
        )
        operations.append(
            _operation(
                ordinal=len(operations),
                kind=BranchOperationKind.TRANSFER,
                trajectory_id=trajectory.trajectory_id,
                source_artifact_id=capsule.capsule_id,
                target_artifact_id=trajectory.trajectory_id,
                logical_offset_ns=offset_ns,
                checksum_sha256=capsule.state_digest.value,
                byte_count=capsule.payload_byte_length,
                evidence_uri=capsule.payload_uri,
                evidence_sha256=capsule.state_digest.value,
                claim_scope=EvidenceClaimScope.DECLARED_ARTIFACT,
                detail="logical state handoff scoped to declared capsule bytes; no physical transfer measured",
            )
        )
        operations.append(
            _operation(
                ordinal=len(operations),
                kind=BranchOperationKind.CHECKSUM,
                trajectory_id=trajectory.trajectory_id,
                source_artifact_id=trajectory.trajectory_id,
                target_artifact_id=trajectory.trajectory_id,
                logical_offset_ns=offset_ns,
                checksum_sha256=trajectory_hash,
                evidence_uri=trajectory.trace_evidence.uri,
                evidence_sha256=trajectory.trace_evidence.digest.value,
                claim_scope=EvidenceClaimScope.DERIVED_CONTENT_IDENTITY,
                detail="canonical TrajectoryCapsule checksum derived from the validated input model",
            )
        )

    baseline = next(
        item
        for item in branch_group.trajectories
        if item.trajectory_id == branch_group.baseline_trajectory_id
    )
    baseline_hash = trajectory_hashes[baseline.trajectory_id]
    for trajectory, offset_ns in zip(branch_group.trajectories, offsets_ns, strict=True):
        if trajectory.trajectory_id == baseline.trajectory_id:
            continue
        trajectory_hash = trajectory_hashes[trajectory.trajectory_id]
        comparison_hash = canonical_digest({"baseline": baseline_hash, "observed": trajectory_hash})
        operations.append(
            _operation(
                ordinal=len(operations),
                kind=BranchOperationKind.DIVERGENCE,
                trajectory_id=trajectory.trajectory_id,
                source_artifact_id=baseline.trajectory_id,
                target_artifact_id=trajectory.trajectory_id,
                logical_offset_ns=offset_ns,
                checksum_sha256=comparison_hash,
                evidence_uri=trajectory.trace_evidence.uri,
                evidence_sha256=trajectory.trace_evidence.digest.value,
                claim_scope=EvidenceClaimScope.DERIVED_CONTENT_IDENTITY,
                detail="canonical trajectory capsule differs from the baseline capsule identity",
            )
        )
    return tuple(operations)


def _autopsy_event(
    operation: BranchOperationEvidence,
    *,
    host: str,
    parent_event_id: str | None,
) -> AutopsyEvent:
    event_types = {
        BranchOperationKind.STATE: EventType.CONTROLLER,
        BranchOperationKind.FORK: EventType.CONTROLLER,
        BranchOperationKind.TRANSFER: EventType.KV_TRANSFER,
        BranchOperationKind.CHECKSUM: EventType.CPU_LAUNCH,
        BranchOperationKind.STORAGE: EventType.STORAGE_READ,
        BranchOperationKind.DIVERGENCE: EventType.CONTROLLER,
    }
    if operation.kind is BranchOperationKind.TRANSFER:
        resource = ResourceRef(resource_id=f"helix-{operation.kind.value}", resource_type="process")
    elif operation.kind is BranchOperationKind.STORAGE:
        resource = ResourceRef(resource_id=f"helix-{operation.kind.value}", resource_type="storage")
    else:
        resource = ResourceRef(
            resource_id=f"helix-{operation.kind.value}", resource_type="logical_queue"
        )
    return AutopsyEvent(
        event_id=operation.operation_id,
        event_type=event_types[operation.kind],
        host=host,
        request_id=operation.trajectory_id,
        operation=f"helix.branch.{operation.kind.value}",
        start_ns=operation.logical_offset_ns,
        end_ns=operation.logical_offset_ns,
        source_clock=SourceClock.SYNTHETIC,
        parent_event_id=parent_event_id,
        resource=resource,
        evidence=EvidenceRef(
            source="sloforge.helix.integrations",
            artifact_uri=operation.evidence_uri,
            sha256=operation.evidence_sha256,
            record_index=operation.ordinal,
        ),
    )


def export_branch_workload_trace(
    branch_group: BranchGroup,
    *,
    raw_branch_group_uri: str,
    scheduled_offsets_ms: tuple[float, ...],
    seed: int,
    topology_fingerprint: str,
    physical_plan_hash: str,
    reference_host: str = "helix-local",
) -> BranchTraceExport:
    """Export validated branch inputs without claiming generated timing measurements."""

    if len(scheduled_offsets_ms) != len(branch_group.trajectories):
        raise ValueError("one scheduled offset is required for every branch trajectory")
    if len(branch_group.trajectories) > 10_000:
        raise ValueError("branch trace export is bounded to 10000 trajectories")
    if any(not math.isfinite(value) or value < 0.0 for value in scheduled_offsets_ms):
        raise ValueError("scheduled offsets must be finite and non-negative")
    trajectory_hashes = {
        trajectory.trajectory_id: canonical_hash(trajectory)
        for trajectory in branch_group.trajectories
    }
    offsets_ns = tuple(round(value * 1_000_000) for value in scheduled_offsets_ms)
    if any(value > 2**63 - 1 for value in offsets_ns):
        raise ValueError("scheduled offsets must fit signed 64-bit nanoseconds")
    requests = tuple(
        BranchWorkloadRequest(
            request_id=trajectory.trajectory_id,
            branch_point_id=branch_group.branch_point.branch_point_id,
            trajectory_id=trajectory.trajectory_id,
            ordinal=ordinal,
            scheduled_offset_ms=scheduled_offsets_ms[ordinal],
            input_digest=branch_group.branch_point.prefix_digest,
            output_digest=Digest(value=trajectory_hashes[trajectory.trajectory_id]),
            status=_status(trajectory),
        )
        for ordinal, trajectory in enumerate(branch_group.trajectories)
    )
    group_hash = canonical_hash(branch_group)
    lineage = _lineage(branch_group, trajectory_hashes)
    trace_payload = {
        "branch_group_id": branch_group.group_id,
        "environment_id": branch_group.branch_point.environment_state.environment_id,
        "seed": seed,
        "requests": [item.model_dump(mode="json") for item in requests],
        "raw_trace_uri": raw_branch_group_uri,
        "raw_trace_digest": group_hash,
    }
    workload_trace = BranchWorkloadTrace(
        trace_id=canonical_digest(trace_payload),
        branch_group_id=branch_group.group_id,
        environment_id=branch_group.branch_point.environment_state.environment_id,
        seed=seed,
        started_at=min(item.started_at for item in branch_group.trajectories),
        completed_at=max(item.completed_at for item in branch_group.trajectories),
        requests=requests,
        raw_trace_uri=raw_branch_group_uri,
        raw_trace_digest=Digest(value=group_hash),
        lineage=lineage,
    )
    operations = _operations(branch_group, offsets_ns, trajectory_hashes)
    events = tuple(
        _autopsy_event(
            operation,
            host=reference_host,
            parent_event_id=operations[index - 1].operation_id if index else None,
        )
        for index, operation in enumerate(operations)
    )
    evidence_refs = {
        (raw_branch_group_uri, group_hash),
        (
            branch_group.branch_point.environment_state.payload_uri,
            branch_group.branch_point.environment_state.state_digest.value,
        ),
    }
    evidence_refs.update(
        (trajectory.trace_evidence.uri, trajectory.trace_evidence.digest.value)
        for trajectory in branch_group.trajectories
    )
    artifacts = tuple(
        EvidenceRef(
            source="sloforge.helix",
            artifact_uri=uri,
            sha256=digest,
        )
        for uri, digest in sorted(evidence_refs)
    )
    autopsy_run = AutopsyRun(
        run_id=f"helix-branch-{workload_trace.trace_id}",
        source="imported_trace",
        topology_fingerprint=topology_fingerprint,
        physical_plan_hash=physical_plan_hash,
        workload_fingerprint=canonical_digest(workload_trace),
        reference_host=reference_host,
        events=events,
        artifacts=artifacts,
        warnings=(
            "Helix schedule offsets are logical inputs; zero-duration events are not latency measurements.",
            "Trajectory identity divergence is not a causal diagnosis.",
        ),
    )
    source_hashes = tuple(
        sorted(
            {
                group_hash,
                branch_group.branch_point.environment_state.state_digest.value,
                *(trajectory_hashes.values()),
                *(
                    trajectory.trace_evidence.digest.value
                    for trajectory in branch_group.trajectories
                ),
            }
        )
    )
    unsealed = BranchTraceExport.model_construct(
        export_id="0" * 64,
        workload_trace=workload_trace,
        operations=operations,
        autopsy_run=autopsy_run,
        source_branch_group_sha256=group_hash,
        source_artifact_hashes=source_hashes,
        assumptions=_ASSUMPTIONS,
    )
    export_id = canonical_digest(unsealed.model_dump(mode="json", exclude={"export_id"}))
    return BranchTraceExport(
        export_id=export_id,
        workload_trace=workload_trace,
        operations=operations,
        autopsy_run=autopsy_run,
        source_branch_group_sha256=group_hash,
        source_artifact_hashes=source_hashes,
        assumptions=_ASSUMPTIONS,
    )


__all__ = ["export_branch_workload_trace"]
