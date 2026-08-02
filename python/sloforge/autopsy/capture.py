"""Normalize Fabric simulator evidence into the canonical Autopsy event model."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from sloforge.fabric.ir import PhysicalExecutionPlan, canonical_hash
from sloforge.fabric.simulation import (
    FabricSimulationOutput,
    FabricSimulationRequest,
    OperationOutcome,
)
from sloforge.util import canonical_json, sha256_file

from .models import (
    AlignmentEstimate,
    AlignmentQuality,
    AutopsyEvent,
    AutopsyRun,
    CounterValue,
    EventType,
    EvidenceRef,
    FaultInterval,
    ResourceRef,
    SourceClock,
)

ResourceType = Literal[
    "cpu_core_group",
    "numa_memory",
    "gpu_compute",
    "gpu_hbm",
    "copy_engine",
    "nvlink",
    "nvswitch",
    "pcie",
    "nic_queue",
    "network_rail",
    "storage",
    "process",
    "logical_queue",
]


def _event_type(operation_id: str, resource_ids: tuple[str, ...]) -> EventType:
    if operation_id.endswith(":launch"):
        return EventType.CPU_LAUNCH
    if operation_id.endswith(":prefill"):
        return EventType.PREFILL
    if operation_id.endswith(":decode"):
        return EventType.DECODE
    if ":kv-" in operation_id:
        return EventType.KV_TRANSFER
    if ":collective-" in operation_id:
        if any("nic-network" in identifier or "rail" in identifier for identifier in resource_ids):
            return EventType.NETWORK_TRANSFER
        if any("nvlink" in identifier for identifier in resource_ids):
            return EventType.NVLINK_TRANSFER
        if any("pcie" in identifier or "gpu-nic" in identifier for identifier in resource_ids):
            return EventType.PCIE_TRANSFER
        return EventType.COLLECTIVE
    return EventType.GPU_COMPUTE


def _resource(outcome: OperationOutcome) -> ResourceRef:
    identifier = outcome.resource_ids[0] if outcome.resource_ids else "logical:dependency-barrier"
    resource_type: ResourceType
    if "compute:" in identifier:
        resource_type = "gpu_compute"
    elif "nvlink" in identifier:
        resource_type = "nvlink"
    elif "pcie" in identifier or "gpu-nic" in identifier:
        resource_type = "pcie"
    elif "nic-network" in identifier or "rail" in identifier:
        resource_type = "network_rail"
    elif "cpu:" in identifier:
        resource_type = "cpu_core_group"
    else:
        resource_type = "logical_queue"
    return ResourceRef(
        resource_id=identifier,
        resource_type=resource_type,
        topology_node_id=identifier,
        contention_domain=identifier,
    )


def _operation_name(operation_id: str) -> str:
    suffix = operation_id.split(":", maxsplit=1)[-1]
    if suffix.startswith("rank-"):
        return suffix.rsplit(":", maxsplit=1)[-1]
    return suffix.split("-", maxsplit=1)[0]


def _counters(outcome: OperationOutcome, event_type: EventType) -> tuple[CounterValue, ...]:
    counters = [
        CounterValue(name="wait_ms", value=outcome.wait_us / 1_000.0, unit="ms"),
        CounterValue(
            name="prediction_uncertainty_ms",
            value=outcome.uncertainty_us / 1_000.0,
            unit="ms",
        ),
    ]
    if event_type is EventType.NETWORK_TRANSFER and outcome.duration_us > 0.0:
        counters.append(
            CounterValue(
                name="network_bandwidth_gbps",
                value=outcome.transferred_bytes * 8.0 / outcome.duration_us / 1_000.0,
                unit="Gbps",
            )
        )
    return tuple(counters)


def capture_simulation_run(
    *,
    run_id: str,
    request: FabricSimulationRequest,
    output: FabricSimulationOutput,
    plan: PhysicalExecutionPlan,
    topology_fingerprint: str,
    workload_fingerprint: str,
    artifact_path: Path,
) -> AutopsyRun:
    """Create event evidence whose metrics are derived from the actual simulator output."""

    if not artifact_path.is_file():
        raise FileNotFoundError(f"simulation evidence does not exist: {artifact_path}")
    evidence = EvidenceRef(
        source="sloforge-fabric-sim",
        artifact_uri=str(artifact_path),
        sha256=sha256_file(artifact_path),
    )
    operations_by_id = {operation.id: operation for operation in request.operations}
    bindings = {f"rank-{binding.rank_id}": binding for binding in plan.rank_placement.bindings}
    events: list[AutopsyEvent] = []
    record_index = 0
    for outcome in output.operations:
        request_operation = operations_by_id[outcome.operation_id]
        event_type = _event_type(outcome.operation_id, outcome.resource_ids)
        ranks: tuple[str | None, ...] = outcome.rank_ids or (None,)
        for rank_label in ranks:
            binding = bindings.get(rank_label or "")
            rank = int(rank_label.removeprefix("rank-")) if rank_label is not None else None
            host = (
                binding.host_id if binding is not None else plan.rank_placement.bindings[0].host_id
            )
            indexed_evidence = evidence.model_copy(update={"record_index": record_index})
            events.append(
                AutopsyEvent(
                    event_id=f"{run_id}:{outcome.operation_id}:rank-{rank if rank is not None else 'all'}",
                    event_type=event_type,
                    host=host,
                    rank=rank,
                    replica=binding.replica_id if binding is not None else None,
                    gpu=binding.gpu_id if binding is not None else None,
                    nic=binding.nic_id if binding is not None else None,
                    numa_domain=binding.numa_domain_id if binding is not None else None,
                    request_id=request_operation.request_id,
                    operation=_operation_name(outcome.operation_id),
                    start_ns=round(outcome.start_us * 1_000.0),
                    end_ns=round(outcome.end_us * 1_000.0),
                    source_clock=SourceClock.SYNTHETIC,
                    normalized_start_ns=round(outcome.start_us * 1_000.0),
                    normalized_end_ns=round(outcome.end_us * 1_000.0),
                    alignment_confidence=1.0,
                    alignment_uncertainty_ns=0,
                    dependency_event_ids=(),
                    resource=_resource(outcome),
                    counters=_counters(outcome, event_type),
                    evidence=indexed_evidence,
                )
            )
            record_index += 1
    hosts = sorted({binding.host_id for binding in plan.rank_placement.bindings})
    alignments = tuple(
        AlignmentEstimate(
            host=host,
            reference_host=hosts[0],
            offset_ns=0.0,
            drift_ppm=0.0,
            reference_local_ns=0,
            uncertainty_ns=0,
            sample_count=1,
            confidence=1.0,
            quality=AlignmentQuality.GOOD,
            residual_p95_ns=0,
        )
        for host in hosts
    )
    fault_intervals = tuple(
        FaultInterval(
            fault_id=fault.id,
            fault_type=fault.ground_truth_label,
            target=(
                getattr(fault.effect, "resource_id", None)
                or getattr(fault.effect, "rank_id", None)
                or getattr(fault.effect, "collective_id", None)
                or "unknown"
            ),
            start_ns=round(fault.start_us * 1_000.0),
            end_ns=round(
                (fault.end_us if fault.end_us is not None else output.metrics.makespan_us) * 1_000.0
            ),
            parameters=(
                CounterValue(
                    name="multiplier",
                    value=float(getattr(fault.effect, "multiplier", 0.0)),
                    unit="ratio",
                ),
            )
            if hasattr(fault.effect, "multiplier")
            else (),
        )
        for fault in request.faults
    )
    request_hash = hashlib.sha256(
        canonical_json(request.model_dump(mode="json")).encode()
    ).hexdigest()
    return AutopsyRun(
        run_id=run_id,
        source="simulator",
        topology_fingerprint=topology_fingerprint,
        physical_plan_hash=canonical_hash(plan),
        workload_fingerprint=workload_fingerprint,
        reference_host=hosts[0],
        events=tuple(events),
        alignments=alignments,
        fault_intervals=fault_intervals,
        artifacts=(evidence,),
        warnings=(f"simulation_request_sha256={request_hash}",),
    )
