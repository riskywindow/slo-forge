"""Conservative champion/challenger and rollback resource bounds."""

from __future__ import annotations

import math

from .model import (
    EvidenceStatus,
    ResourceContract,
    ResourceDemand,
    ResourceEvidence,
    VerificationError,
)


def analyze_resources(contract: ResourceContract, demand: ResourceDemand) -> ResourceEvidence:
    if contract.device_capacity_bytes <= 0 or contract.host_capacity_bytes <= 0:
        raise VerificationError("resource capacities must be positive")
    if not 0 <= contract.safety_margin_fraction < 1:
        raise VerificationError("safety margin must be in [0, 1)")
    if not 0 <= contract.fragmentation_fraction < 1:
        raise VerificationError("fragmentation allowance must be in [0, 1)")
    numeric_demand = (
        demand.model_device_bytes,
        demand.state_device_bytes,
        demand.queue_device_bytes,
        demand.communication_device_bytes,
        demand.workspace_device_bytes,
        demand.challenger_device_bytes,
        demand.host_bytes,
        demand.conversion_overlap_bytes,
        demand.processes,
        demand.threads,
        demand.file_descriptors,
    )
    if any(value < 0 for value in numeric_demand):
        raise VerificationError("resource demands cannot be negative")
    raw_device = sum(numeric_demand[:6]) + demand.conversion_overlap_bytes
    device_peak = math.ceil(raw_device / (1 - contract.fragmentation_fraction))
    host_peak = demand.host_bytes + demand.conversion_overlap_bytes
    usable_device = math.floor(
        contract.device_capacity_bytes * (1 - contract.safety_margin_fraction)
    )
    usable_host = math.floor(contract.host_capacity_bytes * (1 - contract.safety_margin_fraction))
    violations: list[str] = []
    if device_peak > usable_device:
        violations.append("peak_device_memory")
    if host_peak > usable_host:
        violations.append("peak_host_memory")
    if demand.processes > contract.maximum_processes:
        violations.append("process_bound")
    if demand.threads > contract.maximum_threads:
        violations.append("thread_bound")
    if demand.file_descriptors > contract.maximum_file_descriptors:
        violations.append("file_descriptor_bound")
    return ResourceEvidence(
        status=EvidenceStatus.FAILED if violations else EvidenceStatus.PASSED,
        conservative_peak_device_bytes=device_peak,
        usable_device_bytes=usable_device,
        conservative_peak_host_bytes=host_peak,
        usable_host_bytes=usable_host,
        violations=tuple(violations),
        assumptions=(
            "champion_and_challenger_coexist",
            "conversion_buffer_overlaps",
            "fragmentation_applied_to_device_peak",
        ),
    )
