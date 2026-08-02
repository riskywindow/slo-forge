"""Safe, typed physical-fabric fault scenarios for deterministic simulation."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from yaml.tokens import AliasToken, AnchorToken

from sloforge.fabric.simulation import (
    Collective,
    CollectiveDelayFault,
    FabricSimulationRequest,
    FaultEffect,
    RankSlowdownFault,
    ResourceRateFault,
    ResourceUnavailableFault,
    TimedFault,
)


class PhysicalFaultType(StrEnum):
    GPU_PROCESS_CRASH = "gpu_process_crash"
    WORKER_CRASH = "worker_crash"
    GPU_CLOCK_THROTTLE = "gpu_clock_throttle"
    GPU_COMPUTE_SLOWDOWN = "gpu_compute_slowdown"
    GPU_MEMORY_BANDWIDTH_SLOWDOWN = "gpu_memory_bandwidth_slowdown"
    PCIE_BANDWIDTH_DEGRADATION = "pcie_bandwidth_degradation"
    NVLINK_BANDWIDTH_DEGRADATION = "nvlink_bandwidth_degradation"
    NIC_BANDWIDTH_DEGRADATION = "nic_bandwidth_degradation"
    NETWORK_LATENCY_INCREASE = "network_latency_increase"
    NETWORK_JITTER_INCREASE = "network_jitter_increase"
    PACKET_LOSS = "packet_loss"
    NETWORK_RAIL_LOSS = "network_rail_loss"
    NETWORK_BANDWIDTH_DEGRADATION = "network_bandwidth_degradation"
    RANK_SPECIFIC_SLOWDOWN = "rank_specific_slowdown"
    RANK_SPECIFIC_GPU_SLOWDOWN = "rank_specific_gpu_slowdown"
    COLLECTIVE_DELAY = "collective_delay"
    COLLECTIVE_FAILURE = "collective_failure"
    NCCL_INITIALIZATION_DELAY = "nccl_initialization_delay"
    EXPERT_LOAD_SKEW = "expert_load_skew"
    HOT_EXPERT_CONCENTRATION = "hot_expert_concentration"
    PREFILL_WORKER_LOSS = "prefill_worker_loss"
    DECODE_WORKER_LOSS = "decode_worker_loss"
    KV_TRANSFER_SLOWDOWN = "kv_transfer_slowdown"
    STARTUP_SPIKE = "startup_spike"
    NUMA_MISPLACEMENT = "numa_misplacement"
    CPU_SCHEDULING_DELAY = "cpu_scheduling_delay"
    CONTAINER_CPU_THROTTLING = "container_cpu_throttling"
    STORAGE_READ_SLOWDOWN = "storage_read_slowdown"
    SIMULATED_OOM = "simulated_oom"
    MEMORY_FRAGMENTATION_REJECTION = "memory_fragmentation_rejection"


_RANK_FAULTS = {
    PhysicalFaultType.RANK_SPECIFIC_SLOWDOWN,
    PhysicalFaultType.RANK_SPECIFIC_GPU_SLOWDOWN,
    PhysicalFaultType.EXPERT_LOAD_SKEW,
    PhysicalFaultType.HOT_EXPERT_CONCENTRATION,
}
_COLLECTIVE_FAULTS = {
    PhysicalFaultType.COLLECTIVE_DELAY,
    PhysicalFaultType.NCCL_INITIALIZATION_DELAY,
}
_UNAVAILABLE_FAULTS = {
    PhysicalFaultType.GPU_PROCESS_CRASH,
    PhysicalFaultType.WORKER_CRASH,
    PhysicalFaultType.NETWORK_RAIL_LOSS,
    PhysicalFaultType.COLLECTIVE_FAILURE,
    PhysicalFaultType.PREFILL_WORKER_LOSS,
    PhysicalFaultType.DECODE_WORKER_LOSS,
    PhysicalFaultType.SIMULATED_OOM,
    PhysicalFaultType.MEMORY_FRAGMENTATION_REJECTION,
}


class _FaultModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class PhysicalFaultSpec(_FaultModel):
    fault_id: str = Field(min_length=1)
    fault_type: PhysicalFaultType
    target_id: str = Field(min_length=1)
    start_us: Annotated[float, Field(ge=0.0)]
    end_us: Annotated[float, Field(gt=0.0)] | None = None
    degradation_multiplier: Annotated[float, Field(gt=0.0, le=1.0)] | None = None

    @model_validator(mode="after")
    def validate_effect(self) -> Self:
        if self.end_us is not None and self.end_us <= self.start_us:
            raise ValueError("fault end must be after start")
        needs_multiplier = self.fault_type not in _UNAVAILABLE_FAULTS
        if needs_multiplier != (self.degradation_multiplier is not None):
            requirement = "requires" if needs_multiplier else "forbids"
            raise ValueError(f"{self.fault_type.value} {requirement} degradation_multiplier")
        return self


class PhysicalFaultScenario(_FaultModel):
    schema_version: Literal["sloforge.fabric.faults/v1"]
    scenario_id: str = Field(min_length=1)
    execution_mode: Literal["simulation"] = "simulation"
    faults: tuple[PhysicalFaultSpec, ...]

    @model_validator(mode="after")
    def validate_faults(self) -> Self:
        if not self.faults:
            raise ValueError("physical fault scenario cannot be empty")
        identifiers = [fault.fault_id for fault in self.faults]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("physical fault identifiers must be unique")
        return self


MAX_PHYSICAL_FAULT_SCENARIO_BYTES = 4 * 1024 * 1024


def load_physical_fault_scenario(path: Path) -> PhysicalFaultScenario:
    with path.open("rb") as handle:
        raw = handle.read(MAX_PHYSICAL_FAULT_SCENARIO_BYTES + 1)
    if len(raw) > MAX_PHYSICAL_FAULT_SCENARIO_BYTES:
        raise ValueError("physical fault scenario exceeds 4 MiB")
    text = raw.decode("utf-8")
    if any(isinstance(token, (AliasToken, AnchorToken)) for token in yaml.scan(text)):
        raise ValueError("physical fault scenarios do not permit YAML anchors or aliases")
    payload = yaml.safe_load(text)
    # JSON-mode validation retains strict scalar checks while translating YAML
    # sequences into the immutable tuples required by the canonical model.
    return PhysicalFaultScenario.model_validate_json(json.dumps(payload))


def bind_physical_faults(
    scenario: PhysicalFaultScenario, request: FabricSimulationRequest
) -> tuple[TimedFault, ...]:
    """Bind a simulation-only scenario and reject every unknown target."""

    resources = {resource.id for resource in request.resources}
    ranks = {rank for operation in request.operations for rank in operation.rank_ids}
    collectives = {
        operation.kind.collective_id
        for operation in request.operations
        if isinstance(operation.kind, Collective)
    }
    result: list[TimedFault] = []
    for fault in scenario.faults:
        effect: FaultEffect
        if fault.fault_type in _RANK_FAULTS:
            if fault.target_id not in ranks:
                raise ValueError(f"fault {fault.fault_id} references unknown rank")
            assert fault.degradation_multiplier is not None
            effect = RankSlowdownFault(
                rank_id=fault.target_id,
                multiplier=fault.degradation_multiplier,
            )
        elif fault.fault_type in _COLLECTIVE_FAULTS:
            if fault.target_id not in collectives:
                raise ValueError(f"fault {fault.fault_id} references unknown collective")
            assert fault.degradation_multiplier is not None
            effect = CollectiveDelayFault(
                collective_id=fault.target_id,
                multiplier=fault.degradation_multiplier,
            )
        elif fault.fault_type in _UNAVAILABLE_FAULTS:
            if fault.target_id not in resources:
                raise ValueError(f"fault {fault.fault_id} references unknown resource")
            effect = ResourceUnavailableFault(resource_id=fault.target_id)
        else:
            if fault.target_id not in resources:
                raise ValueError(f"fault {fault.fault_id} references unknown resource")
            assert fault.degradation_multiplier is not None
            effect = ResourceRateFault(
                resource_id=fault.target_id,
                multiplier=fault.degradation_multiplier,
            )
        result.append(
            TimedFault(
                id=fault.fault_id,
                start_us=fault.start_us,
                end_us=fault.end_us,
                effect=effect,
                ground_truth_label=fault.fault_type.value,
            )
        )
    return tuple(result)
