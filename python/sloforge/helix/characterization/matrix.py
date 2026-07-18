"""Strict, bounded, restartable characterization experiment matrices.

Matrix dimensions are controlled sweeps. They are never interpreted as an
empirical production distribution; a later trace corpus supplies distributional
weights when those weights are actually observed.
"""

from __future__ import annotations

import hashlib
import json
import random
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_.-]{0,127}$")]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)]
MAX_MATRIX_BYTES = 4 * 1024 * 1024
MAX_EXPANDED_CASES = 100_000


class MatrixModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class EvidenceClass(StrEnum):
    SYNTHETIC = "SYNTHETIC"
    REPLAYED = "REPLAYED"
    HARDWARE_BACKED_REAL = "HARDWARE_BACKED_REAL"
    SIMULATED_HARDWARE = "SIMULATED_HARDWARE"


class TraceLevel(StrEnum):
    DISABLED = "disabled"
    MINIMAL = "minimal"
    FULL = "full"


class WorkloadClass(StrEnum):
    CODING_AGENT = "coding_agent"
    PURE_REASONING = "pure_reasoning"
    VERIFICATION_HEAVY = "verification_heavy"
    CAPACITY_RECLAMATION = "capacity_reclamation"
    HIGHLY_DIVERGENT_NEGATIVE = "highly_divergent_negative"


class DivergencePattern(StrEnum):
    IMMEDIATE = "immediate"
    EARLY = "early"
    MID = "mid"
    LATE = "late"
    HIGHLY_SHARED = "highly_shared"
    HIGHLY_DIVERGENT = "highly_divergent"


class BranchSurvival(StrEnum):
    ALL_COMPLETE = "all_complete"
    AGGRESSIVE_EARLY_PRUNING = "aggressive_early_pruning"
    REWARD_PRUNING = "reward_pruning"
    RANDOM_DEATH = "random_death"
    LONG_TAIL = "long_tail"


class EnvironmentSize(StrEnum):
    NONE = "none"
    TINY = "tiny"
    MEDIUM = "medium"
    LARGE = "large"
    DEPENDENCY_HEAVY = "dependency_heavy"
    SERVICE_HEAVY = "service_heavy"


class ToolBehavior(StrEnum):
    NONE = "none"
    FREQUENT_CHEAP = "frequent_cheap"
    INFREQUENT_EXPENSIVE = "infrequent_expensive"
    BUILD_HEAVY = "build_heavy"
    TEST_HEAVY = "test_heavy"
    SERVICE_HEAVY = "service_heavy"


class PolicyMode(StrEnum):
    SAME_POLICY = "same_policy"
    CROSS_POLICY = "cross_policy"
    STRICT = "strict"
    SEGMENTED = "segmented"


class ResourcePressure(StrEnum):
    IDLE = "idle"
    MODERATE_SERVING = "moderate_serving"
    HEAVY_SERVING = "heavy_serving"
    ROLLOUT_SATURATION = "rollout_saturation"
    CPU_ENVIRONMENT_SATURATION = "cpu_environment_saturation"
    NETWORK_CONTENTION = "network_contention"
    STORAGE_CONTENTION = "storage_contention"


class MigrationBehavior(StrEnum):
    NONE = "none"
    BRANCH_MIGRATION = "branch_migration"
    CAPACITY_RECLAMATION = "capacity_reclamation"
    CHECKPOINT_RESUME = "checkpoint_resume"
    WORKER_FAILURE_RECOVERY = "worker_failure_recovery"


class FaultScenario(StrEnum):
    NONE = "none"
    BRANCH_ABORT = "branch_abort"
    WORKER_FAILURE = "worker_failure"
    TRANSFER_RETRY = "transfer_retry"
    STORAGE_SLOWDOWN = "storage_slowdown"


class ExperimentSpec(MatrixModel):
    workload_class: WorkloadClass
    branch_fanout: int = Field(ge=1, le=4096)
    prefix_tokens: int = Field(ge=1, le=2**24)
    suffix_tokens: int = Field(ge=1, le=2**20)
    divergence_pattern: DivergencePattern
    branch_survival: BranchSurvival
    environment_size: EnvironmentSize
    tool_behavior: ToolBehavior
    policy_mode: PolicyMode
    resource_pressure: ResourcePressure
    migration_behavior: MigrationBehavior
    page_size_bytes: int = Field(ge=4096, le=2**30)
    tensor_parallel: int = Field(ge=1, le=1024)
    state_packing: Identifier
    runtime: Identifier
    device_placement: Identifier
    memory_tier: Identifier
    transport: Identifier
    fault_scenario: FaultScenario
    trace_level: TraceLevel
    evidence_class: EvidenceClass
    hardware: Identifier
    simulated_gpu_state: bool
    timeout_seconds: int = Field(ge=1, le=86_400)
    estimated_cost_usd: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def honest_hardware_classification(self) -> Self:
        if self.simulated_gpu_state and self.evidence_class is EvidenceClass.HARDWARE_BACKED_REAL:
            raise ValueError("simulated GPU state cannot be hardware-backed real evidence")
        if self.hardware == "cpu-local" and self.evidence_class is EvidenceClass.SIMULATED_HARDWARE:
            raise ValueError("a local CPU execution is not simulated hardware")
        return self


class MatrixSweep(MatrixModel):
    sweep_id: Identifier
    field: NonEmpty
    values: Annotated[tuple[Any, ...], Field(min_length=1, max_length=4096)]
    overrides: dict[str, Any] = Field(default_factory=dict)
    seeds: Annotated[tuple[int, ...] | None, Field(default=None, min_length=1, max_length=128)]
    repetitions: int | None = Field(default=None, ge=1, le=10_000)

    @model_validator(mode="after")
    def sweep_field_exists(self) -> Self:
        if self.field not in ExperimentSpec.model_fields:
            raise ValueError(f"unknown experiment field {self.field!r}")
        unknown = set(self.overrides) - set(ExperimentSpec.model_fields)
        if unknown:
            raise ValueError(f"unknown experiment override fields: {sorted(unknown)}")
        return self


class CharacterizationMatrix(MatrixModel):
    schema_version: Literal["sloforge.branchfabric.characterization-matrix/v1"]
    matrix_id: Identifier
    purpose: NonEmpty
    distribution_claim: Literal[False]
    run_order_seed: int = Field(ge=0, le=2**64 - 1)
    seeds: Annotated[tuple[int, ...], Field(min_length=1, max_length=128)]
    repetitions: int = Field(ge=1, le=10_000)
    base: ExperimentSpec
    sweeps: Annotated[tuple[MatrixSweep, ...], Field(min_length=1, max_length=4096)]

    @model_validator(mode="after")
    def unique_sweeps(self) -> Self:
        identifiers = [item.sweep_id for item in self.sweeps]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("sweep identifiers must be unique")
        return self


class ExperimentCase(MatrixModel):
    experiment_id: Identifier
    matrix_id: Identifier
    sweep_id: Identifier
    value_ordinal: int = Field(ge=0)
    repetition: int = Field(ge=0)
    seed: int = Field(ge=0, le=2**64 - 1)
    spec: ExperimentSpec
    expected_artifact_path: NonEmpty


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_matrix(path: Path) -> CharacterizationMatrix:
    with path.open("rb") as handle:
        payload = handle.read(MAX_MATRIX_BYTES + 1)
    if len(payload) > MAX_MATRIX_BYTES:
        raise ValueError("characterization matrix exceeds 4 MiB")
    document = yaml.safe_load(payload)
    if not isinstance(document, dict):
        raise ValueError("characterization matrix must be a mapping")
    return CharacterizationMatrix.model_validate(document)


def expand_matrix(matrix: CharacterizationMatrix) -> tuple[ExperimentCase, ...]:
    cases: list[ExperimentCase] = []
    base = matrix.base.model_dump(mode="python")
    for sweep in matrix.sweeps:
        seeds = sweep.seeds or matrix.seeds
        repetitions = sweep.repetitions or matrix.repetitions
        for value_ordinal, value in enumerate(sweep.values):
            payload = {**base, **sweep.overrides, sweep.field: value}
            spec = ExperimentSpec.model_validate(payload)
            for seed in seeds:
                for repetition in range(repetitions):
                    identity = {
                        "matrix_id": matrix.matrix_id,
                        "sweep_id": sweep.sweep_id,
                        "value_ordinal": value_ordinal,
                        "seed": seed,
                        "repetition": repetition,
                        "spec": spec.model_dump(mode="json"),
                    }
                    digest = hashlib.sha256(_canonical(identity)).hexdigest()[:16]
                    experiment_id = f"{sweep.sweep_id}.{value_ordinal:03d}.{digest}"
                    cases.append(
                        ExperimentCase(
                            experiment_id=experiment_id,
                            matrix_id=matrix.matrix_id,
                            sweep_id=sweep.sweep_id,
                            value_ordinal=value_ordinal,
                            repetition=repetition,
                            seed=seed,
                            spec=spec,
                            expected_artifact_path=(
                                "artifacts/branchfabric/characterization/{run_id}/experiments/"
                                f"{experiment_id}"
                            ),
                        )
                    )
                    if len(cases) > MAX_EXPANDED_CASES:
                        raise ValueError("characterization matrix expands beyond 100000 cases")
    random.Random(matrix.run_order_seed).shuffle(cases)
    return tuple(cases)


__all__ = [
    "MAX_EXPANDED_CASES",
    "MAX_MATRIX_BYTES",
    "CharacterizationMatrix",
    "EvidenceClass",
    "ExperimentCase",
    "ExperimentSpec",
    "TraceLevel",
    "expand_matrix",
    "load_matrix",
]
