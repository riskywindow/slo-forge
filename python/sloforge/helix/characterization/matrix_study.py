"""Execute every controlled matrix cell through the calibrated state model.

This is deliberately an analytical synthetic execution, not a latency benchmark.
It closes the gap between validating a matrix and actually evaluating every cell
while preserving the distinction from hardware-backed workload measurements.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sloforge.helix.characterization.matrix import (
    ExperimentSpec,
    expand_matrix,
    load_matrix,
)
from sloforge.helix.characterization.projection import (
    StateCalibration,
    calibrate_state,
    project_attention_cow,
    project_state_composition,
)

MAX_OUTPUT_BYTES = 256 * 1024 * 1024


class MatrixStudyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class MatrixCellResult(MatrixStudyModel):
    experiment_id: str = Field(min_length=1, max_length=256)
    sweep_id: str = Field(min_length=1, max_length=128)
    value_ordinal: int = Field(ge=0)
    repetition: int = Field(ge=0)
    seed: int = Field(ge=0, le=2**64 - 1)
    spec: ExperimentSpec
    execution_class: Literal["ANALYTICAL_SYNTHETIC_STATE_MODEL"] = (
        "ANALYTICAL_SYNTHETIC_STATE_MODEL"
    )
    timing_measurement_class: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    distribution_claim: Literal[False] = False
    shared_root_logical_bytes: int = Field(ge=0)
    shared_root_physical_bytes: int = Field(ge=0)
    private_suffix_logical_bytes_per_branch: int = Field(ge=0)
    private_suffix_physical_bytes_per_branch: int = Field(ge=0)
    logical_unique_bytes: int = Field(gt=0)
    physical_allocated_bytes: int = Field(gt=0)
    naive_independent_allocation_bytes: int = Field(gt=0)
    cow_faults_projected: int = Field(ge=0)
    cow_copied_bytes_projected: int = Field(ge=0)
    internal_fragmentation_bytes: int = Field(ge=0)
    physical_amplification: float = Field(gt=0, allow_inf_nan=False)
    sharing_efficiency: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    projected_attention_kv_bytes: int = Field(ge=0)


class ControlledMatrixStudy(MatrixStudyModel):
    schema_version: Literal["sloforge.branchfabric.controlled-matrix-study/v1"] = (
        "sloforge.branchfabric.controlled-matrix-study/v1"
    )
    matrix_id: str = Field(min_length=1, max_length=128)
    matrix_reference: str = Field(min_length=1, max_length=4096)
    matrix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration: StateCalibration
    case_count: int = Field(gt=0, le=100_000)
    evaluated_case_count: int = Field(gt=0, le=100_000)
    hardware_executed_case_count: Literal[0] = 0
    distribution_claim: Literal[False] = False
    cases: tuple[MatrixCellResult, ...] = Field(min_length=1, max_length=100_000)
    limitations: tuple[str, ...] = Field(min_length=1)


def evaluate_controlled_matrix(
    matrix_path: Path,
    *,
    calibration_artifact: Path,
) -> ControlledMatrixStudy:
    """Evaluate every expanded cell without inventing performance measurements."""

    matrix = load_matrix(matrix_path)
    calibration = calibrate_state(calibration_artifact)
    cases = expand_matrix(matrix)
    evaluated: list[MatrixCellResult] = []
    for case in cases:
        projection = project_attention_cow(
            calibration,
            branch_fanout=case.spec.branch_fanout,
            prefix_tokens=case.spec.prefix_tokens,
            suffix_tokens=case.spec.suffix_tokens,
            divergence_pattern=case.spec.divergence_pattern,
            page_size_bytes=case.spec.page_size_bytes,
        )
        composition = project_state_composition(
            calibration,
            tokens=case.spec.prefix_tokens,
        )
        evaluated.append(
            MatrixCellResult(
                experiment_id=case.experiment_id,
                sweep_id=case.sweep_id,
                value_ordinal=case.value_ordinal,
                repetition=case.repetition,
                seed=case.seed,
                spec=case.spec,
                shared_root_logical_bytes=projection.shared_root_logical_bytes,
                shared_root_physical_bytes=projection.shared_root_physical_bytes,
                private_suffix_logical_bytes_per_branch=(
                    projection.private_suffix_logical_bytes_per_branch
                ),
                private_suffix_physical_bytes_per_branch=(
                    projection.private_suffix_physical_bytes_per_branch
                ),
                logical_unique_bytes=projection.logical_unique_bytes,
                physical_allocated_bytes=projection.physical_allocated_bytes,
                naive_independent_allocation_bytes=(projection.naive_independent_allocation_bytes),
                cow_faults_projected=projection.cow_faults,
                cow_copied_bytes_projected=projection.cow_copied_bytes,
                internal_fragmentation_bytes=projection.internal_fragmentation_bytes,
                physical_amplification=projection.physical_amplification,
                sharing_efficiency=projection.sharing_efficiency,
                projected_attention_kv_bytes=composition.attention_kv_bytes,
            )
        )
    return ControlledMatrixStudy(
        matrix_id=matrix.matrix_id,
        matrix_reference=matrix_path.as_posix(),
        matrix_sha256=hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        calibration=calibration,
        case_count=len(cases),
        evaluated_case_count=len(evaluated),
        cases=tuple(evaluated),
        limitations=(
            "Every matrix cell is evaluated by a calibrated deterministic state model; no cell is a runtime latency measurement.",
            "The calibration is an eight-token synthetic Continuum capsule with simulated KV bytes, not a production model distribution.",
            "Pressure, tool, fault, policy, transport, runtime, and environment fields remain declared controlled inputs unless separately exercised by a dedicated study.",
            "No CPU projection is evidence of GPU allocation, PCIe, NIC, HBM, CXL, or storage performance.",
        ),
    )


def write_controlled_matrix_study(study: ControlledMatrixStudy, output: Path) -> str:
    """Atomically persist a bounded matrix study and return its SHA-256."""

    payload = (study.model_dump_json(indent=2) + "\n").encode()
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ValueError(f"controlled matrix study exceeds {MAX_OUTPUT_BYTES} bytes")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ControlledMatrixStudy",
    "MatrixCellResult",
    "evaluate_controlled_matrix",
    "write_controlled_matrix_study",
]
