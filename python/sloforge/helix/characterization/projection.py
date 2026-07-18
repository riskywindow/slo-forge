"""Evidence-labelled state and copy-on-write projections.

These helpers turn a captured Continuum capsule into controlled synthetic
projections. They deliberately do not estimate a production distribution.
The calibration artifact, its hash, and every scaling assumption travel with
the result so projected bytes cannot be mistaken for hardware measurements.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.helix.characterization.matrix import DivergencePattern, EvidenceClass

MAX_CAPSULE_BYTES = 64 * 1024 * 1024


class ProjectionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ComponentCalibration(ProjectionModel):
    component_id: str = Field(min_length=1, max_length=512)
    observed_logical_bytes: int = Field(ge=0)


class StateCalibration(ProjectionModel):
    schema_version: Literal["sloforge.branchfabric.state-calibration/v1"]
    evidence_class: Literal[EvidenceClass.SYNTHETIC]
    analysis_method: Literal["captured_synthetic_fixture"]
    source_artifact: str = Field(min_length=1, max_length=4096)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capsule_id: str = Field(min_length=1, max_length=512)
    observed_tokens: int = Field(gt=0)
    components: tuple[ComponentCalibration, ...] = Field(min_length=1)
    attention_bytes_per_token: int = Field(gt=0)
    assumptions: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def components_are_unique(self) -> StateCalibration:
        identifiers = [item.component_id for item in self.components]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("calibration component identifiers must be unique")
        if "state/attention-kv" not in identifiers:
            raise ValueError("calibration requires state/attention-kv")
        return self

    def component_bytes(self, component_id: str) -> int:
        for component in self.components:
            if component.component_id == component_id:
                return component.observed_logical_bytes
        raise KeyError(component_id)


class StateCompositionProjection(ProjectionModel):
    schema_version: Literal["sloforge.branchfabric.state-composition-projection/v1"]
    evidence_class: Literal[EvidenceClass.SYNTHETIC]
    calibration_artifact: str
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokens: int = Field(gt=0)
    attention_kv_bytes: int = Field(ge=0)
    observed_fixed_component_bytes: dict[str, int]
    observed_token_history_bytes: int = Field(ge=0)
    total_without_token_history_extrapolation_bytes: int = Field(ge=0)
    assumptions: tuple[str, ...]


class CowProjection(ProjectionModel):
    schema_version: Literal["sloforge.branchfabric.cow-projection/v1"]
    evidence_class: Literal[EvidenceClass.SYNTHETIC]
    calibration_artifact: str
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_segment: Literal["KV"]
    branch_fanout: int = Field(ge=1)
    prefix_tokens: int = Field(ge=1)
    suffix_tokens: int = Field(ge=1)
    common_suffix_tokens: int = Field(ge=0)
    divergent_suffix_tokens: int = Field(ge=0)
    divergence_pattern: DivergencePattern
    page_size_bytes: int = Field(ge=4096)
    bytes_per_token: int = Field(gt=0)
    shared_root_logical_bytes: int = Field(ge=0)
    shared_root_physical_bytes: int = Field(ge=0)
    private_suffix_logical_bytes_per_branch: int = Field(ge=0)
    private_suffix_physical_bytes_per_branch: int = Field(ge=0)
    logical_unique_bytes: int = Field(gt=0)
    physical_allocated_bytes: int = Field(gt=0)
    naive_independent_allocation_bytes: int = Field(gt=0)
    cow_faults: int = Field(ge=0)
    cow_copied_bytes: int = Field(ge=0)
    duplicated_post_fork_bytes: int = Field(ge=0)
    internal_fragmentation_bytes: int = Field(ge=0)
    metadata_bytes: Literal[0]
    physical_amplification: float = Field(gt=0, allow_inf_nan=False)
    sharing_efficiency: float = Field(ge=-1, le=1, allow_inf_nan=False)
    assumptions: tuple[str, ...] = Field(min_length=1)


def _load_json(path: Path) -> tuple[bytes, dict[str, object]]:
    with path.open("rb") as handle:
        payload = handle.read(MAX_CAPSULE_BYTES + 1)
    if len(payload) > MAX_CAPSULE_BYTES:
        raise ValueError("Continuum capsule artifact exceeds 64 MiB")
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise ValueError("Continuum capsule artifact must be a JSON object")
    return payload, document


def calibrate_state(capsule_artifact: Path) -> StateCalibration:
    """Extract actual logical sizes from one deterministic synthetic capsule."""

    payload, document = _load_json(capsule_artifact)
    raw_capsule = document.get("capsule", document)
    if not isinstance(raw_capsule, dict):
        raise ValueError("capsule field must be an object")
    identity = raw_capsule.get("identity")
    logical_state = raw_capsule.get("logical_state")
    physical_state = raw_capsule.get("physical_state")
    if not all(isinstance(item, dict) for item in (identity, logical_state, physical_state)):
        raise ValueError("capsule is missing identity, logical_state, or physical_state")
    assert isinstance(identity, dict)
    assert isinstance(logical_state, dict)
    assert isinstance(physical_state, dict)

    token_history = logical_state.get("token_history")
    if not isinstance(token_history, dict):
        raise ValueError("capsule is missing token history")
    input_tokens = token_history.get("input_token_ids")
    output_tokens = token_history.get("committed_output_token_ids")
    if not isinstance(input_tokens, list) or not isinstance(output_tokens, list):
        raise ValueError("token history IDs must be arrays")
    observed_tokens = len(input_tokens) + len(output_tokens)
    if observed_tokens <= 0:
        raise ValueError("calibration capsule must contain at least one committed token")

    raw_sizes = physical_state.get("logical_component_sizes")
    if not isinstance(raw_sizes, list) or not raw_sizes:
        raise ValueError("capsule is missing logical component sizes")
    components: list[ComponentCalibration] = []
    for raw_size in raw_sizes:
        if not isinstance(raw_size, dict):
            raise ValueError("logical component size must be an object")
        logical_size_bytes = raw_size.get("logical_size_bytes")
        if not isinstance(logical_size_bytes, int) or logical_size_bytes < 0:
            raise ValueError("logical component size must be a non-negative integer")
        components.append(
            ComponentCalibration(
                component_id=str(raw_size.get("component_id", "")),
                observed_logical_bytes=logical_size_bytes,
            )
        )
    attention_bytes = next(
        (
            item.observed_logical_bytes
            for item in components
            if item.component_id == "state/attention-kv"
        ),
        None,
    )
    if attention_bytes is None or attention_bytes % observed_tokens:
        raise ValueError("attention bytes must be an integral multiple of observed tokens")
    capsule_id = identity.get("capsule_id")
    if not isinstance(capsule_id, str) or not capsule_id:
        raise ValueError("capsule identity is missing capsule_id")
    return StateCalibration(
        schema_version="sloforge.branchfabric.state-calibration/v1",
        evidence_class=EvidenceClass.SYNTHETIC,
        analysis_method="captured_synthetic_fixture",
        source_artifact=capsule_artifact.as_posix(),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        capsule_id=capsule_id,
        observed_tokens=observed_tokens,
        components=tuple(components),
        attention_bytes_per_token=attention_bytes // observed_tokens,
        assumptions=(
            "source is the deterministic Helix CPU fixture, not a production workload",
            "attention KV bytes per token are fixture-specific simulated state",
            "no CPU projection is evidence of GPU HBM allocation or transfer performance",
        ),
    )


def project_state_composition(
    calibration: StateCalibration, *, tokens: int
) -> StateCompositionProjection:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    token_history_bytes = calibration.component_bytes("state/token-history")
    fixed = {
        component.component_id: component.observed_logical_bytes
        for component in calibration.components
        if component.component_id not in {"state/attention-kv", "state/token-history"}
    }
    attention_bytes = calibration.attention_bytes_per_token * tokens
    return StateCompositionProjection(
        schema_version="sloforge.branchfabric.state-composition-projection/v1",
        evidence_class=EvidenceClass.SYNTHETIC,
        calibration_artifact=calibration.source_artifact,
        calibration_sha256=calibration.source_sha256,
        tokens=tokens,
        attention_kv_bytes=attention_bytes,
        observed_fixed_component_bytes=fixed,
        observed_token_history_bytes=token_history_bytes,
        total_without_token_history_extrapolation_bytes=attention_bytes + sum(fixed.values()),
        assumptions=(
            "attention KV scales linearly using the captured fixture layout",
            "fixed components retain their observed sizes",
            "token history is reported only at its observed size and excluded from the total",
            "result is a controlled synthetic projection, not an empirical distribution",
        ),
    )


def common_suffix_tokens(pattern: DivergencePattern, suffix_tokens: int) -> int:
    """Map a controlled divergence label to a declared common suffix length."""

    if suffix_tokens <= 0:
        raise ValueError("suffix_tokens must be positive")
    return {
        DivergencePattern.IMMEDIATE: 0,
        DivergencePattern.EARLY: suffix_tokens // 8,
        DivergencePattern.MID: suffix_tokens // 2,
        DivergencePattern.LATE: (suffix_tokens * 7) // 8,
        DivergencePattern.HIGHLY_SHARED: max(0, suffix_tokens - 1),
        DivergencePattern.HIGHLY_DIVERGENT: 0,
    }[pattern]


def _rounded_up(size_bytes: int, page_size_bytes: int) -> int:
    return math.ceil(size_bytes / page_size_bytes) * page_size_bytes


def project_attention_cow(
    calibration: StateCalibration,
    *,
    branch_fanout: int,
    prefix_tokens: int,
    suffix_tokens: int,
    divergence_pattern: DivergencePattern,
    page_size_bytes: int,
) -> CowProjection:
    """Project page allocation for append-only fixture KV under eager tail-page COW."""

    if branch_fanout <= 0 or prefix_tokens <= 0 or suffix_tokens <= 0:
        raise ValueError("fanout, prefix_tokens, and suffix_tokens must be positive")
    if page_size_bytes < 4096 or page_size_bytes & (page_size_bytes - 1):
        raise ValueError("page_size_bytes must be a power of two of at least 4096")

    bytes_per_token = calibration.attention_bytes_per_token
    root_logical = prefix_tokens * bytes_per_token
    suffix_logical = suffix_tokens * bytes_per_token
    root_physical = _rounded_up(root_logical, page_size_bytes)
    root_remainder = root_logical % page_size_bytes
    cow_faults = branch_fanout if root_remainder and branch_fanout > 1 else 0
    cow_copied_bytes = cow_faults * page_size_bytes
    if branch_fanout == 1:
        private_physical = _rounded_up(root_logical + suffix_logical, page_size_bytes) - (
            root_physical
        )
    elif root_remainder:
        tail_capacity = page_size_bytes - root_remainder
        additional = max(0, suffix_logical - tail_capacity)
        private_physical = page_size_bytes + (
            _rounded_up(additional, page_size_bytes) if additional else 0
        )
    else:
        private_physical = _rounded_up(suffix_logical, page_size_bytes)

    common_tokens = common_suffix_tokens(divergence_pattern, suffix_tokens)
    divergent_tokens = suffix_tokens - common_tokens
    common_bytes = common_tokens * bytes_per_token
    divergent_bytes = divergent_tokens * bytes_per_token
    logical_unique = root_logical + common_bytes + branch_fanout * divergent_bytes
    physical = root_physical + branch_fanout * private_physical
    naive = branch_fanout * _rounded_up(root_logical + suffix_logical, page_size_bytes)
    logical_written = root_logical + branch_fanout * suffix_logical
    return CowProjection(
        schema_version="sloforge.branchfabric.cow-projection/v1",
        evidence_class=EvidenceClass.SYNTHETIC,
        calibration_artifact=calibration.source_artifact,
        calibration_sha256=calibration.source_sha256,
        state_segment="KV",
        branch_fanout=branch_fanout,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        common_suffix_tokens=common_tokens,
        divergent_suffix_tokens=divergent_tokens,
        divergence_pattern=divergence_pattern,
        page_size_bytes=page_size_bytes,
        bytes_per_token=bytes_per_token,
        shared_root_logical_bytes=root_logical,
        shared_root_physical_bytes=root_physical,
        private_suffix_logical_bytes_per_branch=suffix_logical,
        private_suffix_physical_bytes_per_branch=private_physical,
        logical_unique_bytes=logical_unique,
        physical_allocated_bytes=physical,
        naive_independent_allocation_bytes=naive,
        cow_faults=cow_faults,
        cow_copied_bytes=cow_copied_bytes,
        duplicated_post_fork_bytes=(branch_fanout - 1) * common_bytes,
        internal_fragmentation_bytes=physical - logical_written,
        metadata_bytes=0,
        physical_amplification=physical / logical_unique,
        sharing_efficiency=1.0 - physical / naive,
        assumptions=(
            "only attention KV is modeled; environment and non-KV model state are separate",
            "the BranchPoint prefix is shared once and each branch appends its own suffix",
            "a write into a partial root page eagerly copies one full page per branch",
            "identical post-fork suffix bytes are not physically deduplicated",
            "page metadata is excluded until measured by the metadata study",
            "divergence labels are controlled patterns, not production probabilities",
        ),
    )


__all__ = [
    "CowProjection",
    "StateCalibration",
    "StateCompositionProjection",
    "calibrate_state",
    "common_suffix_tokens",
    "project_attention_cow",
    "project_state_composition",
]
