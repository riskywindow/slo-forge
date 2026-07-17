"""Typed branch strategies and proof-carrying state reuse decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.continuum.adapters import CapturedState
from sloforge.continuum.compatibility import CompatibilityRequest
from sloforge.continuum.operations import CheckpointArtifact
from sloforge.continuum.reference.runtime import DeterministicHybridRuntimeAdapter
from sloforge.continuum.transaction import SessionLease
from sloforge.helix.capture.models import canonical_digest

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,255}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class BranchStrategy(StrEnum):
    EXACT_COW = "exact_cow"
    CONTROLLED_RNG_MUTATION = "controlled_rng_mutation"
    COMPATIBLE_CROSS_POLICY = "compatible_cross_policy"
    RECOMPUTE_FROM_HISTORY = "recompute_from_history"


@dataclass(frozen=True, slots=True)
class ExactCowBranch:
    branch_id: str
    policy_epoch_id: str
    lease: SessionLease
    strategy: BranchStrategy = BranchStrategy.EXACT_COW


@dataclass(frozen=True, slots=True)
class RngMutationBranch:
    branch_id: str
    policy_epoch_id: str
    lease: SessionLease
    seed: int
    strategy: BranchStrategy = BranchStrategy.CONTROLLED_RNG_MUTATION

    def __post_init__(self) -> None:
        if not 0 <= self.seed < 2**64:
            raise ValueError("branch RNG seed must be an unsigned 64-bit integer")


@dataclass(frozen=True, slots=True)
class CrossPolicyBranch:
    branch_id: str
    policy_epoch_id: str
    lease: SessionLease
    compatibility: CompatibilityRequest
    destination: DeterministicHybridRuntimeAdapter | None = None
    permit_recomputation: bool = False
    strategy: BranchStrategy = BranchStrategy.COMPATIBLE_CROSS_POLICY


BranchPlan = ExactCowBranch | RngMutationBranch | CrossPolicyBranch


class BranchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class RngActivationOverride(BranchModel):
    algorithm: Literal["continuum-counter-v1"] = "continuum-counter-v1"
    source_seed: int = Field(ge=0, le=2**64 - 1)
    source_counter: int = Field(ge=0)
    branch_seed: int = Field(ge=0, le=2**64 - 1)
    reset_counter_to: int = Field(ge=0)
    apply_before_first_branch_token: Literal[True] = True
    source_sampler_reused: Literal[False] = False

    @model_validator(mode="after")
    def reject_noop(self) -> Self:
        if self.source_seed == self.branch_seed and self.source_counter == self.reset_counter_to:
            raise ValueError("RNG override must mutate the sampler state")
        return self


class StateReuseReport(BranchModel):
    schema_version: Literal["sloforge.helix.state-reuse-report/v1"] = (
        "sloforge.helix.state-reuse-report/v1"
    )
    report_id: Digest
    branch_id: Identifier
    source_capsule_id: Digest
    source_policy_epoch_id: Identifier
    destination_policy_epoch_id: Identifier
    strategy: BranchStrategy
    compatibility_class: str
    source_components: tuple[str, ...]
    directly_reused_components: tuple[str, ...]
    recomputed_components: tuple[str, ...]
    replaced_components: tuple[str, ...]
    unsupported_components: tuple[str, ...]
    verification_obligations: tuple[str, ...]
    source_state_exact: bool
    transcript_equivalence_claimed: Literal[False] = False
    recomputation_evidence_digest: Digest | None = None
    report_digest: Digest

    @model_validator(mode="after")
    def verify_report(self) -> Self:
        partitions = (
            set(self.directly_reused_components),
            set(self.recomputed_components),
            set(self.replaced_components),
            set(self.unsupported_components),
        )
        component_lists = (
            self.source_components,
            self.directly_reused_components,
            self.recomputed_components,
            self.replaced_components,
            self.unsupported_components,
        )
        if any(len(items) != len(set(items)) for items in component_lists):
            raise ValueError("state reuse component identifiers must be unique")
        for index, left in enumerate(partitions):
            if any(left & right for right in partitions[index + 1 :]):
                raise ValueError("state reuse component dispositions overlap")
        if set().union(*partitions) != set(self.source_components):
            raise ValueError("state reuse dispositions must cover every source component")
        if self.unsupported_components:
            raise ValueError("a successful StateReuseReport cannot contain unsupported state")
        if self.source_state_exact and (self.recomputed_components or self.replaced_components):
            raise ValueError("exact source-state reuse cannot recompute or replace components")
        if bool(self.recomputed_components) != (self.recomputation_evidence_digest is not None):
            raise ValueError("recomputed state requires exactly one evidence digest")
        if (self.strategy is BranchStrategy.RECOMPUTE_FROM_HISTORY) != bool(
            self.recomputed_components
        ):
            raise ValueError("recomputation strategy and component dispositions disagree")
        payload = self.model_dump(mode="json", exclude={"report_id", "report_digest"})
        if canonical_digest(payload) != self.report_digest:
            raise ValueError("state reuse report digest is invalid")
        identity = self.model_dump(mode="json", exclude={"report_id"})
        if canonical_digest(identity) != self.report_id:
            raise ValueError("state reuse report identifier is invalid")
        return self


@dataclass(frozen=True, slots=True)
class BranchMember:
    branch_id: str
    policy_epoch_id: str
    checkpoint: CheckpointArtifact
    state_reuse: StateReuseReport
    recomputed: CapturedState | None = None
    rng_override: RngActivationOverride | None = None
    environment_branch: object | None = None


@dataclass(frozen=True, slots=True)
class BranchGroupExecution:
    group_id: str
    branch_point_id: str
    source_capsule_id: str
    members: tuple[BranchMember, ...]
    shared_immutable_digests: tuple[str, ...]
    seed: int
    environment_base_capsule_id: str | None = None


class InterventionKind(StrEnum):
    POLICY = "policy"
    RNG = "rng"
    MODEL = "model"
    ENVIRONMENT = "environment"
    TOOL = "tool"


class BranchIntervention(BranchModel):
    intervention_id: Identifier
    kind: InterventionKind
    target: Identifier
    value_digest: Digest


class BranchMinimizationResult(BranchModel):
    original_intervention_ids: tuple[Identifier, ...]
    minimal_intervention_ids: tuple[Identifier, ...]
    evaluations: int = Field(ge=1)
    search_complete: bool
    witness_digest: Digest

    @model_validator(mode="after")
    def verify_witness(self) -> Self:
        if len(self.original_intervention_ids) != len(set(self.original_intervention_ids)):
            raise ValueError("original intervention identifiers must be unique")
        if len(self.minimal_intervention_ids) != len(set(self.minimal_intervention_ids)):
            raise ValueError("minimal intervention identifiers must be unique")
        if not set(self.minimal_intervention_ids) <= set(self.original_intervention_ids):
            raise ValueError("minimal interventions must be a subset of the original set")
        payload = {
            "schema": "sloforge.helix.branch-minimization/v1",
            "original": self.original_intervention_ids,
            "minimal": self.minimal_intervention_ids,
            "evaluations": self.evaluations,
            "search_complete": self.search_complete,
        }
        if canonical_digest(payload) != self.witness_digest:
            raise ValueError("branch minimization witness digest is invalid")
        return self
