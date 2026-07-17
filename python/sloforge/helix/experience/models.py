"""Strict, hash-bound contracts for deterministic experience selection.

The selector consumes feature summaries and content-addressed evidence references.  It
does not ingest production payloads, infer consent, or authorize side effects.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
NonNegativeFinite = Annotated[float, Field(ge=0.0, le=10**15, allow_inf_nan=False)]


def canonical_digest(value: object) -> str:
    """Return the repository's deterministic digest for a JSON-compatible value."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ExperienceModel(BaseModel):
    """Fail-closed base class for the experience-selection JSON boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class EvidenceSource(StrEnum):
    SYNTHETIC = "synthetic"
    AUTHORIZED_PRODUCTION = "authorized_production"


class PrivacyClass(StrEnum):
    PUBLIC = "public"
    TENANT_PRIVATE = "tenant_private"
    RESTRICTED = "restricted"


class SideEffectRisk(StrEnum):
    PURE = "pure"
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    EXTERNAL = "external"
    IRREVERSIBLE = "irreversible"


class SelectionStrategy(StrEnum):
    RANDOM = "random"
    FAILURE_ONLY = "failure_only"
    UNCERTAINTY_ONLY = "uncertainty_only"
    NOVELTY_ONLY = "novelty_only"
    HELIX_VALUE_AWARE = "helix_value_aware"


# A convenient compatibility name for callers that refer to selection as a policy.
ExperienceSelectionPolicy = SelectionStrategy


class ExclusionReason(StrEnum):
    TENANT_MISMATCH = "tenant_mismatch"
    PRODUCTION_DISABLED = "production_disabled"
    CONSENT_REQUIRED = "consent_required"
    AUTHORIZATION_ARTIFACT_REQUIRED = "authorization_artifact_required"
    REDACTION_REQUIRED = "redaction_required"
    REDACTION_ARTIFACT_REQUIRED = "redaction_artifact_required"
    PRIVACY_NOT_ALLOWED = "privacy_not_allowed"
    SIDE_EFFECT_RISK_NOT_ALLOWED = "side_effect_risk_not_allowed"
    LIVE_SIDE_EFFECT_REQUIRED = "live_side_effect_required"
    BASELINE_FILTERED = "baseline_filtered"
    NON_POSITIVE_LEARNING_VALUE = "non_positive_learning_value"
    BELOW_MINIMUM_SCORE = "below_minimum_score"
    REDUNDANT = "redundant"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CAPACITY_EXHAUSTED = "capacity_exhausted"
    MAXIMUM_COUNT_REACHED = "maximum_count_reached"
    TRAIN_ALL_GUARD = "train_all_guard"


class ArtifactRef(ExperienceModel):
    """A content-addressed reference; production payload bytes never cross this boundary."""

    artifact_id: Identifier
    artifact_uri: Annotated[str, Field(min_length=1, max_length=2048)]
    artifact_sha256: Digest
    sample_ids: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=100_000)]

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("artifact sample identifiers must be unique")
        return self


class ExperienceFeatures(ExperienceModel):
    """Bounded selection signals. Values are supplied predictions, not measurements."""

    failure: bool
    verifier_disagreement: UnitInterval
    policy_uncertainty: UnitInterval
    value_uncertainty: UnitInterval
    novelty: UnitInterval
    rarity: UnitInterval
    safety: UnitInterval
    recurrence: UnitInterval
    autopsy_issue: UnitInterval
    reward_disagreement: UnitInterval
    capability_regression: UnitInterval
    branchability: UnitInterval
    expected_learning_value: NonNegativeFinite


class ExperienceCandidate(ExperienceModel):
    """A trainable evidence candidate with explicit governance and resource metadata."""

    candidate_id: Identifier
    tenant_id: Identifier
    source: EvidenceSource
    privacy: PrivacyClass
    side_effect_risk: SideEffectRisk
    consent_granted: bool
    redaction_applied: bool
    requires_live_side_effects: bool = False
    content_fingerprint: Digest
    artifacts: Annotated[tuple[ArtifactRef, ...], Field(min_length=1, max_length=64)]
    authorization_artifact_sha256: Digest | None = None
    redaction_artifact_sha256: Digest | None = None
    features: ExperienceFeatures
    training_cost_microunits: int = Field(gt=0, le=10**18)
    capacity_units: int = Field(gt=0, le=10**12)

    @model_validator(mode="after")
    def validate_artifacts(self) -> Self:
        artifact_ids = tuple(artifact.artifact_id for artifact in self.artifacts)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("candidate artifact identifiers must be unique")
        return self


class SelectionWeights(ExperienceModel):
    """Auditable Helix value-aware weights; defaults sum to one."""

    failure: UnitInterval = 0.10
    verifier_disagreement: UnitInterval = 0.08
    policy_uncertainty: UnitInterval = 0.08
    value_uncertainty: UnitInterval = 0.08
    novelty: UnitInterval = 0.10
    rarity: UnitInterval = 0.06
    safety: UnitInterval = 0.10
    recurrence: UnitInterval = 0.06
    autopsy_issue: UnitInterval = 0.08
    reward_disagreement: UnitInterval = 0.06
    capability_regression: UnitInterval = 0.08
    branchability: UnitInterval = 0.05
    expected_value_per_cost: UnitInterval = 0.07

    @model_validator(mode="after")
    def validate_positive_total(self) -> Self:
        if self.total <= 0.0:
            raise ValueError("at least one experience-selection weight must be positive")
        return self

    @property
    def total(self) -> float:
        return sum(
            (
                self.failure,
                self.verifier_disagreement,
                self.policy_uncertainty,
                self.value_uncertainty,
                self.novelty,
                self.rarity,
                self.safety,
                self.recurrence,
                self.autopsy_issue,
                self.reward_disagreement,
                self.capability_regression,
                self.branchability,
                self.expected_value_per_cost,
            )
        )


class ExperienceSelectionConstraints(ExperienceModel):
    """Hard selection envelope. Budget, capacity, and item count are all mandatory."""

    budget_microunits: int = Field(ge=0, le=10**30)
    capacity_units: int = Field(ge=0, le=10**18)
    max_selected_experiences: int = Field(gt=0, le=100_000)
    maximum_privacy: PrivacyClass
    allowed_side_effect_risks: Annotated[
        tuple[SideEffectRisk, ...], Field(min_length=1, max_length=3)
    ]
    allow_production_evidence: bool
    minimum_score: UnitInterval = 0.0

    @model_validator(mode="after")
    def validate_effects(self) -> Self:
        if len(self.allowed_side_effect_risks) != len(set(self.allowed_side_effect_risks)):
            raise ValueError("allowed side-effect risks must be unique")
        illegal = {SideEffectRisk.EXTERNAL, SideEffectRisk.IRREVERSIBLE}
        if illegal.intersection(self.allowed_side_effect_risks):
            raise ValueError("external and irreversible effects cannot be authorized for training")
        return self


class ExperienceSelectionRequest(ExperienceModel):
    schema_version: Literal["sloforge.helix.experience-selection-request/v1"] = (
        "sloforge.helix.experience-selection-request/v1"
    )
    request_id: Identifier
    tenant_id: Identifier
    seed: int = Field(ge=0, le=2**64 - 1)
    strategy: SelectionStrategy
    constraints: ExperienceSelectionConstraints
    candidates: Annotated[tuple[ExperienceCandidate, ...], Field(min_length=1, max_length=100_000)]
    weights: SelectionWeights = SelectionWeights()
    assumptions: Annotated[tuple[NonEmpty, ...], Field(max_length=128)] = ()

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("experience candidate identifiers must be unique")
        if len(self.assumptions) != len(set(self.assumptions)):
            raise ValueError("selection assumptions must be unique")
        artifact_hashes: dict[str, str] = {}
        for candidate in self.candidates:
            for artifact in candidate.artifacts:
                existing = artifact_hashes.setdefault(
                    artifact.artifact_id, artifact.artifact_sha256
                )
                if existing != artifact.artifact_sha256:
                    raise ValueError("one artifact identifier cannot claim multiple hashes")
        return self


class SelectionScore(ExperienceModel):
    """Complete score explanation retained for selected and excluded candidates."""

    failure: UnitInterval
    verifier_disagreement: UnitInterval
    policy_uncertainty: UnitInterval
    value_uncertainty: UnitInterval
    novelty: UnitInterval
    rarity: UnitInterval
    safety: UnitInterval
    recurrence: UnitInterval
    autopsy_issue: UnitInterval
    reward_disagreement: UnitInterval
    capability_regression: UnitInterval
    branchability: UnitInterval
    expected_learning_value: NonNegativeFinite
    value_cost_ratio: NonNegativeFinite
    normalized_value_cost: UnitInterval
    strategy_score: UnitInterval
    deterministic_tie_break: Digest


class CandidateDecision(ExperienceModel):
    candidate_id: Identifier
    candidate_digest: Digest
    selected: bool
    selection_rank: int | None = Field(default=None, ge=1, le=100_000)
    score: SelectionScore
    prediction_uncertainty: UnitInterval
    training_cost_microunits: int = Field(gt=0, le=10**18)
    capacity_units: int = Field(gt=0, le=10**12)
    artifact_hashes: Annotated[tuple[Digest, ...], Field(min_length=1, max_length=66)]
    exclusion_reasons: Annotated[tuple[ExclusionReason, ...], Field(max_length=32)] = ()

    @model_validator(mode="after")
    def validate_disposition(self) -> Self:
        if self.selected:
            if self.selection_rank is None or self.exclusion_reasons:
                raise ValueError("selected evidence requires a rank and no exclusion reasons")
        elif self.selection_rank is not None or not self.exclusion_reasons:
            raise ValueError("excluded evidence requires reasons and cannot claim a rank")
        if len(self.exclusion_reasons) != len(set(self.exclusion_reasons)):
            raise ValueError("candidate exclusion reasons must be unique")
        if tuple(sorted(set(self.artifact_hashes))) != self.artifact_hashes:
            raise ValueError("candidate artifact hashes must be sorted and unique")
        return self


class SelectionAccounting(ExperienceModel):
    budget_limit_microunits: int = Field(ge=0, le=10**30)
    budget_used_microunits: int = Field(ge=0, le=10**30)
    budget_remaining_microunits: int = Field(ge=0, le=10**30)
    capacity_limit_units: int = Field(ge=0, le=10**18)
    capacity_used_units: int = Field(ge=0, le=10**18)
    capacity_remaining_units: int = Field(ge=0, le=10**18)
    configured_max_count: int = Field(gt=0, le=100_000)
    effective_max_count: int = Field(ge=0, le=100_000)
    selected_count: int = Field(ge=0, le=100_000)
    excluded_count: int = Field(ge=0, le=100_000)

    @model_validator(mode="after")
    def validate_conservation(self) -> Self:
        if self.budget_used_microunits + self.budget_remaining_microunits != (
            self.budget_limit_microunits
        ):
            raise ValueError("selection budget accounting does not conserve the limit")
        if self.capacity_used_units + self.capacity_remaining_units != self.capacity_limit_units:
            raise ValueError("selection capacity accounting does not conserve the limit")
        if self.effective_max_count > self.configured_max_count:
            raise ValueError("effective count cannot exceed the configured selection limit")
        if self.selected_count > self.effective_max_count:
            raise ValueError("selected count exceeds the effective selection limit")
        return self


class ExperienceSelectionPlan(ExperienceModel):
    """Tamper-evident, complete disposition of an experience candidate pool."""

    schema_version: Literal["sloforge.helix.experience-selection-plan/v1"] = (
        "sloforge.helix.experience-selection-plan/v1"
    )
    plan_id: Digest
    request_digest: Digest
    request_id: Identifier
    tenant_id: Identifier
    seed: int = Field(ge=0, le=2**64 - 1)
    strategy: SelectionStrategy
    selected_candidate_ids: tuple[Identifier, ...]
    decisions: Annotated[tuple[CandidateDecision, ...], Field(min_length=1, max_length=100_000)]
    accounting: SelectionAccounting
    input_artifact_hashes: Annotated[tuple[Digest, ...], Field(min_length=1, max_length=100_000)]
    assumptions: Annotated[tuple[NonEmpty, ...], Field(min_length=1, max_length=256)]
    limitations: Annotated[tuple[NonEmpty, ...], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        decision_ids = tuple(decision.candidate_id for decision in self.decisions)
        if tuple(sorted(decision_ids)) != decision_ids:
            raise ValueError("candidate decisions must be ordered by identifier")
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("candidate decisions must be unique")
        selected = sorted(
            (decision for decision in self.decisions if decision.selected),
            key=lambda decision: decision.selection_rank or 0,
        )
        if tuple(decision.candidate_id for decision in selected) != self.selected_candidate_ids:
            raise ValueError("selected candidate identifiers disagree with decision ranks")
        if tuple(decision.selection_rank for decision in selected) != tuple(
            range(1, len(selected) + 1)
        ):
            raise ValueError("selection ranks must be dense")
        if self.accounting.selected_count != len(selected):
            raise ValueError("selected-count accounting disagrees with decisions")
        if self.accounting.excluded_count != len(self.decisions) - len(selected):
            raise ValueError("excluded-count accounting disagrees with decisions")
        if self.accounting.budget_used_microunits != sum(
            decision.training_cost_microunits for decision in selected
        ):
            raise ValueError("budget usage disagrees with selected evidence")
        if self.accounting.capacity_used_units != sum(
            decision.capacity_units for decision in selected
        ):
            raise ValueError("capacity usage disagrees with selected evidence")
        if len(self.decisions) > 1 and len(selected) == len(self.decisions):
            raise ValueError("train-all plans are prohibited")
        expected_hashes = tuple(
            sorted({digest for decision in self.decisions for digest in decision.artifact_hashes})
        )
        if self.input_artifact_hashes != expected_hashes:
            raise ValueError("plan artifact hashes disagree with candidate decisions")
        identity = self.model_dump(mode="json", exclude={"plan_id"})
        if canonical_digest(identity) != self.plan_id:
            raise ValueError("experience selection plan identifier is invalid")
        return self


__all__ = [
    "ArtifactRef",
    "CandidateDecision",
    "Digest",
    "EvidenceSource",
    "ExclusionReason",
    "ExperienceCandidate",
    "ExperienceFeatures",
    "ExperienceSelectionConstraints",
    "ExperienceSelectionPlan",
    "ExperienceSelectionPolicy",
    "ExperienceSelectionRequest",
    "PrivacyClass",
    "SelectionAccounting",
    "SelectionScore",
    "SelectionStrategy",
    "SelectionWeights",
    "SideEffectRisk",
    "canonical_digest",
]
