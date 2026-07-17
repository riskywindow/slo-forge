"""Deterministic multi-seed H9 experience-selection reference campaign.

The campaign compares the five reference selection strategies under one shared resource
envelope.  Selector scores only choose samples.  Reported learning effects come from a
separate reference-trainer update followed by paired, sampled policy decisions on held-out
cases; predicted learning-value fields never enter the downstream success calculation.
"""

from __future__ import annotations

import argparse
import json
import math
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from sloforge.helix.experience import (
    ArtifactRef,
    EvidenceSource,
    ExperienceCandidate,
    ExperienceFeatures,
    ExperienceSelectionConstraints,
    ExperienceSelectionPlan,
    ExperienceSelectionRequest,
    PrivacyClass,
    SelectionStrategy,
    SelectionWeights,
    SideEffectRisk,
    canonical_digest,
    select_experiences,
)
from sloforge.helix.policy import DeterministicPolicy
from sloforge.helix.scheduler.statistics import student_t_mean_interval_95
from sloforge.helix.trainers import (
    ReferenceTrainer,
    ReferenceTrainingSample,
    TrainingAlgorithm,
    TrainingResult,
)

MAX_CAMPAIGN_SCENARIO_BYTES = 2 * 1024 * 1024
MAX_CAMPAIGN_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_CAMPAIGN_ARTIFACT_BYTES = 64 * 1024 * 1024
_STRATEGIES = tuple(SelectionStrategy)
_BASELINES = tuple(
    strategy for strategy in _STRATEGIES if strategy is not SelectionStrategy.HELIX_VALUE_AWARE
)

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,159}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
_TRAINING_SAMPLES_ADAPTER = TypeAdapter(tuple[ReferenceTrainingSample, ...])


class ExperienceSelectionCampaignValidationError(ValueError):
    """Raised when a persisted campaign fails content or relational validation."""


class _CampaignModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class TrainingSignal(_CampaignModel):
    """Inline synthetic evidence converted into one reference-trainer sample."""

    sample_id: Identifier
    action: Identifier
    advantage: Annotated[float, Field(gt=0.0, le=100.0, allow_inf_nan=False)]
    token_weight: Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False)]
    reward_margin: Annotated[float, Field(gt=0.0, le=100.0, allow_inf_nan=False)]
    failure_family: Identifier


class CampaignCandidate(_CampaignModel):
    candidate_id: Identifier
    diversity_group: Identifier
    redundancy_group: Identifier
    features: ExperienceFeatures
    training_cost_microunits: Annotated[int, Field(gt=0, le=10**18)]
    capacity_units: Annotated[int, Field(gt=0, le=10**12)]
    training_signal: TrainingSignal


class HoldoutCase(_CampaignModel):
    case_id: Identifier
    observation: Annotated[str, Field(min_length=1, max_length=4096)]
    failure_family: Identifier
    successful_actions: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=128)]

    @model_validator(mode="after")
    def unique_actions(self) -> Self:
        if len(self.successful_actions) != len(set(self.successful_actions)):
            raise ValueError("holdout successful actions must be unique")
        return self


class ExperienceSelectionCampaignScenario(_CampaignModel):
    schema_version: Literal["sloforge.helix.experience-selection-campaign-scenario/v1"] = (
        "sloforge.helix.experience-selection-campaign-scenario/v1"
    )
    scenario_id: Identifier
    tenant_id: Identifier
    base_policy: DeterministicPolicy
    constraints: ExperienceSelectionConstraints
    weights: SelectionWeights
    candidates: Annotated[tuple[CampaignCandidate, ...], Field(min_length=2, max_length=128)]
    training_algorithm: Literal["successful_branch_distillation"]
    trainer_learning_rate: Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False)]
    trainer_kl_coefficient: Annotated[float, Field(ge=0.0, le=10.0, allow_inf_nan=False)]
    training_steps: Annotated[int, Field(gt=0, le=256)]
    evaluation_replicates_per_case: Annotated[int, Field(gt=0, le=128)]
    holdout_cases: Annotated[tuple[HoldoutCase, ...], Field(min_length=1, max_length=128)]
    assumptions: Annotated[tuple[str, ...], Field(min_length=1, max_length=64)]
    limitations: Annotated[tuple[str, ...], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def valid_protocol(self) -> Self:
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        sample_ids = tuple(item.training_signal.sample_id for item in self.candidates)
        case_ids = tuple(item.case_id for item in self.holdout_cases)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("campaign candidate identifiers must be unique")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("campaign training sample identifiers must be unique")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("campaign holdout case identifiers must be unique")
        actions = set(self.base_policy.actions)
        if any(item.training_signal.action not in actions for item in self.candidates):
            raise ValueError("campaign training signal references an unknown action")
        if any(not set(item.successful_actions).issubset(actions) for item in self.holdout_cases):
            raise ValueError("campaign holdout case references an unknown action")
        if self.constraints.minimum_score != 0.0:
            raise ValueError("reference campaign requires a zero minimum score for fair baselines")
        if self.constraints.maximum_privacy is not PrivacyClass.PUBLIC:
            raise ValueError("reference campaign permits public synthetic evidence only")
        if self.constraints.allowed_side_effect_risks != (SideEffectRisk.PURE,):
            raise ValueError("reference campaign permits pure evidence only")
        if self.constraints.allow_production_evidence:
            raise ValueError("reference campaign cannot enable production evidence")
        if self.constraints.max_selected_experiences >= len(self.candidates):
            raise ValueError("campaign must retain an unselected holdout from the candidate pool")
        qualifying = {
            SelectionStrategy.RANDOM: tuple(self.candidates),
            SelectionStrategy.FAILURE_ONLY: tuple(
                item for item in self.candidates if item.features.failure
            ),
            SelectionStrategy.UNCERTAINTY_ONLY: tuple(
                item
                for item in self.candidates
                if max(item.features.policy_uncertainty, item.features.value_uncertainty) > 0.0
            ),
            SelectionStrategy.NOVELTY_ONLY: tuple(
                item for item in self.candidates if item.features.novelty > 0.0
            ),
            SelectionStrategy.HELIX_VALUE_AWARE: tuple(
                item for item in self.candidates if item.features.expected_learning_value > 0.0
            ),
        }
        for strategy, candidates in qualifying.items():
            if not candidates:
                raise ValueError(f"campaign has no qualifying candidate for {strategy.value}")
            if not any(
                item.training_cost_microunits <= self.constraints.budget_microunits
                and item.capacity_units <= self.constraints.capacity_units
                for item in candidates
            ):
                raise ValueError(f"campaign cannot admit any candidate for {strategy.value}")
        if len(self.assumptions) != len(set(self.assumptions)):
            raise ValueError("campaign assumptions must be unique")
        if len(self.limitations) != len(set(self.limitations)):
            raise ValueError("campaign limitations must be unique")
        return self


class CandidateEvidence(_CampaignModel):
    schema_version: Literal["sloforge.helix.experience-candidate-evidence/v1"] = (
        "sloforge.helix.experience-candidate-evidence/v1"
    )
    candidate_id: Identifier
    diversity_group: Identifier
    redundancy_group: Identifier
    training_signal: TrainingSignal


ArtifactKind = Literal[
    "scenario",
    "candidate_evidence",
    "selection_request",
    "selection_plan",
    "training_samples",
    "training_result",
    "evaluation_trace",
]


class ArtifactPointer(_CampaignModel):
    kind: ArtifactKind
    path: Annotated[str, Field(min_length=1, max_length=2048)]
    sha256: Digest

    @model_validator(mode="after")
    def portable_path(self) -> Self:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or str(path) != self.path:
            raise ValueError("campaign artifact paths must be normalized and relative")
        return self


class CandidateEvidenceArtifact(_CampaignModel):
    candidate_id: Identifier
    artifact: ArtifactPointer

    @model_validator(mode="after")
    def correct_kind(self) -> Self:
        if self.artifact.kind != "candidate_evidence":
            raise ValueError("candidate evidence requires a candidate-evidence artifact")
        return self


class EvaluationDecision(_CampaignModel):
    case_id: Identifier
    failure_family: Identifier
    replicate: Annotated[int, Field(ge=0, le=127)]
    decision_seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    champion_action: Identifier
    trained_action: Identifier
    champion_success: bool
    trained_success: bool


class DownstreamEvaluationTrace(_CampaignModel):
    schema_version: Literal["sloforge.helix.experience-downstream-evaluation/v1"] = (
        "sloforge.helix.experience-downstream-evaluation/v1"
    )
    scenario_id: Identifier
    campaign_seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    strategy: SelectionStrategy
    champion_policy_epoch_id: Identifier
    champion_weights_sha256: Digest
    trained_policy_epoch_id: Identifier
    trained_weights_sha256: Digest
    training_checkpoint_sha256: Digest
    decisions: Annotated[tuple[EvaluationDecision, ...], Field(min_length=1, max_length=16_384)]
    measured_champion_success_rate: UnitInterval
    measured_trained_success_rate: UnitInterval
    paired_measured_success_change: Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]
    failures_repaired: Annotated[int, Field(ge=0)]
    failures_introduced: Annotated[int, Field(ge=0)]
    repaired_failure_families: tuple[Identifier, ...]

    @model_validator(mode="after")
    def valid_measurements(self) -> Self:
        count = len(self.decisions)
        champion_successes = sum(item.champion_success for item in self.decisions)
        trained_successes = sum(item.trained_success for item in self.decisions)
        repaired = sum(
            not item.champion_success and item.trained_success for item in self.decisions
        )
        introduced = sum(
            item.champion_success and not item.trained_success for item in self.decisions
        )
        expected_families = tuple(
            sorted(
                {
                    item.failure_family
                    for item in self.decisions
                    if not item.champion_success and item.trained_success
                }
            )
        )
        expected = (
            champion_successes / count,
            trained_successes / count,
            (trained_successes - champion_successes) / count,
        )
        observed = (
            self.measured_champion_success_rate,
            self.measured_trained_success_rate,
            self.paired_measured_success_change,
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(expected, observed, strict=True)
        ):
            raise ValueError("downstream rates disagree with raw paired decisions")
        if repaired != self.failures_repaired or introduced != self.failures_introduced:
            raise ValueError("failure repair accounting disagrees with raw paired decisions")
        if expected_families != self.repaired_failure_families:
            raise ValueError("repaired failure families disagree with raw paired decisions")
        return self


class ObservationProvenance(_CampaignModel):
    selection_request: ArtifactPointer
    selection_plan: ArtifactPointer
    training_samples: ArtifactPointer
    training_result: ArtifactPointer
    evaluation_trace: ArtifactPointer

    @model_validator(mode="after")
    def correct_kinds(self) -> Self:
        observed = (
            self.selection_request.kind,
            self.selection_plan.kind,
            self.training_samples.kind,
            self.training_result.kind,
            self.evaluation_trace.kind,
        )
        expected = (
            "selection_request",
            "selection_plan",
            "training_samples",
            "training_result",
            "evaluation_trace",
        )
        if observed != expected:
            raise ValueError("observation provenance artifact kinds are invalid")
        return self

    def pointers(self) -> tuple[ArtifactPointer, ...]:
        return (
            self.selection_request,
            self.selection_plan,
            self.training_samples,
            self.training_result,
            self.evaluation_trace,
        )


class StrategySeedObservation(_CampaignModel):
    campaign_seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    selection_seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    training_seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    strategy: SelectionStrategy
    selected_candidate_ids: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=127)]
    selected_count: Annotated[int, Field(gt=0, le=127)]
    training_cost_microunits: Annotated[int, Field(gt=0, le=10**30)]
    selected_diversity_group_count: Annotated[int, Field(gt=0, le=127)]
    selected_diversity_ratio: UnitInterval
    semantic_redundancy_count: Annotated[int, Field(ge=0, le=127)]
    semantic_redundancy_ratio: UnitInterval
    evaluation_decision_count: Annotated[int, Field(gt=0, le=16_384)]
    measured_champion_success_rate: UnitInterval
    measured_trained_success_rate: UnitInterval
    paired_measured_success_change: Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]
    failures_repaired: Annotated[int, Field(ge=0, le=16_384)]
    failures_introduced: Annotated[int, Field(ge=0, le=16_384)]
    repaired_failure_families: tuple[Identifier, ...]
    provenance: ObservationProvenance

    @model_validator(mode="after")
    def valid_accounting(self) -> Self:
        if self.selection_seed != self.campaign_seed:
            raise ValueError("selection seed must be the explicit campaign seed")
        if self.selected_count != len(self.selected_candidate_ids):
            raise ValueError("selected count disagrees with selected identifiers")
        if len(self.selected_candidate_ids) != len(set(self.selected_candidate_ids)):
            raise ValueError("selected candidate identifiers must be unique")
        if not 1 <= self.selected_diversity_group_count <= self.selected_count:
            raise ValueError("selected diversity count is outside the selected set")
        if not 0 <= self.semantic_redundancy_count < self.selected_count:
            raise ValueError("semantic redundancy count is outside the selected set")
        if self.failures_repaired > self.evaluation_decision_count or (
            self.failures_introduced > self.evaluation_decision_count
        ):
            raise ValueError("failure transition counts exceed evaluated decisions")
        if tuple(sorted(set(self.repaired_failure_families))) != (self.repaired_failure_families):
            raise ValueError("repaired failure families must be sorted and unique")
        provenance_paths = tuple(item.path for item in self.provenance.pointers())
        if len(provenance_paths) != len(set(provenance_paths)):
            raise ValueError("observation provenance paths must be unique")
        expected_diversity = self.selected_diversity_group_count / self.selected_count
        expected_redundancy = self.semantic_redundancy_count / self.selected_count
        if not math.isclose(
            expected_diversity, self.selected_diversity_ratio, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("selected diversity ratio is invalid")
        if not math.isclose(
            expected_redundancy, self.semantic_redundancy_ratio, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("selected redundancy ratio is invalid")
        expected_change = (
            self.failures_repaired - self.failures_introduced
        ) / self.evaluation_decision_count
        if not math.isclose(
            expected_change,
            self.paired_measured_success_change,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("paired success change disagrees with repair accounting")
        if not math.isclose(
            self.measured_trained_success_rate - self.measured_champion_success_rate,
            self.paired_measured_success_change,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("paired success change disagrees with measured rates")
        return self


class StudentTSensitivity(_CampaignModel):
    method: Literal["two_sided_student_t_mean"] = "two_sided_student_t_mean"
    confidence: Annotated[float, Field(ge=0.95, le=0.95)] = 0.95
    mean: float
    lower_95: float
    upper_95: float
    sample_count: Annotated[int, Field(ge=2, le=32)]
    sample_standard_deviation: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def finite_ordered_interval(self) -> Self:
        values = (self.mean, self.lower_95, self.upper_95)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Student-t sensitivity values must be finite")
        if not self.lower_95 <= self.mean <= self.upper_95:
            raise ValueError("Student-t sensitivity interval must contain its mean")
        if self.confidence != 0.95:
            raise ValueError("campaign sensitivity intervals require 95% confidence")
        return self


class StrategySummary(_CampaignModel):
    strategy: SelectionStrategy
    training_cost_microunits: StudentTSensitivity
    selected_diversity_ratio: StudentTSensitivity
    semantic_redundancy_ratio: StudentTSensitivity
    failures_repaired: StudentTSensitivity
    failures_introduced: StudentTSensitivity
    measured_champion_success_rate: StudentTSensitivity
    measured_trained_success_rate: StudentTSensitivity
    paired_measured_success_change: StudentTSensitivity


class PairedHelixBaselineEffect(_CampaignModel):
    campaign_seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    helix_measured_success_change: Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]
    baseline_measured_success_change: Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]
    paired_difference: Annotated[float, Field(ge=-2.0, le=2.0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def valid_difference(self) -> Self:
        expected = self.helix_measured_success_change - self.baseline_measured_success_change
        if not math.isclose(expected, self.paired_difference, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Helix-baseline effect is not paired by seed")
        return self


class HelixBaselineComparison(_CampaignModel):
    baseline_strategy: SelectionStrategy
    effect_definition: Literal[
        "paired difference in measured success change (Helix minus baseline)"
    ] = "paired difference in measured success change (Helix minus baseline)"
    per_seed_effects: Annotated[tuple[PairedHelixBaselineEffect, ...], Field(min_length=2)]
    sensitivity: StudentTSensitivity

    @model_validator(mode="after")
    def valid_baseline(self) -> Self:
        if self.baseline_strategy is SelectionStrategy.HELIX_VALUE_AWARE:
            raise ValueError("Helix cannot be its own baseline")
        return self


class H9Result(_CampaignModel):
    hypothesis_id: Literal["H9"] = "H9"
    status: Literal["not_supported", "inconclusive"]
    claim: Literal[
        "Helix has greater paired measured downstream success change than the predeclared random-selection primary baseline."
    ] = "Helix has greater paired measured downstream success change than the predeclared random-selection primary baseline."
    decision_rule: Literal[
        "not_supported when the Helix-minus-random paired mean is non-positive; otherwise inconclusive because seed-sensitivity cases are non-independent; other baselines are descriptive"
    ] = "not_supported when the Helix-minus-random paired mean is non-positive; otherwise inconclusive because seed-sensitivity cases are non-independent; other baselines are descriptive"
    limitations: tuple[str, ...]


class ExperienceSelectionCampaignRun(_CampaignModel):
    schema_version: Literal["sloforge.helix.experience-selection-campaign/v1"] = (
        "sloforge.helix.experience-selection-campaign/v1"
    )
    campaign_id: Digest
    scenario_id: Identifier
    seeds: Annotated[tuple[int, ...], Field(min_length=2, max_length=32)]
    strategies: Annotated[tuple[SelectionStrategy, ...], Field(min_length=5, max_length=5)]
    scenario_artifact: ArtifactPointer
    candidate_evidence: Annotated[
        tuple[CandidateEvidenceArtifact, ...], Field(min_length=2, max_length=128)
    ]
    observations: Annotated[tuple[StrategySeedObservation, ...], Field(min_length=10)]
    strategy_summaries: Annotated[tuple[StrategySummary, ...], Field(min_length=5, max_length=5)]
    helix_baseline_comparisons: Annotated[
        tuple[HelixBaselineComparison, ...], Field(min_length=4, max_length=4)
    ]
    h9_result: H9Result
    raw_artifacts: Annotated[tuple[ArtifactPointer, ...], Field(min_length=13)]
    measurement_definition: Literal[
        "paired sampled holdout decisions after reference training; selector scores are not gains"
    ]
    validation_class: Literal["deterministic-local-cpu-synthetic-reference-learning"]
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def complete_and_sealed(self) -> Self:
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("campaign seeds must be unique")
        if tuple(sorted(self.seeds)) != self.seeds:
            raise ValueError("campaign seeds must be sorted")
        if any(seed < 0 or seed > 2**63 - 1 for seed in self.seeds):
            raise ValueError("campaign seeds must fit signed 64-bit integers")
        if self.strategies != _STRATEGIES:
            raise ValueError("campaign must preserve the five-strategy reference order")
        expected_matrix = {(seed, strategy) for seed in self.seeds for strategy in _STRATEGIES}
        observed_matrix = {(item.campaign_seed, item.strategy) for item in self.observations}
        if observed_matrix != expected_matrix or len(self.observations) != len(expected_matrix):
            raise ValueError("campaign must contain the complete strategy-by-seed matrix")
        for seed in self.seeds:
            paired = tuple(item for item in self.observations if item.campaign_seed == seed)
            if len({item.training_seed for item in paired}) != 1:
                raise ValueError("strategies must share one trainer seed within each campaign seed")
            if len({item.evaluation_decision_count for item in paired}) != 1:
                raise ValueError("strategies must share one holdout size within each campaign seed")
            if len({item.measured_champion_success_rate for item in paired}) != 1:
                raise ValueError(
                    "strategies must share paired champion measurements within each campaign seed"
                )
        expected_summaries = tuple(
            _summarize_strategy(
                strategy,
                tuple(item for item in self.observations if item.strategy is strategy),
            )
            for strategy in _STRATEGIES
        )
        if self.strategy_summaries != expected_summaries:
            raise ValueError("strategy summaries disagree with raw seed observations")
        expected_comparisons = _compare_helix_to_baselines(self.seeds, self.observations)
        if self.helix_baseline_comparisons != expected_comparisons:
            raise ValueError("Helix comparisons disagree with paired seed observations")
        if self.h9_result != _h9_result(expected_comparisons, self.limitations):
            raise ValueError("H9 result disagrees with its declared decision rule")
        paths = tuple(item.path for item in self.raw_artifacts)
        if tuple(sorted(set(paths))) != paths:
            raise ValueError("raw campaign artifact paths must be sorted and unique")
        ledger = {item.path: item for item in self.raw_artifacts}
        linked = [self.scenario_artifact]
        linked.extend(item.artifact for item in self.candidate_evidence)
        for observation in self.observations:
            linked.extend(observation.provenance.pointers())
        if len(linked) != len(self.raw_artifacts):
            raise ValueError("raw artifact ledger contains unlinked campaign artifacts")
        for pointer in linked:
            if ledger.get(pointer.path) != pointer:
                raise ValueError("campaign provenance pointer is absent from the raw ledger")
        if self.scenario_artifact.kind != "scenario":
            raise ValueError("campaign scenario artifact has the wrong kind")
        candidate_ids = tuple(item.candidate_id for item in self.candidate_evidence)
        if tuple(sorted(set(candidate_ids))) != candidate_ids:
            raise ValueError("candidate evidence must be identifier-sorted and unique")
        identity = self.model_dump(mode="json", exclude={"campaign_id"})
        if canonical_digest(identity) != self.campaign_id:
            raise ValueError("experience-selection campaign identifier is invalid")
        return self


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_document(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (tuple, list)):
        return [_json_document(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_document(item) for key, item in value.items()}
    return value


def _write_document(output: Path, value: object) -> None:
    document = _json_document(value)
    encoded = _canonical_bytes(document) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)


def _write_artifact(
    output_root: Path, relative: str, kind: ArtifactKind, value: object
) -> ArtifactPointer:
    destination = output_root / relative
    document = _json_document(value)
    encoded = _canonical_bytes(document) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded)
    return ArtifactPointer(kind=kind, path=relative, sha256=sha256(encoded).hexdigest())


def _copy_scenario(output_root: Path, source_bytes: bytes) -> ArtifactPointer:
    relative = "raw/scenario.json"
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source_bytes)
    return ArtifactPointer(kind="scenario", path=relative, sha256=sha256(source_bytes).hexdigest())


def load_experience_selection_campaign_scenario(
    path: Path,
) -> ExperienceSelectionCampaignScenario:
    """Load one bounded strict JSON campaign scenario."""

    payload = path.read_bytes()
    if len(payload) > MAX_CAMPAIGN_SCENARIO_BYTES:
        raise ValueError("experience-selection campaign scenario exceeds the byte limit")
    return ExperienceSelectionCampaignScenario.model_validate_json(payload, strict=True)


def _derived_seed(domain: str, scenario_id: str, campaign_seed: int, *parts: object) -> int:
    identity = ":".join((domain, scenario_id, str(campaign_seed), *(str(item) for item in parts)))
    return int.from_bytes(sha256(identity.encode("utf-8")).digest()[:8], "big") & (2**63 - 1)


def _content_fingerprint(signal: TrainingSignal) -> str:
    content = signal.model_dump(mode="json", exclude={"sample_id"})
    return canonical_digest(content)


def _selection_candidate(
    scenario: ExperienceSelectionCampaignScenario,
    item: CampaignCandidate,
    artifact: ArtifactPointer,
) -> ExperienceCandidate:
    return ExperienceCandidate(
        candidate_id=item.candidate_id,
        tenant_id=scenario.tenant_id,
        source=EvidenceSource.SYNTHETIC,
        privacy=PrivacyClass.PUBLIC,
        side_effect_risk=SideEffectRisk.PURE,
        consent_granted=False,
        redaction_applied=False,
        content_fingerprint=_content_fingerprint(item.training_signal),
        artifacts=(
            ArtifactRef(
                artifact_id=f"artifact.{item.candidate_id}",
                artifact_uri=artifact.path,
                artifact_sha256=artifact.sha256,
                sample_ids=(item.training_signal.sample_id,),
            ),
        ),
        features=item.features,
        training_cost_microunits=item.training_cost_microunits,
        capacity_units=item.capacity_units,
    )


def _selection_request(
    scenario: ExperienceSelectionCampaignScenario,
    strategy: SelectionStrategy,
    seed: int,
    candidate_artifacts: dict[str, ArtifactPointer],
) -> ExperienceSelectionRequest:
    candidates = tuple(
        _selection_candidate(scenario, item, candidate_artifacts[item.candidate_id])
        for item in scenario.candidates
    )
    return ExperienceSelectionRequest(
        request_id=f"{scenario.scenario_id}.{strategy.value}.{seed}",
        tenant_id=scenario.tenant_id,
        seed=seed,
        strategy=strategy,
        constraints=scenario.constraints,
        candidates=candidates,
        weights=scenario.weights,
        assumptions=scenario.assumptions,
    )


def _training_samples(
    scenario: ExperienceSelectionCampaignScenario,
    selected_candidate_ids: tuple[str, ...],
) -> tuple[ReferenceTrainingSample, ...]:
    by_id = {item.candidate_id: item for item in scenario.candidates}
    samples: list[ReferenceTrainingSample] = []
    for candidate_id in selected_candidate_ids:
        signal = by_id[candidate_id].training_signal
        samples.append(
            ReferenceTrainingSample(
                sample_id=signal.sample_id,
                trajectory_id=f"candidate-evidence:{candidate_id}",
                branch_point_id=f"candidate-point:{candidate_id}",
                branch_group_id=f"h9:{scenario.scenario_id}",
                policy_epoch_id=scenario.base_policy.policy_epoch_id,
                action=signal.action,
                behavior_log_probability=scenario.base_policy.log_probability(signal.action),
                advantage=signal.advantage,
                token_weight=signal.token_weight,
                reward_margin=signal.reward_margin,
            )
        )
    return tuple(samples)


def _evaluate_downstream(
    scenario: ExperienceSelectionCampaignScenario,
    *,
    strategy: SelectionStrategy,
    campaign_seed: int,
    trained: DeterministicPolicy,
    checkpoint_hash: str,
) -> DownstreamEvaluationTrace:
    decisions: list[EvaluationDecision] = []
    for case in scenario.holdout_cases:
        for replicate in range(scenario.evaluation_replicates_per_case):
            decision_seed = _derived_seed(
                "evaluation", scenario.scenario_id, campaign_seed, case.case_id, replicate
            )
            champion_decision = scenario.base_policy.decide(
                case.observation, seed=decision_seed, rng_counter=replicate
            )
            trained_decision = trained.decide(
                case.observation, seed=decision_seed, rng_counter=replicate
            )
            decisions.append(
                EvaluationDecision(
                    case_id=case.case_id,
                    failure_family=case.failure_family,
                    replicate=replicate,
                    decision_seed=decision_seed,
                    champion_action=champion_decision.action,
                    trained_action=trained_decision.action,
                    champion_success=champion_decision.action in case.successful_actions,
                    trained_success=trained_decision.action in case.successful_actions,
                )
            )
    decision_tuple = tuple(decisions)
    count = len(decision_tuple)
    champion_count = sum(item.champion_success for item in decision_tuple)
    trained_count = sum(item.trained_success for item in decision_tuple)
    repaired = tuple(
        item for item in decision_tuple if not item.champion_success and item.trained_success
    )
    introduced = tuple(
        item for item in decision_tuple if item.champion_success and not item.trained_success
    )
    return DownstreamEvaluationTrace(
        scenario_id=scenario.scenario_id,
        campaign_seed=campaign_seed,
        strategy=strategy,
        champion_policy_epoch_id=scenario.base_policy.policy_epoch_id,
        champion_weights_sha256=scenario.base_policy.weights_hash,
        trained_policy_epoch_id=trained.policy_epoch_id,
        trained_weights_sha256=trained.weights_hash,
        training_checkpoint_sha256=checkpoint_hash,
        decisions=decision_tuple,
        measured_champion_success_rate=champion_count / count,
        measured_trained_success_rate=trained_count / count,
        paired_measured_success_change=(trained_count - champion_count) / count,
        failures_repaired=len(repaired),
        failures_introduced=len(introduced),
        repaired_failure_families=tuple(sorted({item.failure_family for item in repaired})),
    )


def _sensitivity(values: tuple[float, ...]) -> StudentTSensitivity:
    mean, lower, upper, deviation = student_t_mean_interval_95(values)
    return StudentTSensitivity(
        mean=mean,
        lower_95=lower,
        upper_95=upper,
        sample_count=len(values),
        sample_standard_deviation=deviation,
    )


def _summarize_strategy(
    strategy: SelectionStrategy,
    observations: tuple[StrategySeedObservation, ...],
) -> StrategySummary:
    ordered = tuple(sorted(observations, key=lambda item: item.campaign_seed))
    return StrategySummary(
        strategy=strategy,
        training_cost_microunits=_sensitivity(
            tuple(float(item.training_cost_microunits) for item in ordered)
        ),
        selected_diversity_ratio=_sensitivity(
            tuple(item.selected_diversity_ratio for item in ordered)
        ),
        semantic_redundancy_ratio=_sensitivity(
            tuple(item.semantic_redundancy_ratio for item in ordered)
        ),
        failures_repaired=_sensitivity(tuple(float(item.failures_repaired) for item in ordered)),
        failures_introduced=_sensitivity(
            tuple(float(item.failures_introduced) for item in ordered)
        ),
        measured_champion_success_rate=_sensitivity(
            tuple(item.measured_champion_success_rate for item in ordered)
        ),
        measured_trained_success_rate=_sensitivity(
            tuple(item.measured_trained_success_rate for item in ordered)
        ),
        paired_measured_success_change=_sensitivity(
            tuple(item.paired_measured_success_change for item in ordered)
        ),
    )


def _compare_helix_to_baselines(
    seeds: tuple[int, ...],
    observations: tuple[StrategySeedObservation, ...],
) -> tuple[HelixBaselineComparison, ...]:
    by_key = {(item.campaign_seed, item.strategy): item for item in observations}
    comparisons: list[HelixBaselineComparison] = []
    for baseline in _BASELINES:
        effects = tuple(
            PairedHelixBaselineEffect(
                campaign_seed=seed,
                helix_measured_success_change=by_key[
                    seed, SelectionStrategy.HELIX_VALUE_AWARE
                ].paired_measured_success_change,
                baseline_measured_success_change=by_key[
                    seed, baseline
                ].paired_measured_success_change,
                paired_difference=(
                    by_key[seed, SelectionStrategy.HELIX_VALUE_AWARE].paired_measured_success_change
                    - by_key[seed, baseline].paired_measured_success_change
                ),
            )
            for seed in seeds
        )
        comparisons.append(
            HelixBaselineComparison(
                baseline_strategy=baseline,
                per_seed_effects=effects,
                sensitivity=_sensitivity(tuple(item.paired_difference for item in effects)),
            )
        )
    return tuple(comparisons)


def _h9_result(
    comparisons: tuple[HelixBaselineComparison, ...], limitations: tuple[str, ...]
) -> H9Result:
    primary = next(
        item for item in comparisons if item.baseline_strategy is SelectionStrategy.RANDOM
    )
    status: Literal["not_supported", "inconclusive"] = (
        "not_supported" if primary.sensitivity.mean <= 0.0 else "inconclusive"
    )
    return H9Result(status=status, limitations=limitations)


def _render_report(run: ExperienceSelectionCampaignRun) -> str:
    lines = [
        "# Helix H9 experience-selection campaign",
        "",
        f"Campaign ID: `{run.campaign_id}`",
        "",
        f"H9 status: **{run.h9_result.status}**",
        "",
        "Measured effects are paired sampled holdout outcomes after reference training. "
        "Selector scores and expected learning values are not treated as measured gain.",
        "",
        "## Strategy summaries",
        "",
        "| Strategy | Cost mean [95% sensitivity] | Diversity | Redundancy | Repaired | Introduced | Before | After | Paired change |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in run.strategy_summaries:
        cost = summary.training_cost_microunits
        diversity = summary.selected_diversity_ratio
        redundancy = summary.semantic_redundancy_ratio
        repaired = summary.failures_repaired
        introduced = summary.failures_introduced
        before = summary.measured_champion_success_rate
        after = summary.measured_trained_success_rate
        change = summary.paired_measured_success_change
        lines.append(
            f"| {summary.strategy.value} | {cost.mean:.1f} [{cost.lower_95:.1f}, {cost.upper_95:.1f}] "
            f"| {diversity.mean:.3f} | {redundancy.mean:.3f} | {repaired.mean:.2f} "
            f"| {introduced.mean:.2f} | {before.mean:.3f} | {after.mean:.3f} "
            f"| {change.mean:+.3f} [{change.lower_95:+.3f}, {change.upper_95:+.3f}] |"
        )
    lines.extend(
        (
            "",
            "## Paired Helix comparisons",
            "",
            "| Baseline | Helix-minus-baseline mean | 95% Student-t sensitivity interval | Seeds |",
            "| --- | ---: | ---: | ---: |",
        )
    )
    for comparison in run.helix_baseline_comparisons:
        sensitivity = comparison.sensitivity
        lines.append(
            f"| {comparison.baseline_strategy.value} | {sensitivity.mean:+.3f} "
            f"| [{sensitivity.lower_95:+.3f}, {sensitivity.upper_95:+.3f}] "
            f"| {sensitivity.sample_count} |"
        )
    lines.extend(("", "## Raw provenance", ""))
    lines.append(
        f"The campaign JSON contains a path-and-SHA-256 ledger for {len(run.raw_artifacts)} "
        "scenario, candidate, selection, training, and evaluation artifacts."
    )
    lines.extend(("", "## Limitations", ""))
    lines.extend(f"- {item}" for item in run.limitations)
    lines.append("")
    return "\n".join(lines)


def run_experience_selection_campaign(
    scenario_path: Path,
    output: Path,
    *,
    seeds: tuple[int, ...],
) -> ExperienceSelectionCampaignRun:
    """Run the complete bounded strategy-by-seed campaign and write raw evidence."""

    if not 2 <= len(seeds) <= 32:
        raise ValueError("experience-selection campaign requires two to 32 seeds")
    if len(seeds) != len(set(seeds)):
        raise ValueError("experience-selection campaign seeds must be unique")
    if any(seed < 0 or seed > 2**63 - 1 for seed in seeds):
        raise ValueError("experience-selection campaign seeds must fit signed 64-bit integers")
    ordered_seeds = tuple(sorted(seeds))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("campaign output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    scenario_bytes = scenario_path.read_bytes()
    if len(scenario_bytes) > MAX_CAMPAIGN_SCENARIO_BYTES:
        raise ValueError("experience-selection campaign scenario exceeds the byte limit")
    scenario = ExperienceSelectionCampaignScenario.model_validate_json(scenario_bytes, strict=True)

    raw_artifacts: list[ArtifactPointer] = []
    scenario_artifact = _copy_scenario(output, scenario_bytes)
    raw_artifacts.append(scenario_artifact)
    candidate_artifacts: list[CandidateEvidenceArtifact] = []
    artifacts_by_candidate: dict[str, ArtifactPointer] = {}
    for item in sorted(scenario.candidates, key=lambda candidate: candidate.candidate_id):
        evidence = CandidateEvidence(
            candidate_id=item.candidate_id,
            diversity_group=item.diversity_group,
            redundancy_group=item.redundancy_group,
            training_signal=item.training_signal,
        )
        pointer = _write_artifact(
            output,
            f"raw/candidates/{item.candidate_id}.json",
            "candidate_evidence",
            evidence,
        )
        raw_artifacts.append(pointer)
        artifacts_by_candidate[item.candidate_id] = pointer
        candidate_artifacts.append(
            CandidateEvidenceArtifact(candidate_id=item.candidate_id, artifact=pointer)
        )

    candidate_by_id = {item.candidate_id: item for item in scenario.candidates}
    observations: list[StrategySeedObservation] = []
    trainer = ReferenceTrainer(
        learning_rate=scenario.trainer_learning_rate,
        kl_coefficient=scenario.trainer_kl_coefficient,
    )
    for seed in ordered_seeds:
        for strategy in _STRATEGIES:
            relative_root = f"raw/runs/seed-{seed}/{strategy.value}"
            request = _selection_request(scenario, strategy, seed, artifacts_by_candidate)
            request_pointer = _write_artifact(
                output, f"{relative_root}/selection-request.json", "selection_request", request
            )
            plan = select_experiences(request)
            if not plan.selected_candidate_ids:
                raise RuntimeError(f"{strategy.value} selected no trainable experience")
            plan_pointer = _write_artifact(
                output, f"{relative_root}/selection-plan.json", "selection_plan", plan
            )
            samples = _training_samples(scenario, plan.selected_candidate_ids)
            samples_pointer = _write_artifact(
                output,
                f"{relative_root}/training-samples.json",
                "training_samples",
                samples,
            )
            training_seed = _derived_seed("training", scenario.scenario_id, seed)
            result = trainer.train(
                base=scenario.base_policy,
                samples=samples,
                algorithm=TrainingAlgorithm.SUCCESSFUL_BRANCH_DISTILLATION,
                candidate_policy_epoch_id=(
                    f"{scenario.scenario_id}.{strategy.value}.seed-{seed}.candidate"
                ),
                seed=training_seed,
                steps=scenario.training_steps,
            )
            result_pointer = _write_artifact(
                output, f"{relative_root}/training-result.json", "training_result", result
            )
            evaluation = _evaluate_downstream(
                scenario,
                strategy=strategy,
                campaign_seed=seed,
                trained=result.candidate,
                checkpoint_hash=result.checkpoint_hash,
            )
            evaluation_pointer = _write_artifact(
                output,
                f"{relative_root}/evaluation.json",
                "evaluation_trace",
                evaluation,
            )
            raw_artifacts.extend(
                (
                    request_pointer,
                    plan_pointer,
                    samples_pointer,
                    result_pointer,
                    evaluation_pointer,
                )
            )
            selected = tuple(candidate_by_id[item] for item in plan.selected_candidate_ids)
            diversity_count = len({item.diversity_group for item in selected})
            redundancy_count = len(selected) - len({item.redundancy_group for item in selected})
            observations.append(
                StrategySeedObservation(
                    campaign_seed=seed,
                    selection_seed=seed,
                    training_seed=training_seed,
                    strategy=strategy,
                    selected_candidate_ids=plan.selected_candidate_ids,
                    selected_count=len(selected),
                    training_cost_microunits=plan.accounting.budget_used_microunits,
                    selected_diversity_group_count=diversity_count,
                    selected_diversity_ratio=diversity_count / len(selected),
                    semantic_redundancy_count=redundancy_count,
                    semantic_redundancy_ratio=redundancy_count / len(selected),
                    evaluation_decision_count=len(evaluation.decisions),
                    measured_champion_success_rate=evaluation.measured_champion_success_rate,
                    measured_trained_success_rate=evaluation.measured_trained_success_rate,
                    paired_measured_success_change=evaluation.paired_measured_success_change,
                    failures_repaired=evaluation.failures_repaired,
                    failures_introduced=evaluation.failures_introduced,
                    repaired_failure_families=evaluation.repaired_failure_families,
                    provenance=ObservationProvenance(
                        selection_request=request_pointer,
                        selection_plan=plan_pointer,
                        training_samples=samples_pointer,
                        training_result=result_pointer,
                        evaluation_trace=evaluation_pointer,
                    ),
                )
            )

    observation_tuple = tuple(observations)
    summaries = tuple(
        _summarize_strategy(
            strategy,
            tuple(item for item in observation_tuple if item.strategy is strategy),
        )
        for strategy in _STRATEGIES
    )
    comparisons = _compare_helix_to_baselines(ordered_seeds, observation_tuple)
    limitations = tuple(
        dict.fromkeys(
            (
                *scenario.limitations,
                "This is one deterministic local synthetic scenario, not a production estimate.",
                "Seeds are deterministic sensitivity cases and are not claimed to be independent population draws.",
                "Student-t intervals summarize seed sensitivity at small n; they are not multiplicity-adjusted hypothesis tests.",
                "Selection features and expected learning values are input predictions; only paired holdout decisions are reported as measured gain.",
                "Semantic redundancy uses declared scenario groups; the selector itself deduplicates only exact content fingerprints.",
                "Failure-only uses the reference selector's declared secondary ranking behavior when more candidates qualify than fit.",
            )
        )
    )
    h9 = _h9_result(comparisons, limitations)
    unsealed = ExperienceSelectionCampaignRun.model_construct(
        campaign_id="0" * 64,
        scenario_id=scenario.scenario_id,
        seeds=ordered_seeds,
        strategies=_STRATEGIES,
        scenario_artifact=scenario_artifact,
        candidate_evidence=tuple(candidate_artifacts),
        observations=observation_tuple,
        strategy_summaries=summaries,
        helix_baseline_comparisons=comparisons,
        h9_result=h9,
        raw_artifacts=tuple(sorted(raw_artifacts, key=lambda item: item.path)),
        measurement_definition=(
            "paired sampled holdout decisions after reference training; selector scores are not gains"
        ),
        validation_class="deterministic-local-cpu-synthetic-reference-learning",
        assumptions=scenario.assumptions,
        limitations=limitations,
    )
    identity = unsealed.model_dump(mode="json", exclude={"campaign_id"})
    run = ExperienceSelectionCampaignRun(
        campaign_id=canonical_digest(identity),
        scenario_id=unsealed.scenario_id,
        seeds=unsealed.seeds,
        strategies=unsealed.strategies,
        scenario_artifact=unsealed.scenario_artifact,
        candidate_evidence=unsealed.candidate_evidence,
        observations=unsealed.observations,
        strategy_summaries=unsealed.strategy_summaries,
        helix_baseline_comparisons=unsealed.helix_baseline_comparisons,
        h9_result=unsealed.h9_result,
        raw_artifacts=unsealed.raw_artifacts,
        measurement_definition=unsealed.measurement_definition,
        validation_class=unsealed.validation_class,
        assumptions=unsealed.assumptions,
        limitations=unsealed.limitations,
    )
    _write_document(output / "campaign.json", run)
    (output / "report.md").write_text(_render_report(run), encoding="utf-8")
    return run


def _read_validation_file(
    output_root: Path,
    relative: str,
    *,
    maximum_bytes: int,
) -> bytes:
    candidate = output_root / relative
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ExperienceSelectionCampaignValidationError(
            f"campaign artifact is missing: {relative}"
        ) from exc
    if not resolved.is_relative_to(output_root):
        raise ExperienceSelectionCampaignValidationError(
            f"campaign artifact escapes the output directory: {relative}"
        )
    if not resolved.is_file():
        raise ExperienceSelectionCampaignValidationError(
            f"campaign artifact is not a regular file: {relative}"
        )
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise ExperienceSelectionCampaignValidationError(
            f"campaign artifact cannot be inspected: {relative}"
        ) from exc
    if size > maximum_bytes:
        raise ExperienceSelectionCampaignValidationError(
            f"campaign artifact exceeds the byte limit: {relative}"
        )
    try:
        return resolved.read_bytes()
    except OSError as exc:
        raise ExperienceSelectionCampaignValidationError(
            f"campaign artifact cannot be read: {relative}"
        ) from exc


def _strict_artifacts(
    run: ExperienceSelectionCampaignRun,
    output_root: Path,
) -> tuple[
    dict[str, ExperienceSelectionCampaignScenario],
    dict[str, CandidateEvidence],
    dict[str, ExperienceSelectionRequest],
    dict[str, ExperienceSelectionPlan],
    dict[str, tuple[ReferenceTrainingSample, ...]],
    dict[str, TrainingResult],
    dict[str, DownstreamEvaluationTrace],
]:
    scenarios: dict[str, ExperienceSelectionCampaignScenario] = {}
    candidates: dict[str, CandidateEvidence] = {}
    requests: dict[str, ExperienceSelectionRequest] = {}
    plans: dict[str, ExperienceSelectionPlan] = {}
    sample_batches: dict[str, tuple[ReferenceTrainingSample, ...]] = {}
    training_results: dict[str, TrainingResult] = {}
    evaluations: dict[str, DownstreamEvaluationTrace] = {}
    for pointer in run.raw_artifacts:
        maximum = (
            MAX_CAMPAIGN_SCENARIO_BYTES
            if pointer.kind == "scenario"
            else MAX_CAMPAIGN_ARTIFACT_BYTES
        )
        payload = _read_validation_file(output_root, pointer.path, maximum_bytes=maximum)
        if sha256(payload).hexdigest() != pointer.sha256:
            raise ExperienceSelectionCampaignValidationError(
                f"campaign artifact SHA-256 mismatch: {pointer.path}"
            )
        try:
            if pointer.kind == "scenario":
                scenarios[pointer.path] = ExperienceSelectionCampaignScenario.model_validate_json(
                    payload, strict=True
                )
            elif pointer.kind == "candidate_evidence":
                candidates[pointer.path] = CandidateEvidence.model_validate_json(
                    payload, strict=True
                )
            elif pointer.kind == "selection_request":
                requests[pointer.path] = ExperienceSelectionRequest.model_validate_json(
                    payload, strict=True
                )
            elif pointer.kind == "selection_plan":
                plans[pointer.path] = ExperienceSelectionPlan.model_validate_json(
                    payload, strict=True
                )
            elif pointer.kind == "training_samples":
                sample_batches[pointer.path] = _TRAINING_SAMPLES_ADAPTER.validate_json(
                    payload, strict=True
                )
            elif pointer.kind == "training_result":
                training_results[pointer.path] = TrainingResult.model_validate_json(
                    payload, strict=True
                )
            else:
                evaluations[pointer.path] = DownstreamEvaluationTrace.model_validate_json(
                    payload, strict=True
                )
        except (TypeError, ValueError) as exc:
            raise ExperienceSelectionCampaignValidationError(
                f"campaign artifact fails strict {pointer.kind} parsing: {pointer.path}"
            ) from exc
    return (
        scenarios,
        candidates,
        requests,
        plans,
        sample_batches,
        training_results,
        evaluations,
    )


def _expected_seed_observation(
    scenario: ExperienceSelectionCampaignScenario,
    observation: StrategySeedObservation,
    plan: ExperienceSelectionPlan,
    trace: DownstreamEvaluationTrace,
) -> StrategySeedObservation:
    candidate_by_id = {item.candidate_id: item for item in scenario.candidates}
    selected = tuple(candidate_by_id[item] for item in plan.selected_candidate_ids)
    diversity_count = len({item.diversity_group for item in selected})
    redundancy_count = len(selected) - len({item.redundancy_group for item in selected})
    return StrategySeedObservation(
        campaign_seed=observation.campaign_seed,
        selection_seed=observation.campaign_seed,
        training_seed=_derived_seed("training", scenario.scenario_id, observation.campaign_seed),
        strategy=observation.strategy,
        selected_candidate_ids=plan.selected_candidate_ids,
        selected_count=len(selected),
        training_cost_microunits=plan.accounting.budget_used_microunits,
        selected_diversity_group_count=diversity_count,
        selected_diversity_ratio=diversity_count / len(selected),
        semantic_redundancy_count=redundancy_count,
        semantic_redundancy_ratio=redundancy_count / len(selected),
        evaluation_decision_count=len(trace.decisions),
        measured_champion_success_rate=trace.measured_champion_success_rate,
        measured_trained_success_rate=trace.measured_trained_success_rate,
        paired_measured_success_change=trace.paired_measured_success_change,
        failures_repaired=trace.failures_repaired,
        failures_introduced=trace.failures_introduced,
        repaired_failure_families=trace.repaired_failure_families,
        provenance=observation.provenance,
    )


def validate_experience_selection_campaign(output: Path) -> ExperienceSelectionCampaignRun:
    """Reopen, rehash, strictly parse, and deterministically replay a campaign bundle."""

    try:
        output_root = output.resolve(strict=True)
    except OSError as exc:
        raise ExperienceSelectionCampaignValidationError(
            "campaign output directory is missing"
        ) from exc
    if not output_root.is_dir():
        raise ExperienceSelectionCampaignValidationError("campaign output path is not a directory")
    manifest = _read_validation_file(
        output_root, "campaign.json", maximum_bytes=MAX_CAMPAIGN_MANIFEST_BYTES
    )
    try:
        run = ExperienceSelectionCampaignRun.model_validate_json(manifest, strict=True)
    except (TypeError, ValueError) as exc:
        raise ExperienceSelectionCampaignValidationError(
            "campaign manifest fails strict validation"
        ) from exc
    expected_manifest = _canonical_bytes(run.model_dump(mode="json")) + b"\n"
    if manifest != expected_manifest:
        raise ExperienceSelectionCampaignValidationError(
            "campaign manifest is not the canonical sealed document"
        )
    (
        scenarios,
        candidate_documents,
        requests,
        plans,
        sample_batches,
        training_results,
        evaluations,
    ) = _strict_artifacts(run, output_root)
    expected_raw_paths = tuple(item.path for item in run.raw_artifacts)
    raw_root = output_root / "raw"
    actual_raw_paths = tuple(
        sorted(
            path.relative_to(output_root).as_posix()
            for path in raw_root.rglob("*")
            if path.is_file()
        )
    )
    if actual_raw_paths != expected_raw_paths:
        raise ExperienceSelectionCampaignValidationError(
            "raw campaign directory disagrees with the sealed artifact ledger"
        )

    scenario = scenarios.get(run.scenario_artifact.path)
    if scenario is None or scenario.scenario_id != run.scenario_id:
        raise ExperienceSelectionCampaignValidationError(
            "campaign scenario does not match the sealed campaign identity"
        )
    if run.assumptions != scenario.assumptions:
        raise ExperienceSelectionCampaignValidationError(
            "campaign assumptions do not match the raw scenario"
        )
    if not set(scenario.limitations).issubset(run.limitations):
        raise ExperienceSelectionCampaignValidationError("campaign omits a raw scenario limitation")

    scenario_candidates = {item.candidate_id: item for item in scenario.candidates}
    candidate_artifacts = {item.candidate_id: item.artifact for item in run.candidate_evidence}
    for candidate_id, pointer in candidate_artifacts.items():
        definition = scenario_candidates.get(candidate_id)
        document = candidate_documents.get(pointer.path)
        if definition is None or document is None:
            raise ExperienceSelectionCampaignValidationError(
                f"candidate evidence is not linked to the raw scenario: {candidate_id}"
            )
        expected_document = CandidateEvidence(
            candidate_id=definition.candidate_id,
            diversity_group=definition.diversity_group,
            redundancy_group=definition.redundancy_group,
            training_signal=definition.training_signal,
        )
        if document != expected_document:
            raise ExperienceSelectionCampaignValidationError(
                f"candidate evidence disagrees with the raw scenario: {candidate_id}"
            )
    if set(candidate_artifacts) != set(scenario_candidates):
        raise ExperienceSelectionCampaignValidationError(
            "campaign candidate evidence does not cover the raw scenario"
        )

    trainer = ReferenceTrainer(
        learning_rate=scenario.trainer_learning_rate,
        kl_coefficient=scenario.trainer_kl_coefficient,
    )
    for observation in run.observations:
        provenance = observation.provenance
        request = requests[provenance.selection_request.path]
        expected_request = _selection_request(
            scenario,
            observation.strategy,
            observation.campaign_seed,
            candidate_artifacts,
        )
        if request != expected_request:
            raise ExperienceSelectionCampaignValidationError(
                "selection request disagrees with scenario, strategy, or seed"
            )
        plan = plans[provenance.selection_plan.path]
        if plan != select_experiences(request):
            raise ExperienceSelectionCampaignValidationError(
                "selection plan does not reproduce from its raw request"
            )
        samples = sample_batches[provenance.training_samples.path]
        expected_samples = _training_samples(scenario, plan.selected_candidate_ids)
        if samples != expected_samples:
            raise ExperienceSelectionCampaignValidationError(
                "training samples disagree with the selected candidate evidence"
            )
        training = training_results[provenance.training_result.path]
        expected_training = trainer.train(
            base=scenario.base_policy,
            samples=samples,
            algorithm=TrainingAlgorithm.SUCCESSFUL_BRANCH_DISTILLATION,
            candidate_policy_epoch_id=training.candidate.policy_epoch_id,
            seed=observation.training_seed,
            steps=scenario.training_steps,
        )
        if training != expected_training:
            raise ExperienceSelectionCampaignValidationError(
                "training result does not reproduce from selected samples and seed"
            )
        trace = evaluations[provenance.evaluation_trace.path]
        expected_trace = _evaluate_downstream(
            scenario,
            strategy=observation.strategy,
            campaign_seed=observation.campaign_seed,
            trained=training.candidate,
            checkpoint_hash=training.checkpoint_hash,
        )
        if trace != expected_trace:
            raise ExperienceSelectionCampaignValidationError(
                "evaluation trace does not reproduce from trained policy and paired seeds"
            )
        if observation != _expected_seed_observation(scenario, observation, plan, trace):
            raise ExperienceSelectionCampaignValidationError(
                "campaign observation disagrees with linked selection, training, or evaluation"
            )
    return run


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("campaign seeds must be comma-separated integers") from exc
    if not seeds:
        raise ValueError("at least two campaign seeds are required")
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic Helix H9 campaign")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", required=True, help="comma-separated explicit seeds")
    arguments = parser.parse_args(argv)
    try:
        run = run_experience_selection_campaign(
            arguments.scenario,
            arguments.output,
            seeds=_parse_seeds(arguments.seeds),
        )
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "campaign_id": run.campaign_id,
                "h9_status": run.h9_result.status,
                "seed_count": len(run.seeds),
                "strategy_seed_observation_count": len(run.observations),
                "raw_artifact_count": len(run.raw_artifacts),
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MAX_CAMPAIGN_ARTIFACT_BYTES",
    "MAX_CAMPAIGN_MANIFEST_BYTES",
    "MAX_CAMPAIGN_SCENARIO_BYTES",
    "ExperienceSelectionCampaignRun",
    "ExperienceSelectionCampaignScenario",
    "ExperienceSelectionCampaignValidationError",
    "load_experience_selection_campaign_scenario",
    "run_experience_selection_campaign",
    "validate_experience_selection_campaign",
]
