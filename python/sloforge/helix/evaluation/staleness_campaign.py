"""Executed H4 campaign for asynchronous policy-provenance boundaries.

The campaign intentionally keeps three decisions separate: staleness
assessment, batch admission, and trainer admission.  In particular, it does
not relabel an off-policy behavior probability as a current-policy probability
to make the strict reference trainer accept it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from sloforge.continuum.ir import ExactnessClass
from sloforge.helix.datasets.reference import (
    BatchSampleProvenance,
    ReferenceTrainingBatchManifest,
)
from sloforge.helix.ir import (
    Digest as IrDigest,
)
from sloforge.helix.ir import (
    LineageReference,
    LineageRelation,
    PolicyEpoch,
)
from sloforge.helix.ir import (
    StalenessDisposition as PortableStalenessDisposition,
)
from sloforge.helix.ir import (
    StalenessReport as PortableStalenessReport,
)
from sloforge.helix.policy import DeterministicPolicy
from sloforge.helix.staleness import (
    ContinuumCompatibilityEvidence,
    DecisionLogProbability,
    IndexRange,
    LogProbabilityRecomputeEvidence,
    LogProbabilitySource,
    PolicyDistanceEvidence,
    PolicySegment,
    PolicySemantics,
    PolicyVersion,
    SampleKind,
    StalenessAssessmentRequest,
    StalenessDisposition,
    StalenessPolicy,
    TrainingEligibility,
    TrajectoryPolicyProvenance,
    TrajectoryStalenessReport,
    TransitionBoundary,
    assess_staleness,
)
from sloforge.helix.trainers import (
    ReferenceTrainer,
    ReferenceTrainingSample,
    TrainingAlgorithm,
    TrainingResult,
)

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
MAX_CAMPAIGN_ARTIFACT_BYTES = 8 * 1024 * 1024


class _CampaignModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class StalenessCase(StrEnum):
    CURRENT_ON_POLICY = "current_on_policy"
    BOUNDED_STALE = "bounded_stale"
    HARD_REJECTED_STALE = "hard_rejected_stale"
    MISSING_BEHAVIOR_LOGPROB = "missing_behavior_log_probability"
    SEGMENTED_MIXED_POLICY = "segmented_mixed_policy"
    RECOMPUTED_LOGPROB = "recomputed_log_probability"
    INCOMPATIBLE_MODEL_STATE = "incompatible_model_state"


CASE_ORDER = (
    StalenessCase.CURRENT_ON_POLICY,
    StalenessCase.BOUNDED_STALE,
    StalenessCase.HARD_REJECTED_STALE,
    StalenessCase.MISSING_BEHAVIOR_LOGPROB,
    StalenessCase.SEGMENTED_MIXED_POLICY,
    StalenessCase.RECOMPUTED_LOGPROB,
    StalenessCase.INCOMPATIBLE_MODEL_STATE,
)


class BatchDisposition(StrEnum):
    ACCEPTED = "accepted"
    STALE_REJECTED = "stale_rejected"
    RECOMPUTE_REQUIRED = "recompute_required"
    STRICT_POLICY_MIXING_REJECTED = "strict_policy_mixing_rejected"
    INCOMPATIBLE_STATE_REJECTED = "incompatible_state_rejected"


class TrainerDisposition(StrEnum):
    TRAINED = "trained"
    BATCH_REJECTED = "batch_rejected"
    REFERENCE_ADAPTER_REJECTED = "reference_adapter_rejected"


class RawArtifact(_CampaignModel):
    artifact_kind: Literal["policy_epochs", "staleness", "batch", "trainer", "outcome"]
    path: Annotated[str, Field(min_length=1, max_length=4096)]
    sha256: Digest

    @model_validator(mode="after")
    def portable_path(self) -> Self:
        parsed = PurePosixPath(self.path)
        if parsed.is_absolute() or ".." in parsed.parts or self.path == ".":
            raise ValueError("H4 raw artifact paths must be portable and relative")
        return self


class RawPolicyEvidence(_CampaignModel):
    schema_version: Literal["sloforge.helix.h4-policy-evidence/v1"] = (
        "sloforge.helix.h4-policy-evidence/v1"
    )
    source_epochs: Annotated[tuple[PolicyEpoch, ...], Field(min_length=2, max_length=2)]
    reference_policies: Annotated[
        tuple[DeterministicPolicy, ...], Field(min_length=2, max_length=2)
    ]
    candidate_epoch: PolicyEpoch | None = None
    candidate_reference_policy: DeterministicPolicy | None = None

    @model_validator(mode="after")
    def bind_portable_and_reference_policies(self) -> Self:
        if tuple(epoch.epoch for epoch in self.source_epochs) != (0, 1):
            raise ValueError("H4 source policy epochs must contain ordered epochs zero and one")
        for epoch, policy in zip(self.source_epochs, self.reference_policies, strict=True):
            if policy.policy_epoch_id != f"{epoch.policy_id}@{epoch.epoch}":
                raise ValueError("portable and reference policy identities disagree")
            if policy.weights_hash != epoch.policy_digest.value:
                raise ValueError("portable policy digest does not bind reference weights")
        if (self.candidate_epoch is None) != (self.candidate_reference_policy is None):
            raise ValueError("portable and reference candidate evidence must be complete")
        if self.candidate_epoch is not None and self.candidate_reference_policy is not None:
            if self.candidate_epoch.epoch != 2:
                raise ValueError("H4 candidate must be epoch two")
            if self.candidate_epoch.policy_id != self.source_epochs[-1].policy_id:
                raise ValueError("H4 candidate changed the logical policy identity")
            if self.candidate_epoch.parent_policy_digest != self.source_epochs[-1].policy_digest:
                raise ValueError("H4 candidate parent digest does not name the learner")
            if (
                self.candidate_reference_policy.policy_epoch_id
                != f"{self.candidate_epoch.policy_id}@{self.candidate_epoch.epoch}"
                or self.candidate_reference_policy.weights_hash
                != self.candidate_epoch.policy_digest.value
            ):
                raise ValueError("portable candidate epoch does not bind candidate weights")
        return self


class RawStalenessEvidence(_CampaignModel):
    schema_version: Literal["sloforge.helix.h4-staleness-evidence/v1"] = (
        "sloforge.helix.h4-staleness-evidence/v1"
    )
    request: StalenessAssessmentRequest
    report: TrajectoryStalenessReport
    portable_lag_reports: tuple[PortableStalenessReport, ...]
    portable_report_omission_reason: str | None = None

    @model_validator(mode="after")
    def bind_assessment(self) -> Self:
        if assess_staleness(self.request) != self.report:
            raise ValueError("raw H4 staleness report does not match its request")
        if bool(self.portable_lag_reports) == (self.portable_report_omission_reason is not None):
            raise ValueError("portable report evidence and omission reason are inconsistent")
        if any(
            item.trajectory_id != self.report.trajectory_id for item in self.portable_lag_reports
        ):
            raise ValueError("portable lag report names a different trajectory")
        return self


class BatchCandidate(_CampaignModel):
    sample_id: Digest
    action: Literal["hold", "improve"]
    decision: DecisionLogProbability
    target_policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    staleness_updates: Annotated[int, Field(ge=0)]
    eligible_by_staleness: bool
    training_sample: ReferenceTrainingSample | None

    @model_validator(mode="after")
    def preserve_probability_provenance(self) -> Self:
        if self.training_sample is None:
            if self.decision.behavior_source is not LogProbabilitySource.MISSING:
                raise ValueError("only a missing behavior probability can omit a training sample")
            return self
        sample = self.training_sample
        if sample.sample_id != self.sample_id or sample.action != self.action:
            raise ValueError("batch candidate and training sample identities disagree")
        if sample.behavior_log_probability != self.decision.behavior_log_probability:
            raise ValueError("batch candidate changed the recorded behavior log probability")
        expected_epoch = (
            f"{self.decision.behavior_policy.policy_id}@{self.decision.behavior_policy.epoch}"
        )
        if sample.policy_epoch_id != expected_epoch:
            raise ValueError("batch candidate relabeled its behavior policy epoch")
        if sample.eligible != self.eligible_by_staleness:
            raise ValueError("training sample eligibility disagrees with staleness evidence")
        return self


class BatchAttemptEvidence(_CampaignModel):
    schema_version: Literal["sloforge.helix.h4-batch-attempt/v1"] = (
        "sloforge.helix.h4-batch-attempt/v1"
    )
    candidates: Annotated[tuple[BatchCandidate, ...], Field(min_length=1, max_length=2)]
    disposition: BatchDisposition
    manifest: ReferenceTrainingBatchManifest | None
    rejection_reason: str | None

    @model_validator(mode="after")
    def valid_admission(self) -> Self:
        accepted = self.disposition is BatchDisposition.ACCEPTED
        if accepted != (self.manifest is not None):
            raise ValueError("batch disposition and manifest presence disagree")
        if accepted == (self.rejection_reason is not None):
            raise ValueError("batch rejection reason presence is inconsistent")
        if self.manifest is not None:
            candidate_ids = tuple(item.sample_id for item in self.candidates)
            manifest_ids = tuple(item.sample.sample_id for item in self.manifest.samples)
            if candidate_ids != manifest_ids:
                raise ValueError("batch manifest changed its candidate sample set")
        return self


class TrainerAttemptEvidence(_CampaignModel):
    schema_version: Literal["sloforge.helix.h4-trainer-attempt/v1"] = (
        "sloforge.helix.h4-trainer-attempt/v1"
    )
    disposition: TrainerDisposition
    base_policy: DeterministicPolicy | None
    result: TrainingResult | None
    rejection_reason: str | None

    @model_validator(mode="after")
    def valid_attempt(self) -> Self:
        trained = self.disposition is TrainerDisposition.TRAINED
        if trained != (self.result is not None):
            raise ValueError("trainer disposition and result presence disagree")
        if trained == (self.rejection_reason is not None):
            raise ValueError("trainer rejection reason presence is inconsistent")
        if self.disposition is TrainerDisposition.BATCH_REJECTED and self.base_policy is not None:
            raise ValueError("a batch-rejected case must not invoke a trainer")
        if self.disposition is not TrainerDisposition.BATCH_REJECTED and self.base_policy is None:
            raise ValueError("an invoked trainer requires its exact base policy")
        if self.result is not None and self.base_policy is not None:
            if self.result.base_policy_epoch_id != self.base_policy.policy_epoch_id:
                raise ValueError("training result names the wrong immutable base policy")
            if any(
                not math.isfinite(value)
                for metric in self.result.metrics
                for value in (metric.objective, metric.policy_kl)
            ):
                raise ValueError("H4 trainer result contains a non-finite metric")
        return self


class TrainingStability(_CampaignModel):
    metric_count: Annotated[int, Field(gt=0, le=256)]
    minimum_objective: float
    maximum_objective: float
    maximum_absolute_objective: Annotated[float, Field(ge=0.0)]
    maximum_policy_kl: Annotated[float, Field(ge=0.0)]
    final_policy_kl: Annotated[float, Field(ge=0.0)]

    @model_validator(mode="after")
    def finite_and_ordered(self) -> Self:
        values = (
            self.minimum_objective,
            self.maximum_objective,
            self.maximum_absolute_objective,
            self.maximum_policy_kl,
            self.final_policy_kl,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("H4 training stability metrics must be finite")
        if self.minimum_objective > self.maximum_objective:
            raise ValueError("H4 training objective bounds are reversed")
        return self


class EvaluatedPolicyOutcome(_CampaignModel):
    success_action: Literal["improve"] = "improve"
    evaluation_seeds: Annotated[tuple[int, ...], Field(min_length=64, max_length=64)]
    base_success_count: Annotated[int, Field(ge=0, le=64)]
    candidate_success_count: Annotated[int, Field(ge=0, le=64)]
    base_success_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    candidate_success_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    paired_success_rate_delta: Annotated[float, Field(ge=-1.0, le=1.0)]
    base_action_probability: Annotated[float, Field(gt=0.0, le=1.0)]
    candidate_action_probability: Annotated[float, Field(gt=0.0, le=1.0)]

    @model_validator(mode="after")
    def valid_outcome(self) -> Self:
        if len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError("H4 policy evaluation seeds must be unique")
        if self.base_success_rate != self.base_success_count / len(self.evaluation_seeds):
            raise ValueError("base evaluated success rate is inconsistent")
        if self.candidate_success_rate != self.candidate_success_count / len(self.evaluation_seeds):
            raise ValueError("candidate evaluated success rate is inconsistent")
        expected_delta = self.candidate_success_rate - self.base_success_rate
        if not math.isclose(
            self.paired_success_rate_delta, expected_delta, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("paired evaluated policy outcome is inconsistent")
        return self


class CaseObservation(_CampaignModel):
    case_id: Digest
    seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    case: StalenessCase
    staleness_report_id: Digest
    staleness_disposition: StalenessDisposition
    training_eligibility: TrainingEligibility
    batch_disposition: BatchDisposition
    trainer_disposition: TrainerDisposition
    invalid_for_training: bool
    invalid_sample_accepted: bool
    training_stability: TrainingStability | None
    evaluated_policy_outcome: EvaluatedPolicyOutcome | None
    raw_artifacts: Annotated[tuple[RawArtifact, ...], Field(min_length=4, max_length=5)]

    @model_validator(mode="after")
    def complete_case(self) -> Self:
        expected_invalid = self.case in {
            StalenessCase.HARD_REJECTED_STALE,
            StalenessCase.MISSING_BEHAVIOR_LOGPROB,
            StalenessCase.SEGMENTED_MIXED_POLICY,
            StalenessCase.INCOMPATIBLE_MODEL_STATE,
        }
        if self.invalid_for_training != expected_invalid:
            raise ValueError("H4 invalid-case classification is inconsistent")
        trained = self.trainer_disposition is TrainerDisposition.TRAINED
        if self.invalid_sample_accepted != (self.invalid_for_training and trained):
            raise ValueError("invalid sample acceptance measurement is inconsistent")
        if trained != (self.training_stability is not None):
            raise ValueError("trained case and stability evidence presence disagree")
        if trained != (self.evaluated_policy_outcome is not None):
            raise ValueError("trained case and evaluated outcome presence disagree")
        expected_kinds = ["policy_epochs", "staleness", "batch", "trainer"]
        if trained:
            expected_kinds.append("outcome")
        if [item.artifact_kind for item in self.raw_artifacts] != expected_kinds:
            raise ValueError("H4 case raw artifact matrix is incomplete or unordered")
        paths = [item.path for item in self.raw_artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("H4 case raw artifact paths must be unique")
        identity = self.model_dump(mode="json", exclude={"case_id"})
        if _digest(identity) != self.case_id:
            raise ValueError("H4 case identity is invalid")
        return self


class StalenessCampaign(_CampaignModel):
    schema_version: Literal["sloforge.helix.staleness-campaign/v1"] = (
        "sloforge.helix.staleness-campaign/v1"
    )
    campaign_id: Digest
    seeds: Annotated[tuple[int, ...], Field(min_length=2, max_length=32)]
    observations: Annotated[tuple[CaseObservation, ...], Field(min_length=14, max_length=224)]
    invalid_case_count: Annotated[int, Field(ge=0)]
    invalid_sample_acceptance_count: Annotated[int, Field(ge=0)]
    invalid_sample_acceptance_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    staleness_eligible_case_count: Annotated[int, Field(ge=0)]
    trained_case_count: Annotated[int, Field(ge=0)]
    finite_training_case_count: Annotated[int, Field(ge=0)]
    evaluated_policy_outcome_count: Annotated[int, Field(ge=0)]
    limitations: Annotated[tuple[str, ...], Field(min_length=3)]

    @model_validator(mode="after")
    def complete_campaign(self) -> Self:
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("H4 campaign seeds must be unique")
        expected = tuple((seed, case) for seed in self.seeds for case in CASE_ORDER)
        observed = tuple((item.seed, item.case) for item in self.observations)
        if observed != expected:
            raise ValueError("H4 campaign must preserve the complete ordered seed/case matrix")
        invalid = tuple(item for item in self.observations if item.invalid_for_training)
        invalid_accepted = sum(item.invalid_sample_accepted for item in invalid)
        trained = tuple(
            item
            for item in self.observations
            if item.trainer_disposition is TrainerDisposition.TRAINED
        )
        staleness_eligible = sum(
            item.training_eligibility is not TrainingEligibility.INELIGIBLE
            for item in self.observations
        )
        expected_values = (
            self.invalid_case_count == len(invalid),
            self.invalid_sample_acceptance_count == invalid_accepted,
            self.invalid_sample_acceptance_rate
            == (invalid_accepted / len(invalid) if invalid else 0.0),
            self.staleness_eligible_case_count == staleness_eligible,
            self.trained_case_count == len(trained),
            self.finite_training_case_count
            == sum(item.training_stability is not None for item in self.observations),
            self.evaluated_policy_outcome_count
            == sum(item.evaluated_policy_outcome is not None for item in self.observations),
        )
        if not all(expected_values):
            raise ValueError("H4 campaign aggregate measurements are inconsistent")
        if _digest(self.model_dump(mode="json", exclude={"campaign_id"})) != self.campaign_id:
            raise ValueError("H4 campaign identity is invalid")
        return self


@dataclass(frozen=True)
class _PolicyContext:
    portable_old: PolicyEpoch
    portable_current: PolicyEpoch
    old: DeterministicPolicy
    current: DeterministicPolicy
    version_old: PolicyVersion
    version_current: PolicyVersion


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _campaign_artifact_path(output: Path, relative: str) -> Path:
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or relative == ".":
        raise ValueError("H4 campaign artifact path is not portable")
    candidate = output.joinpath(*parsed.parts)
    cursor = output
    for part in parsed.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("H4 campaign artifact path contains a symbolic link")
    resolved_root = output.resolve()
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError("H4 campaign artifact escapes its output root")
    if resolved.stat().st_size > MAX_CAMPAIGN_ARTIFACT_BYTES:
        raise ValueError("H4 campaign artifact exceeds the byte limit")
    return resolved


def _label_digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _derived_seed(seed: int, label: str, ordinal: int = 0) -> int:
    payload = f"sloforge.helix.h4/v1\0{seed}\0{label}\0{ordinal}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _write(
    output: Path,
    path: Path,
    artifact_kind: Literal["policy_epochs", "staleness", "batch", "trainer", "outcome"],
    value: BaseModel,
) -> RawArtifact:
    payload = _canonical_bytes(value.model_dump(mode="json")) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return RawArtifact(
        artifact_kind=artifact_kind,
        path=path.relative_to(output).as_posix(),
        sha256=sha256(payload).hexdigest(),
    )


def _lineage(artifact_id: str) -> LineageReference:
    return LineageReference(
        artifact_id=artifact_id,
        artifact_kind="sloforge.helix.h4/evaluation-fixture",
        relation=LineageRelation.DERIVED_FROM,
        digest=IrDigest(value=_label_digest(f"lineage:{artifact_id}")),
    )


def _policy_context(seed: int, case: StalenessCase) -> _PolicyContext:
    policy_id = f"h4-{seed}-{case.value}"
    old_logits = (2.0, -2.0) if case is StalenessCase.HARD_REJECTED_STALE else (0.2, -0.2)
    old = DeterministicPolicy(
        policy_epoch_id=f"{policy_id}@0",
        actions=("hold", "improve"),
        logits=old_logits,
    )
    current = DeterministicPolicy(
        policy_epoch_id=f"{policy_id}@1",
        actions=old.actions,
        logits=(0.0, 0.0),
    )
    portable_old = PolicyEpoch(
        policy_id=policy_id,
        epoch=0,
        policy_digest=IrDigest(value=old.weights_hash),
        created_at="2026-08-09T00:00:00Z",
        lineage=(_lineage(f"{policy_id}-root"),),
    )
    portable_current = PolicyEpoch(
        policy_id=policy_id,
        epoch=1,
        policy_digest=IrDigest(value=current.weights_hash),
        parent_epoch=0,
        parent_policy_digest=portable_old.policy_digest,
        training_transaction_id=f"h4-{seed}-{case.value}-source-update",
        created_at="2026-08-09T00:01:00Z",
        lineage=(_lineage(f"{policy_id}@0"),),
    )
    version_old = PolicyVersion(
        policy_id=policy_id,
        epoch=0,
        update=0,
        parameter_digest=old.weights_hash,
        continuum_compatibility_fingerprint=_label_digest(f"compatibility:{policy_id}:0"),
        published_at_ms=0,
    )
    version_current = PolicyVersion(
        policy_id=policy_id,
        epoch=1,
        update=1,
        parameter_digest=current.weights_hash,
        continuum_compatibility_fingerprint=_label_digest(f"compatibility:{policy_id}:1"),
        published_at_ms=100,
    )
    return _PolicyContext(
        portable_old=portable_old,
        portable_current=portable_current,
        old=old,
        current=current,
        version_old=version_old,
        version_current=version_current,
    )


def _segment(index: int, policy: PolicyVersion, start: int, end: int) -> PolicySegment:
    return PolicySegment(
        segment_id=f"segment-{index}",
        segment_index=index,
        policy=policy,
        token_range=IndexRange(start=0, end_exclusive=0),
        action_range=IndexRange(start=start, end_exclusive=end),
        collected_at_ms=1_000 + index * 100,
        completed_at_ms=1_050 + index * 100,
    )


def _recorded(
    *,
    seed: int,
    case: StalenessCase,
    index: int,
    behavior: PolicyVersion,
    behavior_log_probability: float,
    target: PolicyVersion | None = None,
    target_log_probability: float | None = None,
) -> DecisionLogProbability:
    return DecisionLogProbability(
        sample_kind=SampleKind.ACTION,
        sample_index=index,
        behavior_policy=behavior,
        behavior_log_probability=behavior_log_probability,
        behavior_source=LogProbabilitySource.RECORDED,
        behavior_evidence_digest=_label_digest(
            f"behavior:{seed}:{case.value}:{index}:{behavior.epoch}"
        ),
        target_policy=target,
        target_log_probability=target_log_probability,
        target_source=(
            LogProbabilitySource.RECORDED if target is not None else LogProbabilitySource.MISSING
        ),
        target_evidence_digest=(
            _label_digest(f"target:{seed}:{case.value}:{index}:{target.epoch}")
            if target is not None
            else None
        ),
    )


def _recomputed(
    *, seed: int, case: StalenessCase, policy: PolicyVersion, value: float
) -> DecisionLogProbability:
    result_digest = _digest(
        {
            "action": "improve",
            "log_probability": value,
            "policy_digest": policy.parameter_digest,
        }
    )
    evidence = LogProbabilityRecomputeEvidence(
        recomputation_id=f"h4-recompute-{seed}",
        subject_kind=SampleKind.ACTION,
        subject_index=0,
        policy=policy,
        token_history_digest=_label_digest(f"history:{seed}:{case.value}"),
        implementation_digest=_label_digest("h4-reference-logprob-recompute/v1"),
        result_digest=result_digest,
        seed=_derived_seed(seed, case.value),
        recomputed_at_ms=1_500,
    )
    return DecisionLogProbability(
        sample_kind=SampleKind.ACTION,
        sample_index=0,
        behavior_policy=policy,
        behavior_log_probability=value,
        behavior_source=LogProbabilitySource.RECOMPUTED,
        behavior_evidence_digest=result_digest,
        behavior_recomputation=evidence,
    )


def _distance(context: _PolicyContext) -> PolicyDistanceEvidence:
    parameter_delta_l2 = math.sqrt(
        math.fsum(
            (old - current) ** 2
            for old, current in zip(context.old.logits, context.current.logits, strict=True)
        )
    )
    kl_divergence = math.fsum(
        old_probability * (math.log(old_probability) - math.log(current_probability))
        for old_probability, current_probability in zip(
            context.old.probabilities(), context.current.probabilities(), strict=True
        )
    )
    measurement = {
        "behavior_policy_digest": context.old.weights_hash,
        "learner_policy_digest": context.current.weights_hash,
        "parameter_delta_l2": parameter_delta_l2,
        "kl_divergence": kl_divergence,
        "sample_count": len(context.old.actions),
    }
    return PolicyDistanceEvidence(
        behavior_policy=context.version_old,
        learner_policy=context.version_current,
        parameter_delta_l2=parameter_delta_l2,
        kl_divergence=kl_divergence,
        sample_count=len(context.old.actions),
        evidence_digest=_digest(measurement),
    )


def _boundary(
    context: _PolicyContext,
    seed: int,
    case: StalenessCase,
    *,
    incompatible: bool,
) -> TransitionBoundary:
    exactness = ExactnessClass.INCOMPATIBLE if incompatible else ExactnessClass.EXACT_SEMANTIC
    compatibility = ContinuumCompatibilityEvidence(
        report_id=f"h4-state-{seed}-{case.value}",
        report_digest=_label_digest(f"state:{seed}:{case.value}:{exactness.value}"),
        compatibility_class=exactness,
        safe=not incompatible,
        source_compatibility_fingerprint=(context.version_old.continuum_compatibility_fingerprint),
        destination_compatibility_fingerprint=(
            context.version_current.continuum_compatibility_fingerprint
        ),
    )
    return TransitionBoundary(
        boundary_id=f"h4-boundary-{seed}",
        from_segment_index=0,
        to_segment_index=1,
        token_index=0,
        action_index=1,
        from_policy=context.version_old,
        to_policy=context.version_current,
        compatibility=compatibility,
    )


def _assessment_request(
    seed: int, case: StalenessCase, context: _PolicyContext
) -> tuple[StalenessAssessmentRequest, tuple[str, ...]]:
    old_improve = context.old.log_probability("improve")
    current_improve = context.current.log_probability("improve")
    if case in {StalenessCase.SEGMENTED_MIXED_POLICY, StalenessCase.INCOMPATIBLE_MODEL_STATE}:
        records = (
            _recorded(
                seed=seed,
                case=case,
                index=0,
                behavior=context.version_old,
                behavior_log_probability=context.old.log_probability("hold"),
                target=context.version_current,
                target_log_probability=context.current.log_probability("hold"),
            ),
            _recorded(
                seed=seed,
                case=case,
                index=1,
                behavior=context.version_current,
                behavior_log_probability=current_improve,
            ),
        )
        trajectory = TrajectoryPolicyProvenance(
            trajectory_id=_label_digest(f"trajectory:{seed}:{case.value}"),
            semantics=PolicySemantics.SEGMENTED,
            token_count=0,
            action_count=2,
            segments=(
                _segment(0, context.version_old, 0, 1),
                _segment(1, context.version_current, 1, 2),
            ),
            transitions=(
                _boundary(
                    context,
                    seed,
                    case,
                    incompatible=case is StalenessCase.INCOMPATIBLE_MODEL_STATE,
                ),
            ),
            log_probabilities=records,
            trace_evidence_digest=_label_digest(f"trace:{seed}:{case.value}"),
        )
        return (
            StalenessAssessmentRequest(
                trajectory=trajectory,
                learner_policy=context.version_current,
                policy=StalenessPolicy(policy_id=f"h4-policy-{case.value}"),
                distance_evidence=(_distance(context),),
                assessed_at_ms=2_000,
                seed=seed,
            ),
            ("hold", "improve"),
        )

    behavior = (
        context.version_old
        if case in {StalenessCase.BOUNDED_STALE, StalenessCase.HARD_REJECTED_STALE}
        else context.version_current
    )
    if case is StalenessCase.MISSING_BEHAVIOR_LOGPROB:
        record = DecisionLogProbability(
            sample_kind=SampleKind.ACTION,
            sample_index=0,
            behavior_policy=behavior,
        )
    elif case is StalenessCase.RECOMPUTED_LOGPROB:
        record = _recomputed(
            seed=seed,
            case=case,
            policy=behavior,
            value=current_improve,
        )
    elif behavior == context.version_old:
        record = _recorded(
            seed=seed,
            case=case,
            index=0,
            behavior=behavior,
            behavior_log_probability=old_improve,
            target=context.version_current,
            target_log_probability=current_improve,
        )
    else:
        record = _recorded(
            seed=seed,
            case=case,
            index=0,
            behavior=behavior,
            behavior_log_probability=current_improve,
        )
    trajectory = TrajectoryPolicyProvenance(
        trajectory_id=_label_digest(f"trajectory:{seed}:{case.value}"),
        semantics=PolicySemantics.STRICT,
        token_count=0,
        action_count=1,
        segments=(_segment(0, behavior, 0, 1),),
        log_probabilities=(record,),
        trace_evidence_digest=_label_digest(f"trace:{seed}:{case.value}"),
    )
    if case is StalenessCase.BOUNDED_STALE:
        staleness_policy = StalenessPolicy(
            policy_id="h4-bounded-stale",
            max_epoch_distance=0,
            max_update_distance=0,
            stale_disposition=StalenessDisposition.PRIORITY_REDUCTION,
        )
    elif case is StalenessCase.HARD_REJECTED_STALE:
        staleness_policy = StalenessPolicy(
            policy_id="h4-hard-stale",
            max_epoch_distance=0,
            max_update_distance=0,
            max_parameter_delta_l2=0.2,
            max_kl_divergence=0.1,
        )
    else:
        staleness_policy = StalenessPolicy(policy_id=f"h4-policy-{case.value}")
    distance = (_distance(context),) if behavior == context.version_old else ()
    return (
        StalenessAssessmentRequest(
            trajectory=trajectory,
            learner_policy=context.version_current,
            policy=staleness_policy,
            distance_evidence=distance,
            assessed_at_ms=2_000,
            seed=seed,
        ),
        ("improve",),
    )


def _sample_id(seed: int, case: StalenessCase, index: int) -> str:
    return _label_digest(f"sample:{seed}:{case.value}:{index}")


def _portable_reports(
    seed: int,
    case: StalenessCase,
    context: _PolicyContext,
    report: TrajectoryStalenessReport,
) -> tuple[tuple[PortableStalenessReport, ...], str | None]:
    if case is StalenessCase.MISSING_BEHAVIOR_LOGPROB:
        return (), "portable lag report would erase the missing behavior-log-probability state"
    if case is StalenessCase.INCOMPATIBLE_MODEL_STATE:
        return (), "portable lag report cannot encode incompatible Continuum transition evidence"
    epochs = {0: context.portable_old, 1: context.portable_current}
    reports: list[PortableStalenessReport] = []
    for segment in report.segment_reports:
        behavior = epochs[segment.behavior_policy.epoch]
        stale = segment.stale
        if segment.disposition is StalenessDisposition.HARD_REJECT:
            disposition = PortableStalenessDisposition.REJECT
            weight = None
        elif segment.disposition is StalenessDisposition.PRIORITY_REDUCTION:
            disposition = PortableStalenessDisposition.REWEIGHT
            weights = segment.importance_weights
            weight = weights[0].applied_ratio if weights else segment.priority
        else:
            disposition = PortableStalenessDisposition.ACCEPT
            weight = None
        sample_id = _sample_id(seed, case, segment.segment_index)
        lineage_ids = tuple(
            dict.fromkeys(
                (
                    sample_id,
                    report.trajectory_id,
                    f"{behavior.policy_id}@{behavior.epoch}",
                    f"{context.portable_current.policy_id}@{context.portable_current.epoch}",
                )
            )
        )
        reports.append(
            PortableStalenessReport(
                report_id=f"h4-lag-{seed}-{case.value}-{segment.segment_index}",
                sample_id=sample_id,
                trajectory_id=report.trajectory_id,
                behavior_policy_epoch=behavior,
                learner_policy_epoch=context.portable_current,
                epoch_lag=context.portable_current.epoch - behavior.epoch,
                maximum_allowed_lag=0 if stale else 1,
                stale=stale,
                disposition=disposition,
                importance_sampling_weight=weight,
                assessed_at="2026-08-09T00:02:00Z",
                lineage=tuple(_lineage(item) for item in lineage_ids),
            )
        )
    return tuple(reports), None


def _candidates_from_request(
    seed: int,
    case: StalenessCase,
    actions: tuple[str, ...],
    request: StalenessAssessmentRequest,
    report: TrajectoryStalenessReport,
) -> tuple[BatchCandidate, ...]:
    records = tuple(
        sorted(
            request.trajectory.log_probabilities,
            key=lambda item: (item.sample_kind.value, item.sample_index),
        )
    )
    candidates: list[BatchCandidate] = []
    for index, (decision, action) in enumerate(zip(records, actions, strict=True)):
        segment = next(item for item in report.segment_reports if item.action_range.contains(index))
        eligible = report.training_eligible and segment.training_eligible
        sample_id = _sample_id(seed, case, index)
        value = decision.behavior_log_probability
        sample = (
            ReferenceTrainingSample(
                sample_id=sample_id,
                trajectory_id=report.trajectory_id,
                branch_point_id=f"h4-point-{seed}-{case.value}",
                branch_group_id=f"h4-group-{seed}-{case.value}",
                policy_epoch_id=(
                    f"{decision.behavior_policy.policy_id}@{decision.behavior_policy.epoch}"
                ),
                action=action,
                behavior_log_probability=value,
                advantage=1.0 if action == "improve" else -0.25,
                token_weight=1.0,
                staleness_updates=(report.learner_policy.update - decision.behavior_policy.update),
                eligible=eligible,
            )
            if value is not None
            else None
        )
        candidates.append(
            BatchCandidate(
                sample_id=sample_id,
                action=cast(Literal["hold", "improve"], action),
                decision=decision,
                target_policy_epoch_id=(
                    f"{report.learner_policy.policy_id}@{report.learner_policy.epoch}"
                ),
                staleness_updates=(report.learner_policy.update - decision.behavior_policy.update),
                eligible_by_staleness=eligible,
                training_sample=sample,
            )
        )
    return tuple(candidates)


def _manifest(
    seed: int,
    case: StalenessCase,
    candidates: tuple[BatchCandidate, ...],
    learner_policy_epoch_id: str,
) -> ReferenceTrainingBatchManifest:
    if any(item.training_sample is None for item in candidates):
        raise ValueError("batch schema requires every candidate behavior log probability")
    training_samples = cast(
        tuple[ReferenceTrainingSample, ...],
        tuple(item.training_sample for item in candidates),
    )
    credit_hash = _label_digest(f"credit:{seed}:{case.value}")
    provenance = tuple(
        BatchSampleProvenance(
            sample=sample,
            tenant_id="h4-evaluation",
            branch_id=f"h4-branch-{seed}-{case.value}-{index}",
            trajectory_hash=sample.trajectory_id,
            reward_id=f"h4-reward-{seed}-{case.value}-{index}",
            reward_hash=_label_digest(f"reward:{seed}:{case.value}:{index}"),
            credit_hash=credit_hash,
            environment_capsule_id=_label_digest(f"environment:{seed}:{case.value}:{index}"),
            source_model_capsule_id=f"h4-model-{seed}-{case.value}",
            state_reuse_report_hash=_label_digest(f"state-reuse:{seed}:{case.value}:{index}"),
            staleness_disposition="accepted",
        )
        for index, sample in enumerate(training_samples)
    )
    data_hash = _digest([item.model_dump(mode="json") for item in provenance])
    draft = ReferenceTrainingBatchManifest.model_construct(
        batch_id="0" * 64,
        tenant_id="h4-evaluation",
        branch_group_id=f"h4-group-{seed}-{case.value}",
        branch_point_id=f"h4-point-{seed}-{case.value}",
        behavior_policy_epoch_id=training_samples[0].policy_epoch_id,
        learner_policy_epoch_id=learner_policy_epoch_id,
        algorithm=TrainingAlgorithm.BRANCH_RELATIVE,
        samples=provenance,
        training_sample_ids=tuple(item.sample_id for item in training_samples),
        holdout_trajectory_ids=(),
        excluded_trajectory_ids=(),
        credit_hash=credit_hash,
        data_hash=data_hash,
        creation_code_version="sloforge.helix.h4-staleness-campaign/v1",
        seed=seed,
    )
    return ReferenceTrainingBatchManifest.model_validate(
        {
            **draft.model_dump(),
            "batch_id": _digest(draft.model_dump(mode="json", exclude={"batch_id"})),
        },
        strict=True,
    )


def _batch_attempt(
    seed: int,
    case: StalenessCase,
    candidates: tuple[BatchCandidate, ...],
    report: TrajectoryStalenessReport,
) -> BatchAttemptEvidence:
    if not report.training_eligible:
        if case is StalenessCase.MISSING_BEHAVIOR_LOGPROB:
            disposition = BatchDisposition.RECOMPUTE_REQUIRED
        elif case is StalenessCase.INCOMPATIBLE_MODEL_STATE:
            disposition = BatchDisposition.INCOMPATIBLE_STATE_REJECTED
        else:
            disposition = BatchDisposition.STALE_REJECTED
        return BatchAttemptEvidence(
            candidates=candidates,
            disposition=disposition,
            manifest=None,
            rejection_reason=(
                f"staleness gate: {report.primary_disposition.value}; "
                + ", ".join(reason.code.value for reason in report.reasons)
            ),
        )
    learner_id = f"{report.learner_policy.policy_id}@{report.learner_policy.epoch}"
    try:
        manifest = _manifest(seed, case, candidates, learner_id)
    except (ValidationError, ValueError) as exc:
        return BatchAttemptEvidence(
            candidates=candidates,
            disposition=BatchDisposition.STRICT_POLICY_MIXING_REJECTED,
            manifest=None,
            rejection_reason=str(exc),
        )
    return BatchAttemptEvidence(
        candidates=candidates,
        disposition=BatchDisposition.ACCEPTED,
        manifest=manifest,
        rejection_reason=None,
    )


def _trainer_attempt(
    seed: int,
    case: StalenessCase,
    context: _PolicyContext,
    batch: BatchAttemptEvidence,
) -> TrainerAttemptEvidence:
    if batch.manifest is None:
        return TrainerAttemptEvidence(
            disposition=TrainerDisposition.BATCH_REJECTED,
            base_policy=None,
            result=None,
            rejection_reason=f"batch gate: {batch.disposition.value}: {batch.rejection_reason}",
        )
    try:
        result = ReferenceTrainer(
            learning_rate=0.35,
            kl_coefficient=0.05,
            ratio_clip=0.2,
            maximum_staleness_updates=2,
            maximum_steps=32,
        ).train(
            base=context.current,
            samples=batch.manifest.trainer_samples(),
            algorithm=batch.manifest.algorithm,
            candidate_policy_epoch_id=f"{context.portable_current.policy_id}@2",
            seed=seed,
            steps=8,
        )
    except ValueError as exc:
        return TrainerAttemptEvidence(
            disposition=TrainerDisposition.REFERENCE_ADAPTER_REJECTED,
            base_policy=context.current,
            result=None,
            rejection_reason=str(exc),
        )
    return TrainerAttemptEvidence(
        disposition=TrainerDisposition.TRAINED,
        base_policy=context.current,
        result=result,
        rejection_reason=None,
    )


def _candidate_epoch(
    context: _PolicyContext, trainer: TrainerAttemptEvidence
) -> PolicyEpoch | None:
    if trainer.result is None:
        return None
    return PolicyEpoch(
        policy_id=context.portable_current.policy_id,
        epoch=2,
        policy_digest=IrDigest(value=trainer.result.candidate.weights_hash),
        parent_epoch=1,
        parent_policy_digest=context.portable_current.policy_digest,
        training_transaction_id=(
            f"h4-training-{context.portable_current.policy_id}-{trainer.result.seed}"
        ),
        created_at="2026-08-09T00:03:00Z",
        lineage=(_lineage(f"{context.portable_current.policy_id}@1"),),
    )


def _stability(trainer: TrainerAttemptEvidence) -> TrainingStability | None:
    if trainer.result is None:
        return None
    objectives = tuple(item.objective for item in trainer.result.metrics)
    policy_kls = tuple(item.policy_kl for item in trainer.result.metrics)
    return TrainingStability(
        metric_count=len(objectives),
        minimum_objective=min(objectives),
        maximum_objective=max(objectives),
        maximum_absolute_objective=max(abs(value) for value in objectives),
        maximum_policy_kl=max(policy_kls),
        final_policy_kl=policy_kls[-1],
    )


def _outcome(
    seed: int,
    case: StalenessCase,
    trainer: TrainerAttemptEvidence,
) -> EvaluatedPolicyOutcome | None:
    if trainer.result is None or trainer.base_policy is None:
        return None
    evaluation_seeds = tuple(
        _derived_seed(seed, f"outcome:{case.value}", index) for index in range(64)
    )
    base_count = 0
    candidate_count = 0
    for index, evaluation_seed in enumerate(evaluation_seeds):
        observation = f"h4-heldout:{case.value}:{index % 11}"
        base_count += (
            trainer.base_policy.decide(
                observation,
                seed=evaluation_seed,
                rng_counter=index,
            ).action
            == "improve"
        )
        candidate_count += (
            trainer.result.candidate.decide(
                observation,
                seed=evaluation_seed,
                rng_counter=index,
            ).action
            == "improve"
        )
    base_rate = base_count / len(evaluation_seeds)
    candidate_rate = candidate_count / len(evaluation_seeds)
    return EvaluatedPolicyOutcome(
        evaluation_seeds=evaluation_seeds,
        base_success_count=base_count,
        candidate_success_count=candidate_count,
        base_success_rate=base_rate,
        candidate_success_rate=candidate_rate,
        paired_success_rate_delta=candidate_rate - base_rate,
        base_action_probability=trainer.base_policy.probabilities()[1],
        candidate_action_probability=trainer.result.candidate.probabilities()[1],
    )


def _case_observation(output: Path, seed: int, case: StalenessCase) -> CaseObservation:
    context = _policy_context(seed, case)
    request, actions = _assessment_request(seed, case, context)
    report = assess_staleness(request)
    portable_reports, omission = _portable_reports(seed, case, context, report)
    candidates = _candidates_from_request(seed, case, actions, request, report)
    batch = _batch_attempt(seed, case, candidates, report)
    trainer = _trainer_attempt(seed, case, context, batch)
    candidate_epoch = _candidate_epoch(context, trainer)
    policy_evidence = RawPolicyEvidence(
        source_epochs=(context.portable_old, context.portable_current),
        reference_policies=(context.old, context.current),
        candidate_epoch=candidate_epoch,
        candidate_reference_policy=(
            trainer.result.candidate if trainer.result is not None else None
        ),
    )
    staleness_evidence = RawStalenessEvidence(
        request=request,
        report=report,
        portable_lag_reports=portable_reports,
        portable_report_omission_reason=omission,
    )
    stability = _stability(trainer)
    outcome = _outcome(seed, case, trainer)
    raw_directory = output / "raw" / f"seed-{seed}" / case.value
    artifacts = [
        _write(
            output,
            raw_directory / "policy_epochs.json",
            "policy_epochs",
            policy_evidence,
        ),
        _write(output, raw_directory / "staleness.json", "staleness", staleness_evidence),
        _write(output, raw_directory / "batch.json", "batch", batch),
        _write(output, raw_directory / "trainer.json", "trainer", trainer),
    ]
    if outcome is not None:
        artifacts.append(_write(output, raw_directory / "outcome.json", "outcome", outcome))
    invalid = case in {
        StalenessCase.HARD_REJECTED_STALE,
        StalenessCase.MISSING_BEHAVIOR_LOGPROB,
        StalenessCase.SEGMENTED_MIXED_POLICY,
        StalenessCase.INCOMPATIBLE_MODEL_STATE,
    }
    draft = CaseObservation.model_construct(
        case_id="0" * 64,
        seed=seed,
        case=case,
        staleness_report_id=report.report_id,
        staleness_disposition=report.primary_disposition,
        training_eligibility=report.training_eligibility,
        batch_disposition=batch.disposition,
        trainer_disposition=trainer.disposition,
        invalid_for_training=invalid,
        invalid_sample_accepted=invalid and trainer.result is not None,
        training_stability=stability,
        evaluated_policy_outcome=outcome,
        raw_artifacts=tuple(artifacts),
    )
    identity = draft.model_dump(mode="json", exclude={"case_id"})
    return CaseObservation(
        case_id=_digest(identity),
        seed=seed,
        case=case,
        staleness_report_id=report.report_id,
        staleness_disposition=report.primary_disposition,
        training_eligibility=report.training_eligibility,
        batch_disposition=batch.disposition,
        trainer_disposition=trainer.disposition,
        invalid_for_training=invalid,
        invalid_sample_accepted=invalid and trainer.result is not None,
        training_stability=stability,
        evaluated_policy_outcome=outcome,
        raw_artifacts=tuple(artifacts),
    )


def run_staleness_campaign(
    output: Path,
    *,
    seeds: tuple[int, ...] = (41, 73, 113),
) -> StalenessCampaign:
    """Execute the deterministic seven-case H4 matrix and retain every raw boundary."""

    if not 2 <= len(seeds) <= 32 or len(seeds) != len(set(seeds)):
        raise ValueError("H4 staleness campaign requires two to 32 unique seeds")
    if any(seed < 0 or seed > 2**63 - 1 for seed in seeds):
        raise ValueError("H4 staleness campaign seeds must fit signed 64-bit integers")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("H4 staleness campaign output must be absent or empty")

    observations = tuple(
        _case_observation(output, seed, case) for seed in seeds for case in CASE_ORDER
    )
    invalid = tuple(item for item in observations if item.invalid_for_training)
    invalid_accepted = sum(item.invalid_sample_accepted for item in invalid)
    trained = tuple(
        item for item in observations if item.trainer_disposition is TrainerDisposition.TRAINED
    )
    draft = StalenessCampaign.model_construct(
        campaign_id="0" * 64,
        seeds=seeds,
        observations=observations,
        invalid_case_count=len(invalid),
        invalid_sample_acceptance_count=invalid_accepted,
        invalid_sample_acceptance_rate=(invalid_accepted / len(invalid) if invalid else 0.0),
        staleness_eligible_case_count=sum(
            item.training_eligibility is not TrainingEligibility.INELIGIBLE for item in observations
        ),
        trained_case_count=len(trained),
        finite_training_case_count=sum(
            item.training_stability is not None for item in observations
        ),
        evaluated_policy_outcome_count=sum(
            item.evaluated_policy_outcome is not None for item in observations
        ),
        limitations=(
            "The campaign uses a synthetic categorical CPU policy and deterministic fixture outcomes.",
            "The strict reference trainer cannot apply bounded off-policy or segmented mixed-policy evidence to the current learner checkpoint; those adapter rejections are retained rather than normalized away.",
            "Finite metrics over eight tiny reference steps do not establish stability for a large asynchronous optimizer.",
            "Portable v1 lag reports cannot encode missing log probabilities or incompatible Continuum transitions; the richer raw staleness request and report remain authoritative.",
        ),
    )
    identity = draft.model_dump(mode="json", exclude={"campaign_id"})
    campaign = StalenessCampaign(
        campaign_id=_digest(identity),
        seeds=seeds,
        observations=observations,
        invalid_case_count=len(invalid),
        invalid_sample_acceptance_count=invalid_accepted,
        invalid_sample_acceptance_rate=(invalid_accepted / len(invalid) if invalid else 0.0),
        staleness_eligible_case_count=sum(
            item.training_eligibility is not TrainingEligibility.INELIGIBLE for item in observations
        ),
        trained_case_count=len(trained),
        finite_training_case_count=sum(
            item.training_stability is not None for item in observations
        ),
        evaluated_policy_outcome_count=sum(
            item.evaluated_policy_outcome is not None for item in observations
        ),
        limitations=draft.limitations,
    )
    payload = _canonical_bytes(campaign.model_dump(mode="json")) + b"\n"
    campaign_path = output / "campaign.json"
    campaign_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_path.write_bytes(payload)
    return campaign


def validate_staleness_campaign(output: Path) -> StalenessCampaign:
    """Reopen a published campaign and validate every provenance artifact."""

    campaign_path = _campaign_artifact_path(output, "campaign.json")
    campaign = StalenessCampaign.model_validate_json(campaign_path.read_bytes(), strict=True)
    parsers: dict[str, type[BaseModel]] = {
        "policy_epochs": RawPolicyEvidence,
        "staleness": RawStalenessEvidence,
        "batch": BatchAttemptEvidence,
        "trainer": TrainerAttemptEvidence,
        "outcome": EvaluatedPolicyOutcome,
    }
    seen: set[str] = set()
    for observation in campaign.observations:
        loaded: dict[str, BaseModel] = {}
        for artifact in observation.raw_artifacts:
            if artifact.path in seen:
                raise ValueError("H4 campaign reused a raw artifact path")
            seen.add(artifact.path)
            path = _campaign_artifact_path(output, artifact.path)
            payload = path.read_bytes()
            if sha256(payload).hexdigest() != artifact.sha256:
                raise ValueError("H4 campaign raw artifact digest mismatch")
            loaded[artifact.artifact_kind] = parsers[artifact.artifact_kind].model_validate_json(
                payload, strict=True
            )
        staleness = loaded["staleness"]
        batch = loaded["batch"]
        trainer = loaded["trainer"]
        policies = loaded["policy_epochs"]
        if not isinstance(policies, RawPolicyEvidence):
            raise TypeError("H4 policy artifact parsed to the wrong evidence type")
        if not isinstance(staleness, RawStalenessEvidence):
            raise TypeError("H4 staleness artifact parsed to the wrong evidence type")
        if not isinstance(batch, BatchAttemptEvidence):
            raise TypeError("H4 batch artifact parsed to the wrong evidence type")
        if not isinstance(trainer, TrainerAttemptEvidence):
            raise TypeError("H4 trainer artifact parsed to the wrong evidence type")
        if (
            staleness.report.report_id != observation.staleness_report_id
            or staleness.report.primary_disposition is not observation.staleness_disposition
            or staleness.report.training_eligibility is not observation.training_eligibility
            or batch.disposition is not observation.batch_disposition
            or trainer.disposition is not observation.trainer_disposition
        ):
            raise ValueError("H4 campaign observation disagrees with raw boundary evidence")
        policy_by_epoch = {
            item.policy_epoch_id: item
            for item in (
                *policies.reference_policies,
                *(
                    (policies.candidate_reference_policy,)
                    if policies.candidate_reference_policy is not None
                    else ()
                ),
            )
        }
        for distance in staleness.request.distance_evidence:
            behavior_id = f"{distance.behavior_policy.policy_id}@{distance.behavior_policy.epoch}"
            learner_id = f"{distance.learner_policy.policy_id}@{distance.learner_policy.epoch}"
            behavior_policy = policy_by_epoch[behavior_id]
            learner_policy = policy_by_epoch[learner_id]
            expected_l2 = math.sqrt(
                math.fsum(
                    (left - right) ** 2
                    for left, right in zip(
                        behavior_policy.logits, learner_policy.logits, strict=True
                    )
                )
            )
            expected_kl = math.fsum(
                left * (math.log(left) - math.log(right))
                for left, right in zip(
                    behavior_policy.probabilities(),
                    learner_policy.probabilities(),
                    strict=True,
                )
            )
            measurement = {
                "behavior_policy_digest": behavior_policy.weights_hash,
                "learner_policy_digest": learner_policy.weights_hash,
                "parameter_delta_l2": expected_l2,
                "kl_divergence": expected_kl,
                "sample_count": len(behavior_policy.actions),
            }
            if (
                not math.isclose(
                    distance.parameter_delta_l2, expected_l2, rel_tol=0.0, abs_tol=1e-12
                )
                or not math.isclose(distance.kl_divergence, expected_kl, rel_tol=0.0, abs_tol=1e-12)
                or distance.sample_count != len(behavior_policy.actions)
                or distance.evidence_digest != _digest(measurement)
            ):
                raise ValueError("H4 policy-distance evidence is not derived from its policies")
        for candidate in batch.candidates:
            decision = candidate.decision
            if decision.behavior_source is not LogProbabilitySource.RECOMPUTED:
                continue
            policy_id = f"{decision.behavior_policy.policy_id}@{decision.behavior_policy.epoch}"
            policy = policy_by_epoch[policy_id]
            expected_probability = policy.log_probability(candidate.action)
            expected_digest = _digest(
                {
                    "action": candidate.action,
                    "log_probability": expected_probability,
                    "policy_digest": policy.weights_hash,
                }
            )
            if (
                decision.behavior_log_probability != expected_probability
                or decision.behavior_evidence_digest != expected_digest
            ):
                raise ValueError("H4 recomputed probability is not derived from its policy")
        if batch.manifest is not None and trainer.base_policy is not None:
            adapter = ReferenceTrainer(
                learning_rate=0.35,
                kl_coefficient=0.05,
                ratio_clip=0.2,
                maximum_staleness_updates=2,
                maximum_steps=32,
            )
            try:
                replayed = adapter.train(
                    base=trainer.base_policy,
                    samples=batch.manifest.trainer_samples(),
                    algorithm=batch.manifest.algorithm,
                    candidate_policy_epoch_id=(f"{staleness.report.learner_policy.policy_id}@2"),
                    seed=observation.seed,
                    steps=8,
                )
            except ValueError as error:
                if (
                    trainer.disposition is not TrainerDisposition.REFERENCE_ADAPTER_REJECTED
                    or trainer.rejection_reason != str(error)
                ):
                    raise ValueError("H4 trainer rejection does not replay") from error
            else:
                if trainer.result != replayed:
                    raise ValueError("H4 trainer result does not replay")
        if _stability(trainer) != observation.training_stability:
            raise ValueError("H4 training stability disagrees with raw trainer evidence")
        outcome = loaded.get("outcome")
        if outcome != observation.evaluated_policy_outcome:
            raise ValueError("H4 campaign policy outcome disagrees with raw evidence")
    expected_files = {
        "campaign.json",
        *(artifact.path for item in campaign.observations for artifact in item.raw_artifacts),
    }
    actual_files: set[str] = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError("H4 campaign output contains a symbolic link")
        if path.is_file():
            actual_files.add(path.relative_to(output).as_posix())
    if actual_files != expected_files:
        raise ValueError("H4 campaign raw artifact inventory is incomplete")
    return campaign


__all__ = [
    "CASE_ORDER",
    "MAX_CAMPAIGN_ARTIFACT_BYTES",
    "BatchDisposition",
    "CaseObservation",
    "StalenessCampaign",
    "StalenessCase",
    "TrainerDisposition",
    "run_staleness_campaign",
    "validate_staleness_campaign",
]
