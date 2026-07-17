"""Multi-seed evaluation assembled only from executed Helix artifacts.

The reference evaluation deliberately separates an exercised observation from
an unresolved hypothesis.  It never fills missing GPU, production, or causal
measurements with simulator output.
"""

from __future__ import annotations

import html
import json
import math
import platform
import subprocess
import sys
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.helix.credit import BranchRelativeCredit
from sloforge.helix.demo import run_cpu_demo
from sloforge.helix.scheduler import (
    SchedulerPlan,
    SchedulerPolicy,
    SchedulerRequest,
    compile_resource_plan,
    load_scheduler_request,
)
from sloforge.helix.scheduler.statistics import student_t_mean_interval_95
from sloforge.helix.trainers import TrainingAlgorithm

if TYPE_CHECKING:
    from sloforge.helix.evaluation.continual_learning import ContinualLearningCampaign
    from sloforge.helix.evaluation.experience_selection import ExperienceSelectionCampaignRun
    from sloforge.helix.evaluation.preservation import PreservationCampaign
    from sloforge.helix.evaluation.staleness_campaign import StalenessCampaign
    from sloforge.helix.evaluation.training_algorithms import TrainingAlgorithmCampaign

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_STATIC_LIMIT_BASIS = (
    "One declared resource-vector unit per learning class; aggregate capacity and budget "
    "constraints remain hard."
)
MAX_REFERENCE_ARTIFACT_BYTES = 32 * 1024 * 1024


class _EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _FlagshipEvidenceModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)


class _FlagshipRejectedCandidate(_FlagshipEvidenceModel):
    success_rate: float
    promotion_state: str


class _FlagshipCorrectedCandidate(_FlagshipEvidenceModel):
    success_rate: float


class _FlagshipReplayResult(_FlagshipEvidenceModel):
    exact_identity_verified: bool


class _FlagshipReplay(_FlagshipEvidenceModel):
    joint: _FlagshipReplayResult
    transcript: _FlagshipReplayResult
    environment: _FlagshipReplayResult
    model: _FlagshipReplayResult


class _FlagshipPromotion(_FlagshipEvidenceModel):
    final_state: str
    incompatible_session_pinned: bool


class _FlagshipSummary(_FlagshipEvidenceModel):
    schema_version: Literal["sloforge.helix.cpu-demo/v1"]
    seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    champion_success_rate: float
    rejected_candidate: _FlagshipRejectedCandidate
    corrected_candidate: _FlagshipCorrectedCandidate
    measured_success_rate_delta: float
    replay: _FlagshipReplay
    promotion: _FlagshipPromotion
    branch_group_id: str
    branch_point_id: str
    credit_evidence_hash: Digest


class Interval(_EvaluationModel):
    mean: float
    lower_95: float
    upper_95: float
    sample_count: Annotated[int, Field(gt=0)]
    sample_standard_deviation: Annotated[float, Field(ge=0.0)]
    method: Literal["two_sided_student_t_mean"] = "two_sided_student_t_mean"
    confidence_level: Annotated[float, Field(ge=0.95, le=0.95)] = 0.95

    @model_validator(mode="after")
    def valid_interval(self) -> Interval:
        values = (self.mean, self.lower_95, self.upper_95, self.sample_standard_deviation)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("evaluation interval values must be finite")
        if not self.lower_95 <= self.mean <= self.upper_95:
            raise ValueError("evaluation interval must contain its mean")
        return self


class RawArtifact(_EvaluationModel):
    artifact_kind: Literal[
        "flagship_summary",
        "scheduler_plan",
        "scheduler_workload",
        "branch_credit",
        "training_algorithm_campaign",
        "staleness_campaign",
        "preservation_campaign",
        "continual_learning_campaign",
        "experience_selection_campaign",
    ]
    path: Annotated[str, Field(min_length=1, max_length=4096)]
    sha256: Digest
    seed: int | None = None

    @model_validator(mode="after")
    def valid_seed_scope(self) -> RawArtifact:
        parsed = PurePosixPath(self.path)
        if parsed.is_absolute() or ".." in parsed.parts or self.path == ".":
            raise ValueError("raw artifact paths must be portable evaluation-relative paths")
        if self.artifact_kind in {
            "scheduler_workload",
            "training_algorithm_campaign",
            "staleness_campaign",
            "preservation_campaign",
            "continual_learning_campaign",
            "experience_selection_campaign",
        }:
            if self.seed is not None:
                raise ValueError("scheduler workload provenance is shared across seeds")
        elif self.seed is None:
            raise ValueError("executed raw artifacts require their deterministic seed")
        elif not 0 <= self.seed <= 2**63 - 1:
            raise ValueError("raw artifact seeds must fit signed 64-bit integers")
        return self


class SeedObservation(_EvaluationModel):
    seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    champion_success_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    rejected_success_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    challenger_success_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    paired_success_rate_difference: Annotated[float, Field(ge=-1.0, le=1.0)]
    summary_artifact_path: str
    summary_artifact_sha256: Digest

    @model_validator(mode="after")
    def valid_effect(self) -> SeedObservation:
        expected = self.challenger_success_rate - self.champion_success_rate
        if not math.isclose(
            expected, self.paired_success_rate_difference, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("paired success-rate effect disagrees with seed observations")
        return self


class HypothesisResult(_EvaluationModel):
    hypothesis_id: Annotated[str, Field(pattern=r"^H(?:10|[1-9])$")]
    status: Literal[
        "supported_within_scope",
        "not_supported",
        "inconclusive",
        "partial",
        "not_exercised",
    ]
    claim_scope: str
    observations: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    limitations: tuple[str, ...]


class SchedulerComparison(_EvaluationModel):
    policy: SchedulerPolicy
    seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    plan_id: Digest
    request_digest: Digest
    serving_predicted_slo_feasible: bool
    completed_work_count: Annotated[int, Field(ge=0)]
    predicted_learning_value: float
    total_cost_microunits: Annotated[int, Field(ge=0)]
    preemption_count: Annotated[int, Field(ge=0)]
    lost_work_ticks: Annotated[int, Field(ge=0)]
    artifact_path: str
    artifact_sha256: Digest
    static_limit_basis: str | None = None

    @model_validator(mode="after")
    def valid_static_method(self) -> SchedulerComparison:
        if (self.policy is SchedulerPolicy.STATIC) != (self.static_limit_basis is not None):
            raise ValueError("only the static baseline may declare a static-limit basis")
        if not math.isfinite(self.predicted_learning_value):
            raise ValueError("scheduler predicted learning value must be finite")
        return self


class EvaluationRun(_EvaluationModel):
    schema_version: Literal["sloforge.helix.reference-evaluation/v1"] = (
        "sloforge.helix.reference-evaluation/v1"
    )
    evaluation_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    seeds: Annotated[tuple[int, ...], Field(min_length=2, max_length=32)]
    seed_observations: Annotated[tuple[SeedObservation, ...], Field(min_length=2, max_length=32)]
    champion_success: Interval
    rejected_success: Interval
    challenger_success: Interval
    success_rate_delta: Interval
    exact_joint_replay_count: Annotated[int, Field(ge=0)]
    non_joint_exact_identity_count: Annotated[int, Field(ge=0)]
    rejected_promotion_count: Annotated[int, Field(ge=0)]
    rollback_count: Annotated[int, Field(ge=0)]
    incompatible_session_pin_count: Annotated[int, Field(ge=0)]
    scheduler_comparisons: tuple[SchedulerComparison, ...]
    hypotheses: Annotated[tuple[HypothesisResult, ...], Field(min_length=10, max_length=10)]
    raw_artifacts: Annotated[tuple[RawArtifact, ...], Field(min_length=1)]
    effect_size_definition: Literal[
        "paired absolute success-rate difference (corrected challenger minus champion)"
    ]
    software_manifest: dict[str, str]
    hardware_manifest: dict[str, str | int]
    validation_class: Literal["deterministic-local-cpu-synthetic"]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def complete_matrix(self) -> EvaluationRun:
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("evaluation seeds must be unique")
        if any(seed < 0 or seed > 2**63 - 1 for seed in self.seeds):
            raise ValueError("evaluation seeds must fit signed 64-bit integers")
        if tuple(item.seed for item in self.seed_observations) != self.seeds:
            raise ValueError("seed observations must preserve evaluation seed order")
        expected = tuple(f"H{index}" for index in range(1, 11))
        observed = tuple(item.hypothesis_id for item in self.hypotheses)
        if observed != expected:
            raise ValueError("evaluation must report H1 through H10 in order")
        matrix = {(item.seed, item.policy) for item in self.scheduler_comparisons}
        expected_matrix = {(seed, policy) for seed in self.seeds for policy in SchedulerPolicy}
        if matrix != expected_matrix or len(self.scheduler_comparisons) != len(expected_matrix):
            raise ValueError("scheduler evaluation must include every policy for every seed")
        interval_inputs = (
            (
                self.champion_success,
                tuple(item.champion_success_rate for item in self.seed_observations),
            ),
            (
                self.rejected_success,
                tuple(item.rejected_success_rate for item in self.seed_observations),
            ),
            (
                self.challenger_success,
                tuple(item.challenger_success_rate for item in self.seed_observations),
            ),
            (
                self.success_rate_delta,
                tuple(item.paired_success_rate_difference for item in self.seed_observations),
            ),
        )
        for interval, values in interval_inputs:
            expected_interval = _interval(values)
            if interval != expected_interval:
                raise ValueError("evaluation interval disagrees with raw seed observations")
        if any(
            count > len(self.seeds)
            for count in (
                self.exact_joint_replay_count,
                self.rejected_promotion_count,
                self.rollback_count,
                self.incompatible_session_pin_count,
            )
        ):
            raise ValueError("per-run evaluation counts cannot exceed the seed count")
        paths = tuple(item.path for item in self.raw_artifacts)
        if tuple(sorted(set(paths))) != paths:
            raise ValueError("raw evaluation artifacts must be path-sorted and unique")
        artifacts = {item.path: item for item in self.raw_artifacts}
        for observation in self.seed_observations:
            raw = artifacts.get(observation.summary_artifact_path)
            if raw is None or raw.sha256 != observation.summary_artifact_sha256:
                raise ValueError("seed observation lacks matching raw summary provenance")
        for comparison in self.scheduler_comparisons:
            raw = artifacts.get(comparison.artifact_path)
            if raw is None or raw.sha256 != comparison.artifact_sha256:
                raise ValueError("scheduler comparison lacks matching raw plan provenance")
        hypothesis_paths = {
            path for hypothesis in self.hypotheses for path in hypothesis.artifact_paths
        }
        if not hypothesis_paths.issubset(artifacts):
            raise ValueError("hypothesis references an artifact without a recorded digest")
        identity = self.model_dump(mode="json", exclude={"evaluation_id"})
        if sha256(_canonical_bytes(identity)).hexdigest() != self.evaluation_id:
            raise ValueError("reference evaluation identifier is invalid")
        return self


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> str:
    document = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    payload = _canonical_bytes(document)
    encoded = payload + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return sha256(encoded).hexdigest()


def _repository_manifest() -> dict[str, str]:
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=no"),
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "unavailable", "tracked_worktree_dirty": "unavailable"}
    return {
        "git_commit": commit,
        "tracked_worktree_dirty": str(bool(status)).lower(),
    }


def _interval(values: tuple[float, ...]) -> Interval:
    mean, lower, upper, deviation = student_t_mean_interval_95(values)
    return Interval(
        mean=mean,
        lower_95=lower,
        upper_95=upper,
        sample_count=len(values),
        sample_standard_deviation=deviation,
    )


def _request_for_policy(
    request: SchedulerRequest, policy: SchedulerPolicy, seed: int
) -> SchedulerRequest:
    payload = request.model_dump(mode="python")
    payload["policy"] = policy
    payload["seed"] = seed
    payload["static_limits"] = (
        request.resource_vectors.model_dump(mode="python")
        if policy is SchedulerPolicy.STATIC
        else None
    )
    return SchedulerRequest.model_validate(payload)


def _scheduler_comparisons(
    workload: Path, raw: Path, seeds: tuple[int, ...]
) -> tuple[tuple[SchedulerComparison, ...], tuple[RawArtifact, ...]]:
    request = load_scheduler_request(workload)
    results: list[SchedulerComparison] = []
    workload_copy = raw / "scheduler" / "source-workload.json"
    workload_payload = workload.read_bytes()
    workload_copy.parent.mkdir(parents=True, exist_ok=True)
    workload_copy.write_bytes(workload_payload)
    artifacts = [
        RawArtifact(
            artifact_kind="scheduler_workload",
            path=workload_copy.relative_to(raw.parent).as_posix(),
            sha256=sha256(workload_payload).hexdigest(),
        )
    ]
    for seed in seeds:
        for policy in SchedulerPolicy:
            comparison_request = _request_for_policy(request, policy, seed)
            plan = compile_resource_plan(comparison_request)
            target = raw / "scheduler" / f"seed-{seed}" / f"{policy.value}.json"
            artifact_sha256 = _write_json(target, plan)
            artifact_path = target.relative_to(raw.parent).as_posix()
            artifacts.append(
                RawArtifact(
                    artifact_kind="scheduler_plan",
                    path=artifact_path,
                    sha256=artifact_sha256,
                    seed=seed,
                )
            )
            results.append(
                SchedulerComparison(
                    policy=policy,
                    seed=seed,
                    plan_id=plan.plan_id,
                    request_digest=plan.request_digest,
                    serving_predicted_slo_feasible=all(
                        tick.serving_slo_satisfied for tick in plan.ticks
                    ),
                    completed_work_count=len(plan.completed_work_ids),
                    predicted_learning_value=plan.predicted_learning_value,
                    total_cost_microunits=plan.budget.total_microunits,
                    preemption_count=len(plan.preemptions),
                    lost_work_ticks=sum(item.lost_work_ticks for item in plan.outcomes),
                    artifact_path=artifact_path,
                    artifact_sha256=artifact_sha256,
                    static_limit_basis=(
                        _STATIC_LIMIT_BASIS if policy is SchedulerPolicy.STATIC else None
                    ),
                )
            )
    return tuple(results), tuple(artifacts)


def _validated_artifact_path(root: Path, artifact: RawArtifact) -> Path:
    parsed = PurePosixPath(artifact.path)
    if (
        not parsed.parts
        or parsed.parts[0] != "raw"
        or str(parsed) != artifact.path
        or "\\" in artifact.path
    ):
        raise ValueError("raw artifact path is not a normalized evaluation-relative path")
    cursor = root
    for part in parsed.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("raw evaluation artifact path contains a symbolic link")
    try:
        path = root.joinpath(*parsed.parts).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"raw evaluation artifact is missing: {artifact.path}") from exc
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("raw artifact path escapes the evaluation output")
    if path.stat().st_size > MAX_REFERENCE_ARTIFACT_BYTES:
        raise ValueError("raw evaluation artifact exceeds the byte limit")
    return path


def _comparison_from_plan(plan: SchedulerPlan, artifact: RawArtifact) -> SchedulerComparison:
    return SchedulerComparison(
        policy=plan.policy,
        seed=plan.seed,
        plan_id=plan.plan_id,
        request_digest=plan.request_digest,
        serving_predicted_slo_feasible=all(tick.serving_slo_satisfied for tick in plan.ticks),
        completed_work_count=len(plan.completed_work_ids),
        predicted_learning_value=plan.predicted_learning_value,
        total_cost_microunits=plan.budget.total_microunits,
        preemption_count=len(plan.preemptions),
        lost_work_ticks=sum(item.lost_work_ticks for item in plan.outcomes),
        artifact_path=artifact.path,
        artifact_sha256=artifact.sha256,
        static_limit_basis=(_STATIC_LIMIT_BASIS if plan.policy is SchedulerPolicy.STATIC else None),
    )


def _validate_flagship_evidence(
    run: EvaluationRun,
    artifacts: dict[str, tuple[RawArtifact, Path, bytes]],
) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
    observations = {item.seed: item for item in run.seed_observations}
    summaries: dict[int, _FlagshipSummary] = {}
    summary_artifacts = tuple(
        item for item in run.raw_artifacts if item.artifact_kind == "flagship_summary"
    )
    if len(summary_artifacts) != len(run.seeds):
        raise ValueError("evaluation must contain one flagship summary per seed")
    for artifact in summary_artifacts:
        if artifact.seed is None or artifact.seed in summaries:
            raise ValueError("flagship summary seed provenance is incomplete or duplicated")
        summary = _FlagshipSummary.model_validate_json(artifacts[artifact.path][2], strict=True)
        observation = observations.get(artifact.seed)
        if observation is None or (
            summary.seed != artifact.seed
            or observation.summary_artifact_path != artifact.path
            or observation.summary_artifact_sha256 != artifact.sha256
            or observation.champion_success_rate != summary.champion_success_rate
            or observation.rejected_success_rate != summary.rejected_candidate.success_rate
            or observation.challenger_success_rate != summary.corrected_candidate.success_rate
            or not math.isclose(
                observation.paired_success_rate_difference,
                summary.measured_success_rate_delta,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("seed observation disagrees with raw flagship summary evidence")
        summaries[artifact.seed] = summary
    if set(summaries) != set(run.seeds):
        raise ValueError("flagship summaries do not preserve the evaluation seed matrix")

    exact_joint = sum(item.replay.joint.exact_identity_verified for item in summaries.values())
    non_joint_exact = sum(
        replay.exact_identity_verified
        for item in summaries.values()
        for replay in (item.replay.transcript, item.replay.environment, item.replay.model)
    )
    rejected = sum(
        item.rejected_candidate.promotion_state == "rejected" for item in summaries.values()
    )
    rollbacks = sum(item.promotion.final_state == "rolled_back" for item in summaries.values())
    pinned = sum(item.promotion.incompatible_session_pinned for item in summaries.values())
    if (
        run.exact_joint_replay_count != exact_joint
        or run.non_joint_exact_identity_count != non_joint_exact
        or run.rejected_promotion_count != rejected
        or run.rollback_count != rollbacks
        or run.incompatible_session_pin_count != pinned
    ):
        raise ValueError("evaluation aggregate counts disagree with flagship summary evidence")

    credit_ledger = tuple(
        item for item in run.raw_artifacts if item.artifact_kind == "branch_credit"
    )
    credit_artifacts = {item.seed: item for item in credit_ledger}
    if (
        len(credit_ledger) != len(run.seeds)
        or len(credit_artifacts) != len(run.seeds)
        or set(credit_artifacts) != set(run.seeds)
    ):
        raise ValueError("evaluation must contain one branch-credit artifact per seed")
    for seed, summary in summaries.items():
        artifact = credit_artifacts[seed]
        expected_path = (
            PurePosixPath(observations[seed].summary_artifact_path).parent
            / "credit"
            / "branch-relative.json"
        ).as_posix()
        credit = BranchRelativeCredit.model_validate_json(artifacts[artifact.path][2], strict=True)
        credit_evidence_hash = sha256(_canonical_bytes(credit.model_dump(mode="json"))).hexdigest()
        if (
            artifact.path != expected_path
            or summary.credit_evidence_hash != credit_evidence_hash
            or summary.branch_group_id != credit.branch_group_id
            or summary.branch_point_id != credit.branch_point_id
        ):
            raise ValueError("branch-credit evidence disagrees with its flagship summary")
    return (
        tuple(observations[seed].summary_artifact_path for seed in run.seeds),
        tuple(credit_artifacts[seed].path for seed in run.seeds),
        exact_joint,
        non_joint_exact,
    )


def _validate_scheduler_evidence(
    run: EvaluationRun,
    artifacts: dict[str, tuple[RawArtifact, Path, bytes]],
) -> None:
    workloads = tuple(
        item for item in run.raw_artifacts if item.artifact_kind == "scheduler_workload"
    )
    plans = tuple(item for item in run.raw_artifacts if item.artifact_kind == "scheduler_plan")
    if len(workloads) != 1 or len(plans) != len(run.scheduler_comparisons):
        raise ValueError("scheduler artifact matrix is incomplete")
    request = load_scheduler_request(artifacts[workloads[0].path][1])
    plan_paths = {item.path for item in plans}
    comparison_paths = {item.artifact_path for item in run.scheduler_comparisons}
    if plan_paths != comparison_paths:
        raise ValueError("scheduler comparisons do not consume the complete plan ledger")
    for comparison in run.scheduler_comparisons:
        artifact, _, payload = artifacts[comparison.artifact_path]
        if artifact.artifact_kind != "scheduler_plan" or artifact.seed != comparison.seed:
            raise ValueError("scheduler comparison has invalid plan provenance")
        plan = SchedulerPlan.model_validate_json(payload, strict=True)
        expected_plan = compile_resource_plan(
            _request_for_policy(request, comparison.policy, comparison.seed)
        )
        if plan != expected_plan:
            raise ValueError("scheduler plan differs from deterministic recompilation")
        if comparison != _comparison_from_plan(plan, artifact):
            raise ValueError("scheduler comparison disagrees with its raw plan")


def _validate_campaign_evidence(
    run: EvaluationRun,
    artifacts: dict[str, tuple[RawArtifact, Path, bytes]],
    *,
    demo_paths: tuple[str, ...],
    credit_paths: tuple[str, ...],
    joint_exact: int,
    non_joint_exact: int,
) -> None:
    # Keep these imports local: importing executable campaign modules from the package
    # initializer causes `python -m ...` to emit runpy's already-imported warning.
    from sloforge.helix.evaluation.continual_learning import (
        validate_continual_learning_campaign,
    )
    from sloforge.helix.evaluation.experience_selection import (
        validate_experience_selection_campaign,
    )
    from sloforge.helix.evaluation.preservation import validate_preservation_campaign
    from sloforge.helix.evaluation.staleness_campaign import validate_staleness_campaign
    from sloforge.helix.evaluation.training_algorithms import (
        validate_training_algorithm_campaign,
    )

    def campaign_artifact(artifact_kind: str) -> tuple[RawArtifact, Path]:
        matches = tuple(item for item in run.raw_artifacts if item.artifact_kind == artifact_kind)
        if len(matches) != 1:
            raise ValueError(f"evaluation requires one {artifact_kind} manifest")
        artifact = matches[0]
        path = artifacts[artifact.path][1]
        if path.name != "campaign.json":
            raise ValueError("campaign artifact must name its campaign manifest")
        return artifact, path

    training_artifact, training_path = campaign_artifact("training_algorithm_campaign")
    staleness_artifact, staleness_path = campaign_artifact("staleness_campaign")
    preservation_artifact, preservation_path = campaign_artifact("preservation_campaign")
    continual_artifact, continual_path = campaign_artifact("continual_learning_campaign")
    experience_artifact, experience_path = campaign_artifact("experience_selection_campaign")
    training_campaign = validate_training_algorithm_campaign(training_path.parent)
    staleness_campaign = validate_staleness_campaign(staleness_path.parent)
    preservation_campaign = validate_preservation_campaign(
        preservation_path.parent,
        scenario_path=(
            _REPOSITORY_ROOT / "scenarios/helix/resource/rollout-preservation-campaign.json"
        ),
    )
    continual_campaign = validate_continual_learning_campaign(continual_path.parent)
    experience_campaign = validate_experience_selection_campaign(experience_path.parent)
    if (
        training_campaign.seeds != run.seeds
        or staleness_campaign.seeds != run.seeds
        or continual_campaign.seeds != run.seeds
        or experience_campaign.seeds != run.seeds
        or preservation_campaign.seeds != run.seeds[: len(preservation_campaign.seeds)]
    ):
        raise ValueError("campaign seed provenance disagrees with the evaluation")
    expected_hypotheses = _hypotheses(
        demo_paths=demo_paths,
        credit_paths=credit_paths,
        scheduler=run.scheduler_comparisons,
        joint_exact=joint_exact,
        non_joint_exact=non_joint_exact,
        training_campaign=training_campaign,
        training_campaign_path=training_artifact.path,
        staleness_campaign=staleness_campaign,
        staleness_campaign_path=staleness_artifact.path,
        preservation_campaign=preservation_campaign,
        preservation_campaign_path=preservation_artifact.path,
        continual_campaign=continual_campaign,
        continual_campaign_path=continual_artifact.path,
        experience_campaign=experience_campaign,
        experience_campaign_path=experience_artifact.path,
    )
    if run.hypotheses != expected_hypotheses:
        raise ValueError("evaluation hypothesis summaries disagree with validated evidence")


def validate_reference_evaluation(output: Path) -> EvaluationRun:
    """Strictly validate a published evaluation and all evidence it summarizes."""

    try:
        root = output.resolve(strict=True)
    except OSError as exc:
        raise ValueError("reference evaluation output is missing") from exc
    if not root.is_dir():
        raise ValueError("reference evaluation output is not a directory")
    manifest_path = root / "evaluation.json"
    if manifest_path.is_symlink():
        raise ValueError("reference evaluation manifest cannot be a symbolic link")
    try:
        if manifest_path.stat().st_size > MAX_REFERENCE_ARTIFACT_BYTES:
            raise ValueError("reference evaluation manifest exceeds the byte limit")
        manifest = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError("reference evaluation manifest is missing") from exc
    run = EvaluationRun.model_validate_json(manifest, strict=True)
    if manifest != _canonical_bytes(run.model_dump(mode="json")) + b"\n":
        raise ValueError("reference evaluation manifest is not canonical")

    artifacts: dict[str, tuple[RawArtifact, Path, bytes]] = {}
    for artifact in run.raw_artifacts:
        path = _validated_artifact_path(root, artifact)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read raw evaluation artifact: {artifact.path}") from exc
        if sha256(payload).hexdigest() != artifact.sha256:
            raise ValueError(f"raw evaluation artifact digest mismatch: {artifact.path}")
        artifacts[artifact.path] = (artifact, path, payload)

    demo_paths, credit_paths, joint_exact, non_joint_exact = _validate_flagship_evidence(
        run, artifacts
    )
    _validate_scheduler_evidence(run, artifacts)
    _validate_campaign_evidence(
        run,
        artifacts,
        demo_paths=demo_paths,
        credit_paths=credit_paths,
        joint_exact=joint_exact,
        non_joint_exact=non_joint_exact,
    )
    return run


def _hypotheses(
    *,
    demo_paths: tuple[str, ...],
    credit_paths: tuple[str, ...],
    scheduler: tuple[SchedulerComparison, ...],
    joint_exact: int,
    non_joint_exact: int,
    training_campaign: TrainingAlgorithmCampaign,
    training_campaign_path: str,
    staleness_campaign: StalenessCampaign,
    staleness_campaign_path: str,
    preservation_campaign: PreservationCampaign,
    preservation_campaign_path: str,
    continual_campaign: ContinualLearningCampaign,
    continual_campaign_path: str,
    experience_campaign: ExperienceSelectionCampaignRun,
    experience_campaign_path: str,
) -> tuple[HypothesisResult, ...]:
    demos = demo_paths
    scheduler_paths = tuple(item.artifact_path for item in scheduler)
    scheduler_preemption_runs = sum(item.preemption_count > 0 for item in scheduler)
    maximum_budget = training_campaign.example_budgets[-1]
    algorithm_success = {
        algorithm: sum(
            run.task_success_rate
            for run in training_campaign.runs
            if run.algorithm is algorithm and run.training_examples == maximum_budget
        )
        / len(training_campaign.seeds)
        for algorithm in TrainingAlgorithm
    }
    preservation = {item.strategy.value: item for item in preservation_campaign.aggregates}
    continual_accepted = sum(item.accepted_update_count for item in continual_campaign.runs)
    continual_rejections = sum(item.candidate_rejection_count for item in continual_campaign.runs)
    continual_recurrence = sum(
        item.failure_recurrence_rate for item in continual_campaign.runs
    ) / len(continual_campaign.runs)
    experience_helix = next(
        item
        for item in experience_campaign.strategy_summaries
        if item.strategy.value == "helix_value_aware"
    )
    experience_random = next(
        item
        for item in experience_campaign.helix_baseline_comparisons
        if item.baseline_strategy.value == "random"
    )
    staleness_cases = {
        item.case
        for item in staleness_campaign.observations
        if item.seed == training_campaign.seeds[0]
    }
    complete_staleness_matrix = (
        staleness_campaign.seeds == training_campaign.seeds
        and bool(staleness_cases)
        and all(
            len(tuple(item for item in staleness_campaign.observations if item.seed == seed))
            == len(staleness_cases)
            and {item.case for item in staleness_campaign.observations if item.seed == seed}
            == staleness_cases
            for seed in training_campaign.seeds
        )
    )
    common = (
        "Reference policy and repository are synthetic local CPU fixtures.",
        "The result does not establish behavior for arbitrary models, tools, or production traffic.",
    )
    return (
        HypothesisResult(
            hypothesis_id="H1",
            status="partial",
            claim_scope="State identity and replay contracts, not downstream trajectory fidelity.",
            observations=(
                f"Joint restoration passed exact identity in {joint_exact}/{len(demo_paths)} runs.",
                f"Non-joint modes passed exact identity in {non_joint_exact} cases.",
                "Transcript equality explicitly did not establish hidden model/environment equivalence.",
            ),
            artifact_paths=demos,
            limitations=(
                "Readiness, prefill work, and divergence under a nondeterministic real model were not measured.",
                *common,
            ),
        ),
        HypothesisResult(
            hypothesis_id="H2",
            status="partial",
            claim_scope="CAS-backed logical copy-on-write isolation and cleanup.",
            observations=("All branch workspaces were isolated and cleaned after each run.",),
            artifact_paths=demos,
            limitations=(
                "No reflink/page-fault or GPU-memory amplification benchmark was run on this host.",
                *common,
            ),
        ),
        HypothesisResult(
            hypothesis_id="H3",
            status=training_campaign.h3_result.status,
            claim_scope=(
                "Comparative sample-budget behavior of four objectives on one categorical CPU "
                "reference fixture."
            ),
            observations=(
                f"At {maximum_budget} examples, mean targeted success was "
                f"{algorithm_success[TrainingAlgorithm.SUCCESSFUL_BRANCH_DISTILLATION]:.6f} "
                "for successful-branch distillation, "
                f"{algorithm_success[TrainingAlgorithm.PAIRWISE_PREFERENCE]:.6f} for pairwise "
                "preference, "
                f"{algorithm_success[TrainingAlgorithm.GROUP_RELATIVE]:.6f} for group-relative, "
                f"and {algorithm_success[TrainingAlgorithm.BRANCH_RELATIVE]:.6f} for "
                "branch-relative optimization.",
                f"The paired campaign decision rule classified H3 as {training_campaign.h3_result.status}.",
            ),
            artifact_paths=(training_campaign_path,),
            limitations=(
                "Training examples are controlled synthetic branch-provenance samples, not GPU-hour-normalized exact-state flagship trajectories.",
                "This reference result does not establish comparative performance for neural policies.",
            ),
        ),
        HypothesisResult(
            hypothesis_id="H4",
            status=(
                "supported_within_scope"
                if complete_staleness_matrix
                and staleness_campaign.invalid_sample_acceptance_count == 0
                else "not_supported"
            ),
            claim_scope="Fail-closed policy provenance and bounded staleness semantics.",
            observations=(
                f"The campaign executed {len(staleness_campaign.observations)} policy-version cases with "
                f"{staleness_campaign.invalid_sample_acceptance_count} invalid sample acceptances.",
                f"{staleness_campaign.staleness_eligible_case_count} cases passed the staleness layer, while strict batch and trainer boundaries separately rejected mixed or unsupported old-policy evidence.",
            ),
            artifact_paths=(staleness_campaign_path,),
            limitations=(
                "The reference trainer intentionally cannot consume bounded off-policy or segmented mixed-policy evidence.",
                "The recomputed-log-probability case uses declared deterministic fixture evidence rather than recomputing a neural policy probability from token history.",
                "Training stability under a large asynchronous neural optimizer was not measured.",
            ),
        ),
        HypothesisResult(
            hypothesis_id="H5",
            status="partial",
            claim_scope="Deterministic scheduler feasibility under identical scenario inputs.",
            observations=(
                "Dedicated, static, utilization, FIFO, and value-aware policies were compiled for every evaluation seed.",
                "Hard feasibility against supplied serving latency and queue predictions is reported independently for every policy, seed, and tick.",
            ),
            artifact_paths=scheduler_paths,
            limitations=(
                "Learning values are scenario predictions with provenance, not observed policy gains.",
                "No GPU-hour causal claim is made.",
            ),
        ),
        HypothesisResult(
            hypothesis_id="H6",
            status="supported_within_scope",
            claim_scope="Executed local rollout preservation during deterministic capacity reclamation.",
            observations=(
                f"A preservation path was selected in {scheduler_preemption_runs}/{len(scheduler)} scheduler policy-seed runs.",
                f"Terminate/restart lost {preservation['terminate_restart'].mean_lost_tokens:.3f} tokens and {preservation['terminate_restart'].mean_lost_tool_work_units:.3f} tool-work units on average.",
                f"Joint Continuum plus environment restoration lost {preservation['joint_continuum_environment'].mean_lost_tokens:.3f} tokens and {preservation['joint_continuum_environment'].mean_lost_tool_work_units:.3f} tool-work units on average.",
            ),
            artifact_paths=(*scheduler_paths, preservation_campaign_path),
            limitations=(
                "Resume and restoration values are deterministic executed-operation ticks, not hardware wall time.",
                "No hardware-backed rollout migration timing was measured.",
            ),
        ),
        HypothesisResult(
            hypothesis_id="H7",
            status="supported_within_scope",
            claim_scope="Local atomic promotion state machine and evidence preservation.",
            observations=(
                "Each local synthetic run rejected an executed below-threshold candidate, promoted a passing candidate after shadow/canary, pinned an incompatible session, and restored the champion during rollback.",
            ),
            artifact_paths=demos,
            limitations=(
                "No live distributed gateway or multi-region control plane was exercised.",
            ),
        ),
        HypothesisResult(
            hypothesis_id="H8",
            status="partial",
            claim_scope="Continual repair and forgetting over changing task distributions.",
            observations=(
                f"Across {len(continual_campaign.runs)} seed runs and {len(continual_campaign.task_sequence)} sequential tasks, {continual_accepted} candidates passed and {continual_rejections} candidates were rejected by the retention gate.",
                f"The unweighted mean per-seed recurrence rate for accepted repairs was {continual_recurrence:.6f}.",
                "Later targeted candidates improved their current task but were rejected for excessive prior-capability loss.",
            ),
            artifact_paths=(continual_campaign_path,),
            limitations=(
                "The categorical policy lacks an observation representation, which limits forward adaptation.",
                "Gate success demonstrates protected retention, not continual-learning superiority.",
            ),
        ),
        HypothesisResult(
            hypothesis_id="H9",
            status=experience_campaign.h9_result.status,
            claim_scope="Measured downstream learning value from governed experience selection.",
            observations=(
                f"Helix value-aware selection produced mean paired measured success change {experience_helix.paired_measured_success_change.mean:.6f} with 95% seed-sensitivity interval [{experience_helix.paired_measured_success_change.lower_95:.6f}, {experience_helix.paired_measured_success_change.upper_95:.6f}].",
                f"Helix minus random had paired mean {experience_random.sensitivity.mean:.6f} with interval [{experience_random.sensitivity.lower_95:.6f}, {experience_random.sensitivity.upper_95:.6f}].",
                f"The campaign decision rule classified H9 as {experience_campaign.h9_result.status}.",
            ),
            artifact_paths=(experience_campaign_path,),
            limitations=(
                "The campaign uses one local synthetic candidate pool and categorical reference trainer.",
                "Small-n Student-t intervals are sensitivity summaries and are not multiplicity-adjusted hypothesis tests.",
            ),
        ),
        HypothesisResult(
            hypothesis_id="H10",
            status="partial",
            claim_scope="Controlled exact-state sibling rewards and credit provenance.",
            observations=(
                "Every branch group shared one committed BranchPoint and preserved component rewards, first divergence, intervention type, assumptions, and exclusions.",
            ),
            artifact_paths=credit_paths,
            limitations=(
                "No independent non-sibling variance baseline was run.",
                "Controlled sibling differences improve comparability but do not prove universal causal identification.",
            ),
        ),
    )


def _markdown(run: EvaluationRun) -> str:
    lines = [
        "# SLOForge Helix reference evaluation",
        "",
        f"Evaluation ID: `{run.evaluation_id}`",
        "",
        "This report is generated from deterministic local CPU artifacts. Synthetic results are not presented as GPU or production measurements.",
        "",
        "## Flagship result",
        "",
        "Intervals are two-sided Student-t 95% intervals over paired deterministic seed runs. They describe seed sensitivity in this fixture, not population uncertainty.",
        "",
        f"Effect size: {run.effect_size_definition}.",
        "",
        "| Metric | Mean | Student-t 95% interval | Seed runs |",
        "|---|---:|---:|---:|",
        f"| Champion task success | {run.champion_success.mean:.3f} | [{run.champion_success.lower_95:.3f}, {run.champion_success.upper_95:.3f}] | {run.champion_success.sample_count} |",
        f"| Rejected challenger task success | {run.rejected_success.mean:.3f} | [{run.rejected_success.lower_95:.3f}, {run.rejected_success.upper_95:.3f}] | {run.rejected_success.sample_count} |",
        f"| Corrected challenger task success | {run.challenger_success.mean:.3f} | [{run.challenger_success.lower_95:.3f}, {run.challenger_success.upper_95:.3f}] | {run.challenger_success.sample_count} |",
        f"| Corrected minus champion paired effect | {run.success_rate_delta.mean:.3f} | [{run.success_rate_delta.lower_95:.3f}, {run.success_rate_delta.upper_95:.3f}] | {run.success_rate_delta.sample_count} |",
        "",
        "## Scheduler comparison",
        "",
        "All scheduler values and SLO results below are modeled predictions, not measurements.",
        "",
        "| Seed | Policy | Predicted SLO feasible | Completed | Predicted value | Cost (microunits) | Lost ticks |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in run.scheduler_comparisons:
        lines.append(
            f"| {item.seed} | {item.policy.value} | "
            f"{str(item.serving_predicted_slo_feasible).lower()} | "
            f"{item.completed_work_count} | {item.predicted_learning_value:.3f} | "
            f"{item.total_cost_microunits} | {item.lost_work_ticks} |"
        )
    lines.extend(("", "## Hypotheses", ""))
    for hypothesis in run.hypotheses:
        lines.extend(
            (
                f"### {hypothesis.hypothesis_id}: {hypothesis.status}",
                "",
                hypothesis.claim_scope,
                "",
                *(f"- Observation: {value}" for value in hypothesis.observations),
                *(f"- Limitation: {value}" for value in hypothesis.limitations),
                "",
            )
        )
    lines.extend(
        (
            "## Raw provenance",
            "",
            "Every executed summary and scheduler plan used above is content-addressed here. Paths are relative to the evaluation output directory.",
            "",
            *(f"- `{item.sha256}`  `{item.path}`" for item in run.raw_artifacts),
            "",
            "## Limitations",
            "",
            *(f"- {value}" for value in run.limitations),
            "",
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def _html(markdown: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>SLOForge Helix evaluation</title>"
        "<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}"
        "pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.4rem}</style>"
        "</head><body><h1>SLOForge Helix evaluation</h1><pre>"
        + html.escape(markdown)
        + "</pre></body></html>\n"
    )


def write_evaluation_reports(run: EvaluationRun, reports: Path) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    main = _markdown(run)
    (reports / "helix-evaluation.md").write_text(main, encoding="utf-8")
    (reports / "helix-evaluation.html").write_text(_html(main), encoding="utf-8")
    mapping = {
        "helix-replay.md": ("H1", "H2"),
        "helix-credit.md": ("H3", "H10"),
        "helix-resource-compiler.md": ("H5", "H6"),
        "helix-promotion.md": ("H4", "H7"),
        "helix-continual-learning.md": ("H8", "H9"),
    }
    by_id = {item.hypothesis_id: item for item in run.hypotheses}
    for filename, ids in mapping.items():
        lines = [f"# SLOForge Helix: {' and '.join(ids)}", ""]
        for hypothesis_id in ids:
            item = by_id[hypothesis_id]
            lines.extend(
                (
                    f"## {hypothesis_id}: {item.status}",
                    "",
                    item.claim_scope,
                    "",
                    *(f"- Observation: {value}" for value in item.observations),
                    *(f"- Limitation: {value}" for value in item.limitations),
                    *(f"- Artifact: `{value}`" for value in item.artifact_paths),
                    "",
                )
            )
        (reports / filename).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_reference_evaluation(
    output: Path,
    *,
    reports: Path,
    workload: Path,
    seeds: tuple[int, ...] = (41, 73, 113),
) -> EvaluationRun:
    """Execute the CPU flagship over multiple seeds and compile honest reports."""

    from sloforge.helix.evaluation.continual_learning import (
        run_continual_learning_campaign,
        validate_continual_learning_campaign,
    )
    from sloforge.helix.evaluation.experience_selection import (
        run_experience_selection_campaign,
        validate_experience_selection_campaign,
    )
    from sloforge.helix.evaluation.preservation import (
        load_preservation_scenario,
        run_preservation_campaign,
        validate_preservation_campaign,
    )
    from sloforge.helix.evaluation.staleness_campaign import (
        run_staleness_campaign,
        validate_staleness_campaign,
    )
    from sloforge.helix.evaluation.training_algorithms import (
        run_training_algorithm_campaign,
        validate_training_algorithm_campaign,
    )

    if len(seeds) < 2 or len(seeds) > 32 or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation requires two to 32 unique seeds")
    if any(seed < 0 or seed > 2**63 - 1 for seed in seeds):
        raise ValueError("evaluation seeds must fit signed 64-bit integers")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("evaluation output must be absent or empty")
    raw = output / "raw"
    demo_summaries: list[dict[str, Any]] = []
    demo_paths: list[Path] = []
    for seed in seeds:
        run_directory = raw / "flagship" / f"seed-{seed}"
        summary = run_cpu_demo(run_directory, seed=seed)
        summary_path = run_directory / "summary.json"
        demo_paths.append(summary_path)
        demo_summaries.append(summary)

    scheduler, scheduler_artifacts = _scheduler_comparisons(workload, raw, seeds)
    campaign_root = raw / "campaigns"
    training_campaign_output = campaign_root / "h3-training-algorithms"
    training_campaign = run_training_algorithm_campaign(training_campaign_output, seeds=seeds)
    if validate_training_algorithm_campaign(training_campaign_output) != training_campaign:
        raise ValueError("H3 campaign changed during publication validation")
    staleness_campaign_output = campaign_root / "h4-staleness"
    staleness_campaign = run_staleness_campaign(staleness_campaign_output, seeds=seeds)
    if validate_staleness_campaign(staleness_campaign_output) != staleness_campaign:
        raise ValueError("H4 campaign changed during publication validation")
    preservation_scenario = (
        _REPOSITORY_ROOT / "scenarios/helix/resource/rollout-preservation-campaign.json"
    )
    preservation_seed_limit = load_preservation_scenario(preservation_scenario).maximum_seeds
    preservation_campaign_output = campaign_root / "h6-preservation"
    preservation_campaign = run_preservation_campaign(
        preservation_campaign_output,
        scenario_path=preservation_scenario,
        seeds=seeds[:preservation_seed_limit],
    )
    if (
        validate_preservation_campaign(
            preservation_campaign_output, scenario_path=preservation_scenario
        )
        != preservation_campaign
    ):
        raise ValueError("H6 campaign changed during publication validation")
    continual_campaign_output = campaign_root / "h8-continual-learning"
    continual_campaign = run_continual_learning_campaign(continual_campaign_output, seeds=seeds)
    if validate_continual_learning_campaign(continual_campaign_output) != continual_campaign:
        raise ValueError("H8 campaign changed during publication validation")
    experience_campaign_output = campaign_root / "h9-experience-selection"
    experience_campaign = run_experience_selection_campaign(
        _REPOSITORY_ROOT / "scenarios/helix/experience/h9-reference-campaign.json",
        experience_campaign_output,
        seeds=seeds,
    )
    if validate_experience_selection_campaign(experience_campaign_output) != experience_campaign:
        raise ValueError("H9 campaign changed during publication validation")
    seed_observations = tuple(
        SeedObservation(
            seed=seed,
            champion_success_rate=float(summary["champion_success_rate"]),
            rejected_success_rate=float(dict(summary["rejected_candidate"])["success_rate"]),
            challenger_success_rate=float(dict(summary["corrected_candidate"])["success_rate"]),
            paired_success_rate_difference=(
                float(dict(summary["corrected_candidate"])["success_rate"])
                - float(summary["champion_success_rate"])
            ),
            summary_artifact_path=path.relative_to(output).as_posix(),
            summary_artifact_sha256=sha256(path.read_bytes()).hexdigest(),
        )
        for seed, summary, path in zip(seeds, demo_summaries, demo_paths, strict=True)
    )
    champion = tuple(item.champion_success_rate for item in seed_observations)
    rejected = tuple(item.rejected_success_rate for item in seed_observations)
    challenger = tuple(item.challenger_success_rate for item in seed_observations)
    delta = tuple(item.paired_success_rate_difference for item in seed_observations)
    joint_exact = sum(
        bool(dict(dict(item["replay"])["joint"])["exact_identity_verified"])
        for item in demo_summaries
    )
    non_joint_exact = sum(
        bool(dict(dict(item["replay"])[mode])["exact_identity_verified"])
        for item in demo_summaries
        for mode in ("transcript", "environment", "model")
    )
    demo_artifact_paths = tuple(path.relative_to(output).as_posix() for path in demo_paths)
    credit_paths = tuple(path.parent / "credit" / "branch-relative.json" for path in demo_paths)
    credit_artifact_paths = tuple(path.relative_to(output).as_posix() for path in credit_paths)
    training_campaign_path = (
        (training_campaign_output / "campaign.json").relative_to(output).as_posix()
    )
    staleness_campaign_path = (
        (staleness_campaign_output / "campaign.json").relative_to(output).as_posix()
    )
    preservation_campaign_path = (
        (preservation_campaign_output / "campaign.json").relative_to(output).as_posix()
    )
    continual_campaign_path = (
        (continual_campaign_output / "campaign.json").relative_to(output).as_posix()
    )
    experience_campaign_path = (
        (experience_campaign_output / "campaign.json").relative_to(output).as_posix()
    )
    hypotheses = _hypotheses(
        demo_paths=demo_artifact_paths,
        credit_paths=credit_artifact_paths,
        scheduler=scheduler,
        joint_exact=joint_exact,
        non_joint_exact=non_joint_exact,
        training_campaign=training_campaign,
        training_campaign_path=training_campaign_path,
        staleness_campaign=staleness_campaign,
        staleness_campaign_path=staleness_campaign_path,
        preservation_campaign=preservation_campaign,
        preservation_campaign_path=preservation_campaign_path,
        continual_campaign=continual_campaign,
        continual_campaign_path=continual_campaign_path,
        experience_campaign=experience_campaign,
        experience_campaign_path=experience_campaign_path,
    )
    champion_interval = _interval(champion)
    rejected_interval = _interval(rejected)
    challenger_interval = _interval(challenger)
    delta_interval = _interval(delta)
    raw_artifacts = [
        RawArtifact(
            artifact_kind="flagship_summary",
            path=item.summary_artifact_path,
            sha256=item.summary_artifact_sha256,
            seed=item.seed,
        )
        for item in seed_observations
    ]
    raw_artifacts.extend(scheduler_artifacts)
    campaign_artifacts: tuple[
        tuple[
            Literal[
                "training_algorithm_campaign",
                "staleness_campaign",
                "preservation_campaign",
                "continual_learning_campaign",
                "experience_selection_campaign",
            ],
            str,
        ],
        ...,
    ] = (
        ("training_algorithm_campaign", training_campaign_path),
        ("staleness_campaign", staleness_campaign_path),
        ("preservation_campaign", preservation_campaign_path),
        ("continual_learning_campaign", continual_campaign_path),
        ("experience_selection_campaign", experience_campaign_path),
    )
    for artifact_kind, artifact_path in campaign_artifacts:
        path = output / artifact_path
        raw_artifacts.append(
            RawArtifact(
                artifact_kind=artifact_kind,
                path=artifact_path,
                sha256=sha256(path.read_bytes()).hexdigest(),
            )
        )
    for seed, credit_path, credit_artifact_path in zip(
        seeds, credit_paths, credit_artifact_paths, strict=True
    ):
        raw_artifacts.append(
            RawArtifact(
                artifact_kind="branch_credit",
                path=credit_artifact_path,
                sha256=sha256(credit_path.read_bytes()).hexdigest(),
                seed=seed,
            )
        )
    ordered_artifacts = tuple(sorted(raw_artifacts, key=lambda item: item.path))
    rejected_promotion_count = sum(
        dict(item["rejected_candidate"])["promotion_state"] == "rejected" for item in demo_summaries
    )
    rollback_count = sum(
        dict(item["promotion"])["final_state"] == "rolled_back" for item in demo_summaries
    )
    incompatible_session_pin_count = sum(
        bool(dict(item["promotion"])["incompatible_session_pinned"]) for item in demo_summaries
    )
    software_manifest = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        **_repository_manifest(),
    }
    hardware_manifest: dict[str, str | int] = {
        "machine": platform.machine(),
        "system": platform.system(),
        "logical_cpu_count": int(__import__("os").cpu_count() or 0),
        "gpu_exercised": "false",
    }
    effect_size_definition: Literal[
        "paired absolute success-rate difference (corrected challenger minus champion)"
    ] = "paired absolute success-rate difference (corrected challenger minus champion)"
    limitations = (
        "No GPU or hardware-backed model path was available on this host.",
        "No production data, external API, live promotion, or external side effect was authorized.",
        "Student-t intervals summarize deterministic seed sensitivity with a small sample; seeds are not asserted to be independent population draws.",
        "H1-H10 statuses are descriptive evidence classifications, not multiplicity-adjusted statistical hypothesis tests.",
        "Predicted scheduler learning value and predicted serving-SLO feasibility are not observed quality or serving measurements.",
        "The static baseline uses one declared resource-vector unit per learning class because the source workload is authored for the value-aware policy.",
    )
    unsealed = EvaluationRun.model_construct(
        evaluation_id="0" * 64,
        seeds=seeds,
        seed_observations=seed_observations,
        champion_success=champion_interval,
        rejected_success=rejected_interval,
        challenger_success=challenger_interval,
        success_rate_delta=delta_interval,
        exact_joint_replay_count=joint_exact,
        non_joint_exact_identity_count=non_joint_exact,
        rejected_promotion_count=rejected_promotion_count,
        rollback_count=rollback_count,
        incompatible_session_pin_count=incompatible_session_pin_count,
        scheduler_comparisons=scheduler,
        hypotheses=hypotheses,
        raw_artifacts=ordered_artifacts,
        effect_size_definition=effect_size_definition,
        software_manifest=software_manifest,
        hardware_manifest=hardware_manifest,
        validation_class="deterministic-local-cpu-synthetic",
        limitations=limitations,
    )
    identity = unsealed.model_dump(mode="json", exclude={"evaluation_id"})
    run = EvaluationRun(
        evaluation_id=sha256(_canonical_bytes(identity)).hexdigest(),
        seeds=seeds,
        seed_observations=seed_observations,
        champion_success=champion_interval,
        rejected_success=rejected_interval,
        challenger_success=challenger_interval,
        success_rate_delta=delta_interval,
        exact_joint_replay_count=joint_exact,
        non_joint_exact_identity_count=non_joint_exact,
        rejected_promotion_count=rejected_promotion_count,
        rollback_count=rollback_count,
        incompatible_session_pin_count=incompatible_session_pin_count,
        scheduler_comparisons=scheduler,
        hypotheses=hypotheses,
        raw_artifacts=ordered_artifacts,
        effect_size_definition=effect_size_definition,
        software_manifest=software_manifest,
        hardware_manifest=hardware_manifest,
        validation_class="deterministic-local-cpu-synthetic",
        limitations=limitations,
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "evaluation.json", run)
    write_evaluation_reports(run, reports)
    return run


__all__ = [
    "MAX_REFERENCE_ARTIFACT_BYTES",
    "EvaluationRun",
    "run_reference_evaluation",
    "validate_reference_evaluation",
    "write_evaluation_reports",
]
