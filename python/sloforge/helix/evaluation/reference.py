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
import sys
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.helix.demo import run_cpu_demo
from sloforge.helix.scheduler import (
    SchedulerPolicy,
    SchedulerRequest,
    compile_resource_plan,
    load_scheduler_request,
)
from sloforge.helix.scheduler.statistics import student_t_mean_interval_95

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


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
        "flagship_summary", "scheduler_plan", "scheduler_workload", "branch_credit"
    ]
    path: Annotated[str, Field(min_length=1, max_length=4096)]
    sha256: Digest
    seed: int | None = None

    @model_validator(mode="after")
    def valid_seed_scope(self) -> RawArtifact:
        parsed = PurePosixPath(self.path)
        if parsed.is_absolute() or ".." in parsed.parts or self.path == ".":
            raise ValueError("raw artifact paths must be portable evaluation-relative paths")
        if self.artifact_kind == "scheduler_workload":
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
    status: Literal["supported_within_scope", "not_supported", "partial", "not_exercised"]
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
    static_basis = (
        "One declared resource-vector unit per learning class; aggregate capacity and budget "
        "constraints remain hard."
    )
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
                    static_limit_basis=(static_basis if policy is SchedulerPolicy.STATIC else None),
                )
            )
    return tuple(results), tuple(artifacts)


def _hypotheses(
    *,
    demo_paths: tuple[str, ...],
    credit_paths: tuple[str, ...],
    scheduler: tuple[SchedulerComparison, ...],
    joint_exact: int,
    non_joint_exact: int,
) -> tuple[HypothesisResult, ...]:
    demos = demo_paths
    scheduler_paths = tuple(item.artifact_path for item in scheduler)
    scheduler_preemption_runs = sum(item.preemption_count > 0 for item in scheduler)
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
            status="not_exercised",
            claim_scope="Comparative sample efficiency across four training algorithms.",
            observations=("The flagship exercised branch-relative optimization only.",),
            artifact_paths=demos,
            limitations=("A controlled multi-algorithm learning-curve experiment is absent.",),
        ),
        HypothesisResult(
            hypothesis_id="H4",
            status="not_exercised",
            claim_scope="Fail-closed policy provenance and bounded staleness semantics.",
            observations=(
                "The reference evaluation did not execute a multi-policy asynchronous staleness campaign.",
            ),
            artifact_paths=(),
            limitations=(
                "Source-level tests are not counted as executed evaluation evidence.",
                "Training stability under a large asynchronous optimizer was not measured.",
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
            status="partial" if scheduler_preemption_runs else "not_exercised",
            claim_scope="Deterministic capacity-reclamation accounting.",
            observations=(
                f"A preservation path was selected in {scheduler_preemption_runs}/{len(scheduler)} scheduler policy-seed runs.",
                "The workload declares restart, checkpoint, and Continuum alternatives, but declaration alone is not execution evidence.",
            ),
            artifact_paths=scheduler_paths,
            limitations=("No hardware-backed rollout migration timing was measured.",),
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
            status="not_exercised",
            claim_scope="Continual repair and forgetting over changing task distributions.",
            observations=("One targeted repair transaction was exercised per seed.",),
            artifact_paths=demos,
            limitations=("A longitudinal task-distribution sequence is absent.",),
        ),
        HypothesisResult(
            hypothesis_id="H9",
            status="not_exercised",
            claim_scope="Governed deterministic experience selection.",
            observations=(
                "The reference evaluation did not run a controlled comparison of experience-selection strategies.",
            ),
            artifact_paths=(),
            limitations=(
                "Source-level selector tests are not counted as executed evaluation evidence.",
                "Downstream learning gain from selection strategies was not measured.",
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
    hypotheses = _hypotheses(
        demo_paths=demo_artifact_paths,
        credit_paths=credit_artifact_paths,
        scheduler=scheduler,
        joint_exact=joint_exact,
        non_joint_exact=non_joint_exact,
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


__all__ = ["EvaluationRun", "run_reference_evaluation", "write_evaluation_reports"]
