"""Executed CPU comparison of the four reference post-training objectives.

The campaign measures policy outcomes produced by the trainer.  It never uses
the training objective itself as a proxy for task success.
"""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.helix.policy import DeterministicPolicy
from sloforge.helix.scheduler.statistics import student_t_mean_interval_95
from sloforge.helix.trainers import (
    ReferenceTrainer,
    ReferenceTrainingSample,
    TrainingAlgorithm,
    TrainingResult,
)

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
MAX_CAMPAIGN_ARTIFACT_BYTES = 8 * 1024 * 1024


class _CampaignModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AlgorithmRun(_CampaignModel):
    seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    algorithm: TrainingAlgorithm
    training_examples: Annotated[int, Field(gt=0, le=1024)]
    candidate_policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    checkpoint_hash: Digest
    data_hash: Digest
    base_task_success_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    base_capability_retention_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    task_success_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    capability_retention_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    policy_kl: Annotated[float, Field(ge=0.0)]
    accepted_samples: Annotated[int, Field(gt=0)]
    rejected_samples: Annotated[int, Field(ge=0)]
    task_evaluation_count: Annotated[int, Field(gt=0)]
    retention_evaluation_count: Annotated[int, Field(gt=0)]
    evidence_path: Annotated[str, Field(min_length=1, max_length=4096)]
    evidence_sha256: Digest

    @model_validator(mode="after")
    def finite_metrics(self) -> Self:
        if not math.isfinite(self.policy_kl):
            raise ValueError("algorithm comparison KL must be finite")
        return self


class PairedAlgorithmEffect(_CampaignModel):
    baseline_algorithm: TrainingAlgorithm
    effect_definition: Literal[
        "branch-relative minus baseline mean task success across declared example budgets, paired by seed"
    ] = "branch-relative minus baseline mean task success across declared example budgets, paired by seed"
    mean: float
    lower_95: float
    upper_95: float
    sample_count: Annotated[int, Field(ge=2, le=32)]
    sample_standard_deviation: Annotated[float, Field(ge=0.0)]

    @model_validator(mode="after")
    def valid_effect(self) -> Self:
        values = (self.mean, self.lower_95, self.upper_95, self.sample_standard_deviation)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("H3 paired effect values must be finite")
        if not self.lower_95 <= self.mean <= self.upper_95:
            raise ValueError("H3 paired effect interval must contain its mean")
        if self.baseline_algorithm is TrainingAlgorithm.BRANCH_RELATIVE:
            raise ValueError("H3 branch-relative algorithm cannot be its own baseline")
        return self


class H3Result(_CampaignModel):
    hypothesis_id: Literal["H3"] = "H3"
    status: Literal["not_supported", "inconclusive"]
    decision_rule: Literal[
        "not_supported when any branch-relative paired mean is non-positive; otherwise inconclusive because training is deterministic and budgets also change optimizer steps"
    ] = "not_supported when any branch-relative paired mean is non-positive; otherwise inconclusive because training is deterministic and budgets also change optimizer steps"
    comparisons: Annotated[tuple[PairedAlgorithmEffect, ...], Field(min_length=3, max_length=3)]


class TrainingAlgorithmCampaign(_CampaignModel):
    schema_version: str = "sloforge.helix.training-algorithm-campaign/v1"
    campaign_id: Digest
    seeds: Annotated[tuple[int, ...], Field(min_length=2, max_length=32)]
    example_budgets: Annotated[tuple[int, ...], Field(min_length=2, max_length=16)]
    base_policy_hash: Digest
    base_metric_seed: Literal[0] = 0
    base_task_success_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    base_capability_retention_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    runs: Annotated[tuple[AlgorithmRun, ...], Field(min_length=16, max_length=4096)]
    h3_result: H3Result
    task_success_definition: str
    retention_definition: str
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def complete_campaign(self) -> Self:
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("algorithm campaign seeds must be unique")
        if tuple(sorted(set(self.example_budgets))) != self.example_budgets:
            raise ValueError("algorithm campaign budgets must be unique and sorted")
        expected = {
            (seed, algorithm, budget)
            for seed in self.seeds
            for algorithm in TrainingAlgorithm
            for budget in self.example_budgets
        }
        observed = {(run.seed, run.algorithm, run.training_examples) for run in self.runs}
        if observed != expected or len(self.runs) != len(expected):
            raise ValueError(
                "algorithm campaign must contain the full seed/algorithm/budget matrix"
            )
        if self.h3_result != _h3_result(self.seeds, self.example_budgets, self.runs):
            raise ValueError("H3 result disagrees with paired learning-curve observations")
        identity = self.model_dump(mode="json", exclude={"campaign_id"})
        if _digest(identity) != self.campaign_id:
            raise ValueError("algorithm campaign identity is invalid")
        return self


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


def _write(path: Path, value: object) -> str:
    document = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    payload = _canonical_bytes(document) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def _artifact_path(root: Path, relative: str) -> Path:
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or relative == ".":
        raise ValueError("algorithm campaign artifact path is not portable")
    candidate = root.joinpath(*parsed.parts)
    cursor = root
    for part in parsed.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("algorithm campaign artifact path contains a symbolic link")
    resolved_root = root.resolve()
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError("algorithm campaign artifact escapes its output root")
    if resolved.stat().st_size > MAX_CAMPAIGN_ARTIFACT_BYTES:
        raise ValueError("algorithm campaign artifact exceeds the byte limit")
    return resolved


def _base_policy() -> DeterministicPolicy:
    return DeterministicPolicy(
        policy_epoch_id="algorithm-campaign-champion@0",
        actions=("unsafe_shortcut", "guarded_fix", "safe_fallback"),
        logits=(0.9, 0.0, -0.3),
    )


def _evaluation_cases(
    policy: DeterministicPolicy,
    *,
    campaign_seed: int,
    observation_prefix: str,
    accepted_actions: frozenset[str],
    count: int = 128,
) -> tuple[dict[str, object], ...]:
    cases: list[dict[str, object]] = []
    for ordinal in range(count):
        decision_seed = (campaign_seed * 1_000_003 + ordinal * 97 + 17) % 2**64
        decision = policy.decide(
            f"{observation_prefix}:{ordinal % 11}",
            seed=decision_seed,
            rng_counter=ordinal,
        )
        cases.append(
            {
                "ordinal": ordinal,
                "observation": f"{observation_prefix}:{ordinal % 11}",
                "decision": decision.model_dump(mode="json"),
                "success": decision.action in accepted_actions,
            }
        )
    return tuple(cases)


def _case_rate(cases: tuple[dict[str, object], ...]) -> float:
    return sum(bool(case["success"]) for case in cases) / len(cases)


def _h3_result(
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    runs: tuple[AlgorithmRun, ...],
) -> H3Result:
    by_key = {(item.seed, item.algorithm, item.training_examples): item for item in runs}
    comparisons: list[PairedAlgorithmEffect] = []
    for baseline in TrainingAlgorithm:
        if baseline is TrainingAlgorithm.BRANCH_RELATIVE:
            continue
        effects = tuple(
            math.fsum(
                by_key[seed, TrainingAlgorithm.BRANCH_RELATIVE, budget].task_success_rate
                - by_key[seed, baseline, budget].task_success_rate
                for budget in budgets
            )
            / len(budgets)
            for seed in seeds
        )
        mean, lower, upper, deviation = student_t_mean_interval_95(effects)
        comparisons.append(
            PairedAlgorithmEffect(
                baseline_algorithm=baseline,
                mean=mean,
                lower_95=lower,
                upper_95=upper,
                sample_count=len(effects),
                sample_standard_deviation=deviation,
            )
        )
    typed = tuple(comparisons)
    status: Literal["not_supported", "inconclusive"] = (
        "not_supported" if any(item.mean <= 0.0 for item in typed) else "inconclusive"
    )
    return H3Result(status=status, comparisons=typed)


def _evaluation_rate(
    policy: DeterministicPolicy,
    *,
    campaign_seed: int,
    observation_prefix: str,
    accepted_actions: frozenset[str],
    count: int = 128,
) -> float:
    return _case_rate(
        _evaluation_cases(
            policy,
            campaign_seed=campaign_seed,
            observation_prefix=observation_prefix,
            accepted_actions=accepted_actions,
            count=count,
        )
    )


def _samples(
    base: DeterministicPolicy,
    *,
    algorithm: TrainingAlgorithm,
    budget: int,
    seed: int,
) -> tuple[ReferenceTrainingSample, ...]:
    actions = ("guarded_fix", "unsafe_shortcut", "safe_fallback")
    advantages = {"guarded_fix": 1.0, "unsafe_shortcut": -1.0, "safe_fallback": 0.25}
    samples: list[ReferenceTrainingSample] = []
    for ordinal in range(budget):
        action = actions[ordinal % len(actions)]
        paired = algorithm is TrainingAlgorithm.PAIRWISE_PREFERENCE
        chosen = "guarded_fix" if paired else None
        rejected = (
            "unsafe_shortcut"
            if paired and ordinal % 2 == 0
            else "safe_fallback"
            if paired
            else None
        )
        samples.append(
            ReferenceTrainingSample(
                sample_id=f"h3-{seed}-{algorithm.value}-{budget}-{ordinal}",
                trajectory_id=f"h3-trajectory-{seed}-{budget}-{ordinal}",
                branch_point_id=f"h3-branchpoint-{seed}",
                branch_group_id=f"h3-branchgroup-{seed}",
                policy_epoch_id=base.policy_epoch_id,
                action="guarded_fix" if paired else action,
                behavior_log_probability=base.log_probability("guarded_fix" if paired else action),
                advantage=advantages[action],
                token_weight=1.0 if action != "safe_fallback" else 0.6,
                reward_margin=(1.0 if paired else max(0.0, advantages[action])),
                chosen_action=chosen,
                rejected_action=rejected,
            )
        )
    return tuple(samples)


def _kl(reference: DeterministicPolicy, candidate: DeterministicPolicy) -> float:
    return math.fsum(
        left * (math.log(left) - math.log(right))
        for left, right in zip(reference.probabilities(), candidate.probabilities(), strict=True)
    )


def run_training_algorithm_campaign(
    output: Path,
    *,
    seeds: tuple[int, ...] = (41, 73, 113),
    example_budgets: tuple[int, ...] = (1, 2, 4, 8),
) -> TrainingAlgorithmCampaign:
    """Run a full deterministic seed/algorithm/example-budget matrix."""

    if len(seeds) < 2 or len(seeds) > 32 or len(set(seeds)) != len(seeds):
        raise ValueError("algorithm campaign requires two to 32 unique seeds")
    if any(seed < 0 or seed > 2**63 - 1 for seed in seeds):
        raise ValueError("algorithm campaign seeds must fit signed 64-bit integers")
    if tuple(sorted(set(example_budgets))) != example_budgets:
        raise ValueError("example budgets must be unique and sorted")
    if not example_budgets or any(budget <= 0 or budget > 1024 for budget in example_budgets):
        raise ValueError("example budgets must be in 1..1024")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("algorithm campaign output must be absent or empty")

    base = _base_policy()
    base_task = _evaluation_rate(
        base,
        campaign_seed=0,
        observation_prefix="targeted-repair",
        accepted_actions=frozenset({"guarded_fix"}),
    )
    base_retention = _evaluation_rate(
        base,
        campaign_seed=0,
        observation_prefix="capability-retention",
        accepted_actions=frozenset({"guarded_fix", "safe_fallback"}),
    )
    runs: list[AlgorithmRun] = []
    trainer = ReferenceTrainer(learning_rate=0.45, kl_coefficient=0.05)
    for seed in seeds:
        for algorithm in TrainingAlgorithm:
            for budget in example_budgets:
                samples = _samples(base, algorithm=algorithm, budget=budget, seed=seed)
                result = trainer.train(
                    base=base,
                    samples=samples,
                    algorithm=algorithm,
                    candidate_policy_epoch_id=f"h3-{algorithm.value}-{seed}-{budget}@1",
                    seed=seed,
                    steps=min(32, 2 + budget * 2),
                )
                base_task_cases = _evaluation_cases(
                    base,
                    campaign_seed=seed,
                    observation_prefix="targeted-repair",
                    accepted_actions=frozenset({"guarded_fix"}),
                )
                base_retention_cases = _evaluation_cases(
                    base,
                    campaign_seed=seed,
                    observation_prefix="capability-retention",
                    accepted_actions=frozenset({"guarded_fix", "safe_fallback"}),
                )
                task_cases = _evaluation_cases(
                    result.candidate,
                    campaign_seed=seed,
                    observation_prefix="targeted-repair",
                    accepted_actions=frozenset({"guarded_fix"}),
                )
                retention_cases = _evaluation_cases(
                    result.candidate,
                    campaign_seed=seed,
                    observation_prefix="capability-retention",
                    accepted_actions=frozenset({"guarded_fix", "safe_fallback"}),
                )
                evidence_path = (
                    output / "raw" / f"seed-{seed}" / algorithm.value / f"budget-{budget}.json"
                )
                evidence = {
                    "schema_version": "sloforge.helix.training-algorithm-evidence/v1",
                    "seed": seed,
                    "training_examples": budget,
                    "base_policy": base.model_dump(mode="json"),
                    "training_samples": [item.model_dump(mode="json") for item in samples],
                    "training_result": result.model_dump(mode="json"),
                    "base_task_cases": base_task_cases,
                    "base_retention_cases": base_retention_cases,
                    "candidate_task_cases": task_cases,
                    "candidate_retention_cases": retention_cases,
                }
                evidence_hash = _write(evidence_path, evidence)
                final_metric = result.metrics[-1]
                runs.append(
                    AlgorithmRun(
                        seed=seed,
                        algorithm=algorithm,
                        training_examples=budget,
                        candidate_policy_epoch_id=result.candidate.policy_epoch_id,
                        checkpoint_hash=result.checkpoint_hash,
                        data_hash=result.data_hash,
                        base_task_success_rate=_case_rate(base_task_cases),
                        base_capability_retention_rate=_case_rate(base_retention_cases),
                        task_success_rate=_case_rate(task_cases),
                        capability_retention_rate=_case_rate(retention_cases),
                        policy_kl=_kl(base, result.candidate),
                        accepted_samples=final_metric.accepted_samples,
                        rejected_samples=final_metric.rejected_samples,
                        task_evaluation_count=128,
                        retention_evaluation_count=128,
                        evidence_path=evidence_path.relative_to(output).as_posix(),
                        evidence_sha256=evidence_hash,
                    )
                )
    typed_runs = tuple(runs)
    h3_result = _h3_result(seeds, example_budgets, typed_runs)
    unsealed = TrainingAlgorithmCampaign.model_construct(
        campaign_id="0" * 64,
        seeds=seeds,
        example_budgets=example_budgets,
        base_policy_hash=base.weights_hash,
        base_metric_seed=0,
        base_task_success_rate=base_task,
        base_capability_retention_rate=base_retention,
        runs=typed_runs,
        h3_result=h3_result,
        task_success_definition="fraction of 128 deterministic held-out decisions selecting guarded_fix",
        retention_definition="fraction of 128 disjoint decisions avoiding unsafe_shortcut",
        limitations=(
            "The policy is a synthetic categorical CPU reference model.",
            "Example budgets change both sample count and bounded optimizer steps.",
            "The campaign measures fixture outcomes, not GPU-hour sample efficiency.",
            "Campaign seeds vary held-out evaluation decisions; the deterministic trainer does not use its seed for initialization or sample order.",
            "Example budgets also change bounded optimizer steps, so the curve is descriptive rather than compute-normalized sample efficiency.",
            "Student-t intervals summarize non-independent evaluation-seed sensitivity and are not multiplicity-adjusted tests.",
            "Capability retention is reported per run but is not part of the H3 decision rule.",
        ),
    )
    campaign = TrainingAlgorithmCampaign(
        campaign_id=_digest(unsealed.model_dump(mode="json", exclude={"campaign_id"})),
        seeds=seeds,
        example_budgets=example_budgets,
        base_policy_hash=base.weights_hash,
        base_metric_seed=0,
        base_task_success_rate=base_task,
        base_capability_retention_rate=base_retention,
        runs=typed_runs,
        h3_result=h3_result,
        task_success_definition=unsealed.task_success_definition,
        retention_definition=unsealed.retention_definition,
        limitations=unsealed.limitations,
    )
    _write(output / "campaign.json", campaign)
    return campaign


def validate_training_algorithm_campaign(output: Path) -> TrainingAlgorithmCampaign:
    """Reopen the campaign and rehash every trainer result."""

    campaign_path = _artifact_path(output, "campaign.json")
    campaign = TrainingAlgorithmCampaign.model_validate_json(
        campaign_path.read_bytes(), strict=True
    )
    canonical_base = _base_policy()
    if (
        campaign.base_policy_hash != canonical_base.weights_hash
        or campaign.base_task_success_rate
        != _evaluation_rate(
            canonical_base,
            campaign_seed=0,
            observation_prefix="targeted-repair",
            accepted_actions=frozenset({"guarded_fix"}),
        )
        or campaign.base_capability_retention_rate
        != _evaluation_rate(
            canonical_base,
            campaign_seed=0,
            observation_prefix="capability-retention",
            accepted_actions=frozenset({"guarded_fix", "safe_fallback"}),
        )
    ):
        raise ValueError("algorithm campaign canonical base-policy evidence is invalid")
    paths: set[str] = set()
    for run in campaign.runs:
        if run.evidence_path in paths:
            raise ValueError("algorithm campaign reused an evidence path")
        paths.add(run.evidence_path)
        artifact = _artifact_path(output, run.evidence_path)
        payload = artifact.read_bytes()
        if sha256(payload).hexdigest() != run.evidence_sha256:
            raise ValueError("algorithm campaign evidence digest mismatch")
        document = json.loads(payload)
        base = DeterministicPolicy.model_validate_json(
            _canonical_bytes(document["base_policy"]), strict=True
        )
        result = TrainingResult.model_validate_json(
            _canonical_bytes(document["training_result"]), strict=True
        )
        samples = tuple(
            ReferenceTrainingSample.model_validate_json(_canonical_bytes(item), strict=True)
            for item in document["training_samples"]
        )
        replayed_result = ReferenceTrainer(learning_rate=0.45, kl_coefficient=0.05).train(
            base=base,
            samples=samples,
            algorithm=run.algorithm,
            candidate_policy_epoch_id=run.candidate_policy_epoch_id,
            seed=run.seed,
            steps=min(32, 2 + run.training_examples * 2),
        )
        expected_base_task = _evaluation_cases(
            base,
            campaign_seed=run.seed,
            observation_prefix="targeted-repair",
            accepted_actions=frozenset({"guarded_fix"}),
        )
        expected_base_retention = _evaluation_cases(
            base,
            campaign_seed=run.seed,
            observation_prefix="capability-retention",
            accepted_actions=frozenset({"guarded_fix", "safe_fallback"}),
        )
        expected_task = _evaluation_cases(
            result.candidate,
            campaign_seed=run.seed,
            observation_prefix="targeted-repair",
            accepted_actions=frozenset({"guarded_fix"}),
        )
        expected_retention = _evaluation_cases(
            result.candidate,
            campaign_seed=run.seed,
            observation_prefix="capability-retention",
            accepted_actions=frozenset({"guarded_fix", "safe_fallback"}),
        )
        if (
            document["seed"] != run.seed
            or document["training_examples"] != run.training_examples
            or base != canonical_base
            or len(samples) != run.training_examples
            or samples
            != _samples(
                canonical_base,
                algorithm=run.algorithm,
                budget=run.training_examples,
                seed=run.seed,
            )
            or replayed_result != result
            or tuple(document["base_task_cases"]) != expected_base_task
            or tuple(document["base_retention_cases"]) != expected_base_retention
            or tuple(document["candidate_task_cases"]) != expected_task
            or tuple(document["candidate_retention_cases"]) != expected_retention
            or result.algorithm is not run.algorithm
            or result.candidate.policy_epoch_id != run.candidate_policy_epoch_id
            or result.checkpoint_hash != run.checkpoint_hash
            or result.data_hash != run.data_hash
            or result.metrics[-1].accepted_samples != run.accepted_samples
            or result.metrics[-1].rejected_samples != run.rejected_samples
            or len(expected_task) != run.task_evaluation_count
            or len(expected_retention) != run.retention_evaluation_count
            or not math.isclose(
                _kl(base, result.candidate), run.policy_kl, rel_tol=0.0, abs_tol=1e-12
            )
            or _case_rate(expected_base_task) != run.base_task_success_rate
            or _case_rate(expected_base_retention) != run.base_capability_retention_rate
            or _case_rate(expected_task) != run.task_success_rate
            or _case_rate(expected_retention) != run.capability_retention_rate
        ):
            raise ValueError("algorithm campaign summary disagrees with trainer evidence")
    expected_files = {"campaign.json", *paths}
    actual_files: set[str] = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError("algorithm campaign output contains a symbolic link")
        if path.is_file():
            actual_files.add(path.relative_to(output).as_posix())
    if actual_files != expected_files:
        raise ValueError("algorithm campaign raw artifact inventory is incomplete")
    return campaign


__all__ = [
    "MAX_CAMPAIGN_ARTIFACT_BYTES",
    "AlgorithmRun",
    "H3Result",
    "PairedAlgorithmEffect",
    "TrainingAlgorithmCampaign",
    "run_training_algorithm_campaign",
    "validate_training_algorithm_campaign",
]
