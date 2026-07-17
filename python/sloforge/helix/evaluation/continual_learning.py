"""Longitudinal deterministic reference campaign for Helix continual repair."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.helix.policy import DeterministicPolicy
from sloforge.helix.trainers import (
    ReferenceTrainer,
    ReferenceTrainingSample,
    TrainingAlgorithm,
    TrainingResult,
)

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
MAX_CAMPAIGN_ARTIFACT_BYTES = 8 * 1024 * 1024


class _ContinualModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ContinualPhase(_ContinualModel):
    phase_index: Annotated[int, Field(ge=0, le=64)]
    task_id: Annotated[str, Field(min_length=1, max_length=160)]
    required_action: Annotated[str, Field(min_length=1, max_length=160)]
    base_policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    candidate_policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    champion_policy_epoch_id_after_gate: Annotated[str, Field(min_length=1, max_length=160)]
    target_success_before: Annotated[float, Field(ge=0.0, le=1.0)]
    target_success_candidate: Annotated[float, Field(ge=0.0, le=1.0)]
    prior_capability_before: Annotated[float, Field(ge=0.0, le=1.0)]
    prior_capability_candidate: Annotated[float, Field(ge=0.0, le=1.0)]
    accepted: bool
    candidate_rejected: bool
    training_examples: Annotated[int, Field(gt=0)]
    optimizer_steps: Annotated[int, Field(gt=0)]
    policy_kl: Annotated[float, Field(ge=0.0)]
    checkpoint_hash: Digest
    evidence_path: Annotated[str, Field(min_length=1, max_length=4096)]
    evidence_sha256: Digest

    @model_validator(mode="after")
    def valid_gate(self) -> Self:
        if self.accepted == self.candidate_rejected:
            raise ValueError("continual phase must either accept or reject its candidate")
        if not math.isfinite(self.policy_kl):
            raise ValueError("continual phase KL must be finite")
        expected = self.candidate_policy_epoch_id if self.accepted else self.base_policy_epoch_id
        if self.champion_policy_epoch_id_after_gate != expected:
            raise ValueError("continual phase champion does not match its gate disposition")
        return self


class ContinualSeedRun(_ContinualModel):
    seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    phases: Annotated[tuple[ContinualPhase, ...], Field(min_length=3, max_length=64)]
    final_task_success: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]]
    accepted_update_count: Annotated[int, Field(ge=0)]
    candidate_rejection_count: Annotated[int, Field(ge=0)]
    failure_recurrence_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    final_audit_path: Annotated[str, Field(min_length=1, max_length=4096)]
    final_audit_sha256: Digest

    @model_validator(mode="after")
    def complete_run(self) -> Self:
        if tuple(item.phase_index for item in self.phases) != tuple(range(len(self.phases))):
            raise ValueError("continual phases must be contiguous")
        if self.accepted_update_count != sum(item.accepted for item in self.phases):
            raise ValueError("accepted continual update count is inconsistent")
        if self.candidate_rejection_count != sum(item.candidate_rejected for item in self.phases):
            raise ValueError("continual candidate rejection count is inconsistent")
        return self


class ContinualLearningCampaign(_ContinualModel):
    schema_version: str = "sloforge.helix.continual-learning-campaign/v1"
    campaign_id: Digest
    seeds: Annotated[tuple[int, ...], Field(min_length=2, max_length=32)]
    task_sequence: Annotated[tuple[str, ...], Field(min_length=3, max_length=64)]
    runs: Annotated[tuple[ContinualSeedRun, ...], Field(min_length=2, max_length=32)]
    target_improvement_gate: Annotated[float, Field(gt=0.0, le=1.0)]
    retention_drop_gate: Annotated[float, Field(ge=0.0, le=1.0)]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def complete_campaign(self) -> Self:
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("continual campaign seeds must be unique")
        if tuple(item.seed for item in self.runs) != self.seeds:
            raise ValueError("continual runs must preserve seed order")
        if any(
            tuple(phase.task_id for phase in run.phases) != self.task_sequence for run in self.runs
        ):
            raise ValueError("continual runs must execute the declared task sequence")
        if _digest(self.model_dump(mode="json", exclude={"campaign_id"})) != self.campaign_id:
            raise ValueError("continual campaign identity is invalid")
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
        raise ValueError("continual campaign artifact path is not portable")
    candidate = root.joinpath(*parsed.parts)
    cursor = root
    for part in parsed.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("continual campaign artifact path contains a symbolic link")
    resolved_root = root.resolve()
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError("continual campaign artifact escapes its output root")
    if resolved.stat().st_size > MAX_CAMPAIGN_ARTIFACT_BYTES:
        raise ValueError("continual campaign artifact exceeds the byte limit")
    return resolved


def _base_policy() -> DeterministicPolicy:
    return DeterministicPolicy(
        policy_epoch_id="continual-champion@0",
        actions=("fast_path", "guarded_fix", "safe_fallback"),
        logits=(0.8, -0.1, -0.5),
    )


def _success_cases(
    policy: DeterministicPolicy,
    *,
    task_id: str,
    required_action: str,
    seed: int,
    count: int = 128,
) -> tuple[dict[str, object], ...]:
    cases: list[dict[str, object]] = []
    for ordinal in range(count):
        decision_seed = (seed * 1_000_033 + ordinal * 193 + 29) % 2**64
        decision = policy.decide(
            f"continual:{task_id}:{ordinal % 13}",
            seed=decision_seed,
            rng_counter=ordinal,
        )
        cases.append(
            {
                "ordinal": ordinal,
                "observation": f"continual:{task_id}:{ordinal % 13}",
                "decision": decision.model_dump(mode="json"),
                "success": decision.action == required_action,
            }
        )
    return tuple(cases)


def _case_rate(cases: tuple[dict[str, object], ...]) -> float:
    return sum(bool(case["success"]) for case in cases) / len(cases)


def _success_rate(
    policy: DeterministicPolicy,
    *,
    task_id: str,
    required_action: str,
    seed: int,
    count: int = 128,
) -> float:
    return _case_rate(
        _success_cases(
            policy,
            task_id=task_id,
            required_action=required_action,
            seed=seed,
            count=count,
        )
    )


def _training_samples(
    policy: DeterministicPolicy,
    *,
    phase_index: int,
    required_action: str,
    seed: int,
    count: int = 6,
) -> tuple[ReferenceTrainingSample, ...]:
    samples: list[ReferenceTrainingSample] = []
    actions = (required_action, *(action for action in policy.actions if action != required_action))
    for ordinal in range(count):
        action = actions[ordinal % len(actions)]
        samples.append(
            ReferenceTrainingSample(
                sample_id=f"h8-{seed}-{phase_index}-{ordinal}",
                trajectory_id=f"h8-trajectory-{seed}-{phase_index}-{ordinal}",
                branch_point_id=f"h8-branchpoint-{seed}-{phase_index}",
                branch_group_id=f"h8-branchgroup-{seed}-{phase_index}",
                policy_epoch_id=policy.policy_epoch_id,
                action=action,
                behavior_log_probability=policy.log_probability(action),
                advantage=1.0 if action == required_action else -0.5,
                token_weight=1.0,
                reward_margin=1.0 if action == required_action else 0.0,
            )
        )
    return tuple(samples)


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(values) / len(values) if values else 1.0


def _audit_seed(seed: int, task_id: str) -> int:
    payload = f"sloforge.helix.h8/final-audit/v1\0{seed}\0{task_id}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _kl(reference: DeterministicPolicy, candidate: DeterministicPolicy) -> float:
    return math.fsum(
        left * (math.log(left) - math.log(right))
        for left, right in zip(reference.probabilities(), candidate.probabilities(), strict=True)
    )


def run_continual_learning_campaign(
    output: Path,
    *,
    seeds: tuple[int, ...] = (41, 73, 113),
    target_improvement_gate: float = 0.02,
    retention_drop_gate: float = 0.03,
) -> ContinualLearningCampaign:
    """Train sequential challengers and apply an explicit retention gate."""

    if len(seeds) < 2 or len(seeds) > 32 or len(set(seeds)) != len(seeds):
        raise ValueError("continual campaign requires two to 32 unique seeds")
    if any(seed < 0 or seed > 2**63 - 1 for seed in seeds):
        raise ValueError("continual campaign seeds must fit signed 64-bit integers")
    if not 0.0 < target_improvement_gate <= 1.0:
        raise ValueError("target improvement gate must be in (0, 1]")
    if not 0.0 <= retention_drop_gate <= 1.0:
        raise ValueError("retention drop gate must be in [0, 1]")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("continual campaign output must be absent or empty")

    tasks = (
        ("retry-correctness", "guarded_fix"),
        ("latency-path", "fast_path"),
        ("tool-safety", "safe_fallback"),
    )
    runs: list[ContinualSeedRun] = []
    for seed in seeds:
        champion = _base_policy()
        accepted_tasks: list[tuple[str, str, float]] = []
        phases: list[ContinualPhase] = []
        for phase_index, (task_id, required_action) in enumerate(tasks):
            base = champion
            target_before_cases = _success_cases(
                base,
                task_id=task_id,
                required_action=required_action,
                seed=seed,
            )
            target_before = _case_rate(target_before_cases)
            samples = _training_samples(
                base,
                phase_index=phase_index,
                required_action=required_action,
                seed=seed,
            )
            result = ReferenceTrainer(
                learning_rate=0.55,
                kl_coefficient=0.05,
                ratio_clip=0.2,
            ).train(
                base=base,
                samples=samples,
                algorithm=TrainingAlgorithm.BRANCH_RELATIVE,
                candidate_policy_epoch_id=f"continual-{seed}-{phase_index}@1",
                seed=seed + phase_index,
                steps=10,
            )
            candidate = result.candidate
            target_candidate_cases = _success_cases(
                candidate,
                task_id=task_id,
                required_action=required_action,
                seed=seed,
            )
            target_candidate = _case_rate(target_candidate_cases)
            prior_cases: list[dict[str, object]] = []
            prior_before_rates: list[float] = []
            prior_candidate_rates: list[float] = []
            for prior_task, prior_action, accepted_success in accepted_tasks:
                base_cases = _success_cases(
                    base,
                    task_id=prior_task,
                    required_action=prior_action,
                    seed=seed,
                )
                candidate_cases = _success_cases(
                    candidate,
                    task_id=prior_task,
                    required_action=prior_action,
                    seed=seed,
                )
                prior_cases.append(
                    {
                        "task_id": prior_task,
                        "required_action": prior_action,
                        "accepted_success": accepted_success,
                        "base_cases": base_cases,
                        "candidate_cases": candidate_cases,
                    }
                )
                prior_before_rates.append(_case_rate(base_cases))
                prior_candidate_rates.append(_case_rate(candidate_cases))
            prior_before = _mean(tuple(prior_before_rates))
            prior_candidate = _mean(tuple(prior_candidate_rates))
            accepted = (
                target_candidate - target_before >= target_improvement_gate
                and prior_candidate >= prior_before - retention_drop_gate
            )
            champion = candidate if accepted else base
            if accepted:
                accepted_tasks.append((task_id, required_action, target_candidate))
            evidence = {
                "schema_version": "sloforge.helix.continual-phase-evidence/v1",
                "seed": seed,
                "phase_index": phase_index,
                "task_id": task_id,
                "required_action": required_action,
                "base_policy": base.model_dump(mode="json"),
                "training_samples": [item.model_dump(mode="json") for item in samples],
                "training_result": result.model_dump(mode="json"),
                "target_before_cases": target_before_cases,
                "target_candidate_cases": target_candidate_cases,
                "prior_cases": tuple(prior_cases),
                "target_success_before": target_before,
                "target_success_candidate": target_candidate,
                "prior_capability_before": prior_before,
                "prior_capability_candidate": prior_candidate,
                "target_improvement_gate": target_improvement_gate,
                "retention_drop_gate": retention_drop_gate,
                "accepted": accepted,
                "candidate_rejected": not accepted,
                "champion_after_gate": champion.model_dump(mode="json"),
            }
            evidence_path = output / "raw" / f"seed-{seed}" / f"phase-{phase_index}.json"
            evidence_hash = _write(evidence_path, evidence)
            phases.append(
                ContinualPhase(
                    phase_index=phase_index,
                    task_id=task_id,
                    required_action=required_action,
                    base_policy_epoch_id=base.policy_epoch_id,
                    candidate_policy_epoch_id=candidate.policy_epoch_id,
                    champion_policy_epoch_id_after_gate=champion.policy_epoch_id,
                    target_success_before=target_before,
                    target_success_candidate=target_candidate,
                    prior_capability_before=prior_before,
                    prior_capability_candidate=prior_candidate,
                    accepted=accepted,
                    candidate_rejected=not accepted,
                    training_examples=len(samples),
                    optimizer_steps=len(result.metrics),
                    policy_kl=_kl(base, candidate),
                    checkpoint_hash=result.checkpoint_hash,
                    evidence_path=evidence_path.relative_to(output).as_posix(),
                    evidence_sha256=evidence_hash,
                )
            )
        final_audit_cases = {
            task_id: _success_cases(
                champion,
                task_id=task_id,
                required_action=required_action,
                seed=_audit_seed(seed, task_id),
            )
            for task_id, required_action in tasks
        }
        final_task_success = {
            task_id: _case_rate(final_audit_cases[task_id]) for task_id, _required_action in tasks
        }
        recurrences = sum(
            final_task_success[task_id] < accepted_success - retention_drop_gate
            for task_id, _, accepted_success in accepted_tasks
        )
        final_audit_path = output / "raw" / f"seed-{seed}" / "final-audit.json"
        final_audit_hash = _write(
            final_audit_path,
            {
                "schema_version": "sloforge.helix.continual-final-audit/v1",
                "seed": seed,
                "champion_policy": champion.model_dump(mode="json"),
                "accepted_tasks": accepted_tasks,
                "task_cases": final_audit_cases,
            },
        )
        runs.append(
            ContinualSeedRun(
                seed=seed,
                phases=tuple(phases),
                final_task_success=final_task_success,
                accepted_update_count=sum(item.accepted for item in phases),
                candidate_rejection_count=sum(item.candidate_rejected for item in phases),
                failure_recurrence_rate=(
                    recurrences / len(accepted_tasks) if accepted_tasks else 0.0
                ),
                final_audit_path=final_audit_path.relative_to(output).as_posix(),
                final_audit_sha256=final_audit_hash,
            )
        )
    unsealed = ContinualLearningCampaign.model_construct(
        campaign_id="0" * 64,
        seeds=seeds,
        task_sequence=tuple(task for task, _ in tasks),
        runs=tuple(runs),
        target_improvement_gate=target_improvement_gate,
        retention_drop_gate=retention_drop_gate,
        limitations=(
            "The categorical policy has no learned observation representation.",
            "The task sequence and gates are predeclared deterministic CPU fixtures.",
            "The retention gate protects accepted capabilities but may prevent forward adaptation.",
            "Final recurrence uses domain-separated audit decisions that are not used by the candidate gate.",
        ),
    )
    campaign = ContinualLearningCampaign(
        campaign_id=_digest(unsealed.model_dump(mode="json", exclude={"campaign_id"})),
        seeds=seeds,
        task_sequence=unsealed.task_sequence,
        runs=tuple(runs),
        target_improvement_gate=target_improvement_gate,
        retention_drop_gate=retention_drop_gate,
        limitations=unsealed.limitations,
    )
    _write(output / "campaign.json", campaign)
    return campaign


def validate_continual_learning_campaign(output: Path) -> ContinualLearningCampaign:
    """Reopen the campaign and rehash every phase and checkpoint record."""

    campaign_path = _artifact_path(output, "campaign.json")
    campaign = ContinualLearningCampaign.model_validate_json(
        campaign_path.read_bytes(), strict=True
    )
    canonical_tasks = (
        ("retry-correctness", "guarded_fix"),
        ("latency-path", "fast_path"),
        ("tool-safety", "safe_fallback"),
    )
    if campaign.task_sequence != tuple(item[0] for item in canonical_tasks):
        raise ValueError("continual campaign changed the canonical task sequence")
    paths: set[str] = set()
    for run in campaign.runs:
        current_champion = _base_policy()
        accepted_tasks: list[tuple[str, str, float]] = []
        for phase, (expected_task, expected_action) in zip(
            run.phases, canonical_tasks, strict=True
        ):
            if (phase.task_id, phase.required_action) != (expected_task, expected_action):
                raise ValueError("continual campaign changed a canonical task/action binding")
            if phase.evidence_path in paths:
                raise ValueError("continual campaign reused a phase evidence path")
            paths.add(phase.evidence_path)
            artifact = _artifact_path(output, phase.evidence_path)
            payload = artifact.read_bytes()
            if sha256(payload).hexdigest() != phase.evidence_sha256:
                raise ValueError("continual campaign phase evidence digest mismatch")
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
            replayed_result = ReferenceTrainer(
                learning_rate=0.55,
                kl_coefficient=0.05,
                ratio_clip=0.2,
            ).train(
                base=base,
                samples=samples,
                algorithm=TrainingAlgorithm.BRANCH_RELATIVE,
                candidate_policy_epoch_id=phase.candidate_policy_epoch_id,
                seed=run.seed + phase.phase_index,
                steps=10,
            )
            champion_after = DeterministicPolicy.model_validate_json(
                _canonical_bytes(document["champion_after_gate"]), strict=True
            )
            if base != current_champion:
                raise ValueError("continual campaign phase did not begin from the gated champion")
            expected_target_before = _success_cases(
                base,
                task_id=phase.task_id,
                required_action=phase.required_action,
                seed=run.seed,
            )
            expected_target_candidate = _success_cases(
                result.candidate,
                task_id=phase.task_id,
                required_action=phase.required_action,
                seed=run.seed,
            )
            expected_prior_before: list[float] = []
            expected_prior_candidate: list[float] = []
            raw_prior_cases = document["prior_cases"]
            if len(raw_prior_cases) != len(accepted_tasks):
                raise ValueError("continual campaign prior-capability set is incomplete")
            for prior, accepted_task in zip(raw_prior_cases, accepted_tasks, strict=True):
                if (
                    prior["task_id"],
                    prior["required_action"],
                    prior["accepted_success"],
                ) != accepted_task:
                    raise ValueError("continual campaign changed an accepted prior task")
                base_cases = _success_cases(
                    base,
                    task_id=prior["task_id"],
                    required_action=prior["required_action"],
                    seed=run.seed,
                )
                candidate_cases = _success_cases(
                    result.candidate,
                    task_id=prior["task_id"],
                    required_action=prior["required_action"],
                    seed=run.seed,
                )
                if (
                    tuple(prior["base_cases"]) != base_cases
                    or tuple(prior["candidate_cases"]) != candidate_cases
                ):
                    raise ValueError("continual campaign prior-capability samples do not replay")
                expected_prior_before.append(_case_rate(base_cases))
                expected_prior_candidate.append(_case_rate(candidate_cases))
            if (
                document["seed"] != run.seed
                or samples
                != _training_samples(
                    base,
                    phase_index=phase.phase_index,
                    required_action=phase.required_action,
                    seed=run.seed,
                )
                or replayed_result != result
                or document["phase_index"] != phase.phase_index
                or document["task_id"] != phase.task_id
                or document["required_action"] != phase.required_action
                or tuple(document["target_before_cases"]) != expected_target_before
                or tuple(document["target_candidate_cases"]) != expected_target_candidate
                or _case_rate(expected_target_before) != phase.target_success_before
                or _case_rate(expected_target_candidate) != phase.target_success_candidate
                or _mean(tuple(expected_prior_before)) != phase.prior_capability_before
                or _mean(tuple(expected_prior_candidate)) != phase.prior_capability_candidate
                or document["accepted"] != phase.accepted
                or document["candidate_rejected"] != phase.candidate_rejected
                or document["target_improvement_gate"] != campaign.target_improvement_gate
                or document["retention_drop_gate"] != campaign.retention_drop_gate
                or phase.accepted
                != (
                    phase.target_success_candidate - phase.target_success_before
                    >= campaign.target_improvement_gate
                    and phase.prior_capability_candidate
                    >= phase.prior_capability_before - campaign.retention_drop_gate
                )
                or result.base_policy_epoch_id != phase.base_policy_epoch_id
                or result.checkpoint_hash != phase.checkpoint_hash
                or result.candidate.policy_epoch_id != phase.candidate_policy_epoch_id
                or phase.training_examples
                != result.metrics[-1].accepted_samples + result.metrics[-1].rejected_samples
                or phase.optimizer_steps != len(result.metrics)
                or not math.isclose(
                    _kl(base, result.candidate), phase.policy_kl, rel_tol=0.0, abs_tol=1e-12
                )
                or champion_after != (result.candidate if phase.accepted else base)
            ):
                raise ValueError("continual campaign summary disagrees with phase evidence")
            current_champion = champion_after
            if phase.accepted:
                accepted_tasks.append(
                    (phase.task_id, phase.required_action, phase.target_success_candidate)
                )
        task_actions = {
            "retry-correctness": "guarded_fix",
            "latency-path": "fast_path",
            "tool-safety": "safe_fallback",
        }
        if run.final_audit_path in paths:
            raise ValueError("continual campaign reused its final audit path")
        paths.add(run.final_audit_path)
        final_path = _artifact_path(output, run.final_audit_path)
        final_payload = final_path.read_bytes()
        if sha256(final_payload).hexdigest() != run.final_audit_sha256:
            raise ValueError("continual campaign final audit digest mismatch")
        final_document = json.loads(final_payload)
        final_champion = DeterministicPolicy.model_validate_json(
            _canonical_bytes(final_document["champion_policy"]), strict=True
        )
        recomputed_cases = {
            task_id: _success_cases(
                current_champion,
                task_id=task_id,
                required_action=task_actions[task_id],
                seed=_audit_seed(run.seed, task_id),
            )
            for task_id in campaign.task_sequence
        }
        recomputed_final = {
            task_id: _case_rate(recomputed_cases[task_id]) for task_id in campaign.task_sequence
        }
        if (
            final_document["seed"] != run.seed
            or final_champion != current_champion
            or tuple(tuple(item) for item in final_document["accepted_tasks"])
            != tuple(accepted_tasks)
            or {task_id: tuple(cases) for task_id, cases in final_document["task_cases"].items()}
            != recomputed_cases
        ):
            raise ValueError("continual campaign final audit evidence does not replay")
        recurrences = sum(
            recomputed_final[task_id] < accepted_success - campaign.retention_drop_gate
            for task_id, _, accepted_success in accepted_tasks
        )
        expected_recurrence = recurrences / len(accepted_tasks) if accepted_tasks else 0.0
        if recomputed_final != run.final_task_success or not math.isclose(
            expected_recurrence, run.failure_recurrence_rate, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("continual campaign final outcomes do not replay")
    expected_files = {"campaign.json", *paths}
    actual_files: set[str] = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError("continual campaign output contains a symbolic link")
        if path.is_file():
            actual_files.add(path.relative_to(output).as_posix())
    if actual_files != expected_files:
        raise ValueError("continual campaign raw artifact inventory is incomplete")
    return campaign


__all__ = [
    "MAX_CAMPAIGN_ARTIFACT_BYTES",
    "ContinualLearningCampaign",
    "ContinualPhase",
    "ContinualSeedRun",
    "run_continual_learning_campaign",
    "validate_continual_learning_campaign",
]
