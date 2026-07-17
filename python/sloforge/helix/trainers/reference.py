"""Deterministic tiny trainer implementing the four required Helix objectives."""

from __future__ import annotations

import json
import math
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.helix.policy import DeterministicPolicy


class _TrainerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TrainingAlgorithm(StrEnum):
    SUCCESSFUL_BRANCH_DISTILLATION = "successful_branch_distillation"
    PAIRWISE_PREFERENCE = "pairwise_preference"
    GROUP_RELATIVE = "group_relative"
    BRANCH_RELATIVE = "branch_relative"


class ReferenceTrainingSample(_TrainerModel):
    sample_id: Annotated[str, Field(min_length=1, max_length=160)]
    trajectory_id: Annotated[str, Field(min_length=1, max_length=160)]
    branch_point_id: Annotated[str, Field(min_length=1, max_length=160)]
    branch_group_id: Annotated[str, Field(min_length=1, max_length=160)]
    policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    action: Annotated[str, Field(min_length=1, max_length=160)]
    behavior_log_probability: float
    advantage: float
    token_weight: Annotated[float, Field(ge=0.0, le=1.0)]
    reward_margin: Annotated[float, Field(ge=0.0)] = 0.0
    chosen_action: str | None = None
    rejected_action: str | None = None
    staleness_updates: Annotated[int, Field(ge=0)] = 0
    eligible: bool = True

    @model_validator(mode="after")
    def validate_sample(self) -> Self:
        values = (
            self.behavior_log_probability,
            self.advantage,
            self.token_weight,
            self.reward_margin,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("training sample objective inputs must be finite")
        if self.behavior_log_probability > 0.0:
            raise ValueError("behavior log probability must be non-positive")
        paired = self.chosen_action is not None or self.rejected_action is not None
        if paired and (self.chosen_action is None or self.rejected_action is None):
            raise ValueError("preference samples require both chosen and rejected actions")
        if paired and self.chosen_action == self.rejected_action:
            raise ValueError("preference actions must be distinct")
        return self


class TrainingMetric(_TrainerModel):
    step: Annotated[int, Field(ge=0)]
    objective: float
    policy_kl: Annotated[float, Field(ge=0.0)]
    accepted_samples: Annotated[int, Field(ge=0)]
    rejected_samples: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_finite_metric(self) -> Self:
        if not math.isfinite(self.objective) or not math.isfinite(self.policy_kl):
            raise ValueError("training metrics must be finite")
        return self


class TrainingResult(_TrainerModel):
    schema_version: str = "sloforge.helix.reference-training/v1"
    algorithm: TrainingAlgorithm
    base_policy_epoch_id: str
    candidate: DeterministicPolicy
    data_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    checkpoint_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    metrics: Annotated[tuple[TrainingMetric, ...], Field(min_length=1, max_length=10_000)]


class ReferenceTrainer:
    """A bounded categorical trainer, not a general-purpose RL framework."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.2,
        kl_coefficient: float = 0.05,
        ratio_clip: float = 0.2,
        maximum_staleness_updates: int = 4,
        maximum_steps: int = 256,
    ) -> None:
        if not math.isfinite(learning_rate) or not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning rate must be in (0, 1]")
        if not math.isfinite(kl_coefficient) or kl_coefficient < 0.0:
            raise ValueError("KL coefficient must be non-negative")
        if not math.isfinite(ratio_clip) or not 0.0 < ratio_clip < 1.0:
            raise ValueError("ratio clip must be in (0, 1)")
        if maximum_staleness_updates < 0:
            raise ValueError("maximum staleness must be non-negative")
        if not 1 <= maximum_steps <= 10_000:
            raise ValueError("maximum trainer steps must be bounded")
        self.learning_rate = learning_rate
        self.kl_coefficient = kl_coefficient
        self.ratio_clip = ratio_clip
        self.maximum_staleness_updates = maximum_staleness_updates
        self.maximum_steps = maximum_steps

    @staticmethod
    def _data_hash(samples: tuple[ReferenceTrainingSample, ...]) -> str:
        payload = "\n".join(
            sample.model_dump_json() for sample in sorted(samples, key=lambda x: x.sample_id)
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _kl(reference: tuple[float, ...], current: tuple[float, ...]) -> float:
        terms: list[float] = []
        for ref, cur in zip(reference, current, strict=True):
            if ref == 0.0:
                continue
            if cur <= 0.0:
                return math.inf
            terms.append(ref * (math.log(ref) - math.log(cur)))
        return max(0.0, math.fsum(terms))

    @staticmethod
    def _effective_sample(sample: ReferenceTrainingSample, algorithm: TrainingAlgorithm) -> bool:
        if algorithm is TrainingAlgorithm.SUCCESSFUL_BRANCH_DISTILLATION:
            return (
                sample.advantage > 0.0
                and sample.token_weight > 0.0
                and max(sample.reward_margin, sample.advantage) > 0.0
            )
        if algorithm is TrainingAlgorithm.PAIRWISE_PREFERENCE:
            return sample.token_weight > 0.0 and sample.reward_margin > 0.0
        if algorithm is TrainingAlgorithm.GROUP_RELATIVE:
            return sample.advantage != 0.0
        return sample.token_weight > 0.0 and sample.advantage != 0.0

    @staticmethod
    def _log_sigmoid(value: float) -> tuple[float, float]:
        if value >= 0.0:
            exp_negative = math.exp(-value)
            return -math.log1p(exp_negative), 1.0 / (1.0 + exp_negative)
        exp_value = math.exp(value)
        return value - math.log1p(exp_value), exp_value / (1.0 + exp_value)

    def _ppo_surrogate(self, *, log_ratio: float, advantage: float) -> tuple[float, float]:
        lower = 1.0 - self.ratio_clip
        upper = 1.0 + self.ratio_clip
        if advantage > 0.0 and log_ratio >= math.log(upper):
            return upper * advantage, 0.0
        if advantage < 0.0 and log_ratio <= math.log(lower):
            return lower * advantage, 0.0
        try:
            ratio = math.exp(log_ratio)
        except OverflowError as exc:
            raise FloatingPointError("importance ratio overflowed the bounded trainer") from exc
        surrogate = min(ratio * advantage, min(upper, max(lower, ratio)) * advantage)
        signal = ratio * advantage
        if not math.isfinite(surrogate) or not math.isfinite(signal):
            raise FloatingPointError("importance-weighted objective became non-finite")
        return surrogate, signal

    def train(
        self,
        *,
        base: DeterministicPolicy,
        samples: tuple[ReferenceTrainingSample, ...],
        algorithm: TrainingAlgorithm,
        candidate_policy_epoch_id: str,
        seed: int,
        steps: int = 8,
    ) -> TrainingResult:
        if not samples:
            raise ValueError("training requires at least one provenance-complete sample")
        if not 1 <= steps <= self.maximum_steps:
            raise ValueError("requested steps exceed the trainer bound")
        if seed < 0 or seed > 2**64 - 1:
            raise ValueError("seed must fit an unsigned 64-bit value")
        if candidate_policy_epoch_id == base.policy_epoch_id:
            raise ValueError("candidate policy epoch must differ from the immutable base epoch")
        sample_ids = [sample.sample_id for sample in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("training sample identifiers must be unique")
        if any(sample.policy_epoch_id != base.policy_epoch_id for sample in samples):
            raise ValueError("reference trainer rejects mixed behavior policy epochs")
        if any(sample.action not in base.actions for sample in samples):
            raise ValueError("training sample references an unknown action")
        if any(
            not math.isclose(
                sample.behavior_log_probability,
                base.log_probability(sample.action),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for sample in samples
        ):
            raise ValueError("behavior log probability disagrees with the immutable policy epoch")
        provenance_accepted = tuple(
            sample
            for sample in samples
            if sample.eligible and sample.staleness_updates <= self.maximum_staleness_updates
        )
        if algorithm is TrainingAlgorithm.PAIRWISE_PREFERENCE:
            if any(
                sample.chosen_action is None or sample.rejected_action is None
                for sample in provenance_accepted
            ):
                raise ValueError("pairwise preference requires chosen/rejected provenance")
            if any(
                sample.chosen_action not in base.actions
                or sample.rejected_action not in base.actions
                for sample in provenance_accepted
            ):
                raise ValueError("preference sample references an unknown action")
        accepted = tuple(
            sample for sample in provenance_accepted if self._effective_sample(sample, algorithm)
        )
        if not accepted:
            raise ValueError("all training samples were rejected by provenance or staleness policy")
        rejected_count = len(samples) - len(accepted)
        logits = list(base.logits)
        reference_probabilities = base.probabilities()
        metrics: list[TrainingMetric] = []
        for step in range(steps):
            current = DeterministicPolicy(
                policy_epoch_id=base.policy_epoch_id,
                actions=base.actions,
                logits=tuple(logits),
                temperature=base.temperature,
            )
            probabilities = current.probabilities()
            gradient = [0.0 for _ in logits]
            current_kl = self._kl(reference_probabilities, probabilities)
            if not math.isfinite(current_kl):
                raise FloatingPointError("policy KL became non-finite")
            objective = -self.kl_coefficient * current_kl
            for sample in accepted:
                action_index = base.actions.index(sample.action)
                current_log_probability = math.log(probabilities[action_index])
                if algorithm is TrainingAlgorithm.SUCCESSFUL_BRANCH_DISTILLATION:
                    weight = sample.token_weight * max(sample.reward_margin, sample.advantage)
                    signal = weight / base.temperature
                    objective += weight * current_log_probability
                elif algorithm is TrainingAlgorithm.PAIRWISE_PREFERENCE:
                    assert sample.chosen_action is not None
                    assert sample.rejected_action is not None
                    chosen = base.actions.index(sample.chosen_action)
                    rejected = base.actions.index(sample.rejected_action)
                    logit_delta = (logits[chosen] - logits[rejected]) / base.temperature
                    log_sigmoid, sigmoid = self._log_sigmoid(logit_delta)
                    weight = sample.token_weight * sample.reward_margin
                    signal = weight * (1.0 - sigmoid) / base.temperature
                    gradient[chosen] += signal
                    gradient[rejected] -= signal
                    objective += weight * log_sigmoid
                    continue
                else:
                    weight = (
                        sample.token_weight
                        if algorithm is TrainingAlgorithm.BRANCH_RELATIVE
                        else 1.0
                    )
                    surrogate, ratio_signal = self._ppo_surrogate(
                        log_ratio=current_log_probability - sample.behavior_log_probability,
                        advantage=sample.advantage,
                    )
                    signal = weight * ratio_signal / base.temperature
                    objective += weight * surrogate
                for index, probability in enumerate(probabilities):
                    gradient[index] -= signal * probability
                gradient[action_index] += signal
            logits = [
                value
                + self.learning_rate
                * (
                    delta / len(accepted)
                    - self.kl_coefficient * (probability - reference_probability) / base.temperature
                )
                for value, delta, probability, reference_probability in zip(
                    logits,
                    gradient,
                    probabilities,
                    reference_probabilities,
                    strict=True,
                )
            ]
            if any(not math.isfinite(value) for value in logits):
                raise FloatingPointError("trainer update produced non-finite logits")
            updated = DeterministicPolicy(
                policy_epoch_id=candidate_policy_epoch_id,
                actions=base.actions,
                logits=tuple(logits),
                temperature=base.temperature,
            )
            metrics.append(
                TrainingMetric(
                    step=step,
                    objective=objective,
                    policy_kl=self._kl(reference_probabilities, updated.probabilities()),
                    accepted_samples=len(accepted),
                    rejected_samples=rejected_count,
                )
            )
        candidate = DeterministicPolicy(
            policy_epoch_id=candidate_policy_epoch_id,
            actions=base.actions,
            logits=tuple(logits),
            temperature=base.temperature,
        )
        data_hash = self._data_hash(samples)
        checkpoint_identity = {
            "schema_version": "sloforge.helix.reference-checkpoint/v2",
            "algorithm": algorithm.value,
            "base": base.model_dump(mode="json"),
            "candidate": candidate.model_dump(mode="json"),
            "data_hash": data_hash,
            "seed": seed,
            "steps": steps,
            "trainer": {
                "learning_rate": self.learning_rate,
                "kl_coefficient": self.kl_coefficient,
                "ratio_clip": self.ratio_clip,
                "maximum_staleness_updates": self.maximum_staleness_updates,
            },
        }
        checkpoint_hash = sha256(
            json.dumps(
                checkpoint_identity,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return TrainingResult(
            algorithm=algorithm,
            base_policy_epoch_id=base.policy_epoch_id,
            candidate=candidate,
            data_hash=data_hash,
            checkpoint_hash=checkpoint_hash,
            seed=seed,
            metrics=tuple(metrics),
        )
