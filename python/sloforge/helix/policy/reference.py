"""Deterministic categorical reference policy used by CPU tests and demos."""

from __future__ import annotations

import math
import random
from hashlib import sha256
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PolicyDecision(_PolicyModel):
    policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    observation_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    action: Annotated[str, Field(min_length=1, max_length=160)]
    action_index: Annotated[int, Field(ge=0)]
    behavior_log_probability: float
    probability: Annotated[float, Field(gt=0.0, le=1.0)]
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    rng_counter: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_log_probability(self) -> Self:
        if not math.isfinite(self.behavior_log_probability):
            raise ValueError("behavior log probability must be finite")
        if abs(math.log(self.probability) - self.behavior_log_probability) > 1e-12:
            raise ValueError("behavior log probability does not match action probability")
        return self


class DeterministicPolicy(_PolicyModel):
    """A deliberately small policy whose complete state is inspectable and hashable."""

    schema_version: str = "sloforge.helix.reference-policy/v1"
    policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    actions: Annotated[tuple[str, ...], Field(min_length=2, max_length=128)]
    logits: Annotated[tuple[float, ...], Field(min_length=2, max_length=128)]
    temperature: Annotated[float, Field(gt=0.0, le=100.0)] = 1.0

    @field_validator("actions")
    @classmethod
    def validate_actions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("policy actions must be unique")
        if any(not item or len(item) > 160 for item in value):
            raise ValueError("policy actions must be bounded non-empty identifiers")
        return value

    @model_validator(mode="after")
    def validate_logits(self) -> Self:
        if len(self.actions) != len(self.logits):
            raise ValueError("each action requires exactly one logit")
        if any(not math.isfinite(value) for value in self.logits):
            raise ValueError("policy logits must be finite")
        return self

    @property
    def weights_hash(self) -> str:
        payload = self.model_dump_json(exclude={"policy_epoch_id"})
        return sha256(payload.encode("utf-8")).hexdigest()

    def probabilities(self) -> tuple[float, ...]:
        scaled = tuple(value / self.temperature for value in self.logits)
        offset = max(scaled)
        weights = tuple(math.exp(value - offset) for value in scaled)
        total = sum(weights)
        return tuple(value / total for value in weights)

    def log_probability(self, action: str) -> float:
        try:
            index = self.actions.index(action)
        except ValueError as exc:
            raise ValueError(f"unknown policy action {action!r}") from exc
        return math.log(self.probabilities()[index])

    def decide(
        self,
        observation: str | bytes,
        *,
        seed: int,
        rng_counter: int = 0,
        forced_action: str | None = None,
    ) -> PolicyDecision:
        if seed < 0 or seed > 2**64 - 1:
            raise ValueError("seed must fit an unsigned 64-bit value")
        if rng_counter < 0:
            raise ValueError("rng counter must be non-negative")
        encoded = observation.encode("utf-8") if isinstance(observation, str) else observation
        observation_hash = sha256(encoded).hexdigest()
        probabilities = self.probabilities()
        if forced_action is None:
            rng = random.Random(f"helix-policy/v1:{seed}:{rng_counter}:{observation_hash}")
            draw = rng.random()
            cumulative = 0.0
            index = len(probabilities) - 1
            for candidate, probability in enumerate(probabilities):
                cumulative += probability
                if draw < cumulative:
                    index = candidate
                    break
        else:
            try:
                index = self.actions.index(forced_action)
            except ValueError as exc:
                raise ValueError(f"forced action {forced_action!r} is not in this policy") from exc
        probability = probabilities[index]
        return PolicyDecision(
            policy_epoch_id=self.policy_epoch_id,
            observation_hash=observation_hash,
            action=self.actions[index],
            action_index=index,
            behavior_log_probability=math.log(probability),
            probability=probability,
            seed=seed,
            rng_counter=rng_counter,
        )
