"""Bounded deterministic rollout worker for normal CPU CI and local demonstrations."""

from __future__ import annotations

import json
import math
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sloforge.helix.effects import AuditEvidence, Effect, EffectClass, EffectLedger
from sloforge.helix.environments import EnvironmentBranch
from sloforge.helix.environments.models import content_digest
from sloforge.helix.policy import DeterministicPolicy


class _RolloutModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RolloutEventKind(StrEnum):
    OBSERVATION = "observation"
    GENERATED_TOKEN = "generated_token"
    ACTION = "action"
    ENVIRONMENT_MUTATION = "environment_mutation"
    TERMINAL = "terminal"


class ActionMutation(_RolloutModel):
    path: Annotated[str, Field(min_length=1, max_length=4096)]
    content: Annotated[str, Field(max_length=8 * 1024 * 1024)]
    expected_before_hash: Annotated[str | None, Field(pattern=r"^[0-9a-f]{64}$")] = None
    mode: Annotated[int, Field(ge=0, le=0o7777)] | None = None

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if value.startswith("/") or ".." in value.split("/"):
            raise ValueError("rollout mutation path must stay inside the environment branch")
        return value


class CandidateAction(_RolloutModel):
    action: Annotated[str, Field(min_length=1, max_length=160)]
    tool_id: Annotated[str, Field(min_length=1, max_length=160)]
    effect_class: EffectClass
    mutations: Annotated[tuple[ActionMutation, ...], Field(max_length=128)] = ()
    compensation: Annotated[str | None, Field(max_length=2048)] = None

    @model_validator(mode="after")
    def validate_effect_contract(self) -> Self:
        if self.mutations and self.effect_class not in {
            EffectClass.IDEMPOTENT_WRITE,
            EffectClass.COMPENSATABLE_WRITE,
        }:
            raise ValueError("filesystem mutations require a declared write effect")
        if self.effect_class is EffectClass.COMPENSATABLE_WRITE and not self.compensation:
            raise ValueError("compensatable rollout action requires a compensation recipe")
        if self.effect_class in {
            EffectClass.IRREVERSIBLE_WRITE,
            EffectClass.EXTERNAL_UNKNOWN,
        }:
            raise ValueError("the local reference worker cannot run unsafe external effects")
        return self


class RolloutEvent(_RolloutModel):
    event_index: Annotated[int, Field(ge=0)]
    kind: RolloutEventKind
    policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    payload_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    previous_event_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    event_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    virtual_time_ns: Annotated[int, Field(ge=0)]


class TokenRecord(_RolloutModel):
    token_index: Annotated[int, Field(ge=0)]
    token: Annotated[str, Field(min_length=1, max_length=160)]
    policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    behavior_log_probability: float
    sampler_seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    rng_counter: Annotated[int, Field(ge=0)]
    event_index: Annotated[int, Field(ge=0)]

    @field_validator("behavior_log_probability")
    @classmethod
    def finite_log_probability(cls, value: float) -> float:
        if not math.isfinite(value) or value > 0.0:
            raise ValueError("behavior log probability must be finite and non-positive")
        return value


class ActionRecord(_RolloutModel):
    action_index: Annotated[int, Field(ge=0)]
    action: Annotated[str, Field(min_length=1, max_length=160)]
    tool_id: Annotated[str, Field(min_length=1, max_length=160)]
    policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    observation_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    behavior_log_probability: float
    effect_class: EffectClass
    environment_transition_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    event_index: Annotated[int, Field(ge=0)]

    @field_validator("behavior_log_probability")
    @classmethod
    def finite_log_probability(cls, value: float) -> float:
        if not math.isfinite(value) or value > 0.0:
            raise ValueError("behavior log probability must be finite and non-positive")
        return value


class ReferenceTrajectory(_RolloutModel):
    schema_version: Literal["sloforge.helix.reference-trajectory/v1"] = (
        "sloforge.helix.reference-trajectory/v1"
    )
    trajectory_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    tenant_id: Annotated[str, Field(min_length=1, max_length=160)]
    branch_id: Annotated[str, Field(min_length=1, max_length=160)]
    branch_group_id: Annotated[str, Field(min_length=1, max_length=160)]
    branch_point_id: Annotated[str, Field(min_length=1, max_length=160)]
    branch_point_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_model_capsule_id: Annotated[str, Field(min_length=1, max_length=256)]
    state_reuse_report_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    initial_environment_capsule_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    final_environment_capsule_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    policy_mode: Literal["strict"] = "strict"
    policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    events: Annotated[tuple[RolloutEvent, ...], Field(min_length=4, max_length=1024)]
    tokens: Annotated[tuple[TokenRecord, ...], Field(min_length=1, max_length=256)]
    actions: Annotated[tuple[ActionRecord, ...], Field(min_length=1, max_length=128)]
    effect_ledger_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    event_chain_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    terminal_status: Literal["completed"] = "completed"

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        if [event.event_index for event in self.events] != list(range(len(self.events))):
            raise ValueError("trajectory events must be contiguous from zero")
        if any(event.policy_epoch_id != self.policy_epoch_id for event in self.events):
            raise ValueError("strict trajectory events must use exactly one policy epoch")
        if any(token.policy_epoch_id != self.policy_epoch_id for token in self.tokens):
            raise ValueError("strict trajectory tokens must use exactly one policy epoch")
        if any(action.policy_epoch_id != self.policy_epoch_id for action in self.actions):
            raise ValueError("strict trajectory actions must use exactly one policy epoch")
        if self.events[-1].event_hash != self.event_chain_hash:
            raise ValueError("event chain head does not match the last event")
        previous = "0" * 64
        for event in self.events:
            if event.previous_event_hash != previous:
                raise ValueError("event chain predecessor mismatch")
            expected = _event_hash(
                event.event_index,
                event.kind,
                event.policy_epoch_id,
                event.payload_hash,
                previous,
                event.virtual_time_ns,
            )
            if expected != event.event_hash:
                raise ValueError("event chain content hash mismatch")
            previous = event.event_hash
        identity = self.model_dump(mode="json", exclude={"trajectory_id"})
        if _canonical_hash(identity) != self.trajectory_id:
            raise ValueError("trajectory content identity mismatch")
        return self


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _event_hash(
    event_index: int,
    kind: RolloutEventKind,
    policy_epoch_id: str,
    payload_hash: str,
    previous_event_hash: str,
    virtual_time_ns: int,
) -> str:
    return _canonical_hash(
        {
            "event_index": event_index,
            "kind": kind.value,
            "policy_epoch_id": policy_epoch_id,
            "payload_hash": payload_hash,
            "previous_event_hash": previous_event_hash,
            "virtual_time_ns": virtual_time_ns,
        }
    )


class ReferenceRolloutWorker:
    """Executes one bounded local decision and records proof-carrying provenance."""

    def __init__(self, *, tenant_id: str = "default", max_events: int = 1024) -> None:
        if not 4 <= max_events <= 100_000:
            raise ValueError("rollout event bound must be in 4..100000")
        self.tenant_id = tenant_id
        self.max_events = max_events

    @staticmethod
    def _append_event(
        events: list[RolloutEvent],
        *,
        kind: RolloutEventKind,
        policy_epoch_id: str,
        payload: object,
        virtual_time_ns: int,
    ) -> RolloutEvent:
        previous = events[-1].event_hash if events else "0" * 64
        payload_hash = _canonical_hash(payload)
        event_index = len(events)
        event = RolloutEvent(
            event_index=event_index,
            kind=kind,
            policy_epoch_id=policy_epoch_id,
            payload_hash=payload_hash,
            previous_event_hash=previous,
            event_hash=_event_hash(
                event_index,
                kind,
                policy_epoch_id,
                payload_hash,
                previous,
                virtual_time_ns,
            ),
            virtual_time_ns=virtual_time_ns,
        )
        events.append(event)
        return event

    def run(
        self,
        *,
        branch: EnvironmentBranch,
        initial_environment_capsule_id: str,
        branch_group_id: str,
        branch_point_id: str,
        branch_point_hash: str,
        source_model_capsule_id: str,
        state_reuse_report_hash: str,
        policy: DeterministicPolicy,
        observation: str,
        candidates: tuple[CandidateAction, ...],
        seed: int,
        rng_counter: int = 0,
        forced_action: str | None = None,
    ) -> ReferenceTrajectory:
        if not 0 <= seed <= 2**64 - 1:
            raise ValueError("rollout seed must fit an unsigned 64-bit value")
        if len(candidates) != len(policy.actions):
            raise ValueError("rollout candidate set must cover every policy action")
        candidate_by_name = {candidate.action: candidate for candidate in candidates}
        if set(candidate_by_name) != set(policy.actions):
            raise ValueError("rollout candidates do not match the loaded policy")
        if len(candidate_by_name) != len(candidates):
            raise ValueError("rollout candidate actions must be unique")
        ledger = EffectLedger(
            tenant_id=self.tenant_id,
            external_side_effects_enabled=False,
            max_records=max(1, len(candidates) * 2),
        )
        events: list[RolloutEvent] = []
        self._append_event(
            events,
            kind=RolloutEventKind.OBSERVATION,
            policy_epoch_id=policy.policy_epoch_id,
            payload={"observation_hash": content_digest(observation.encode("utf-8"))},
            virtual_time_ns=0,
        )
        decision = policy.decide(
            observation,
            seed=seed,
            rng_counter=rng_counter,
            forced_action=forced_action,
        )
        candidate = candidate_by_name[decision.action]
        token_event = self._append_event(
            events,
            kind=RolloutEventKind.GENERATED_TOKEN,
            policy_epoch_id=policy.policy_epoch_id,
            payload={
                "token": decision.action,
                "behavior_log_probability": decision.behavior_log_probability,
                "seed": seed,
                "rng_counter": rng_counter,
            },
            virtual_time_ns=1,
        )
        effect = Effect.build(
            candidate.effect_class,
            candidate.tool_id,
            target=f"branch://{self.tenant_id}/{branch.branch_id}",
            real_external=False,
            idempotency_key=(
                f"{branch.branch_id}:{decision.action}"
                if candidate.effect_class is EffectClass.IDEMPOTENT_WRITE
                else None
            ),
            compensation=candidate.compensation,
            tenant_id=self.tenant_id,
        )
        ledger.require_speculatable((effect,))
        effect_record = ledger.apply(
            effect,
            evidence=(
                AuditEvidence.build(
                    kind="policy-decision",
                    payload=decision.model_dump_json(),
                    metadata={"policy_epoch_id": policy.policy_epoch_id},
                ),
            )
            if candidate.effect_class not in {EffectClass.PURE, EffectClass.READ_ONLY}
            else (),
            speculative=True,
            virtual_time_ns=2,
        )
        action_event = self._append_event(
            events,
            kind=RolloutEventKind.ACTION,
            policy_epoch_id=policy.policy_epoch_id,
            payload={
                "action": decision.action,
                "tool_id": candidate.tool_id,
                "effect_id": effect.effect_id,
            },
            virtual_time_ns=2,
        )
        transition_parts: list[dict[str, object]] = []
        for offset, mutation in enumerate(candidate.mutations, start=1):
            before = branch.read_bytes(mutation.path) if mutation.expected_before_hash else b""
            if (
                mutation.expected_before_hash is not None
                and content_digest(before) != mutation.expected_before_hash
            ):
                raise ValueError(f"mutation precondition failed for {mutation.path}")
            branch.write_text(mutation.path, mutation.content, mode=mutation.mode)
            after_hash = content_digest(mutation.content.encode("utf-8"))
            transition_parts.append({"path": mutation.path, "after_hash": after_hash})
            self._append_event(
                events,
                kind=RolloutEventKind.ENVIRONMENT_MUTATION,
                policy_epoch_id=policy.policy_epoch_id,
                payload=transition_parts[-1],
                virtual_time_ns=2 + offset,
            )
        if len(events) + 1 > self.max_events:
            raise ValueError("rollout exceeds the configured event bound")
        environment_transition_hash = _canonical_hash(transition_parts)
        final_capsule = branch.checkpoint()
        ledger.commit(effect_record.sequence)
        terminal_event = self._append_event(
            events,
            kind=RolloutEventKind.TERMINAL,
            policy_epoch_id=policy.policy_epoch_id,
            payload={
                "status": "completed",
                "final_environment_capsule_id": final_capsule.capsule_id,
            },
            virtual_time_ns=len(events) + 3,
        )
        ledger_hash = _canonical_hash(ledger.audit())
        draft: dict[str, object] = {
            "schema_version": "sloforge.helix.reference-trajectory/v1",
            "tenant_id": self.tenant_id,
            "branch_id": branch.branch_id,
            "branch_group_id": branch_group_id,
            "branch_point_id": branch_point_id,
            "branch_point_hash": branch_point_hash,
            "source_model_capsule_id": source_model_capsule_id,
            "state_reuse_report_hash": state_reuse_report_hash,
            "initial_environment_capsule_id": initial_environment_capsule_id,
            "final_environment_capsule_id": final_capsule.capsule_id,
            "policy_mode": "strict",
            "policy_epoch_id": policy.policy_epoch_id,
            "seed": seed,
            "events": tuple(events),
            "tokens": (
                TokenRecord(
                    token_index=0,
                    token=decision.action,
                    policy_epoch_id=policy.policy_epoch_id,
                    behavior_log_probability=decision.behavior_log_probability,
                    sampler_seed=seed,
                    rng_counter=rng_counter,
                    event_index=token_event.event_index,
                ),
            ),
            "actions": (
                ActionRecord(
                    action_index=0,
                    action=decision.action,
                    tool_id=candidate.tool_id,
                    policy_epoch_id=policy.policy_epoch_id,
                    observation_hash=decision.observation_hash,
                    behavior_log_probability=decision.behavior_log_probability,
                    effect_class=candidate.effect_class,
                    environment_transition_hash=environment_transition_hash,
                    event_index=action_event.event_index,
                ),
            ),
            "effect_ledger_hash": ledger_hash,
            "event_chain_hash": terminal_event.event_hash,
            "terminal_status": "completed",
        }
        trajectory_id = _canonical_hash(
            {
                key: (
                    [item.model_dump(mode="json") for item in value]
                    if isinstance(value, tuple) and value and isinstance(value[0], BaseModel)
                    else value
                )
                for key, value in draft.items()
            }
        )
        return ReferenceTrajectory.model_validate(
            {"trajectory_id": trajectory_id, **draft}, strict=True
        )
