"""Replay traces and scoped comparison evidence."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.helix.capture.models import canonical_digest

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,255}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ReplayMode(StrEnum):
    EXACT = "exact"
    CAUSAL = "causal"
    SEMANTIC = "semantic"


class ComparisonScope(StrEnum):
    TRANSCRIPT = "transcript"
    ENVIRONMENT_ONLY = "environment_only"
    MODEL_ONLY = "model_only"
    JOINT = "joint"


class DivergenceKind(StrEnum):
    TOKEN = "token"
    ACTION = "action"
    ENVIRONMENT = "environment"
    REWARD = "reward"
    OUTCOME = "outcome"
    RESOURCE = "resource"


class ReplayModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ReplayIdentity(ReplayModel):
    policy_epoch_id: Identifier
    policy_digest: Digest
    runtime_name: NonEmpty
    runtime_version: NonEmpty
    runtime_build_hash: Digest
    model_hash: Digest
    model_state_digest: Digest
    environment_capsule_id: Identifier
    environment_state_digest: Digest
    rng_algorithm: NonEmpty
    rng_seed: int = Field(ge=0, le=2**64 - 1)
    rng_counter: int = Field(ge=0)
    tool_contract_hash: Digest


class ReplayEvent(ReplayModel):
    event_id: Identifier
    kind: NonEmpty
    payload_digest: Digest
    semantic_digest: Digest
    causal_parent_id: Identifier | None = None


class ReplayToken(ReplayModel):
    token_index: int = Field(ge=0)
    token_id: int = Field(ge=0)


class ResourceObservation(ReplayModel):
    name: Identifier
    value: float = Field(allow_inf_nan=False)
    unit: NonEmpty


class ReplayFrame(ReplayModel):
    action_index: int = Field(ge=0)
    action: ReplayEvent
    model_tokens: Annotated[tuple[ReplayToken, ...], Field(max_length=100_000)] = ()
    environment_events: Annotated[tuple[ReplayEvent, ...], Field(max_length=100_000)] = ()
    reward: float = Field(allow_inf_nan=False)
    outcome: NonEmpty
    resources: Annotated[tuple[ResourceObservation, ...], Field(max_length=10_000)] = ()

    @model_validator(mode="after")
    def validate_uniqueness(self) -> Self:
        ids = [self.action.event_id, *(item.event_id for item in self.environment_events)]
        if len(ids) != len(set(ids)):
            raise ValueError("replay frame event identifiers must be unique")
        resource_names = [item.name for item in self.resources]
        if len(resource_names) != len(set(resource_names)):
            raise ValueError("replay frame resource names must be unique")
        return self


class ReplayTrace(ReplayModel):
    schema_version: Literal["sloforge.helix.replay-trace/v1"] = "sloforge.helix.replay-trace/v1"
    trace_id: Digest
    branch_point_id: Digest
    identity: ReplayIdentity
    frames: Annotated[tuple[ReplayFrame, ...], Field(min_length=1, max_length=100_000)]
    terminal_outcome: NonEmpty
    trace_digest: Digest

    @model_validator(mode="after")
    def verify_trace(self) -> Self:
        token_count = sum(len(frame.model_tokens) for frame in self.frames)
        event_count = sum(len(frame.environment_events) + 1 for frame in self.frames)
        resource_count = sum(len(frame.resources) for frame in self.frames)
        if max(token_count, event_count, resource_count) > 1_000_000:
            raise ValueError("replay trace exceeds a total nested observation bound")
        action_indices = tuple(frame.action_index for frame in self.frames)
        if action_indices != tuple(
            range(action_indices[0], action_indices[0] + len(action_indices))
        ):
            raise ValueError("replay frame action indices must be dense and ordered")
        token_indices = tuple(
            token.token_index for frame in self.frames for token in frame.model_tokens
        )
        if token_indices and token_indices != tuple(
            range(token_indices[0], token_indices[0] + len(token_indices))
        ):
            raise ValueError("model token indices must be dense and ordered")
        seen_events: set[str] = set()
        for frame in self.frames:
            events = (frame.action, *frame.environment_events)
            for event in events:
                if event.event_id in seen_events:
                    raise ValueError("replay event identifiers must be trace-unique")
                if event.causal_parent_id is not None and event.causal_parent_id not in seen_events:
                    raise ValueError("replay causal parent must precede its child event")
                seen_events.add(event.event_id)
        payload = self.model_dump(mode="json", exclude={"trace_id", "trace_digest"})
        if canonical_digest(payload) != self.trace_digest:
            raise ValueError("replay trace digest is invalid")
        identity = self.model_dump(mode="json", exclude={"trace_id"})
        if canonical_digest(identity) != self.trace_id:
            raise ValueError("replay trace identifier is invalid")
        return self


class ReplayDivergence(ReplayModel):
    kind: DivergenceKind
    action_index: int = Field(ge=0)
    path: NonEmpty
    expected_digest: Digest
    observed_digest: Digest
    explanation: NonEmpty


class ReplayEvidence(ReplayModel):
    schema_version: Literal["sloforge.helix.replay-evidence/v1"] = (
        "sloforge.helix.replay-evidence/v1"
    )
    evidence_id: Digest
    expected_trace_id: Digest
    observed_trace_id: Digest
    mode: ReplayMode
    scope: ComparisonScope
    matched: bool
    exact_identity_verified: bool
    identity_differences: tuple[str, ...]
    first_token_divergence: ReplayDivergence | None = None
    first_action_divergence: ReplayDivergence | None = None
    first_environment_divergence: ReplayDivergence | None = None
    first_reward_divergence: ReplayDivergence | None = None
    first_outcome_divergence: ReplayDivergence | None = None
    first_resource_divergence: ReplayDivergence | None = None
    transcript_evidence_digest: Digest | None = None
    environment_state_evidence_digest: Digest | None = None
    model_state_evidence_digest: Digest | None = None
    joint_evidence_digest: Digest | None = None
    environment_state_equal: bool | None = None
    model_state_equal: bool | None = None
    transcript_establishes_state_equivalence: Literal[False] = False
    limitations: tuple[NonEmpty, ...]
    evidence_digest: Digest

    @model_validator(mode="after")
    def verify_evidence(self) -> Self:
        if not self.limitations:
            raise ValueError("replay evidence must declare its limitations")
        required_digests = {
            "transcript": self.scope in {ComparisonScope.TRANSCRIPT, ComparisonScope.JOINT},
            "environment": self.scope in {ComparisonScope.ENVIRONMENT_ONLY, ComparisonScope.JOINT},
            "model": self.scope in {ComparisonScope.MODEL_ONLY, ComparisonScope.JOINT},
            "joint": self.scope is ComparisonScope.JOINT,
        }
        observed_digests = {
            "transcript": self.transcript_evidence_digest,
            "environment": self.environment_state_evidence_digest,
            "model": self.model_state_evidence_digest,
            "joint": self.joint_evidence_digest,
        }
        if any(
            required != (observed_digests[name] is not None)
            for name, required in required_digests.items()
        ):
            raise ValueError("replay evidence digests must exactly match the declared scope")
        if self.scope in {ComparisonScope.ENVIRONMENT_ONLY, ComparisonScope.JOINT}:
            if self.environment_state_equal is None:
                raise ValueError("environment comparison requires a state equality result")
        elif self.environment_state_equal is not None:
            raise ValueError("non-environment comparison cannot compare environment state")
        if self.scope in {ComparisonScope.MODEL_ONLY, ComparisonScope.JOINT}:
            if self.model_state_equal is None:
                raise ValueError("model comparison requires a state equality result")
        elif self.model_state_equal is not None:
            raise ValueError("non-model comparison cannot compare model state")
        if self.mode is ReplayMode.EXACT:
            if self.identity_differences or not self.exact_identity_verified:
                raise ValueError("exact replay requires one fully verified identity contract")
        elif self.exact_identity_verified:
            raise ValueError("non-exact replay cannot claim exact identity verification")
        divergences = {
            "token": self.first_token_divergence,
            "action": self.first_action_divergence,
            "environment": self.first_environment_divergence,
            "reward": self.first_reward_divergence,
            "outcome": self.first_outcome_divergence,
            "resource": self.first_resource_divergence,
        }
        selected_names = {
            ComparisonScope.TRANSCRIPT: ("token", "action"),
            ComparisonScope.ENVIRONMENT_ONLY: (
                "environment",
                "reward",
                "outcome",
                "resource",
            ),
            ComparisonScope.MODEL_ONLY: ("token", "reward", "outcome", "resource"),
            ComparisonScope.JOINT: (
                "token",
                "action",
                "environment",
                "reward",
                "outcome",
                "resource",
            ),
        }[self.scope]
        state_mismatch = self.environment_state_equal is False or self.model_state_equal is False
        expected_match = not any(divergences[name] is not None for name in selected_names)
        expected_match = expected_match and not state_mismatch
        if self.matched != expected_match:
            raise ValueError("replay matched flag disagrees with scoped divergence evidence")
        payload = self.model_dump(mode="json", exclude={"evidence_id", "evidence_digest"})
        if canonical_digest(payload) != self.evidence_digest:
            raise ValueError("replay evidence digest is invalid")
        identity = self.model_dump(mode="json", exclude={"evidence_id"})
        if canonical_digest(identity) != self.evidence_id:
            raise ValueError("replay evidence identifier is invalid")
        return self


def build_replay_trace(
    *,
    branch_point_id: str,
    identity: ReplayIdentity,
    frames: tuple[ReplayFrame, ...],
    terminal_outcome: str,
) -> ReplayTrace:
    draft = {
        "schema_version": "sloforge.helix.replay-trace/v1",
        "trace_id": "0" * 64,
        "branch_point_id": branch_point_id,
        "identity": identity.model_dump(mode="json"),
        "frames": tuple(frame.model_dump(mode="json") for frame in frames),
        "terminal_outcome": terminal_outcome,
        "trace_digest": "0" * 64,
    }
    trace_digest = canonical_digest(
        {key: value for key, value in draft.items() if key not in {"trace_id", "trace_digest"}}
    )
    draft["trace_digest"] = trace_digest
    trace_id = canonical_digest({key: value for key, value in draft.items() if key != "trace_id"})
    return ReplayTrace(
        trace_id=trace_id,
        branch_point_id=branch_point_id,
        identity=identity,
        frames=frames,
        terminal_outcome=terminal_outcome,
        trace_digest=trace_digest,
    )
