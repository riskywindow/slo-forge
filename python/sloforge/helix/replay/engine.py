"""Exact, causal, and semantic replay comparison with first-divergence evidence."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from sloforge.helix.capture.models import canonical_digest

from .models import (
    ComparisonScope,
    DivergenceKind,
    ReplayDivergence,
    ReplayEvent,
    ReplayEvidence,
    ReplayFrame,
    ReplayMode,
    ReplayTrace,
)


class ReplayError(RuntimeError):
    code = "helix_replay_error"


class ExactReplayIdentityMismatch(ReplayError):
    code = "helix_exact_replay_identity_mismatch"


class ReplayResourceLimit(ReplayError):
    code = "helix_replay_resource_limit"


@dataclass(frozen=True, slots=True)
class ReplayTolerances:
    reward_absolute: float = 0.0
    resource_absolute: float = 0.0

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.reward_absolute, self.resource_absolute)
        ):
            raise ValueError("replay tolerances must be finite and non-negative")


_EXACT_TOLERANCES = ReplayTolerances()


def _digest(value: object) -> str:
    return canonical_digest(value)


def _identity_differences(expected: ReplayTrace, observed: ReplayTrace) -> tuple[str, ...]:
    left = expected.identity.model_dump(mode="json")
    right = observed.identity.model_dump(mode="json")
    differences = [key for key in left if left[key] != right[key]]
    if expected.branch_point_id != observed.branch_point_id:
        differences.append("branch_point_id")
    return tuple(differences)


def _event_equal(left: ReplayEvent, right: ReplayEvent, mode: ReplayMode) -> bool:
    if mode is ReplayMode.EXACT:
        return left == right
    if mode is ReplayMode.CAUSAL:
        return (
            left.kind == right.kind
            and left.semantic_digest == right.semantic_digest
            and left.causal_parent_id == right.causal_parent_id
        )
    return left.kind == right.kind and left.semantic_digest == right.semantic_digest


def _divergence(
    kind: DivergenceKind,
    action_index: int,
    path: str,
    expected: object,
    observed: object,
    explanation: str,
) -> ReplayDivergence:
    return ReplayDivergence(
        kind=kind,
        action_index=action_index,
        path=path,
        expected_digest=_digest(expected),
        observed_digest=_digest(observed),
        explanation=explanation,
    )


def _first_action(
    expected: tuple[ReplayFrame, ...], observed: tuple[ReplayFrame, ...], mode: ReplayMode
) -> ReplayDivergence | None:
    for index in range(max(len(expected), len(observed))):
        if index >= len(expected) or index >= len(observed):
            action_index = (
                expected[index].action_index
                if index < len(expected)
                else observed[index].action_index
                if index < len(observed)
                else 0
            )
            return _divergence(
                DivergenceKind.ACTION,
                action_index,
                f"frames[{index}].action",
                expected[index].action.model_dump(mode="json") if index < len(expected) else None,
                observed[index].action.model_dump(mode="json") if index < len(observed) else None,
                "action stream length differs",
            )
        if not _event_equal(expected[index].action, observed[index].action, mode):
            return _divergence(
                DivergenceKind.ACTION,
                expected[index].action_index,
                f"frames[{index}].action",
                expected[index].action.model_dump(mode="json"),
                observed[index].action.model_dump(mode="json"),
                f"first {mode.value} action mismatch",
            )
    return None


def _flatten_tokens(frames: tuple[ReplayFrame, ...]) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (frame.action_index, token.token_index, token.token_id)
        for frame in frames
        for token in frame.model_tokens
    )


def _first_token(
    expected: tuple[ReplayFrame, ...], observed: tuple[ReplayFrame, ...]
) -> ReplayDivergence | None:
    left = _flatten_tokens(expected)
    right = _flatten_tokens(observed)
    for ordinal in range(max(len(left), len(right))):
        expected_item = left[ordinal] if ordinal < len(left) else None
        observed_item = right[ordinal] if ordinal < len(right) else None
        if expected_item != observed_item:
            action_index = (
                expected_item[0]
                if expected_item is not None
                else observed_item[0]
                if observed_item is not None
                else 0
            )
            return _divergence(
                DivergenceKind.TOKEN,
                action_index,
                f"model_tokens[{ordinal}]",
                expected_item,
                observed_item,
                "first emitted model token mismatch",
            )
    return None


def _flatten_environment(
    frames: tuple[ReplayFrame, ...],
) -> tuple[tuple[int, ReplayEvent], ...]:
    return tuple(
        (frame.action_index, event) for frame in frames for event in frame.environment_events
    )


def _first_environment(
    expected: tuple[ReplayFrame, ...], observed: tuple[ReplayFrame, ...], mode: ReplayMode
) -> ReplayDivergence | None:
    left = _flatten_environment(expected)
    right = _flatten_environment(observed)
    for ordinal in range(max(len(left), len(right))):
        expected_item = left[ordinal] if ordinal < len(left) else None
        observed_item = right[ordinal] if ordinal < len(right) else None
        if (
            expected_item is None
            or observed_item is None
            or not _event_equal(expected_item[1], observed_item[1], mode)
        ):
            action_index = (
                expected_item[0]
                if expected_item is not None
                else observed_item[0]
                if observed_item is not None
                else 0
            )
            return _divergence(
                DivergenceKind.ENVIRONMENT,
                action_index,
                f"environment_events[{ordinal}]",
                expected_item[1].model_dump(mode="json") if expected_item else None,
                observed_item[1].model_dump(mode="json") if observed_item else None,
                f"first {mode.value} environment event mismatch",
            )
    return None


def _first_reward(
    expected: tuple[ReplayFrame, ...],
    observed: tuple[ReplayFrame, ...],
    tolerance: float,
) -> ReplayDivergence | None:
    for index in range(max(len(expected), len(observed))):
        left = expected[index].reward if index < len(expected) else None
        right = observed[index].reward if index < len(observed) else None
        if left is None or right is None or abs(left - right) > tolerance:
            action_index = (
                expected[index].action_index
                if index < len(expected)
                else observed[index].action_index
                if index < len(observed)
                else 0
            )
            return _divergence(
                DivergenceKind.REWARD,
                action_index,
                f"frames[{index}].reward",
                left,
                right,
                "first reward mismatch outside the configured tolerance",
            )
    return None


def _first_outcome(
    expected: tuple[ReplayFrame, ...],
    observed: tuple[ReplayFrame, ...],
    mode: ReplayMode,
) -> ReplayDivergence | None:
    for index in range(max(len(expected), len(observed))):
        left = expected[index].outcome if index < len(expected) else None
        right = observed[index].outcome if index < len(observed) else None
        comparable_left = (
            left if mode is ReplayMode.EXACT or left is None else left.casefold().strip()
        )
        comparable_right = (
            right if mode is ReplayMode.EXACT or right is None else right.casefold().strip()
        )
        if comparable_left != comparable_right:
            action_index = (
                expected[index].action_index
                if index < len(expected)
                else observed[index].action_index
                if index < len(observed)
                else 0
            )
            return _divergence(
                DivergenceKind.OUTCOME,
                action_index,
                f"frames[{index}].outcome",
                left,
                right,
                f"first {mode.value} outcome mismatch",
            )
    return None


def _resource_map(frame: ReplayFrame) -> dict[tuple[str, str], float]:
    return {(item.name, item.unit): item.value for item in frame.resources}


def _first_resource(
    expected: tuple[ReplayFrame, ...],
    observed: tuple[ReplayFrame, ...],
    tolerance: float,
) -> ReplayDivergence | None:
    for index in range(max(len(expected), len(observed))):
        left = _resource_map(expected[index]) if index < len(expected) else {}
        right = _resource_map(observed[index]) if index < len(observed) else {}
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right or abs(left[key] - right[key]) > tolerance:
                action_index = (
                    expected[index].action_index
                    if index < len(expected)
                    else observed[index].action_index
                    if index < len(observed)
                    else 0
                )
                return _divergence(
                    DivergenceKind.RESOURCE,
                    action_index,
                    f"frames[{index}].resources[{key[0]}:{key[1]}]",
                    left.get(key),
                    right.get(key),
                    "first resource observation mismatch outside the configured tolerance",
                )
    return None


def _present(divergences: Iterable[ReplayDivergence | None]) -> tuple[str, ...]:
    return tuple(item.kind.value for item in divergences if item is not None)


def compare_replay(
    expected: ReplayTrace,
    observed: ReplayTrace,
    *,
    mode: ReplayMode,
    scope: ComparisonScope = ComparisonScope.JOINT,
    tolerances: ReplayTolerances = _EXACT_TOLERANCES,
) -> ReplayEvidence:
    """Compare traces without treating transcript equality as environment/model-state equality."""

    identity_differences = _identity_differences(expected, observed)
    if mode is ReplayMode.EXACT and identity_differences:
        raise ExactReplayIdentityMismatch(
            "exact replay identity differs in: " + ", ".join(identity_differences)
        )
    reward_tolerance = 0.0 if mode is ReplayMode.EXACT else tolerances.reward_absolute
    resource_tolerance = 0.0 if mode is ReplayMode.EXACT else tolerances.resource_absolute
    token = _first_token(expected.frames, observed.frames)
    action = _first_action(expected.frames, observed.frames, mode)
    environment = _first_environment(expected.frames, observed.frames, mode)
    reward = _first_reward(expected.frames, observed.frames, reward_tolerance)
    outcome = _first_outcome(expected.frames, observed.frames, mode)
    if outcome is None:
        terminal_left = expected.terminal_outcome
        terminal_right = observed.terminal_outcome
        comparable_left = (
            terminal_left if mode is ReplayMode.EXACT else terminal_left.casefold().strip()
        )
        comparable_right = (
            terminal_right if mode is ReplayMode.EXACT else terminal_right.casefold().strip()
        )
        if comparable_left != comparable_right:
            outcome = _divergence(
                DivergenceKind.OUTCOME,
                expected.frames[-1].action_index,
                "terminal_outcome",
                terminal_left,
                terminal_right,
                f"{mode.value} terminal outcome mismatch",
            )
    resource = _first_resource(expected.frames, observed.frames, resource_tolerance)
    transcript_divergences = (token, action)
    environment_divergences = (environment, reward, outcome, resource)
    model_divergences = (token, reward, outcome, resource)
    joint_divergences = (token, action, environment, reward, outcome, resource)
    selected = {
        ComparisonScope.TRANSCRIPT: transcript_divergences,
        ComparisonScope.ENVIRONMENT_ONLY: environment_divergences,
        ComparisonScope.MODEL_ONLY: model_divergences,
        ComparisonScope.JOINT: joint_divergences,
    }[scope]
    environment_state_equal = (
        expected.identity.environment_capsule_id == observed.identity.environment_capsule_id
        and expected.identity.environment_state_digest == observed.identity.environment_state_digest
    )
    model_state_equal = (
        expected.identity.model_hash == observed.identity.model_hash
        and expected.identity.model_state_digest == observed.identity.model_state_digest
    )
    selected_state_mismatch = (
        scope in {ComparisonScope.ENVIRONMENT_ONLY, ComparisonScope.JOINT}
        and not environment_state_equal
    ) or (scope in {ComparisonScope.MODEL_ONLY, ComparisonScope.JOINT} and not model_state_equal)
    digest_base = {
        "expected_trace_id": expected.trace_id,
        "observed_trace_id": observed.trace_id,
        "mode": mode.value,
    }
    transcript_evidence = (
        _digest(
            {**digest_base, "scope": "transcript", "divergences": _present(transcript_divergences)}
        )
        if scope in {ComparisonScope.TRANSCRIPT, ComparisonScope.JOINT}
        else None
    )
    environment_evidence = (
        _digest(
            {
                **digest_base,
                "scope": "environment_state",
                "expected_state": expected.identity.environment_state_digest,
                "observed_state": observed.identity.environment_state_digest,
                "divergences": _present(environment_divergences),
            }
        )
        if scope in {ComparisonScope.ENVIRONMENT_ONLY, ComparisonScope.JOINT}
        else None
    )
    model_evidence = (
        _digest(
            {
                **digest_base,
                "scope": "model_state",
                "expected_state": expected.identity.model_state_digest,
                "observed_state": observed.identity.model_state_digest,
                "divergences": _present(model_divergences),
            }
        )
        if scope in {ComparisonScope.MODEL_ONLY, ComparisonScope.JOINT}
        else None
    )
    joint_evidence = (
        _digest({**digest_base, "scope": "joint", "divergences": _present(joint_divergences)})
        if scope is ComparisonScope.JOINT
        else None
    )
    draft = {
        "schema_version": "sloforge.helix.replay-evidence/v1",
        "evidence_id": "0" * 64,
        "expected_trace_id": expected.trace_id,
        "observed_trace_id": observed.trace_id,
        "mode": mode.value,
        "scope": scope.value,
        "matched": not any(selected) and not selected_state_mismatch,
        "exact_identity_verified": mode is ReplayMode.EXACT and not identity_differences,
        "identity_differences": identity_differences,
        "first_token_divergence": token.model_dump(mode="json") if token else None,
        "first_action_divergence": action.model_dump(mode="json") if action else None,
        "first_environment_divergence": environment.model_dump(mode="json")
        if environment
        else None,
        "first_reward_divergence": reward.model_dump(mode="json") if reward else None,
        "first_outcome_divergence": outcome.model_dump(mode="json") if outcome else None,
        "first_resource_divergence": resource.model_dump(mode="json") if resource else None,
        "transcript_evidence_digest": transcript_evidence,
        "environment_state_evidence_digest": environment_evidence,
        "model_state_evidence_digest": model_evidence,
        "joint_evidence_digest": joint_evidence,
        "environment_state_equal": (
            environment_state_equal
            if scope in {ComparisonScope.ENVIRONMENT_ONLY, ComparisonScope.JOINT}
            else None
        ),
        "model_state_equal": (
            model_state_equal
            if scope in {ComparisonScope.MODEL_ONLY, ComparisonScope.JOINT}
            else None
        ),
        "transcript_establishes_state_equivalence": False,
        "limitations": (
            "transcript equality is not evidence of environment-state equality",
            "transcript equality is not evidence of hidden model-state equality",
            "causal and semantic modes make no bitwise-equivalence claim",
            "causal mode compares declared event semantics and parent topology, not causation",
            "a non-exact match does not imply runtime, policy, RNG, or tool identity equality",
            "trace comparison does not attest that the runner restored underlying state bytes",
            "semantic equality depends on the declared semantic event digests",
        ),
        "evidence_digest": "0" * 64,
    }
    evidence_digest = canonical_digest(
        {
            key: value
            for key, value in draft.items()
            if key not in {"evidence_id", "evidence_digest"}
        }
    )
    draft["evidence_digest"] = evidence_digest
    evidence_id = canonical_digest(
        {key: value for key, value in draft.items() if key != "evidence_id"}
    )
    return ReplayEvidence(
        evidence_id=evidence_id,
        expected_trace_id=expected.trace_id,
        observed_trace_id=observed.trace_id,
        mode=mode,
        scope=scope,
        matched=not any(selected) and not selected_state_mismatch,
        exact_identity_verified=mode is ReplayMode.EXACT and not identity_differences,
        identity_differences=identity_differences,
        first_token_divergence=token,
        first_action_divergence=action,
        first_environment_divergence=environment,
        first_reward_divergence=reward,
        first_outcome_divergence=outcome,
        first_resource_divergence=resource,
        transcript_evidence_digest=transcript_evidence,
        environment_state_evidence_digest=environment_evidence,
        model_state_evidence_digest=model_evidence,
        joint_evidence_digest=joint_evidence,
        environment_state_equal=(
            environment_state_equal
            if scope in {ComparisonScope.ENVIRONMENT_ONLY, ComparisonScope.JOINT}
            else None
        ),
        model_state_equal=(
            model_state_equal
            if scope in {ComparisonScope.MODEL_ONLY, ComparisonScope.JOINT}
            else None
        ),
        limitations=(
            "transcript equality is not evidence of environment-state equality",
            "transcript equality is not evidence of hidden model-state equality",
            "causal and semantic modes make no bitwise-equivalence claim",
            "causal mode compares declared event semantics and parent topology, not causation",
            "a non-exact match does not imply runtime, policy, RNG, or tool identity equality",
            "trace comparison does not attest that the runner restored underlying state bytes",
            "semantic equality depends on the declared semantic event digests",
        ),
        evidence_digest=evidence_digest,
    )


ReplayRunner = Callable[[ReplayTrace, ReplayMode, int], ReplayTrace]


def replay_and_compare(
    expected: ReplayTrace,
    runner: ReplayRunner,
    *,
    mode: ReplayMode,
    scope: ComparisonScope = ComparisonScope.JOINT,
    seed: int,
    max_frames: int = 10_000,
    tolerances: ReplayTolerances = _EXACT_TOLERANCES,
) -> ReplayEvidence:
    """Invoke one bounded runner and compare its trace without attesting runner authority."""

    if not 0 <= seed < 2**64:
        raise ValueError("replay seed must be an unsigned 64-bit integer")
    if not 1 <= max_frames <= 100_000:
        raise ValueError("max_frames must be in 1..100000")
    if len(expected.frames) > max_frames:
        raise ReplayResourceLimit("expected replay trace exceeds the configured frame bound")
    observed = runner(expected, mode, seed)
    if len(observed.frames) > max_frames:
        raise ReplayResourceLimit("observed replay trace exceeds the configured frame bound")
    return compare_replay(
        expected,
        observed,
        mode=mode,
        scope=scope,
        tolerances=tolerances,
    )
