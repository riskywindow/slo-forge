"""CPU-only end-to-end Helix coding-agent demonstration.

Every policy choice, filesystem mutation, verifier result, training update, gate,
promotion transition, and rollback in this module is executed.  The demo does
not depend on a network service, GPU, or generated success transcript.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeAlias, cast

from pydantic import BaseModel

from sloforge.continuum.adapters import ReferenceTokenMajorAdapter, SessionLifecycle
from sloforge.continuum.compatibility import (
    CompatibilityDecision,
    CompatibilityRequest,
    ExactnessClass,
    ModelSemantics,
    RuntimeCapabilities,
    StateDependencyEvidence,
    analyze_compatibility,
)
from sloforge.continuum.operations import load_checkpoint_artifact
from sloforge.continuum.storage import MemoryContentStore
from sloforge.continuum.transaction import SessionLease
from sloforge.helix.branching import (
    ExactCowBranch,
    RngMutationBranch,
    create_branch_group,
)
from sloforge.helix.capture import (
    CaptureBoundary,
    CaptureSources,
    CoordinatedCaptureCoordinator,
    CoordinatedCaptureRequest,
    VerifiedCaptureArtifact,
)
from sloforge.helix.credit import BranchOutcome, assign_branch_relative_credit
from sloforge.helix.datasets import build_reference_training_batch
from sloforge.helix.effects import Effect, EffectClass, EffectLedger
from sloforge.helix.environments import (
    EnvironmentBackend,
    EnvironmentBranch,
    EnvironmentStateCapsule,
)
from sloforge.helix.environments.models import content_digest
from sloforge.helix.ir import Digest, LineageReference, LineageRelation, PolicyEpoch
from sloforge.helix.policy import DeterministicPolicy
from sloforge.helix.promotion import (
    CompatibilityClass,
    GateEvidence,
    PolicyRegistry,
    PromotionArtifactSource,
    PromotionState,
    build_policy_promotion_capsule,
    validate_policy_promotion_capsule,
)
from sloforge.helix.promotion.capsule import GateName
from sloforge.helix.replay import (
    ComparisonScope,
    ReplayEvent,
    ReplayFrame,
    ReplayIdentity,
    ReplayMode,
    ReplayToken,
    build_replay_trace,
    compare_replay,
)
from sloforge.helix.rewards import (
    DeterministicRewardWorker,
    HiddenCase,
    RewardRun,
    VerifierCommand,
    compute_evaluator_digest,
)
from sloforge.helix.rollouts import (
    ActionMutation,
    CandidateAction,
    ReferenceRolloutWorker,
    ReferenceTrajectory,
)
from sloforge.helix.trainers import ReferenceTrainer, TrainingAlgorithm
from sloforge.helix.transactions import (
    ArtifactReference,
    LearningState,
    LearningTransactionStore,
)

_STAMP = "2026-08-03T00:00:00Z"
_OBSERVATION = (
    "Implement bounded Retry-After handling for HTTP 429 while preserving existing 5xx behavior."
)
_QUALITY_THRESHOLD = 0.75
RewardSpec: TypeAlias = tuple[tuple[VerifierCommand, ...], tuple[HiddenCase, ...]]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _named_seed(label: str, *, lower: int = 0, span: int = 2**31) -> int:
    if not label or lower < 0 or span < 1:
        raise ValueError("named seed inputs are invalid")
    return lower + int(sha256(label.encode("utf-8")).hexdigest()[:16], 16) % span


def _source_commit() -> str:
    repository = Path(__file__).resolve().parents[3]
    marker = repository / ".sloforge-source-commit"
    if marker.is_file() and not marker.is_symlink():
        commit = marker.read_text(encoding="utf-8").strip()
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise RuntimeError("Helix source marker is not a lowercase Git SHA-1")
        return commit
    task_environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    result = subprocess.run(
        ("git", "-c", "core.hooksPath=/dev/null", "rev-parse", "HEAD"),
        cwd=repository,
        env=task_environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=2.0,
    )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("Helix demo source commit is not a lowercase Git SHA-1")
    return commit


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _write_json(path: Path, value: object) -> str:
    document = _json_value(value)
    payload = _canonical_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload + b"\n")
    return sha256(payload).hexdigest()


def _write_sealed_json(path: Path, value: object) -> str:
    """Write exact bytes for a gate that will be independently re-hashed."""

    payload = _canonical_bytes(_json_value(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def _policy_state_compatibility(
    champion: DeterministicPolicy,
    challenger: DeterministicPolicy,
) -> CompatibilityDecision:
    def model(policy: DeterministicPolicy) -> ModelSemantics:
        return ModelSemantics(
            model_id="helix/reference-coding-agent",
            architecture="categorical_policy",
            weights_hash=policy.weights_hash,
            state_producing_weights_hash=policy.weights_hash,
            output_head_hash=policy.weights_hash,
            tokenizer_hash=_hash("reference-action-tokenizer-v1"),
            special_tokens_hash=_hash("reference-action-special-tokens-v1"),
            positional_encoding="none",
            rope_fingerprint="none",
            attention_mask_semantics="decision_boundary",
            layer_count=1,
            head_count=1,
            kv_head_count=1,
            head_dim=len(policy.actions),
            state_dtype="float64",
            quantization="none",
            sampler_algorithm="python_random_v1",
        )

    runtime = RuntimeCapabilities(
        runtime_name="helix-reference-rollout",
        runtime_version="1.0.0",
        adapter_version="1.0.0",
        supported_state_types=("policy.action_logits", "sampler"),
        supported_dtypes=("float64",),
        can_recompute_from_token_history=True,
    )
    decision = analyze_compatibility(
        CompatibilityRequest(
            source=model(champion),
            destination=model(challenger),
            source_runtime=runtime,
            destination_runtime=runtime,
            source_layout_fingerprint=_hash("helix-reference-policy-layout-v1"),
            destination_layout_fingerprint=_hash("helix-reference-policy-layout-v1"),
            required_state_types=("policy.action_logits", "sampler"),
            required_exactness=ExactnessClass.RECOMPUTATION_ASSISTED,
            dependency_evidence=StateDependencyEvidence(
                dependency_graph_hash=_hash("helix-reference-state-dependencies-v1"),
                changed_components=("output_head", "policy_weights"),
                state_producing_components=("output_head", "policy_weights"),
                affected_state_components=("policy.action_logits", "sampler"),
                recomputable_state_components=("policy.action_logits", "sampler"),
                output_head_is_state_sink=True,
                token_history_available=True,
            ),
            allow_recomputation=True,
        )
    )
    if (
        not decision.safe
        or decision.compatibility_class is not ExactnessClass.RECOMPUTATION_ASSISTED
    ):
        raise RuntimeError("Continuum did not require safe cross-policy state recomputation")
    return decision


def _lineage(artifact_id: str, artifact_kind: str, digest: str) -> LineageReference:
    return LineageReference(
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        relation=LineageRelation.DERIVED_FROM,
        digest=Digest(value=digest),
    )


def _write_fixture(root: Path) -> bytes:
    root.mkdir(parents=True, exist_ok=False)
    original = (
        b"def retry_delay(status: int, retry_after: str | None) -> int:\n"
        b'    """Return seconds before retrying a response."""\n'
        b"    if status >= 500:\n"
        b"        return 2\n"
        b"    return 0\n"
    )
    (root / "retry_policy.py").write_bytes(original)
    (root / "test_retry_policy.py").write_text(
        "import unittest\n"
        "from retry_policy import retry_delay\n\n"
        "class RetryPolicyTests(unittest.TestCase):\n"
        "    def test_declared_429_case(self):\n"
        "        self.assertEqual(retry_delay(429, '3'), 3)\n\n"
        "    def test_existing_5xx_contract(self):\n"
        "        self.assertEqual(retry_delay(503, None), 2)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )
    (root / "probe.py").write_text(
        "import sys\n"
        "from retry_policy import retry_delay\n"
        "print(retry_delay(int(sys.argv[1]), sys.argv[2]))\n"
    )
    return original


def _candidate_actions(original: bytes) -> tuple[CandidateAction, ...]:
    before = content_digest(original)
    bodies = {
        "naive_parse": (
            "def retry_delay(status: int, retry_after: str | None) -> int:\n"
            '    """Return seconds before retrying a response."""\n'
            "    if status == 429:\n"
            "        try:\n"
            "            return int(retry_after or '')\n"
            "        except ValueError:\n"
            "            return 1\n"
            "    if status >= 500:\n"
            "        return 2\n"
            "    return 0\n"
        ),
        "guarded_parse": (
            "def retry_delay(status: int, retry_after: str | None) -> int:\n"
            '    """Return seconds before retrying a response."""\n'
            "    if status == 429:\n"
            "        try:\n"
            "            delay = int(retry_after or '')\n"
            "        except ValueError:\n"
            "            return 1\n"
            "        return delay if delay >= 0 else 1\n"
            "    if status >= 500:\n"
            "        return 2\n"
            "    return 0\n"
        ),
        "ast_guided": (
            "def retry_delay(status: int, retry_after: str | None) -> int:\n"
            '    """Return seconds before retrying a response."""\n'
            "    if status == 429:\n"
            "        candidate = (retry_after or '').strip()\n"
            "        if candidate.isdecimal():\n"
            "            return int(candidate)\n"
            "        return 1\n"
            "    if status >= 500:\n"
            "        return 2\n"
            "    return 0\n"
        ),
        "verifier_assisted": (
            "def retry_delay(status: int, retry_after: str | None) -> int:\n"
            '    """Return seconds before retrying a response."""\n'
            "    if status == 429:\n"
            "        try:\n"
            "            delay = int((retry_after or '').strip())\n"
            "        except ValueError:\n"
            "            delay = 1\n"
            "        return max(0, delay) if delay >= 0 else 1\n"
            "    if status >= 500:\n"
            "        return 2\n"
            "    return 0\n"
        ),
    }
    tools = {
        "naive_parse": "direct-edit",
        "guarded_parse": "forced-alternative-edit",
        "ast_guided": "ast-aware-edit",
        "verifier_assisted": "test-guided-edit",
    }
    return tuple(
        CandidateAction(
            action=action,
            tool_id=tools[action],
            effect_class=EffectClass.IDEMPOTENT_WRITE,
            mutations=(
                ActionMutation(
                    path="retry_policy.py",
                    content=body,
                    expected_before_hash=before,
                ),
            ),
        )
        for action, body in bodies.items()
    )


def _policy() -> DeterministicPolicy:
    return DeterministicPolicy(
        policy_epoch_id="coding-agent@0",
        actions=("naive_parse", "guarded_parse", "ast_guided", "verifier_assisted"),
        logits=(0.6, 0.4, 0.2, 0.0),
    )


def _failure_seed(policy: DeterministicPolicy, seed: int) -> int:
    for offset in range(256):
        candidate = (seed + offset) % (2**63 - 1)
        if policy.decide(_OBSERVATION, seed=candidate).action == "naive_parse":
            return candidate
    raise RuntimeError("bounded production-failure seed search found no failing policy action")


def _lease(runtime: ReferenceTokenMajorAdapter, session_id: str) -> SessionLease:
    metadata = runtime.inspect_session("production-session")
    return SessionLease(
        session_id=session_id,
        owner_runtime=runtime.identity.runtime_name,
        owner_epoch=1 if session_id != "production-session" else metadata.owner_epoch,
        fencing_token=1 if session_id != "production-session" else metadata.owner_epoch,
        expiration_ms=120_000,
        coordinator_version=1,
        last_committed_state_version=metadata.state_version,
        last_committed_token_index=metadata.committed_output_index,
    )


def _training_reward_spec() -> RewardSpec:
    return (
        (
            VerifierCommand(
                verifier_id="visible-unit-and-retention-tests",
                argv=("{python}", "-m", "unittest", "-q", "test_retry_policy.py"),
                source_version="helix-coding-fixture/v1",
            ),
        ),
        (
            HiddenCase(
                case_id="hidden-negative-retry-after",
                runner="probe.py",
                arguments=("429", "-1"),
                expected_stdout="1",
                score_on_pass=1.0,
                score_on_fail=-1.0,
            ),
        ),
    )


def _holdout_reward_spec() -> RewardSpec:
    commands, _ = _training_reward_spec()
    return (
        commands,
        (
            HiddenCase(
                case_id="holdout-negative-retry-after",
                runner="probe.py",
                arguments=("429", "-17"),
                expected_stdout="1",
                score_on_pass=1.0,
                score_on_fail=-1.0,
            ),
        ),
    )


def _score(
    worker: DeterministicRewardWorker,
    *,
    trajectory: ReferenceTrajectory,
    workspace: Path,
    output: Path,
    seed: int,
    reward_spec: RewardSpec,
) -> RewardRun:
    commands, hidden = reward_spec
    return worker.verify(
        reward_id=f"reward-{trajectory.trajectory_id[:24]}",
        trajectory_id=trajectory.trajectory_id,
        policy_epoch_id=trajectory.policy_epoch_id,
        tenant_id=trajectory.tenant_id,
        source=workspace,
        evidence_directory=output,
        commands=commands,
        hidden_cases=hidden,
        seed=seed,
    )


def _evaluate_policy(
    *,
    name: str,
    policy: DeterministicPolicy,
    seeds: tuple[int, ...],
    backend: EnvironmentBackend,
    base_environment: EnvironmentStateCapsule,
    actions: tuple[CandidateAction, ...],
    output: Path,
    trusted_evaluator_digest: str,
    reward_spec: RewardSpec,
) -> dict[str, Any]:
    worker = ReferenceRolloutWorker(tenant_id=base_environment.tenant_id)
    reward_worker = DeterministicRewardWorker(
        trusted_evaluator_digests=frozenset({trusted_evaluator_digest})
    )
    cases: list[dict[str, object]] = []
    fresh_report = {
        "mode": "fresh_request_recompute",
        "policy_epoch_id": policy.policy_epoch_id,
        "model_state_reused": False,
        "reason": "evaluation begins at a fresh request boundary",
    }
    report_hash = _hash(fresh_report)
    _write_json(output / "fresh-state-report.json", fresh_report)
    for ordinal, case_seed in enumerate(seeds):
        branch_id = f"eval-{name}-{ordinal:03d}"
        branch = backend.fork(base_environment, branch_id=branch_id, seed=case_seed)
        try:
            trajectory = worker.run(
                branch=branch,
                initial_environment_capsule_id=base_environment.capsule_id,
                branch_group_id=f"evaluation-{name}",
                branch_point_id=f"fresh-{name}",
                branch_point_hash=_hash({"evaluation": name, "seed": case_seed}),
                source_model_capsule_id="fresh-request-boundary",
                state_reuse_report_hash=report_hash,
                policy=policy,
                observation=_OBSERVATION,
                candidates=actions,
                seed=case_seed,
            )
            reward = _score(
                reward_worker,
                trajectory=trajectory,
                workspace=branch.workspace,
                output=output / "reward-evidence" / f"{ordinal:03d}",
                seed=case_seed,
                reward_spec=reward_spec,
            )
            cases.append(
                {
                    "seed": case_seed,
                    "action": trajectory.actions[0].action,
                    "trajectory_id": trajectory.trajectory_id,
                    "reward_id": reward.reward_id,
                    "reward": reward.total_score,
                    "visible_passed": reward.components[0].passed,
                    "hidden_passed": reward.components[1].passed,
                    "task_success": all(item.passed for item in reward.components),
                    "immutable_verifier_input": reward.immutable_source,
                }
            )
        finally:
            branch.cleanup()
    iterations = 1000
    start = time.perf_counter_ns()
    for index in range(iterations):
        policy.decide(_OBSERVATION, seed=seeds[index % len(seeds)], rng_counter=index)
    duration_ns = time.perf_counter_ns() - start
    success_count = sum(bool(item["task_success"]) for item in cases)
    visible_count = sum(bool(item["visible_passed"]) for item in cases)
    return {
        "policy_epoch_id": policy.policy_epoch_id,
        "policy_weights_hash": policy.weights_hash,
        "seeds": seeds,
        "cases": cases,
        "task_success_rate": success_count / len(cases),
        "capability_retention_rate": visible_count / len(cases),
        "mean_policy_decision_latency_ms": duration_ns / iterations / 1_000_000.0,
        "latency_iterations": iterations,
        "hardware_class": "local-cpu-reference-policy",
        "synthetic": True,
    }


def _gate(
    name: str,
    *,
    artifact_hash: str,
    measured: float,
    threshold: float,
    comparator: str,
    sample_count: int,
    seed: int,
    detail: str,
) -> GateEvidence:
    passed = {
        "le": measured <= threshold,
        "ge": measured >= threshold,
        "eq": measured == threshold,
    }[comparator]
    return GateEvidence.model_validate(
        {
            "gate": name,
            "tenant_id": "tenant-helix-demo",
            "evidence_id": f"{name}-{artifact_hash[:24]}",
            "artifact_hash": artifact_hash,
            "passed": passed,
            "sample_count": sample_count,
            "measured_value": measured,
            "threshold": threshold,
            "comparator": comparator,
            "deterministic_seed": seed,
            "detail": detail,
        },
        strict=True,
    )


def _artifact(
    transaction: LearningTransactionStore,
    transaction_id: str,
    *,
    artifact_id: str,
    kind: str,
    path: Path,
    digest: str,
) -> None:
    transaction.add_artifact(
        transaction_id,
        ArtifactReference(
            artifact_id=artifact_id,
            artifact_kind=kind,
            sha256=digest,
            uri=path.as_posix(),
        ),
    )


def _advance(
    store: LearningTransactionStore,
    transaction_id: str,
    states: Iterable[LearningState],
    *,
    start_time: int,
) -> int:
    observed = start_time
    for state in states:
        observed += 1
        store.transition(
            transaction_id,
            target=state,
            reason=f"verified evidence accepted for {state.value}",
            observed_at_ms=observed,
        )
    return observed


def _replay_evidence(
    *,
    branch_point_id: str,
    trajectory: ReferenceTrajectory,
    reward: RewardRun,
    policy: DeterministicPolicy,
    runtime: ReferenceTokenMajorAdapter,
    output: Path,
) -> dict[str, Any]:
    action = trajectory.actions[0]
    event = ReplayEvent(
        event_id="action-0",
        kind="coding_action",
        payload_digest=_hash(action.action),
        semantic_digest=_hash("valid-retry-policy-edit"),
    )
    environment = ReplayEvent(
        event_id="environment-0",
        kind="filesystem_transition",
        payload_digest=action.environment_transition_hash,
        semantic_digest=_hash("workspace-edit-completed"),
        causal_parent_id="action-0",
    )
    frame = ReplayFrame(
        action_index=0,
        action=event,
        model_tokens=(ReplayToken(token_index=0, token_id=int(_hash(action.action)[:8], 16)),),
        environment_events=(environment,),
        reward=reward.total_score,
        outcome="success" if all(item.passed for item in reward.components) else "failure",
    )
    base_identity = ReplayIdentity(
        policy_epoch_id=policy.policy_epoch_id,
        policy_digest=policy.weights_hash,
        runtime_name=runtime.identity.runtime_name,
        runtime_version=runtime.identity.runtime_version,
        runtime_build_hash=runtime.identity.build_hash,
        model_hash=runtime.config.model.model_hash,
        model_state_digest=trajectory.source_model_capsule_id,
        environment_capsule_id=trajectory.initial_environment_capsule_id,
        environment_state_digest=trajectory.initial_environment_capsule_id,
        rng_algorithm="continuum-counter-v1",
        rng_seed=trajectory.seed,
        rng_counter=0,
        tool_contract_hash=_hash(policy.actions),
    )
    expected = build_replay_trace(
        branch_point_id=branch_point_id,
        identity=base_identity,
        frames=(frame,),
        terminal_outcome=frame.outcome,
    )
    transcript_model = _hash(
        {
            "mode": "transcript-refeed",
            "policy": policy.weights_hash,
            "observation": _OBSERVATION,
        }
    )
    reconstructed_environment = _hash(
        {"mode": "transcript-files-unspecified", "observation": _OBSERVATION}
    )
    identities = {
        "transcript": base_identity.model_copy(
            update={
                "model_state_digest": transcript_model,
                "environment_capsule_id": "transcript-environment",
                "environment_state_digest": reconstructed_environment,
            }
        ),
        "environment": base_identity.model_copy(update={"model_state_digest": transcript_model}),
        "model": base_identity.model_copy(
            update={
                "environment_capsule_id": "model-only-environment",
                "environment_state_digest": reconstructed_environment,
            }
        ),
        "joint": base_identity,
    }
    scopes = {
        "transcript": ComparisonScope.TRANSCRIPT,
        "environment": ComparisonScope.ENVIRONMENT_ONLY,
        "model": ComparisonScope.MODEL_ONLY,
        "joint": ComparisonScope.JOINT,
    }
    result: dict[str, Any] = {}
    for name, identity in identities.items():
        observed = build_replay_trace(
            branch_point_id=branch_point_id,
            identity=identity,
            frames=(frame,),
            terminal_outcome=frame.outcome,
        )
        mode = ReplayMode.EXACT if name == "joint" else ReplayMode.CAUSAL
        evidence = compare_replay(expected, observed, mode=mode, scope=scopes[name])
        result[name] = evidence.model_dump(mode="json")
    _write_json(output, result)
    return result


def run_cpu_demo(output: Path, *, seed: int) -> dict[str, Any]:
    """Run the exercised vertical slice and return its artifact-derived summary."""

    if not 0 <= seed <= 2**63 - 1:
        raise ValueError("demo seed must fit a signed 64-bit integer")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Helix demo output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    git_commit = _source_commit()
    source = output / "fixture" / "repository"
    original = _write_fixture(source)
    training_reward_spec = _training_reward_spec()
    holdout_reward_spec = _holdout_reward_spec()
    training_evaluator_digest = compute_evaluator_digest(
        source=source,
        commands=training_reward_spec[0],
        hidden_cases=training_reward_spec[1],
    )
    holdout_evaluator_digest = compute_evaluator_digest(
        source=source,
        commands=holdout_reward_spec[0],
        hidden_cases=holdout_reward_spec[1],
    )
    actions = _candidate_actions(original)
    champion = _policy()
    champion_policy_hash = _write_json(output / "policies" / "champion.json", champion)
    production_seed = _failure_seed(champion, seed)
    production_decision = champion.decide(_OBSERVATION, seed=production_seed)
    if production_decision.action != "naive_parse":
        raise RuntimeError("the executed champion decision did not reproduce the failure")

    runtime = ReferenceTokenMajorAdapter()
    runtime.create_session(
        session_id="production-session",
        request_id="production-request",
        tenant_id="tenant-helix-demo",
        input_token_ids=(11, 13, 17, 19),
        seed=production_seed,
    )
    for token in runtime.stream_tokens("production-session", count=4):
        runtime.acknowledge_gateway(
            "production-session",
            token_index=token.token_index,
            owner_epoch=token.owner_epoch,
        )
    metadata = runtime.inspect_session("production-session")
    environment_backend = EnvironmentBackend(
        output / "environment-store", tenant_id="tenant-helix-demo"
    )
    environment_holder: list[EnvironmentStateCapsule] = []

    def capture_environment(capture_seed: int) -> Any:
        capsule = environment_backend.capture(
            source,
            seed=capture_seed,
            event_watermark=0,
            allowed_tools=(
                "direct-edit",
                "forced-alternative-edit",
                "ast-aware-edit",
                "structured-edit",
            ),
        )
        if environment_holder and environment_holder[0] != capsule:
            raise RuntimeError("idempotent environment capture changed at the barrier")
        if not environment_holder:
            environment_holder.append(capsule)
        return VerifiedCaptureArtifact(
            reference=environment_backend.artifact_watermark(capsule),
            payload=environment_backend.artifact_payload(capsule),
        )

    effects = EffectLedger(tenant_id="tenant-helix-demo")
    effects.record(
        Effect.build(
            EffectClass.READ_ONLY,
            "inspect-repository",
            target="sandbox://repository",
            tenant_id="tenant-helix-demo",
        )
    )
    effects.commit(0)

    def capture_effects() -> VerifiedCaptureArtifact:
        return VerifiedCaptureArtifact(
            reference=effects.artifact_watermark(),
            payload=effects.artifact_payload(),
        )

    boundary = CaptureBoundary(
        action_watermark=0,
        model_token_watermark=metadata.committed_output_index,
        environment_event_watermark=0,
        effect_watermark=0,
    )
    model_store = MemoryContentStore()
    sources = CaptureSources(
        model=runtime,
        model_store=model_store,
        lease=_lease(runtime, "production-session"),
        read_boundary=lambda: boundary,
        capture_environment=capture_environment,
        capture_effects=capture_effects,
        expected_tenant_id="tenant-helix-demo",
    )
    request = CoordinatedCaptureRequest(
        capture_id="coding-failure-capture",
        session_id="production-session",
        source_trajectory_id="production-failure-trajectory",
        policy_epoch_id=champion.policy_epoch_id,
        boundary=boundary,
        seed=production_seed,
        max_quiescence_polls=4,
        published_at_ms=10,
        capture_timestamp=_STAMP,
        git_commit=git_commit,
        continuum_version="0.1.0",
        created_at=_STAMP,
        reason="capture immediately before the incorrect Retry-After edit",
    )
    capture_artifacts = output / "capture"
    with CoordinatedCaptureCoordinator(
        output / "capture.sqlite",
        artifact_directory=capture_artifacts,
        require_verified_artifacts=True,
    ) as coordinator:
        coordinator.propose(request)
        branch_point = coordinator.execute(request.capture_id, sources)
        capture_journal = tuple(
            item.model_dump(mode="json") for item in coordinator.journal(request.capture_id)
        )
    if runtime.inspect_session("production-session").lifecycle is not SessionLifecycle.ACTIVE:
        raise RuntimeError("coordinated capture did not resume the production source")
    branch_point_path = output / "capture" / "branchpoint.json"
    branch_point_hash = _write_json(branch_point_path, branch_point)
    capture_journal_path = output / "capture" / "journal.json"
    capture_journal_hash = _write_json(capture_journal_path, capture_journal)
    parent = load_checkpoint_artifact(capture_artifacts / f"{request.capture_id}.continuum.json")
    environment = environment_holder[0]

    rng_seed = _failure_seed(champion, production_seed + 1)
    plans = (
        RngMutationBranch(
            "different-rng", champion.policy_epoch_id, _lease(runtime, "different-rng"), rng_seed
        ),
        ExactCowBranch(
            "forced-alternative", champion.policy_epoch_id, _lease(runtime, "forced-alternative")
        ),
        ExactCowBranch(
            "alternative-tool", champion.policy_epoch_id, _lease(runtime, "alternative-tool")
        ),
        ExactCowBranch(
            "verifier-assisted",
            champion.policy_epoch_id,
            _lease(runtime, "verifier-assisted"),
        ),
    )
    group = create_branch_group(
        parent,
        branch_point_id=branch_point.branch_point_id,
        source_policy_epoch_id=champion.policy_epoch_id,
        plans=plans,
        store=model_store,
        expected_tenant_id="tenant-helix-demo",
        expected_model=runtime.config.model,
        seed=production_seed,
        published_at_ms=20,
        capture_timestamp=_STAMP,
        git_commit=git_commit,
        continuum_version="0.1.0",
        environment_backend=environment_backend,
        environment_capsule=environment,
    )
    forced = {
        "different-rng": None,
        "forced-alternative": "guarded_parse",
        "alternative-tool": "ast_guided",
        "verifier-assisted": "verifier_assisted",
    }
    rollout_worker = ReferenceRolloutWorker(tenant_id="tenant-helix-demo")
    reward_worker = DeterministicRewardWorker(
        trusted_evaluator_digests=frozenset({training_evaluator_digest})
    )
    trajectories: list[ReferenceTrajectory] = []
    rewards: list[RewardRun] = []
    branch_paths: dict[str, Path] = {}
    try:
        for index, member in enumerate(group.members):
            if member.environment_branch is None:
                raise RuntimeError("joint branch group omitted its environment branch")
            branch = cast(EnvironmentBranch, member.environment_branch)
            trajectory = rollout_worker.run(
                branch=branch,
                initial_environment_capsule_id=environment.capsule_id,
                branch_group_id=group.group_id,
                branch_point_id=branch_point.branch_point_id,
                branch_point_hash=branch_point_hash,
                source_model_capsule_id=group.source_capsule_id,
                state_reuse_report_hash=member.state_reuse.report_digest,
                policy=champion,
                observation=_OBSERVATION,
                candidates=actions,
                seed=(rng_seed if member.branch_id == "different-rng" else seed + index + 1),
                forced_action=forced[member.branch_id],
            )
            reward = _score(
                reward_worker,
                trajectory=trajectory,
                workspace=branch.workspace,
                output=output / "rewards" / member.branch_id,
                seed=seed + index + 100,
                reward_spec=training_reward_spec,
            )
            trajectories.append(trajectory)
            rewards.append(reward)
            trajectory_path = output / "trajectories" / f"{member.branch_id}.json"
            reward_path = output / "rewards" / f"{member.branch_id}.json"
            _write_json(trajectory_path, trajectory)
            _write_json(reward_path, reward)
            _write_json(output / "state-reuse" / f"{member.branch_id}.json", member.state_reuse)
            branch_paths[member.branch_id] = branch.workspace

        outcomes = tuple(
            BranchOutcome(
                branch_id=trajectory.branch_id,
                trajectory_id=trajectory.trajectory_id,
                policy_epoch_id=trajectory.policy_epoch_id,
                action=trajectory.actions[0].action,
                behavior_log_probability=trajectory.actions[0].behavior_log_probability,
                reward_components={
                    component.component_id: component.score for component in reward.components
                },
                first_divergent_action_index=0,
                suffix_action_count=1,
                process_score=None,
                intervention=(
                    "controlled_rng"
                    if trajectory.branch_id == "different-rng"
                    else "controlled_tool"
                    if trajectory.branch_id in {"alternative-tool", "verifier-assisted"}
                    else "controlled_action"
                ),
            )
            for trajectory, reward in zip(trajectories, rewards, strict=True)
        )
        credit = assign_branch_relative_credit(
            branch_group_id=group.group_id,
            branch_point_id=branch_point.branch_point_id,
            outcomes=outcomes,
        )
        credit_path = output / "credit" / "branch-relative.json"
        credit_hash = _write_json(credit_path, credit)
        batch = build_reference_training_batch(
            trajectories=tuple(trajectories),
            rewards=tuple(rewards),
            credit=credit,
            algorithm=TrainingAlgorithm.BRANCH_RELATIVE,
            learner_policy_epoch_id=champion.policy_epoch_id,
            staleness_updates={item.trajectory_id: 0 for item in trajectories},
            maximum_staleness_updates=2,
            holdout_trajectory_ids=(f"holdout-{holdout_evaluator_digest[:24]}",),
            creation_code_version=git_commit,
            seed=seed,
        )
        batch_path = output / "training" / "batch.json"
        batch_hash = _write_json(batch_path, batch)
        trainer = ReferenceTrainer(learning_rate=0.5, kl_coefficient=0.02)
        rejected_training = trainer.train(
            base=champion,
            samples=batch.trainer_samples(),
            algorithm=batch.algorithm,
            candidate_policy_epoch_id="coding-agent-rejected@1",
            seed=seed,
            steps=1,
        )
        corrected_training = trainer.train(
            base=champion,
            samples=batch.trainer_samples(),
            algorithm=batch.algorithm,
            candidate_policy_epoch_id="coding-agent@1",
            seed=seed,
            steps=32,
        )
        rejected_training_path = output / "training" / "rejected-candidate.json"
        corrected_training_path = output / "training" / "corrected-candidate.json"
        _write_json(rejected_training_path, rejected_training)
        corrected_training_hash = _write_json(corrected_training_path, corrected_training)
        _write_json(output / "policies" / "rejected.json", rejected_training.candidate)
        _write_json(output / "policies" / "challenger.json", corrected_training.candidate)
    finally:
        for member in group.members:
            if member.environment_branch is not None:
                cast(EnvironmentBranch, member.environment_branch).cleanup()

    hidden_seed = _named_seed("sloforge-helix-hidden-holdout-v1", lower=200, span=800)
    evaluation_seeds = tuple(hidden_seed + index for index in range(24))
    champion_evaluation = _evaluate_policy(
        name="champion",
        policy=champion,
        seeds=evaluation_seeds,
        backend=environment_backend,
        base_environment=environment,
        actions=actions,
        output=output / "evaluations" / "champion",
        trusted_evaluator_digest=holdout_evaluator_digest,
        reward_spec=holdout_reward_spec,
    )
    rejected_evaluation = _evaluate_policy(
        name="rejected",
        policy=rejected_training.candidate,
        seeds=evaluation_seeds,
        backend=environment_backend,
        base_environment=environment,
        actions=actions,
        output=output / "evaluations" / "rejected",
        trusted_evaluator_digest=holdout_evaluator_digest,
        reward_spec=holdout_reward_spec,
    )
    corrected_evaluation = _evaluate_policy(
        name="corrected",
        policy=corrected_training.candidate,
        seeds=evaluation_seeds,
        backend=environment_backend,
        base_environment=environment,
        actions=actions,
        output=output / "evaluations" / "corrected",
        trusted_evaluator_digest=holdout_evaluator_digest,
        reward_spec=holdout_reward_spec,
    )
    champion_evaluation_path = output / "evaluations" / "champion.json"
    rejected_evaluation_path = output / "evaluations" / "rejected.json"
    corrected_evaluation_path = output / "evaluations" / "corrected.json"
    _write_json(champion_evaluation_path, champion_evaluation)
    rejected_evaluation_hash = _write_json(rejected_evaluation_path, rejected_evaluation)
    corrected_evaluation_hash = _write_json(corrected_evaluation_path, corrected_evaluation)
    rejected_rate = float(rejected_evaluation["task_success_rate"])
    corrected_rate = float(corrected_evaluation["task_success_rate"])
    if rejected_rate >= _QUALITY_THRESHOLD:
        raise RuntimeError("the intentionally under-trained candidate unexpectedly passed quality")
    if corrected_rate < _QUALITY_THRESHOLD:
        raise RuntimeError("the corrected candidate did not pass actual hidden evaluation")

    best_index = max(range(len(rewards)), key=lambda index: rewards[index].total_score)
    replay = _replay_evidence(
        branch_point_id=branch_point.branch_point_id,
        trajectory=trajectories[best_index],
        reward=rewards[best_index],
        policy=champion,
        runtime=runtime,
        output=output / "replay" / "comparison.json",
    )
    replay_path = output / "replay" / "comparison.json"
    replay_hash = sha256(replay_path.read_bytes().rstrip(b"\n")).hexdigest()

    compatibility_decision = _policy_state_compatibility(champion, corrected_training.candidate)
    promotion_evidence_root = output / "promotion" / "evidence"
    promotion_artifact_hashes = {
        "lineage": _write_sealed_json(
            promotion_evidence_root / "lineage.json",
            batch,
        ),
        "reward_integrity": _write_sealed_json(
            promotion_evidence_root / "reward_integrity.json",
            {
                "reward_ids": tuple(item.reward_id for item in rewards),
                "training_evaluator_sha256": training_evaluator_digest,
                "holdout_evaluator_sha256": holdout_evaluator_digest,
                "all_evaluators_trusted": all(item.evaluator_trusted for item in rewards),
                "all_sources_immutable": all(item.immutable_source for item in rewards),
                "hidden_expected_values_exposed": any(
                    item.hidden_expected_values_exposed for item in rewards
                ),
            },
        ),
        "quality": _write_sealed_json(
            promotion_evidence_root / "quality.json", corrected_evaluation
        ),
        "safety": _write_sealed_json(
            promotion_evidence_root / "safety.json",
            {
                "branch_point_hash": branch_point_hash,
                "branch_effect_classes": tuple(
                    trajectory.actions[0].effect_class.value for trajectory in trajectories
                ),
                "isolated_targets": True,
                "external_side_effects_enabled": False,
            },
        ),
        "serving": _write_sealed_json(
            promotion_evidence_root / "serving.json", corrected_evaluation
        ),
        "compatibility": _write_sealed_json(
            promotion_evidence_root / "compatibility.json", compatibility_decision
        ),
    }

    transaction_store = LearningTransactionStore(output / "learning-transactions.sqlite")
    trigger_hash = _hash(
        {
            "observation": _OBSERVATION,
            "decision": production_decision.model_dump(mode="json"),
            "hidden_failure": "negative Retry-After produced an illegal negative delay",
        }
    )
    transaction_store.create(
        transaction_id="learning-tx-accepted",
        deployment="coding-agent-prod",
        champion_policy_epoch_id=champion.policy_epoch_id,
        trigger_hash=trigger_hash,
        seed=seed,
        observed_at_ms=1,
    )
    for artifact_id, kind, path, digest in (
        ("branchpoint", "BranchPoint", branch_point_path, branch_point_hash),
        ("capture-journal", "CaptureJournal", capture_journal_path, capture_journal_hash),
        ("credit", "CreditAssignmentEvidence", credit_path, credit_hash),
        ("training-batch", "TrainingBatchManifest", batch_path, batch_hash),
        (
            "corrected-training",
            "TrainingResult",
            corrected_training_path,
            corrected_training_hash,
        ),
        (
            "corrected-evaluation",
            "EvaluationEvidence",
            corrected_evaluation_path,
            corrected_evaluation_hash,
        ),
        ("replay-comparison", "ReplayEvidence", replay_path, replay_hash),
    ):
        _artifact(
            transaction_store,
            "learning-tx-accepted",
            artifact_id=artifact_id,
            kind=kind,
            path=path,
            digest=digest,
        )
    now = _advance(
        transaction_store,
        "learning-tx-accepted",
        (
            LearningState.CAPTURE_PROPOSED,
            LearningState.CAPTURED,
            LearningState.BRANCH_PLAN_CREATED,
            LearningState.FORKING,
            LearningState.ROLLOUTS_RUNNING,
            LearningState.ROLLOUTS_COMPLETED,
            LearningState.REWARDS_VALIDATING,
            LearningState.CREDIT_ASSIGNING,
            LearningState.TRAINING_BATCH_BUILDING,
            LearningState.TRAINING,
        ),
        start_time=1,
    )
    transaction_store.set_candidate(
        "learning-tx-accepted", corrected_training.candidate.policy_epoch_id
    )
    now = _advance(
        transaction_store,
        "learning-tx-accepted",
        (
            LearningState.CANDIDATE_READY,
            LearningState.ALGORITHM_VALIDATING,
            LearningState.QUALITY_VALIDATING,
            LearningState.SERVING_VALIDATING,
            LearningState.COMPATIBILITY_VALIDATING,
            LearningState.PROMOTION_CAPSULE_READY,
        ),
        start_time=now,
    )

    transaction_store.create(
        transaction_id="learning-tx-rejected",
        deployment="coding-agent-prod",
        champion_policy_epoch_id=champion.policy_epoch_id,
        trigger_hash=trigger_hash,
        seed=seed + 1,
        observed_at_ms=1,
    )
    _artifact(
        transaction_store,
        "learning-tx-rejected",
        artifact_id="rejected-evaluation",
        kind="EvaluationEvidence",
        path=rejected_evaluation_path,
        digest=rejected_evaluation_hash,
    )
    rejected_now = _advance(
        transaction_store,
        "learning-tx-rejected",
        _SUCCESS_PREFIX_TO_QUALITY[:-3],
        start_time=1,
    )
    transaction_store.set_candidate(
        "learning-tx-rejected", rejected_training.candidate.policy_epoch_id
    )
    rejected_now = _advance(
        transaction_store,
        "learning-tx-rejected",
        _SUCCESS_PREFIX_TO_QUALITY[-3:],
        start_time=rejected_now,
    )
    transaction_store.transition(
        "learning-tx-rejected",
        target=LearningState.QUALITY_REGRESSION,
        reason=(
            f"actual hidden success rate {rejected_rate:.6f} was below "
            f"the {_QUALITY_THRESHOLD:.6f} quality contract"
        ),
        observed_at_ms=rejected_now + 1,
        evidence_artifact_ids=("rejected-evaluation",),
    )

    registry = PolicyRegistry(output / "policy-registry.sqlite", tenant_id="tenant-helix-demo")
    registry.register_policy(
        champion, parent_policy_epoch_id=None, status="champion", created_at_ms=1
    )
    registry.register_policy(
        rejected_training.candidate,
        parent_policy_epoch_id=champion.policy_epoch_id,
        status="challenger",
        created_at_ms=2,
    )
    registry.register_policy(
        corrected_training.candidate,
        parent_policy_epoch_id=champion.policy_epoch_id,
        status="challenger",
        created_at_ms=3,
    )
    registry.create_deployment("coding-agent-prod", champion.policy_epoch_id)
    active_old = registry.open_session(
        deployment="coding-agent-prod", session_id="active-incompatible", opened_at_ms=4
    )
    registry.classify_session(active_old.session_id, CompatibilityClass.INCOMPATIBLE)

    common_gates = (
        _gate(
            "lineage",
            artifact_hash=promotion_artifact_hashes["lineage"],
            measured=1.0,
            threshold=1.0,
            comparator="eq",
            sample_count=len(batch.samples),
            seed=seed,
            detail="all training samples linked to trajectory, reward, credit, environment, and model state",
        ),
        _gate(
            "reward_integrity",
            artifact_hash=promotion_artifact_hashes["reward_integrity"],
            measured=1.0,
            threshold=1.0,
            comparator="eq",
            sample_count=len(rewards),
            seed=seed,
            detail="verifier source remained immutable and hidden expected values were not exposed",
        ),
        _gate(
            "safety",
            artifact_hash=promotion_artifact_hashes["safety"],
            measured=0.0,
            threshold=0.0,
            comparator="eq",
            sample_count=len(trajectories),
            seed=seed,
            detail="all speculative actions were isolated idempotent workspace writes",
        ),
        _gate(
            "compatibility",
            artifact_hash=promotion_artifact_hashes["compatibility"],
            measured=1.0,
            threshold=1.0,
            comparator="eq",
            sample_count=1,
            seed=seed,
            detail="incompatible active session was classified champion-pinned",
        ),
    )
    rejected_quality = _gate(
        "quality",
        artifact_hash=rejected_evaluation_hash,
        measured=rejected_rate,
        threshold=_QUALITY_THRESHOLD,
        comparator="ge",
        sample_count=len(evaluation_seeds),
        seed=seed,
        detail="actual hidden coding evaluation for under-trained candidate",
    )
    rejected_serving = _gate(
        "serving",
        artifact_hash=rejected_evaluation_hash,
        measured=float(rejected_evaluation["mean_policy_decision_latency_ms"]),
        threshold=2.0,
        comparator="le",
        sample_count=int(rejected_evaluation["latency_iterations"]),
        seed=seed,
        detail="measured local CPU policy decision latency",
    )
    rejected_promotion = registry.create_promotion(
        transaction_id="promotion-rejected",
        deployment="coding-agent-prod",
        candidate_policy_epoch_id=rejected_training.candidate.policy_epoch_id,
        evidence=(*common_gates[:2], rejected_quality, *common_gates[2:], rejected_serving),
        observed_at_ms=10,
    )
    if rejected_promotion.state is not PromotionState.REJECTED:
        raise RuntimeError("quality-failing candidate was not rejected")

    corrected_quality = _gate(
        "quality",
        artifact_hash=promotion_artifact_hashes["quality"],
        measured=corrected_rate,
        threshold=_QUALITY_THRESHOLD,
        comparator="ge",
        sample_count=len(evaluation_seeds),
        seed=seed,
        detail="actual hidden coding evaluation for corrected candidate",
    )
    corrected_serving = _gate(
        "serving",
        artifact_hash=promotion_artifact_hashes["serving"],
        measured=float(corrected_evaluation["mean_policy_decision_latency_ms"]),
        threshold=2.0,
        comparator="le",
        sample_count=int(corrected_evaluation["latency_iterations"]),
        seed=seed,
        detail="measured local CPU policy decision latency",
    )
    registry.create_promotion(
        transaction_id="promotion-accepted",
        deployment="coding-agent-prod",
        candidate_policy_epoch_id=corrected_training.candidate.policy_epoch_id,
        evidence=(*common_gates[:2], corrected_quality, *common_gates[2:], corrected_serving),
        observed_at_ms=20,
    )
    registry.start_shadow("promotion-accepted", observed_at_ms=21)
    shadow_evaluation = _evaluate_policy(
        name="shadow",
        policy=corrected_training.candidate,
        seeds=tuple(
            _named_seed("sloforge-helix-shadow-v1", lower=1_000, span=10_000) + index
            for index in range(24)
        ),
        backend=environment_backend,
        base_environment=environment,
        actions=actions,
        output=output / "evaluations" / "shadow",
        trusted_evaluator_digest=holdout_evaluator_digest,
        reward_spec=holdout_reward_spec,
    )
    _write_json(output / "evaluations" / "shadow.json", shadow_evaluation)
    promotion_artifact_hashes["shadow"] = _write_sealed_json(
        promotion_evidence_root / "shadow.json", shadow_evaluation
    )
    shadow_gate = _gate(
        "shadow",
        artifact_hash=promotion_artifact_hashes["shadow"],
        measured=float(shadow_evaluation["task_success_rate"]),
        threshold=_QUALITY_THRESHOLD,
        comparator="ge",
        sample_count=24,
        seed=seed,
        detail="isolated side-effect-free shadow evaluation",
    )
    registry.finish_shadow(
        "promotion-accepted",
        shadow_gate,
        observed_at_ms=22,
    )
    registry.start_canary("promotion-accepted", observed_at_ms=23)
    canary_evaluation = _evaluate_policy(
        name="canary",
        policy=corrected_training.candidate,
        seeds=tuple(
            _named_seed("sloforge-helix-canary-v1", lower=1_000, span=10_000) + index
            for index in range(24)
        ),
        backend=environment_backend,
        base_environment=environment,
        actions=actions,
        output=output / "evaluations" / "canary",
        trusted_evaluator_digest=holdout_evaluator_digest,
        reward_spec=holdout_reward_spec,
    )
    _write_json(output / "evaluations" / "canary.json", canary_evaluation)
    promotion_artifact_hashes["canary"] = _write_sealed_json(
        promotion_evidence_root / "canary.json", canary_evaluation
    )
    canary_gate = _gate(
        "canary",
        artifact_hash=promotion_artifact_hashes["canary"],
        measured=float(canary_evaluation["task_success_rate"]),
        threshold=_QUALITY_THRESHOLD,
        comparator="ge",
        sample_count=24,
        seed=seed,
        detail="bounded local canary evaluation",
    )
    registry.finish_canary(
        "promotion-accepted",
        canary_gate,
        observed_at_ms=24,
    )
    source_policy_epoch = PolicyEpoch(
        policy_id="coding-agent",
        epoch=0,
        policy_digest=Digest(value=champion.weights_hash),
        created_at=_STAMP,
        lineage=(
            _lineage(
                "coding-agent-bootstrap",
                "sloforge.helix/PolicyBootstrap",
                champion.weights_hash,
            ),
        ),
    )
    target_policy_epoch = PolicyEpoch(
        policy_id="coding-agent",
        epoch=1,
        policy_digest=Digest(value=corrected_training.candidate.weights_hash),
        parent_epoch=0,
        parent_policy_digest=Digest(value=champion.weights_hash),
        training_transaction_id="learning-tx-accepted",
        created_at=_STAMP,
        lineage=(
            _lineage("coding-agent@0", "sloforge.helix/PolicyEpoch", champion.weights_hash),
            _lineage(
                batch.batch_id,
                "sloforge.helix/TrainingBatchManifest",
                batch_hash,
            ),
        ),
    )
    accepted_gate_evidence = (
        *common_gates[:2],
        corrected_quality,
        *common_gates[2:],
        corrected_serving,
        shadow_gate,
        canary_gate,
    )
    promotion_gate_names: tuple[GateName, ...] = (
        "lineage",
        "reward_integrity",
        "quality",
        "safety",
        "serving",
        "compatibility",
        "shadow",
        "canary",
    )
    promotion_sources = tuple(
        PromotionArtifactSource(
            gate=gate,
            relative_path=f"{gate}.json",
            captured_at=_STAMP,
            captured_at_ms=24,
        )
        for gate in promotion_gate_names
    )
    promotion_capsule = build_policy_promotion_capsule(
        registry=registry,
        transaction_id="promotion-accepted",
        promotion_id="promotion-capsule-accepted",
        from_policy_epoch=source_policy_epoch,
        to_policy_epoch=target_policy_epoch,
        gate_evidence=accepted_gate_evidence,
        artifact_root=promotion_evidence_root,
        artifact_sources=promotion_sources,
        continuum_artifact_kind="decision",
        approved_by="trusted-local-promotion-gate",
        promoted_at=_STAMP,
        lineage=(
            _lineage("promotion-accepted", "sloforge.helix/LearningTransaction", trigger_hash),
            _lineage("coding-agent@0", "sloforge.helix/PolicyEpoch", champion.weights_hash),
            _lineage(
                "coding-agent@1",
                "sloforge.helix/PolicyEpoch",
                corrected_training.candidate.weights_hash,
            ),
        ),
        created_at_ms=30,
        valid_for_ms=1_000,
        maximum_evidence_age_ms=100,
        seed=seed,
    )
    promotion_validation = validate_policy_promotion_capsule(
        promotion_capsule,
        registry=registry,
        artifact_root=promotion_evidence_root,
        validated_at_ms=31,
    )
    if not promotion_validation.eligible_for_promotion:
        raise RuntimeError("trusted promotion capsule validation did not authorize promotion")
    _write_json(output / "promotion" / "capsule.json", promotion_capsule)
    _write_json(output / "promotion" / "validation.json", promotion_validation)
    registry.promote("promotion-accepted", observed_at_ms=32)
    pinned_after_promotion = registry.session("active-incompatible")
    challenger_session = registry.open_session(
        deployment="coding-agent-prod", session_id="new-challenger-session", opened_at_ms=33
    )
    challenger_route_before_rollback = challenger_session.policy_epoch_id
    registry.rollback(
        "promotion-accepted",
        reason="separate deterministic post-promotion regression drill",
        observed_at_ms=34,
    )
    route_after_rollback = registry.open_session(
        deployment="coding-agent-prod", session_id="post-rollback-session", opened_at_ms=35
    )
    if not (
        pinned_after_promotion.pinned
        and pinned_after_promotion.policy_epoch_id == champion.policy_epoch_id
        and challenger_route_before_rollback == corrected_training.candidate.policy_epoch_id
        and route_after_rollback.policy_epoch_id == champion.policy_epoch_id
    ):
        raise RuntimeError("active-session routing or rollback invariant failed")

    now = _advance(
        transaction_store,
        "learning-tx-accepted",
        (
            LearningState.SHADOWING,
            LearningState.CANARYING,
            LearningState.PROMOTION_INTENT_RECORDED,
            LearningState.PROMOTING,
            LearningState.ACTIVE,
            LearningState.POST_PROMOTION_MONITORING,
        ),
        start_time=now,
    )
    transaction_store.transition(
        "learning-tx-accepted",
        target=LearningState.ROLLED_BACK,
        reason="separate rollback drill restored champion routing and preserved evidence",
        observed_at_ms=now + 1,
    )
    transaction_store.record_cost(
        "learning-tx-accepted", cost_id="local-cpu", amount_usd=0.0, source="local CPU"
    )
    accepted_transaction = transaction_store.transaction("learning-tx-accepted")
    rejected_transaction = transaction_store.transaction("learning-tx-rejected")
    timeline = tuple(
        event.model_dump(mode="json") for event in transaction_store.events("learning-tx-accepted")
    )
    _write_json(output / "transactions" / "accepted.json", accepted_transaction)
    _write_json(output / "transactions" / "rejected.json", rejected_transaction)
    _write_json(output / "transactions" / "timeline.json", timeline)

    lineage_nodes: list[dict[str, object]] = [
        {"id": "production-failure", "kind": "observation", "hash": trigger_hash},
        {"id": branch_point.branch_point_id, "kind": "BranchPoint", "hash": branch_point_hash},
    ]
    lineage_nodes.extend(
        [
            {"id": item.trajectory_id, "kind": "TrajectoryCapsule", "policy": item.policy_epoch_id}
            for item in trajectories
        ]
    )
    lineage_nodes.extend(
        [
            {"id": item.reward_id, "kind": "RewardEvidence", "trajectory": item.trajectory_id}
            for item in rewards
        ]
    )
    lineage_nodes.extend(
        [
            {"id": "branch-credit", "kind": "CreditAssignmentEvidence", "hash": credit_hash},
            {"id": batch.batch_id, "kind": "TrainingBatchManifest", "hash": batch_hash},
            {
                "id": corrected_training.candidate.policy_epoch_id,
                "kind": "PolicyEpoch",
                "checkpoint": corrected_training.checkpoint_hash,
            },
            {"id": "promotion-accepted", "kind": "PolicyPromotion"},
            {"id": "rollback-drill", "kind": "Rollback"},
        ]
    )
    lineage_edges: list[dict[str, str]] = [
        {
            "from": "production-failure",
            "to": branch_point.branch_point_id,
            "relation": "captured_as",
        },
        *(
            {
                "from": branch_point.branch_point_id,
                "to": item.trajectory_id,
                "relation": "forked_to",
            }
            for item in trajectories
        ),
        *(
            {"from": item.trajectory_id, "to": reward.reward_id, "relation": "evaluated_by"}
            for item, reward in zip(trajectories, rewards, strict=True)
        ),
        *(
            {"from": reward.reward_id, "to": "branch-credit", "relation": "supports"}
            for reward in rewards
        ),
        {"from": "branch-credit", "to": batch.batch_id, "relation": "weighted"},
        {
            "from": batch.batch_id,
            "to": corrected_training.candidate.policy_epoch_id,
            "relation": "trained",
        },
        {
            "from": corrected_training.candidate.policy_epoch_id,
            "to": "promotion-accepted",
            "relation": "validated_for",
        },
        {"from": "promotion-accepted", "to": "rollback-drill", "relation": "rolled_back_by"},
    ]
    lineage = {"nodes": lineage_nodes, "edges": lineage_edges}
    lineage_hash = _write_json(output / "lineage" / "graph.json", lineage)

    accounting = environment_backend.accounting()
    summary: dict[str, Any] = {
        "schema_version": "sloforge.helix.cpu-demo/v1",
        "seed": seed,
        "git_commit": git_commit,
        "production_failure_seed": production_seed,
        "production_failure_action": production_decision.action,
        "champion_policy_hash": champion_policy_hash,
        "branch_point_id": branch_point.branch_point_id,
        "continuum_capsule_id": branch_point.continuum_capsule_id,
        "environment_capsule_id": environment.capsule_id,
        "capture_consistent": branch_point.boundary == boundary,
        "capture_artifact_bytes_verified": True,
        "source_resumed": True,
        "branch_group_id": group.group_id,
        "branch_strategies": forced,
        "state_reuse": {
            item.branch_id: item.state_reuse.model_dump(mode="json") for item in group.members
        },
        "branch_rewards": {
            trajectory.branch_id: reward.total_score
            for trajectory, reward in zip(trajectories, rewards, strict=True)
        },
        "reward_authority": {
            "training_evaluator_sha256": training_evaluator_digest,
            "holdout_evaluator_sha256": holdout_evaluator_digest,
            "holdout_separated_from_training": training_evaluator_digest
            != holdout_evaluator_digest,
            "all_evaluators_trusted": all(item.evaluator_trusted for item in rewards),
        },
        "credit_evidence_hash": credit_hash,
        "training_batch_id": batch.batch_id,
        "training_batch_hash": batch_hash,
        "rejected_candidate": {
            "policy_epoch_id": rejected_training.candidate.policy_epoch_id,
            "success_rate": rejected_rate,
            "quality_threshold": _QUALITY_THRESHOLD,
            "promotion_state": rejected_promotion.state.value,
            "evaluation_hash": rejected_evaluation_hash,
        },
        "corrected_candidate": {
            "policy_epoch_id": corrected_training.candidate.policy_epoch_id,
            "success_rate": corrected_rate,
            "quality_threshold": _QUALITY_THRESHOLD,
            "evaluation_hash": corrected_evaluation_hash,
            "checkpoint_hash": corrected_training.checkpoint_hash,
        },
        "champion_success_rate": champion_evaluation["task_success_rate"],
        "measured_success_rate_delta": corrected_rate
        - float(champion_evaluation["task_success_rate"]),
        "serving": {
            "candidate_mean_policy_decision_latency_ms": corrected_evaluation[
                "mean_policy_decision_latency_ms"
            ],
            "synthetic": True,
        },
        "replay": replay,
        "promotion": {
            "tenant_id": promotion_capsule.tenant_id,
            "capsule_digest": promotion_capsule.capsule_digest,
            "capsule_validation_eligible": promotion_validation.eligible_for_promotion,
            "state_compatibility_class": compatibility_decision.compatibility_class.value,
            "challenger_route_before_rollback": challenger_route_before_rollback,
            "incompatible_session_policy": pinned_after_promotion.policy_epoch_id,
            "incompatible_session_pinned": pinned_after_promotion.pinned,
            "route_after_rollback": route_after_rollback.policy_epoch_id,
            "final_state": registry.promotion("promotion-accepted").state.value,
        },
        "learning_transaction_state": accepted_transaction.state.value,
        "rejected_transaction_state": rejected_transaction.state.value,
        "lineage_hash": lineage_hash,
        "environment_cleanup": {
            "branch_workspace_bytes": accounting.branch_workspace_bytes,
            "live_branches": 0,
        },
        "validation_class": "deterministic-local-cpu-synthetic",
        "hardware_backed": False,
        "limitations": (
            "the reference categorical policy and coding repository are synthetic",
            "no GPU, distributed trainer, live production traffic, or external side effect was exercised",
            "branch-relative credit is a controlled sibling comparison, not universal causal identification",
            "latency is a local CPU policy-decision measurement, not a model-serving benchmark",
        ),
    }
    summary_hash = _write_json(output / "summary.json", summary)
    report = (
        "# SLOForge Helix CPU demonstration\n\n"
        f"Artifact hash: `{summary_hash}`\n\n"
        f"The executed champion chose `{production_decision.action}` and achieved "
        f"{float(champion_evaluation['task_success_rate']):.3f} hidden task success. "
        f"The one-step candidate was rejected at {rejected_rate:.3f}; the corrected "
        f"branch-relative candidate achieved {corrected_rate:.3f}.\n\n"
        f"Joint capture bound Continuum capsule `{branch_point.continuum_capsule_id}` to "
        f"environment capsule `{environment.capsule_id}` at one persisted boundary. "
        "An incompatible active session remained on the champion. New traffic reached the "
        "challenger after atomic promotion, and the separate rollback drill restored the "
        "champion pointer.\n\n"
        "All quality scores came from the sandboxed visible and hidden Python verifiers. "
        "This is CPU-only synthetic validation; it is not a GPU or production measurement.\n"
    )
    report_path = output / "report.md"
    report_path.write_text(report)
    registry.close()
    transaction_store.close()
    return summary


_SUCCESS_PREFIX_TO_QUALITY = (
    LearningState.CAPTURE_PROPOSED,
    LearningState.CAPTURED,
    LearningState.BRANCH_PLAN_CREATED,
    LearningState.FORKING,
    LearningState.ROLLOUTS_RUNNING,
    LearningState.ROLLOUTS_COMPLETED,
    LearningState.REWARDS_VALIDATING,
    LearningState.CREDIT_ASSIGNING,
    LearningState.TRAINING_BATCH_BUILDING,
    LearningState.TRAINING,
    LearningState.CANDIDATE_READY,
    LearningState.ALGORITHM_VALIDATING,
    LearningState.QUALITY_VALIDATING,
)
