"""Semantics-preserving measurement wrappers for the real Helix CPU demo.

This module intentionally does not duplicate the demo.  It temporarily wraps the
actual capture, fork, rollout, environment, reward, training, transaction, and
promotion operations invoked by :func:`sloforge.helix.demo.run_cpu_demo`.  The
wrappers observe arguments/results and restore every original callable before the
harness returns, including on failure.
"""

from __future__ import annotations

import json
import os
import random
import socket
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from sloforge.continuum.operations import CheckpointArtifact, load_checkpoint_artifact
from sloforge.helix import demo
from sloforge.helix.branching import create_branch_group as _create_branch_group
from sloforge.helix.capture import CoordinatedCaptureCoordinator
from sloforge.helix.environments import EnvironmentBranch
from sloforge.helix.environments.models import EntryKind, EnvironmentStateCapsule
from sloforge.helix.promotion import PolicyRegistry
from sloforge.helix.rewards import DeterministicRewardWorker
from sloforge.helix.rollouts import ReferenceRolloutWorker
from sloforge.helix.trainers import ReferenceTrainer
from sloforge.helix.transactions import LearningTransactionStore

from .analysis import analyze_branch_state_sharing
from .recorder import JsonValue, LifecycleRecorder, TraceStream

_T = TypeVar("_T")
_PATCH_LOCK = threading.Lock()


class TraceLevel(StrEnum):
    DISABLED = "disabled"
    MINIMAL = "minimal"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class CharacterizedRun:
    summary: dict[str, Any]
    trace_id: str
    trace_level: TraceLevel
    wall_time_ns: int
    cpu_time_ns: int
    branch_event_count: int
    state_event_count: int
    semantic_digest: str
    sharing_analysis: dict[str, object] | None
    migration_observed: bool = False
    migration_note: str = "The exercised Helix CPU demo does not migrate a post-fork branch."


@dataclass(frozen=True, slots=True)
class OverheadSample:
    trace_level: TraceLevel
    repetition: int
    order_index: int
    seed: int
    wall_time_ns: int
    cpu_time_ns: int
    branch_event_count: int
    state_event_count: int
    semantic_digest: str
    artifact_path: str


@dataclass(frozen=True, slots=True)
class _Timing:
    monotonic_start_ns: int
    duration_ns: int
    cpu_time_ns: int


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _model_document(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _measured_document(value: object) -> tuple[int, str]:
    payload = _canonical_bytes(_model_document(value))
    return len(payload), sha256(payload).hexdigest()


def _call_timed(function: Callable[..., _T], *args: Any, **kwargs: Any) -> tuple[_T, _Timing]:
    start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    result = function(*args, **kwargs)
    cpu_end = time.process_time_ns()
    end = time.perf_counter_ns()
    return result, _Timing(start, end - start, cpu_end - cpu_start)


def _segment_sizes(artifact: CheckpointArtifact) -> dict[str, tuple[str, int]]:
    return {
        manifest.segment_id: (
            manifest.segment_hash.value,
            sum(chunk.size_bytes for chunk in manifest.chunks),
        )
        for manifest in artifact.capsule.segment_manifests
    }


def _unique_chunk_bytes(artifact: CheckpointArtifact) -> int:
    return sum({item.digest: item.size_bytes for item in artifact.chunk_references}.values())


def _workspace_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    )


def _environment_bytes(capsule: EnvironmentStateCapsule) -> int:
    return sum(
        entry.size_bytes
        for entry in capsule.files
        if entry.kind in {EntryKind.FILE, EntryKind.REDACTED}
    )


def _environment_content(capsule: EnvironmentStateCapsule) -> dict[str, tuple[str, int]]:
    return {
        entry.path: (cast(str, entry.content_hash), entry.size_bytes)
        for entry in capsule.files
        if entry.kind in {EntryKind.FILE, EntryKind.REDACTED}
    }


def _semantic_digest(summary: Mapping[str, Any]) -> str:
    """Hash state/decision semantics while excluding explicitly measured latencies."""

    def semantic_value(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: semantic_value(item)
                for key, item in value.items()
                if key not in {"evaluation_hash", "capsule_digest", "lineage_hash"}
                and "latency" not in key
                and "duration" not in key
            }
        if isinstance(value, list):
            return [semantic_value(item) for item in value]
        return value

    projection = {
        key: semantic_value(summary.get(key))
        for key in (
            "production_failure_action",
            "capture_consistent",
            "source_resumed",
            "branch_group_id",
            "branch_actions",
            "branch_rewards",
            "state_reuse",
            "corrected_candidate",
            "rejected_candidate",
            "promotion",
            "learning_transaction_state",
            "rejected_transaction_state",
        )
    }
    return sha256(_canonical_bytes(projection)).hexdigest()


@dataclass(slots=True)
class _Emitter:
    recorder: LifecycleRecorder
    level: TraceLevel
    trace_id: str
    seed: int
    origin_ns: int
    sequence: int = 0
    branch_event_count: int = 0
    state_event_count: int = 0
    branch_point_id: str | None = None
    branch_group_id: str | None = None
    branch_order: list[str] = field(default_factory=list)
    rewards: dict[str, float] = field(default_factory=dict)
    environment_hashes_seen: set[str] = field(default_factory=set)
    branch_events: list[dict[str, JsonValue]] = field(default_factory=list)
    state_events: list[dict[str, JsonValue]] = field(default_factory=list)

    def emit(
        self,
        stream: TraceStream,
        operation_type: str,
        timing: _Timing,
        **values: JsonValue,
    ) -> None:
        if self.level is TraceLevel.DISABLED:
            return
        event: dict[str, JsonValue] = {
            "trace_id": self.trace_id,
            "session_id": "production-session",
            "branch_group_id": self.branch_group_id,
            "branch_id": None,
            "parent_branch_id": "production-session",
            "policy_epoch": None,
            "environment_id": None,
            "transaction_id": None,
            "host": socket.gethostname(),
            "process": os.getpid(),
            "rank": 0,
            "device": "cpu",
            "monotonic_timestamp_ns": timing.monotonic_start_ns,
            "normalized_timestamp_ns": timing.monotonic_start_ns - self.origin_ns,
            "duration_ns": timing.duration_ns,
            "cpu_time_ns": timing.cpu_time_ns,
            "clock_source": "time.perf_counter_ns",
            "alignment_confidence": 1.0,
            "operation_type": operation_type,
            "event_sequence": self.sequence,
            "schema_version": "lifecycle-observation/v1",
            "trace_producer_version": "sloforge-helix-characterization-lifecycle/v1",
            "measurement_source": "SYNTHETIC",
            "workload_evidence_class": "SYNTHETIC",
            "timing_measurement_class": "HARDWARE_BACKED_REAL",
            "simulated_gpu_state": True,
            "seed": self.seed,
            "result": "success",
        }
        event.update(values)
        event["content_hash"] = sha256(_canonical_bytes(event)).hexdigest()
        self.sequence += 1
        if stream is TraceStream.BRANCH_WORKLOAD:
            self.branch_event_count += 1
            self.branch_events.append(dict(event))
        else:
            self.state_event_count += 1
            self.state_events.append(dict(event))
        self.recorder.record(stream, event)


@contextmanager
def _patched_attribute(owner: object, name: str, replacement: object) -> Iterator[None]:
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


def _install_wrappers(stack: ExitStack, emitter: _Emitter) -> None:
    original_capture = CoordinatedCaptureCoordinator.execute

    def capture_execute(instance: Any, capture_id: str, sources: Any) -> Any:
        result, timing = _call_timed(original_capture, instance, capture_id, sources)
        emitter.branch_point_id = result.branch_point_id
        size, digest = _measured_document(result)
        emitter.emit(
            TraceStream.BRANCH_WORKLOAD,
            "BRANCH_POINT_CAPTURE",
            timing,
            logical_state_id=result.continuum_capsule_id,
            physical_state_id=result.continuum_capsule_id,
            environment_id=result.environment.artifact_id,
            policy_epoch=result.policy_epoch_id,
            logical_bytes=size,
            metadata_bytes=size,
            content_digest=digest,
        )
        artifact_path = Path(instance._artifact_directory) / f"{capture_id}.continuum.json"
        checkpoint = load_checkpoint_artifact(artifact_path)
        descriptor_bytes = artifact_path.stat().st_size
        emitter.emit(
            TraceStream.STATE_OPERATION,
            "STATE_SNAPSHOT",
            timing,
            logical_state_id=checkpoint.capsule.identity.capsule_id,
            branch_id="production-session",
            tenant="tenant-helix-demo",
            state_segment="model",
            bytes=sum(size for _digest, size in _segment_sizes(checkpoint).values()),
            logical_bytes=sum(size for _digest, size in _segment_sizes(checkpoint).values()),
            physical_bytes=_unique_chunk_bytes(checkpoint),
            metadata_bytes=descriptor_bytes,
            source_physical_representation="reference_runtime_memory",
            destination_physical_representation="content_addressed_checkpoint",
            state_epoch=checkpoint.capsule.transaction.source_epoch,
        )
        return result

    stack.enter_context(
        _patched_attribute(CoordinatedCaptureCoordinator, "execute", capture_execute)
    )

    original_group = _create_branch_group

    def create_group(parent: CheckpointArtifact, **kwargs: Any) -> Any:
        group, timing = _call_timed(original_group, parent, **kwargs)
        emitter.branch_group_id = group.group_id
        emitter.branch_order = [member.branch_id for member in group.members]
        source_segments = _segment_sizes(parent)
        source_chunks = {item.digest for item in parent.chunk_references}
        seen_chunks = set(source_chunks)
        source_physical_bytes = _unique_chunk_bytes(parent)
        for member in group.members:
            segments = _segment_sizes(member.checkpoint)
            shared = sum(
                size
                for digest, size in segments.values()
                if digest in set(group.shared_immutable_digests)
            )
            logical = sum(size for _digest, size in segments.values())
            newly_allocated = sum(
                reference.size_bytes
                for reference in member.checkpoint.chunk_references
                if reference.digest not in seen_chunks
            )
            seen_chunks.update(item.digest for item in member.checkpoint.chunk_references)
            max_cow_refs = max(
                (
                    entry.copy_on_write_reference_count
                    for table in member.checkpoint.capsule.physical_state.page_tables
                    for entry in table.entries
                ),
                default=0,
            )
            checkpoint_metadata_bytes, _checkpoint_digest = _measured_document(
                member.checkpoint.capsule
            )
            emitter.emit(
                TraceStream.STATE_OPERATION,
                "STATE_FORK",
                timing,
                branch_id=member.branch_id,
                logical_state_id=member.checkpoint.capsule.identity.capsule_id,
                physical_state_id=member.checkpoint.store_manifest.manifest_id,
                tenant="tenant-helix-demo",
                state_segment="model",
                bytes=logical,
                logical_bytes=logical,
                physical_bytes=newly_allocated,
                source_physical_bytes=source_physical_bytes,
                shared_logical_bytes=shared,
                private_logical_bytes=logical - shared,
                naive_independent_bytes=sum(size for _digest, size in source_segments.values()),
                metadata_bytes=checkpoint_metadata_bytes,
                copy_on_write_reference_count=max_cow_refs,
                source_physical_representation="content_addressed_checkpoint",
                destination_physical_representation="content_addressed_fork",
                state_epoch=member.checkpoint.capsule.transaction.source_epoch,
                timing_span_id=f"{emitter.trace_id}:combined-branch-group-fork",
                timing_scope="combined_model_and_environment_group_fork",
                duration_attribution="shared_span_do_not_sum_across_branch_events",
            )
            environment = cast(EnvironmentBranch, member.environment_branch)
            base = environment._backend._branch_state(member.branch_id).base
            emitter.environment_hashes_seen.update(
                digest for digest, _size in _environment_content(base).values()
            )
            emitter.emit(
                TraceStream.BRANCH_WORKLOAD,
                "ENVIRONMENT_FORK",
                timing,
                branch_id=member.branch_id,
                logical_state_id=base.capsule_id,
                environment_id=base.capsule_id,
                state_segment="filesystem",
                logical_bytes=_environment_bytes(base),
                physical_bytes=_workspace_bytes(environment.workspace),
                metadata_bytes=len(environment._backend.artifact_payload(base)),
                fork_implementation="eager_restore",
                timing_span_id=f"{emitter.trace_id}:combined-branch-group-fork",
                timing_scope="combined_model_and_environment_group_fork",
                duration_attribution="shared_span_do_not_sum_across_branch_events",
            )
            emitter.emit(
                TraceStream.BRANCH_WORKLOAD,
                "BRANCH_FORK",
                timing,
                branch_id=member.branch_id,
                logical_state_id=member.checkpoint.capsule.identity.capsule_id,
                physical_state_id=member.checkpoint.store_manifest.manifest_id,
                environment_id=base.capsule_id,
                policy_epoch=member.policy_epoch_id,
                shared_root=True,
                private_suffix=False,
                branch_strategy=member.state_reuse.strategy.value,
                timing_span_id=f"{emitter.trace_id}:combined-branch-group-fork",
                timing_scope="combined_model_and_environment_group_fork",
                duration_attribution="shared_span_do_not_sum_across_branch_events",
            )
            emitter.emit(
                TraceStream.BRANCH_WORKLOAD,
                "BRANCH_READY",
                _Timing(
                    timing.monotonic_start_ns + timing.duration_ns,
                    0,
                    0,
                ),
                branch_id=member.branch_id,
                logical_state_id=member.checkpoint.capsule.identity.capsule_id,
                environment_id=base.capsule_id,
                policy_epoch=member.policy_epoch_id,
                wait_latency_ns=timing.duration_ns,
                readiness_scope="combined_model_and_environment_group_fork",
                duration_attribution="zero-duration completion marker",
            )
        return group

    stack.enter_context(_patched_attribute(demo, "create_branch_group", create_group))

    original_rollout = ReferenceRolloutWorker.run

    def rollout_run(instance: Any, **kwargs: Any) -> Any:
        trajectory, timing = _call_timed(original_rollout, instance, **kwargs)
        if trajectory.branch_id not in emitter.branch_order:
            return trajectory
        size, digest = _measured_document(trajectory)
        emitter.emit(
            TraceStream.BRANCH_WORKLOAD,
            "ROLLOUT_COMPLETE",
            timing,
            branch_id=trajectory.branch_id,
            logical_state_id=trajectory.source_model_capsule_id,
            environment_id=trajectory.final_environment_capsule_id,
            policy_epoch=trajectory.policy_epoch_id,
            logical_bytes=size,
            metadata_bytes=size,
            content_digest=digest,
            generated_tokens=len(trajectory.tokens),
            action_count=len(trajectory.actions),
            first_divergent_action=0,
            private_suffix=True,
        )
        return trajectory

    stack.enter_context(_patched_attribute(ReferenceRolloutWorker, "run", rollout_run))

    original_reward = DeterministicRewardWorker.verify

    def reward_verify(instance: Any, **kwargs: Any) -> Any:
        reward, timing = _call_timed(original_reward, instance, **kwargs)
        branch_id = str(kwargs["trajectory_id"])
        # Trajectory ids are hashes, so recover the exact branch from the evidence path.
        evidence = Path(kwargs["evidence_directory"])
        branch_id = evidence.name if evidence.name in emitter.branch_order else branch_id
        if branch_id not in emitter.branch_order:
            return reward
        emitter.rewards[branch_id] = reward.total_score
        size, digest = _measured_document(reward)
        emitter.emit(
            TraceStream.BRANCH_WORKLOAD,
            "REWARD_COMPLETE",
            timing,
            branch_id=branch_id,
            policy_epoch=reward.policy_epoch_id,
            logical_bytes=size,
            metadata_bytes=size,
            content_digest=digest,
            reward=reward.total_score,
        )
        return reward

    stack.enter_context(_patched_attribute(DeterministicRewardWorker, "verify", reward_verify))

    original_train = ReferenceTrainer.train

    def trainer_train(instance: Any, **kwargs: Any) -> Any:
        result, timing = _call_timed(original_train, instance, **kwargs)
        size, digest = _measured_document(result)
        emitter.emit(
            TraceStream.BRANCH_WORKLOAD,
            "TRAINING_COMPLETE",
            timing,
            policy_epoch=result.candidate.policy_epoch_id,
            logical_state_id=result.checkpoint_hash,
            logical_bytes=size,
            metadata_bytes=size,
            content_digest=digest,
            training_steps=len(result.metrics),
        )
        return result

    stack.enter_context(_patched_attribute(ReferenceTrainer, "train", trainer_train))

    original_evaluate = demo._evaluate_policy

    def evaluate_policy(**kwargs: Any) -> Any:
        result, timing = _call_timed(original_evaluate, **kwargs)
        size, digest = _measured_document(result)
        emitter.emit(
            TraceStream.BRANCH_WORKLOAD,
            "EVALUATE",
            timing,
            policy_epoch=str(result["policy_epoch_id"]),
            logical_bytes=size,
            metadata_bytes=size,
            content_digest=digest,
            evaluation_name=str(kwargs["name"]),
            evaluation_case_count=len(cast(Sequence[object], result["cases"])),
            evaluation_hardware_class=str(result["hardware_class"]),
        )
        return result

    stack.enter_context(_patched_attribute(demo, "_evaluate_policy", evaluate_policy))

    original_transition = LearningTransactionStore.transition

    def transaction_transition(instance: Any, transaction_id: str, **kwargs: Any) -> Any:
        result, timing = _call_timed(original_transition, instance, transaction_id, **kwargs)
        emitter.emit(
            TraceStream.BRANCH_WORKLOAD,
            "LEARNING_TRANSACTION_STAGE",
            timing,
            transaction_id=transaction_id,
            policy_epoch=result.candidate_policy_epoch_id or result.champion_policy_epoch_id,
            transaction_state=result.state.value,
            transaction_sequence=result.sequence,
        )
        return result

    stack.enter_context(
        _patched_attribute(LearningTransactionStore, "transition", transaction_transition)
    )

    original_promote = PolicyRegistry.promote

    def promote(instance: Any, transaction_id: str, **kwargs: Any) -> Any:
        result, timing = _call_timed(original_promote, instance, transaction_id, **kwargs)
        emitter.emit(
            TraceStream.BRANCH_WORKLOAD,
            "PROMOTION_COMPLETE",
            timing,
            transaction_id=transaction_id,
            policy_epoch=result.candidate_policy_epoch_id,
            promotion_state=result.state.value,
        )
        return result

    stack.enter_context(_patched_attribute(PolicyRegistry, "promote", promote))

    if emitter.level is not TraceLevel.FULL:
        return

    original_write = EnvironmentBranch.write_bytes

    def write_bytes(
        instance: EnvironmentBranch, path: str, data: bytes, *, mode: int | None = None
    ) -> None:
        if instance.branch_id not in emitter.branch_order:
            original_write(instance, path, data, mode=mode)
            return
        target = instance.workspace / path
        old = target.read_bytes() if target.is_file() and not target.is_symlink() else b""
        _result, timing = _call_timed(original_write, instance, path, data, mode=mode)
        emitter.emit(
            TraceStream.STATE_OPERATION,
            "STATE_COW",
            timing,
            branch_id=instance.branch_id,
            logical_state_id=f"environment:{instance.info.base_capsule_id}:{path}",
            physical_state_id=sha256(data).hexdigest(),
            tenant="tenant-helix-demo",
            state_segment="filesystem",
            bytes=len(data),
            logical_bytes=len(data),
            physical_bytes=target.stat().st_size,
            old_bytes=len(old),
            old_content_hash=sha256(old).hexdigest(),
            alignment=1,
            page_size=0,
            source_physical_representation="eager_branch_workspace",
            destination_physical_representation="atomic_replacement_file",
            cow_granularity="whole_file",
        )

    stack.enter_context(_patched_attribute(EnvironmentBranch, "write_bytes", write_bytes))

    original_checkpoint = EnvironmentBranch.checkpoint

    def checkpoint(instance: EnvironmentBranch) -> EnvironmentStateCapsule:
        if instance.branch_id not in emitter.branch_order:
            return original_checkpoint(instance)
        base = instance._backend._branch_state(instance.branch_id).base
        result, timing = _call_timed(original_checkpoint, instance)
        base_content = _environment_content(base)
        result_content = _environment_content(result)
        dirty = sum(
            size
            for path, (digest, size) in result_content.items()
            if base_content.get(path) != (digest, size)
        )
        new_bytes = sum(
            size
            for digest, size in result_content.values()
            if digest not in emitter.environment_hashes_seen
        )
        emitter.environment_hashes_seen.update(digest for digest, _size in result_content.values())
        payload = instance._backend.artifact_payload(result)
        emitter.emit(
            TraceStream.STATE_OPERATION,
            "STATE_SNAPSHOT",
            timing,
            branch_id=instance.branch_id,
            logical_state_id=result.capsule_id,
            physical_state_id=sha256(payload).hexdigest(),
            tenant="tenant-helix-demo",
            state_segment="filesystem",
            bytes=_environment_bytes(result),
            logical_bytes=_environment_bytes(result),
            physical_bytes=new_bytes,
            dirty_bytes=dirty,
            metadata_bytes=len(payload),
            source_physical_representation="eager_branch_workspace",
            destination_physical_representation="content_addressed_incremental_capsule",
            state_epoch=result.event_watermark,
        )
        return result

    stack.enter_context(_patched_attribute(EnvironmentBranch, "checkpoint", checkpoint))

    original_cleanup = EnvironmentBranch.cleanup

    def cleanup(instance: EnvironmentBranch) -> None:
        if instance.branch_id not in emitter.branch_order:
            original_cleanup(instance)
            return
        bytes_before = _workspace_bytes(instance.workspace)
        _result, timing = _call_timed(original_cleanup, instance)
        winner = next(
            (
                branch_id
                for branch_id in emitter.branch_order
                if emitter.rewards.get(branch_id)
                == max(emitter.rewards.values(), default=float("-inf"))
            ),
            None,
        )
        pruned = winner is not None and instance.branch_id != winner
        emitter.emit(
            TraceStream.BRANCH_WORKLOAD,
            "BRANCH_PRUNE" if pruned else "BRANCH_COMPLETE",
            timing,
            branch_id=instance.branch_id,
            physical_bytes=bytes_before,
            reward=emitter.rewards.get(instance.branch_id),
            prune_reason="dominated_by_measured_reward" if pruned else None,
        )
        emitter.emit(
            TraceStream.STATE_OPERATION,
            "STATE_FREE",
            timing,
            branch_id=instance.branch_id,
            logical_state_id=f"environment-branch:{instance.branch_id}",
            tenant="tenant-helix-demo",
            state_segment="filesystem",
            bytes=bytes_before,
            logical_bytes=bytes_before,
            physical_bytes=bytes_before,
            source_physical_representation="eager_branch_workspace",
            destination_physical_representation="reclaimed",
        )

    stack.enter_context(_patched_attribute(EnvironmentBranch, "cleanup", cleanup))


def run_characterized_cpu_demo(
    output: Path,
    *,
    seed: int,
    recorder: LifecycleRecorder,
    trace_level: TraceLevel = TraceLevel.FULL,
    trace_id: str | None = None,
) -> CharacterizedRun:
    """Run the real demo once while observing its actual lifecycle operations."""

    if not 0 <= seed <= 2**63 - 1:
        raise ValueError("demo seed must fit a signed 64-bit integer")
    chosen_trace_id = (
        trace_id or sha256(f"sloforge-helix-characterized-cpu/v1\0{seed}".encode()).hexdigest()
    )
    origin = time.perf_counter_ns()
    cpu_origin = time.process_time_ns()
    emitter = _Emitter(recorder, trace_level, chosen_trace_id, seed, origin)
    if trace_level is TraceLevel.DISABLED:
        summary = demo.run_cpu_demo(output, seed=seed)
    else:
        # Patches are process-global but bounded to this synchronous demo invocation.
        with _PATCH_LOCK, ExitStack() as stack:
            _install_wrappers(stack, emitter)
            summary = demo.run_cpu_demo(output, seed=seed)
    end = time.perf_counter_ns()
    cpu_end = time.process_time_ns()
    sharing = (
        analyze_branch_state_sharing(emitter.branch_events, emitter.state_events)
        if trace_level is TraceLevel.FULL
        else None
    )
    return CharacterizedRun(
        summary=summary,
        trace_id=chosen_trace_id,
        trace_level=trace_level,
        wall_time_ns=end - origin,
        cpu_time_ns=cpu_end - cpu_origin,
        branch_event_count=emitter.branch_event_count,
        state_event_count=emitter.state_event_count,
        semantic_digest=_semantic_digest(summary),
        sharing_analysis=sharing,
    )


def measure_cpu_demo_overhead(
    output_root: Path,
    *,
    seed: int,
    recorder_factory: Callable[[], LifecycleRecorder],
    repetitions: int = 3,
    levels: Sequence[TraceLevel] = (
        TraceLevel.DISABLED,
        TraceLevel.MINIMAL,
        TraceLevel.FULL,
    ),
) -> tuple[OverheadSample, ...]:
    """Run randomized repeated raw samples for disabled/minimal/full tracing.

    The caller owns aggregation and confidence intervals; this function deliberately
    returns every raw sample and its artifact path.  Order randomization is itself
    deterministic from the explicit seed.
    """

    if repetitions < 1:
        raise ValueError("overhead measurement requires at least one repetition")
    schedule = [(level, repetition) for repetition in range(repetitions) for level in levels]
    random.Random(seed).shuffle(schedule)
    samples: list[OverheadSample] = []
    for order_index, (level, repetition) in enumerate(schedule):
        artifact = output_root / f"{order_index:03d}-{level.value}-r{repetition:03d}"
        run = run_characterized_cpu_demo(
            artifact,
            seed=seed,
            recorder=recorder_factory(),
            trace_level=level,
            trace_id=sha256(
                f"sloforge-overhead/v1\0{seed}\0{level.value}\0{repetition}".encode()
            ).hexdigest(),
        )
        samples.append(
            OverheadSample(
                trace_level=level,
                repetition=repetition,
                order_index=order_index,
                seed=seed,
                wall_time_ns=run.wall_time_ns,
                cpu_time_ns=run.cpu_time_ns,
                branch_event_count=run.branch_event_count,
                state_event_count=run.state_event_count,
                semantic_digest=run.semantic_digest,
                artifact_path=os.fspath(artifact),
            )
        )
    return tuple(samples)
