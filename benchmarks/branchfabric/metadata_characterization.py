"""Experiment 003 same-GPU metadata characterization runner.

The readiness parent is a single continuous interval.  Runtime instrumentation
is version-scoped in ``vllm_metadata_0230`` and no vLLM object is serialized.
"""

from __future__ import annotations

import cProfile
import hashlib
import json
import os
import pstats
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from sloforge.continuum.adapters.real_runtime import (
    ConcurrentDecodeResult,
    GpuMemoryState,
    KvStateClass,
    PhysicalKvLayoutSnapshot,
    RuntimeSessionRef,
    SharedRootReference,
)
from sloforge.continuum.adapters.vllm_0230_metadata import (
    TimingDuration,
    hierarchy_from_vllm_observation,
)

Implementation = Literal["baseline", "optimized"]
TrialMode = Literal["independent_prefill", "shared_root"]


class MetadataAdapter0230(Protocol):
    def start_session(
        self, session_id: str, *, token_ids: tuple[int, ...], seed: int, timeout_s: float
    ) -> RuntimeSessionRef: ...

    def prefill_session(self, session_id: str, *, timeout_s: float) -> RuntimeSessionRef: ...

    def pause_at_safe_decode_boundary(
        self, session_id: str, *, timeout_s: float
    ) -> RuntimeSessionRef: ...

    def create_shared_root_reference(
        self, session_id: str, *, prefix_token_count: int, timeout_s: float
    ) -> SharedRootReference: ...

    def fork_same_policy_session(
        self,
        root_reference_id: str,
        branch_session_id: str,
        *,
        divergent_token_id: int,
        seed: int,
        timeout_s: float,
    ) -> RuntimeSessionRef: ...

    def fork_same_policy_sessions_0230(
        self,
        root_reference_id: str,
        branch_specs: tuple[tuple[str, int, int], ...],
        *,
        timeout_s: float,
    ) -> tuple[RuntimeSessionRef, ...]: ...

    def capture_branchpoint(
        self,
        root_reference_id: str,
        branch_session_ids: tuple[str, ...],
        *,
        timeout_s: float,
    ) -> object: ...

    def inspect_logical_state(self, session_id: str, *, timeout_s: float) -> object: ...

    def inspect_physical_kv_layout(
        self, session_ids: tuple[str, ...], *, timeout_s: float
    ) -> PhysicalKvLayoutSnapshot: ...

    def inspect_gpu_memory_state(self, *, timeout_s: float) -> GpuMemoryState: ...

    def begin_post_root_metadata_observation(
        self, *, branch_count: int, prefix_block_count: int
    ) -> None: ...

    def end_post_root_metadata_observation(self) -> None: ...

    def metadata_span(
        self, name: str, category: str, attributes: dict[str, object] | None = None
    ) -> Any: ...

    def inspect_metadata_observation(self) -> dict[str, object]: ...

    def emit_branch_workload_trace(self) -> tuple[object, ...]: ...

    def emit_state_operation_trace(self) -> tuple[object, ...]: ...

    def inspect_trace_self_overhead(self) -> dict[str, int | str]: ...

    def run_concurrent_branches(
        self,
        branch_session_ids: tuple[str, ...],
        *,
        maximum_new_tokens: int,
        seed: int,
        timeout_s: float,
        on_all_branches_ready: Any = None,
    ) -> ConcurrentDecodeResult: ...

    def run_concurrent_independent_sessions(
        self,
        session_ids: tuple[str, ...],
        *,
        maximum_new_tokens: int,
        seed: int,
        timeout_s: float,
        on_all_branches_ready: Any = None,
    ) -> ConcurrentDecodeResult: ...

    def destroy_branch(self, branch_session_id: str, *, timeout_s: float) -> None: ...

    def destroy_session(self, session_id: str, *, timeout_s: float) -> None: ...

    def cleanup_runtime(self, *, timeout_s: float) -> None: ...


@dataclass(frozen=True, slots=True)
class MetadataTrialConfiguration:
    attempt_id: str
    mode: TrialMode
    implementation: Implementation
    seed: int
    prefix_token_ids: tuple[int, ...]
    divergent_token_ids: tuple[int, ...]
    suffix_tokens: int
    timeout_s: float
    trace_level: Literal["disabled", "minimal", "full"]

    def __post_init__(self) -> None:
        if not self.attempt_id or len(self.attempt_id) > 128:
            raise ValueError("attempt_id must be a bounded non-empty string")
        if len(self.prefix_token_ids) != 16_384:
            raise ValueError("Experiment 003 requires exactly 16K prefix tokens")
        if not self.divergent_token_ids or len(self.divergent_token_ids) > 64:
            raise ValueError("fanout must be in 1..64")
        if self.suffix_tokens != 256:
            raise ValueError("Experiment 003 requires a 256-token suffix")
        if not 0 <= self.seed < 1 << 63:
            raise ValueError("seed must be in [0, 2**63)")
        if not 0 < self.timeout_s <= 3600:
            raise ValueError("timeout_s must be in (0, 3600]")

    @property
    def fanout(self) -> int:
        return len(self.divergent_token_ids)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: tuple[object, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for row in rows:
            payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            handle.write(_canonical_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _deadline(timeout_s: float) -> int:
    return time.monotonic_ns() + int(timeout_s * 1_000_000_000)


def _remaining(deadline_ns: int) -> float:
    remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
    if remaining <= 0:
        raise TimeoutError("Experiment 003 trial exceeded its wall-time bound")
    return remaining


def _block_rows(snapshot: PhysicalKvLayoutSnapshot) -> list[dict[str, object]]:
    return [block.model_dump(mode="json") for block in snapshot.blocks]


def _snapshot_payload(snapshot: PhysicalKvLayoutSnapshot) -> dict[str, object]:
    return {
        **snapshot.model_dump(mode="json"),
        "blocks": _block_rows(snapshot),
    }


def _profile_payload(profile: cProfile.Profile, *, limit: int = 120) -> dict[str, object]:
    stats = cast(Any, pstats.Stats(profile))
    rows: list[dict[str, object]] = []
    for (filename, line, function), values in stats.stats.items():
        primitive_calls, total_calls, self_s, cumulative_s, _ = values
        rows.append(
            {
                "filename": filename,
                "line": line,
                "function": function,
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_seconds": self_s,
                "cumulative_seconds": cumulative_s,
            }
        )
    rows.sort(key=lambda row: (-cast(float, row["cumulative_seconds"]), str(row["function"])))
    return {
        "schema_version": "sloforge.branchfabric.cprofile-summary/v1",
        "sort": "cumulative_seconds_descending",
        "rows": rows[:limit],
    }


def _assert_shared_semantics(
    root: SharedRootReference,
    fanout: int,
    snapshots: tuple[PhysicalKvLayoutSnapshot, ...],
) -> dict[str, object]:
    if not snapshots:
        raise RuntimeError("shared trial emitted no physical allocation snapshots")
    divergence = snapshots[0]
    root_ids = set(root.block_ids)
    if set(divergence.shared_prefix_block_ids) != root_ids:
        raise RuntimeError("shared trial did not retain the exact published root IDs")
    by_id = {block.runtime_block_id: block for block in divergence.blocks}
    if any(
        len(by_id[block_id].branch_ids) != fanout or by_id[block_id].refcount < fanout
        for block_id in root_ids
    ):
        raise RuntimeError("shared root ownership/refcounts do not cover every branch")
    private = [
        block for block in divergence.blocks if block.state_class is KvStateClass.PRIVATE_SUFFIX
    ]
    if not private:
        raise RuntimeError("shared trial allocated no private divergent suffix state")
    private_owner_sets = [set(block.branch_ids) for block in private]
    if any(len(owners) != 1 for owners in private_owner_sets):
        raise RuntimeError("a mutable suffix block is shared by multiple branches")
    return {
        "exact_shared_root_block_ids": True,
        "runtime_native_refcounts_cover_fanout": True,
        "private_suffix_blocks_are_single_owner": True,
        "root_block_count": len(root.block_ids),
        "root_physical_bytes": root.physical_bytes,
        "private_block_count_at_divergence": len(private),
    }


def _assert_independent_semantics(
    prefix_tokens: int,
    snapshots: tuple[PhysicalKvLayoutSnapshot, ...],
) -> dict[str, object]:
    if not snapshots:
        raise RuntimeError("independent trial emitted no physical allocation snapshots")
    divergence = snapshots[0]
    if divergence.shared_prefix_block_ids:
        raise RuntimeError("independent trial exposed shared prefix blocks")
    prefix_blocks = (prefix_tokens + 15) // 16
    owners: dict[str, set[str]] = {}
    for block in divergence.blocks:
        if block.logical_token_start < prefix_tokens:
            owners.setdefault(block.runtime_block_id, set()).update(block.branch_ids)
    if any(len(branches) != 1 for branches in owners.values()):
        raise RuntimeError("independent prefix state aliases across branches")
    return {
        "prefix_cache_reuse_absent": True,
        "independent_prefix_blocks_single_owner": True,
        "prefix_blocks_per_branch": prefix_blocks,
    }


def _readiness_span(observation: dict[str, object]) -> dict[str, object] | None:
    spans = cast(list[dict[str, object]], observation.get("spans", []))
    matches = [row for row in spans if row.get("category") == "POST_ROOT_READY"]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError("metadata observation contains multiple POST_ROOT_READY parents")
    return matches[0]


def run_metadata_trial(
    adapter: MetadataAdapter0230,
    config: MetadataTrialConfiguration,
    output: Path,
) -> dict[str, object]:
    """Run one fresh-engine trial and persist raw, metrics, profile, and native data."""

    output.mkdir(parents=True, exist_ok=False)
    deadline_ns = _deadline(config.timeout_s)
    branch_ids = tuple(f"{config.attempt_id}-branch-{index:02d}" for index in range(config.fanout))
    root_id = f"{config.attempt_id}-root"
    root: SharedRootReference | None = None
    root_snapshot: PhysicalKvLayoutSnapshot | None = None
    root_inclusive_start_ns = time.monotonic_ns()
    profile = cProfile.Profile() if config.trace_level == "full" else None
    readiness_ended_ns: int | None = None
    readiness_process_end_ns: int | None = None
    readiness_thread_end_ns: int | None = None
    post_root_start_ns: int
    decode_call_start_ns: int
    result: ConcurrentDecodeResult
    try:
        if config.mode == "shared_root":
            adapter.start_session(
                root_id,
                token_ids=config.prefix_token_ids,
                seed=config.seed,
                timeout_s=_remaining(deadline_ns),
            )
            adapter.prefill_session(root_id, timeout_s=_remaining(deadline_ns))
            adapter.pause_at_safe_decode_boundary(root_id, timeout_s=_remaining(deadline_ns))
            root = adapter.create_shared_root_reference(
                root_id,
                prefix_token_count=len(config.prefix_token_ids),
                timeout_s=_remaining(deadline_ns),
            )
            root_snapshot = adapter.inspect_physical_kv_layout(
                (root_id,), timeout_s=_remaining(deadline_ns)
            )
            if set(root_snapshot.shared_prefix_block_ids) != set(root.block_ids):
                raise RuntimeError("published root physical IDs changed before branch creation")
            prefix_block_count = len(root.block_ids)
        else:
            prefix_block_count = (len(config.prefix_token_ids) + 15) // 16

        post_root_start_ns = time.monotonic_ns()
        post_root_process_start_ns = time.process_time_ns()
        post_root_thread_start_ns = time.thread_time_ns()
        adapter.begin_post_root_metadata_observation(
            branch_count=config.fanout,
            prefix_block_count=prefix_block_count,
        )
        if profile is not None:
            profile.enable()

        if config.mode == "shared_root":
            assert root is not None
            branch_specs = tuple(
                (branch_id, config.divergent_token_ids[index], config.seed + index + 1)
                for index, branch_id in enumerate(branch_ids)
            )
            if config.implementation == "optimized":
                adapter.fork_same_policy_sessions_0230(
                    root.root_reference_id,
                    branch_specs,
                    timeout_s=_remaining(deadline_ns),
                )
            else:
                for branch_id, divergent_token, seed in branch_specs:
                    with adapter.metadata_span(
                        "helix_branch_fork_and_validation",
                        "HELIX_ORCHESTRATION",
                        {"branch_id": branch_id, "prefix_tokens": len(config.prefix_token_ids)},
                    ):
                        adapter.fork_same_policy_session(
                            root.root_reference_id,
                            branch_id,
                            divergent_token_id=divergent_token,
                            seed=seed,
                            timeout_s=_remaining(deadline_ns),
                        )
                        logical = adapter.inspect_logical_state(
                            branch_id, timeout_s=_remaining(deadline_ns)
                        )
                        token_ids = tuple(getattr(logical, "token_ids", ()))
                        if token_ids != (*config.prefix_token_ids, divergent_token):
                            raise RuntimeError("baseline branch logical history changed at fork")
            with adapter.metadata_span(
                "helix_branchpoint_capture",
                "HELIX_ORCHESTRATION",
                {"branch_count": config.fanout},
            ):
                adapter.capture_branchpoint(
                    root.root_reference_id,
                    branch_ids,
                    timeout_s=_remaining(deadline_ns),
                )
        else:
            for index, branch_id in enumerate(branch_ids):
                with adapter.metadata_span(
                    "helix_independent_session_construction",
                    "HELIX_ORCHESTRATION",
                    {"branch_id": branch_id},
                ):
                    adapter.start_session(
                        branch_id,
                        token_ids=(
                            *config.prefix_token_ids,
                            config.divergent_token_ids[index],
                        ),
                        seed=config.seed + index + 1,
                        timeout_s=_remaining(deadline_ns),
                    )

        def all_ready() -> None:
            nonlocal readiness_ended_ns, readiness_process_end_ns, readiness_thread_end_ns
            readiness_ended_ns = time.monotonic_ns()
            readiness_process_end_ns = time.process_time_ns()
            readiness_thread_end_ns = time.thread_time_ns()
            adapter.end_post_root_metadata_observation()
            if profile is not None:
                profile.disable()

        decode_call_start_ns = time.monotonic_ns()
        if config.mode == "shared_root":
            result = adapter.run_concurrent_branches(
                branch_ids,
                maximum_new_tokens=config.suffix_tokens,
                seed=config.seed,
                timeout_s=_remaining(deadline_ns),
                on_all_branches_ready=all_ready,
            )
        else:
            result = adapter.run_concurrent_independent_sessions(
                branch_ids,
                maximum_new_tokens=config.suffix_tokens,
                seed=config.seed,
                timeout_s=_remaining(deadline_ns),
                on_all_branches_ready=all_ready,
            )
        if (
            readiness_ended_ns is None
            or readiness_process_end_ns is None
            or readiness_thread_end_ns is None
        ):
            raise RuntimeError("runtime never called the all-branches-ready boundary")

        # This correctness audit is deliberately outside POST_ROOT_READY.
        for index, branch_id in enumerate(branch_ids):
            logical = adapter.inspect_logical_state(branch_id, timeout_s=_remaining(deadline_ns))
            token_ids = tuple(getattr(logical, "token_ids", ()))
            expected = (*config.prefix_token_ids, config.divergent_token_ids[index])
            if token_ids[: len(expected)] != expected:
                raise RuntimeError("branch logical history changed after readiness")

        observation = adapter.inspect_metadata_observation()
        readiness_span = _readiness_span(observation)
        measured_post_root_ns = readiness_ended_ns - post_root_start_ns
        if readiness_span is not None:
            traced_value = readiness_span["wall_time_ns"]
            if not isinstance(traced_value, int):
                raise RuntimeError("POST_ROOT_READY trace duration is not an integer")
            traced_ns = traced_value
            if abs(traced_ns - measured_post_root_ns) > max(
                100_000, measured_post_root_ns // 1_000
            ):
                raise RuntimeError(
                    "POST_ROOT_READY trace disagrees with external continuous timing"
                )

        timing_payload: dict[str, object] | None = None
        normalized_operations: dict[str, object] | None = None
        if readiness_span is not None:
            hierarchy = hierarchy_from_vllm_observation(observation)
            decomposition = hierarchy.decomposition()
            decomposition.assert_parent_cross_check(
                TimingDuration(
                    wall_ns=measured_post_root_ns,
                    process_cpu_ns=readiness_process_end_ns - post_root_process_start_ns,
                    thread_cpu_ns=readiness_thread_end_ns - post_root_thread_start_ns,
                )
            )
            timing_payload = hierarchy.as_dict()
            normalized_operations = hierarchy.root.operations.normalized(
                branch_count=config.fanout,
                prefix_block_count=prefix_block_count,
                token_count=len(config.prefix_token_ids) * config.fanout,
                request_count=config.fanout,
                fanout=config.fanout,
            )

        if config.mode == "shared_root":
            assert root is not None
            semantics = _assert_shared_semantics(root, config.fanout, result.allocation_snapshots)
        else:
            semantics = _assert_independent_semantics(
                len(config.prefix_token_ids), result.allocation_snapshots
            )
        final_snapshot = adapter.inspect_physical_kv_layout(
            branch_ids, timeout_s=_remaining(deadline_ns)
        )
        hbm = adapter.inspect_gpu_memory_state(timeout_s=_remaining(deadline_ns))
        first_branch_ready_ns = (
            decode_call_start_ns - post_root_start_ns + min(result.branch_ready_latency_ns.values())
        )
        all_branches_ready_ns = measured_post_root_ns
        first_token_latency_ns = (
            decode_call_start_ns - post_root_start_ns + min(result.first_token_latency_ns.values())
        )
        root_inclusive_ready_ns = readiness_ended_ns - root_inclusive_start_ns

        for branch_id in branch_ids:
            if config.mode == "shared_root":
                adapter.destroy_branch(branch_id, timeout_s=_remaining(deadline_ns))
            else:
                adapter.destroy_session(branch_id, timeout_s=_remaining(deadline_ns))
        if config.mode == "shared_root":
            adapter.destroy_session(root_id, timeout_s=_remaining(deadline_ns))
        released = adapter.inspect_gpu_memory_state(timeout_s=_remaining(deadline_ns))
        if released.kv_assigned_bytes != 0:
            raise RuntimeError("runtime retained assigned KV bytes after trial teardown")
        lifecycle_observation = adapter.inspect_metadata_observation()

        summary: dict[str, object] = {
            "schema_version": "sloforge.branchfabric.metadata-trial-summary/v1",
            "configuration": asdict(config),
            "fanout": config.fanout,
            "root_inclusive_ready_ns": root_inclusive_ready_ns,
            "post_root_ready_ns": measured_post_root_ns,
            "first_branch_ready_ns": first_branch_ready_ns,
            "all_branches_ready_ns": all_branches_ready_ns,
            "first_token_latency_ns": first_token_latency_ns,
            "request_admitted_latency_ns": result.request_admitted_latency_ns,
            "first_decode_token_started_latency_ns": result.first_decode_token_started_latency_ns,
            "branch_ready_latency_ns": result.branch_ready_latency_ns,
            "first_token_latency_by_branch_ns": result.first_token_latency_ns,
            "decode_throughput_tokens_per_second": result.throughput_tokens_per_second,
            "steady_decode_throughput_tokens_per_second": (
                result.decode_active_tokens / (result.decode_active_elapsed_ns / 1_000_000_000)
            ),
            "decode_active_tokens": result.decode_active_tokens,
            "decode_active_elapsed_ns": result.decode_active_elapsed_ns,
            "physical_block_count": len(final_snapshot.blocks),
            "shared_root_block_count": len(final_snapshot.shared_prefix_block_ids),
            "private_block_count": len(final_snapshot.private_suffix_block_ids),
            "physical_assigned_bytes": final_snapshot.physical_assigned_bytes,
            "shared_prefix_bytes": final_snapshot.shared_prefix_bytes,
            "private_suffix_bytes": final_snapshot.private_suffix_bytes,
            "runtime_hbm": hbm.model_dump(mode="json"),
            "final_runtime_assigned_kv_bytes": released.kv_assigned_bytes,
            "semantics": semantics,
            "metadata_operation_counters": observation.get("post_root_ready_operation_counters"),
            "metadata_operation_normalization": normalized_operations,
            "lifecycle_metadata_operation_counters": lifecycle_observation.get(
                "operation_counters"
            ),
            "post_root_process_cpu_ns": (readiness_process_end_ns - post_root_process_start_ns),
            "post_root_thread_cpu_ns": (readiness_thread_end_ns - post_root_thread_start_ns),
            "metadata_observation_sha256": hashlib.sha256(
                _canonical_bytes(observation)
            ).hexdigest(),
            "lifecycle_metadata_observation_sha256": hashlib.sha256(
                _canonical_bytes(lifecycle_observation)
            ).hexdigest(),
        }
        _write_json(output / "metrics/trial-summary.json", summary)
        _write_json(output / "metadata/runtime-metadata-observation.json", observation)
        _write_json(
            output / "metadata/runtime-metadata-lifecycle-observation.json",
            lifecycle_observation,
        )
        if timing_payload is not None:
            _write_json(output / "metrics/readiness-decomposition.json", timing_payload)
        _write_json(
            output / "native-metrics/runtime-native-metrics.json",
            {
                "schema_version": "sloforge.branchfabric.runtime-native-metrics/v1",
                "runtime": "vllm",
                "runtime_version": "0.23.0",
                "observations": observation.get("runtime_native_metrics", []),
                "post_root_ready_observation_count": observation.get(
                    "post_root_ready_native_metric_count"
                ),
            },
        )
        if config.trace_level != "disabled":
            _write_jsonl(
                output / "traces/branch-workload-trace-v1.jsonl",
                adapter.emit_branch_workload_trace(),
            )
            _write_jsonl(
                output / "traces/state-operation-trace-v1.jsonl",
                adapter.emit_state_operation_trace(),
            )
            _write_json(
                output / "traces/trace-self-overhead.json",
                adapter.inspect_trace_self_overhead(),
            )
        _write_json(
            output / "raw/physical-layouts.json",
            {
                "root": None if root_snapshot is None else _snapshot_payload(root_snapshot),
                "decode": [_snapshot_payload(item) for item in result.allocation_snapshots],
                "final": _snapshot_payload(final_snapshot),
            },
        )
        if profile is not None:
            _write_json(output / "profiles/cprofile-summary.json", _profile_payload(profile))
        return summary
    finally:
        adapter.cleanup_runtime(timeout_s=60.0)


__all__ = [
    "MetadataAdapter0230",
    "MetadataTrialConfiguration",
    "run_metadata_trial",
]
