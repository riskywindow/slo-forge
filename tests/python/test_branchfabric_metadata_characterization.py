from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from sloforge.continuum.adapters.real_runtime import (
    ConcurrentDecodeResult,
    GpuMemoryState,
    KvStateClass,
    PhysicalKvBlock,
    PhysicalKvLayoutSnapshot,
    RefcountEvidence,
    SharedRootReference,
)

_RUNNER_PATH = (
    Path(__file__).parents[2] / "benchmarks" / "branchfabric" / "metadata_characterization.py"
)
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "sloforge_branchfabric_metadata_characterization", _RUNNER_PATH
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = _RUNNER
_RUNNER_SPEC.loader.exec_module(_RUNNER)
MetadataTrialConfiguration = _RUNNER.MetadataTrialConfiguration
run_metadata_trial = _RUNNER.run_metadata_trial


_PREFIX = tuple(index % 32_000 for index in range(16_384))
_ROOT_IDS = tuple(f"root-{index:04d}" for index in range(1024))


def _block(
    block_id: str,
    index: int,
    branch_ids: tuple[str, ...],
    *,
    private: bool,
) -> PhysicalKvBlock:
    logical_start = 16_384 if private else index * 16
    return PhysicalKvBlock(
        runtime_block_id=block_id,
        block_index=index,
        kv_cache_group=0,
        layer_ids=("model.layers.0.self_attn",),
        device="cuda:0",
        dtype="bfloat16",
        bytes=100,
        block_size_tokens=16,
        logical_token_start=logical_start,
        logical_token_end_exclusive=logical_start + (1 if private else 16),
        branch_ids=branch_ids,
        refcount=len(branch_ids),
        refcount_evidence=RefcountEvidence.RUNTIME_NATIVE,
        state_class=(KvStateClass.PRIVATE_SUFFIX if private else KvStateClass.SHARED_PREFIX),
        allocation_epoch=index + 1,
        cache_resident=not private,
    )


def _snapshot(session_ids: tuple[str, ...], *, private: bool) -> PhysicalKvLayoutSnapshot:
    roots = tuple(
        _block(block_id, index, session_ids, private=False)
        for index, block_id in enumerate(_ROOT_IDS)
    )
    suffix = (_block("private-0", 1024, (session_ids[0],), private=True),) if private else ()
    blocks = (*roots, *suffix)
    return PhysicalKvLayoutSnapshot(
        runtime="vllm",
        runtime_version="0.23.0",
        snapshot_epoch=1,
        observed_at_monotonic_ns=time.monotonic_ns(),
        session_ids=session_ids,
        blocks=blocks,
        shared_prefix_block_ids=_ROOT_IDS,
        private_suffix_block_ids=("private-0",) if private else (),
        root_reference_count=len(session_ids),
        physical_assigned_bytes=len(blocks) * 100,
        shared_prefix_bytes=1024 * 100,
        private_suffix_bytes=100 if private else 0,
        kv_pool_reserved_bytes=1_000_000,
    )


class _RecordedSpan(AbstractContextManager[None]):
    def __init__(self, adapter: _MetadataFixtureAdapter, name: str, category: str) -> None:
        self.adapter = adapter
        self.name = name
        self.category = category
        self.wall = 0
        self.process = 0
        self.thread = 0

    def __enter__(self) -> None:
        self.wall = time.monotonic_ns()
        self.process = time.process_time_ns()
        self.thread = time.thread_time_ns()

    def __exit__(self, *args: object) -> None:
        del args
        wall_end = time.monotonic_ns()
        process_end = time.process_time_ns()
        thread_end = time.thread_time_ns()
        self.adapter.sequence += 1
        self.adapter.spans.append(
            {
                "span_id": f"child-{self.adapter.sequence}",
                "parent_span_id": "root",
                "name": self.name,
                "category": self.category,
                "wall_start_ns": self.wall,
                "wall_end_ns": wall_end,
                "process_cpu_start_ns": self.process,
                "process_cpu_end_ns": process_end,
                "thread_cpu_start_ns": self.thread,
                "thread_cpu_end_ns": thread_end,
                "wall_time_ns": wall_end - self.wall,
                "process_cpu_time_ns": process_end - self.process,
                "thread_cpu_time_ns": thread_end - self.thread,
                "thread_id": 1,
                "attributes": {},
            }
        )


class _MetadataFixtureAdapter:
    def __init__(self) -> None:
        self.sessions: dict[str, tuple[int, ...]] = {}
        self.destroyed: set[str] = set()
        self.spans: list[dict[str, object]] = []
        self.sequence = 0
        self.root_start: tuple[int, int, int] | None = None
        self.cleaned = False

    def start_session(
        self, session_id: str, *, token_ids: tuple[int, ...], seed: int, timeout_s: float
    ) -> object:
        del seed, timeout_s
        self.sessions[session_id] = token_ids
        return object()

    def prefill_session(self, session_id: str, *, timeout_s: float) -> object:
        del session_id, timeout_s
        return object()

    def pause_at_safe_decode_boundary(self, session_id: str, *, timeout_s: float) -> object:
        del session_id, timeout_s
        return object()

    def create_shared_root_reference(
        self, session_id: str, *, prefix_token_count: int, timeout_s: float
    ) -> SharedRootReference:
        del timeout_s
        return SharedRootReference(
            root_reference_id="root-reference",
            source_session_id=session_id,
            prefix_token_count=prefix_token_count,
            block_ids=_ROOT_IDS,
            physical_bytes=1024 * 100,
            adapter_owned_reference_count=1,
            created_at_monotonic_ns=time.monotonic_ns(),
        )

    def fork_same_policy_session(
        self,
        root_reference_id: str,
        branch_session_id: str,
        *,
        divergent_token_id: int,
        seed: int,
        timeout_s: float,
    ) -> object:
        del root_reference_id, seed, timeout_s
        self.sessions[branch_session_id] = (*_PREFIX, divergent_token_id)
        return object()

    def fork_same_policy_sessions_0230(self, *args: object, **kwargs: object) -> tuple[()]:
        del args, kwargs
        raise AssertionError("baseline fixture must not use optimized bulk fork")

    def capture_branchpoint(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        return object()

    def inspect_logical_state(self, session_id: str, *, timeout_s: float) -> object:
        del timeout_s
        return SimpleNamespace(token_ids=self.sessions[session_id])

    def inspect_physical_kv_layout(
        self, session_ids: tuple[str, ...], *, timeout_s: float
    ) -> PhysicalKvLayoutSnapshot:
        del timeout_s
        return _snapshot(session_ids, private=session_ids[0] != "trial-root")

    def inspect_gpu_memory_state(self, *, timeout_s: float) -> GpuMemoryState:
        del timeout_s
        assigned = 0 if self.destroyed == set(self.sessions) else 102_500
        return GpuMemoryState(
            device="cuda:0",
            observed_at_monotonic_ns=time.monotonic_ns(),
            kv_pool_reserved_bytes=1_000_000,
            kv_assigned_bytes=assigned,
            kv_unassigned_bytes=1_000_000 - assigned,
            sample_source="fixture",
        )

    def begin_post_root_metadata_observation(
        self, *, branch_count: int, prefix_block_count: int
    ) -> None:
        assert branch_count == 1
        assert prefix_block_count == 1024
        self.root_start = (
            time.monotonic_ns(),
            time.process_time_ns(),
            time.thread_time_ns(),
        )

    def end_post_root_metadata_observation(self) -> None:
        assert self.root_start is not None
        wall_end = time.monotonic_ns()
        process_end = time.process_time_ns()
        thread_end = time.thread_time_ns()
        wall, process, thread = self.root_start
        self.spans.append(
            {
                "span_id": "root",
                "parent_span_id": None,
                "name": "POST_ROOT_READY",
                "category": "POST_ROOT_READY",
                "wall_start_ns": wall,
                "wall_end_ns": wall_end,
                "process_cpu_start_ns": process,
                "process_cpu_end_ns": process_end,
                "thread_cpu_start_ns": thread,
                "thread_cpu_end_ns": thread_end,
                "wall_time_ns": wall_end - wall,
                "process_cpu_time_ns": process_end - process,
                "thread_cpu_time_ns": thread_end - thread,
                "thread_id": 1,
                "attributes": {"branch_count": 1, "prefix_block_count": 1024},
            }
        )

    def metadata_span(
        self, name: str, category: str, attributes: dict[str, object] | None = None
    ) -> AbstractContextManager[None]:
        del attributes
        return _RecordedSpan(self, name, category)

    def inspect_metadata_observation(self) -> dict[str, object]:
        counters = {
            "block_table_writes": 1025,
            "prefix_blocks_bound": 1024,
            "refcount_increments": 1025,
            "private_suffix_allocations": 1,
            "scheduler_queue_inserts": 1,
            "scheduler_queue_removals": 1,
        }
        lifecycle_counters = {
            **counters,
            "refcount_decrements": 1025 * len(self.destroyed),
            "block_frees": 1 if self.destroyed else 0,
        }
        return {
            "spans": sorted(self.spans, key=lambda row: cast(int, row["wall_start_ns"])),
            "operation_counters": lifecycle_counters,
            "post_root_ready_operation_counters": counters,
            "runtime_native_metrics": [{"num_running_reqs": 1, "num_waiting_reqs": 0}],
            "post_root_ready_native_metric_count": 1,
        }

    def emit_branch_workload_trace(self) -> tuple[object, ...]:
        return ({"schema_version": "fixture-branch-trace/v1"},)

    def emit_state_operation_trace(self) -> tuple[object, ...]:
        return ({"schema_version": "fixture-state-trace/v1"},)

    def inspect_trace_self_overhead(self) -> dict[str, int | str]:
        return {"schema_version": "fixture-overhead/v1", "wall_time_ns": 10}

    def run_concurrent_branches(
        self,
        branch_session_ids: tuple[str, ...],
        *,
        maximum_new_tokens: int,
        seed: int,
        timeout_s: float,
        on_all_branches_ready: Any = None,
    ) -> ConcurrentDecodeResult:
        del seed, timeout_s
        with self.metadata_span("request", "REQUEST_BUILD"):
            pass
        with self.metadata_span("prefix", "PREFIX_LOOKUP"):
            pass
        with self.metadata_span("scheduler", "SCHEDULER_SELECT"):
            pass
        with self.metadata_span("gpu", "GPU_EXECUTION"):
            pass
        on_all_branches_ready()
        branch_id = branch_session_ids[0]
        return ConcurrentDecodeResult(
            branchpoint_id="branchpoint",
            branch_output_token_ids={branch_id: tuple(range(maximum_new_tokens))},
            request_admitted_latency_ns={branch_id: 100},
            first_decode_token_started_latency_ns={branch_id: 200},
            branch_ready_latency_ns={branch_id: 300},
            first_token_latency_ns={branch_id: 400},
            elapsed_ns=1_000_000,
            total_output_tokens=maximum_new_tokens,
            throughput_tokens_per_second=256_000.0,
            decode_active_elapsed_ns=900_000,
            decode_active_boundary_output_counts={branch_id: 1},
            decode_active_tokens=maximum_new_tokens - 1,
            completed_branch_ids=branch_session_ids,
            allocation_snapshots=(_snapshot(branch_session_ids, private=True),),
        )

    def run_concurrent_independent_sessions(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("shared-root fixture must not run independent sessions")

    def destroy_branch(self, branch_session_id: str, *, timeout_s: float) -> None:
        del timeout_s
        self.destroyed.add(branch_session_id)

    def destroy_session(self, session_id: str, *, timeout_s: float) -> None:
        del timeout_s
        self.destroyed.add(session_id)

    def cleanup_runtime(self, *, timeout_s: float) -> None:
        assert timeout_s == 60.0
        self.cleaned = True


def test_metadata_trial_runner_serializes_provenance_and_exact_decomposition(
    tmp_path: Path,
) -> None:
    adapter = _MetadataFixtureAdapter()
    config = MetadataTrialConfiguration(
        attempt_id="trial",
        mode="shared_root",
        implementation="baseline",
        seed=41,
        prefix_token_ids=_PREFIX,
        divergent_token_ids=(31_999,),
        suffix_tokens=256,
        timeout_s=30.0,
        trace_level="minimal",
    )
    output = tmp_path / "attempt"

    summary = run_metadata_trial(adapter, config, output)

    assert adapter.cleaned
    assert summary["fanout"] == 1
    assert summary["final_runtime_assigned_kv_bytes"] == 0
    assert summary["semantics"] == {
        "exact_shared_root_block_ids": True,
        "runtime_native_refcounts_cover_fanout": True,
        "private_suffix_blocks_are_single_owner": True,
        "root_block_count": 1024,
        "root_physical_bytes": 102_400,
        "private_block_count_at_divergence": 1,
    }
    decomposition = json.loads(
        (output / "metrics/readiness-decomposition.json").read_text(encoding="utf-8")
    )["decomposition"]
    assert decomposition["invariant"]["sum_equals_parent"] is True
    assert (
        sum(stage["wall_ns"] for stage in decomposition["stages"].values())
        == (decomposition["parent"]["wall_ns"])
    )
    assert (output / "metadata/runtime-metadata-observation.json").is_file()
    lifecycle_path = output / "metadata/runtime-metadata-lifecycle-observation.json"
    assert lifecycle_path.is_file()
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    assert lifecycle["operation_counters"]["refcount_decrements"] == 2050
    assert summary["lifecycle_metadata_operation_counters"] == lifecycle["operation_counters"]
    assert (
        summary["lifecycle_metadata_observation_sha256"]
        == hashlib.sha256(lifecycle_path.read_bytes()).hexdigest()
    )
    assert (output / "native-metrics/runtime-native-metrics.json").is_file()
    assert (output / "traces/branch-workload-trace-v1.jsonl").is_file()
    assert (output / "raw/physical-layouts.json").is_file()
