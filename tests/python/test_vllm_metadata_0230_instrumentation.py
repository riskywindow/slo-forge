from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from sloforge.continuum.adapters.vllm_0230_metadata import (
    ReadinessStage,
    hierarchy_from_vllm_observation,
)
from sloforge.continuum.adapters.vllm_metadata_0230 import (
    MetadataInstrumentationLevel,
    MetadataOptimization,
    RequestScope0230,
    VllmMetadataRecorder0230,
)


class _StepClock:
    """Independent deterministic monotonic clock for one timing dimension."""

    def __init__(self, *, start: int, step: int) -> None:
        self.value = start - step
        self.step = step

    def __call__(self) -> int:
        self.value += self.step
        return self.value


@dataclass
class _Block:
    block_id: int
    ref_cnt: int


@dataclass
class _BlockGroups:
    blocks: tuple[tuple[_Block, ...], ...]


@dataclass
class _Request:
    request_id: str
    block_hashes: tuple[str, ...] = ()
    num_tokens: int = 33


class _BlockPool:
    def __init__(self) -> None:
        self.freed: tuple[_Block, ...] = ()

    def get_cached_block(
        self, block_hash: str, kv_cache_group_ids: list[int] | None = None
    ) -> _Block | None:
        del kv_cache_group_ids
        return _Block(10, 8) if block_hash == "hit" else None

    def touch(self, blocks: tuple[_Block, ...]) -> None:
        for block in blocks:
            block.ref_cnt += 1

    def get_new_blocks(self, count: int) -> list[_Block]:
        return [_Block(100 + index, 1) for index in range(count)]

    def free_blocks(self, blocks: Any, prepend: bool = False) -> None:
        # The real method accepts an iterable. The probe must not exhaust a
        # one-shot generator while counting it.
        del prepend
        self.freed = tuple(blocks)
        for block in self.freed:
            block.ref_cnt -= 1


class _Coordinator:
    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[tuple[_Block, ...], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        del (
            request_id,
            new_computed_blocks,
            num_local_computed_tokens,
            num_external_computed_tokens,
        )

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[_Block], ...]:
        del request_id, num_tokens_main_model, num_encoder_tokens
        return ([_Block(200 + index, 1) for index in range(num_tokens)],)


class _Manager:
    def __init__(self) -> None:
        self.coordinator = _Coordinator()
        self.block_pool = _BlockPool()
        self.enable_caching = True

    def get_computed_blocks(self, request: _Request) -> tuple[tuple[object, ...], int]:
        hits = 0
        for block_hash in request.block_hashes:
            if self.block_pool.get_cached_block(block_hash, [0]) is None:
                break
            hits += 1
        return (), hits * 16

    def allocate_slots(self, request: _Request) -> object:
        del request
        return object()

    def free(self, request: _Request) -> None:
        del request


class _Runner:
    class _WorkerTableGroup:
        blocks_per_kv_block = 1

    class _WorkerBlockTable:
        def __init__(self) -> None:
            self.block_tables = [_Runner._WorkerTableGroup()]
            self.rows: list[tuple[tuple[list[int], ...], int, str]] = []

        def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
            self.rows.append((block_ids, row_idx, "add"))

        def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
            self.rows.append((block_ids, row_idx, "append"))

    def __init__(self) -> None:
        self.input_batch = SimpleNamespace(block_table=self._WorkerBlockTable())

    def _model_forward(self, scheduler_output: object) -> str:
        del scheduler_output
        return "gpu-output"

    def execute_model(self, scheduler_output: object) -> str:
        return self._model_forward(scheduler_output)

    def sample_tokens(self, grammar_output: object) -> str:
        del grammar_output
        return "sampled-output"


class _Executor:
    def __init__(self, runner: _Runner) -> None:
        self.runner = runner

    def execute_model(self, scheduler_output: object) -> str:
        return self.runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output: object) -> str:
        return self.runner.sample_tokens(grammar_output)


class _Core:
    def __init__(self, runner: _Runner) -> None:
        self.model_executor = _Executor(runner)

    def request_block_hasher(self, request: _Request) -> list[str]:
        del request
        return ["root-hash-0", "root-hash-1"]

    def preprocess_add_request(self, request: _Request) -> _Request:
        request.block_hashes = tuple(self.request_block_hasher(request))
        return request


@dataclass
class _Output:
    new_token_ids: tuple[int, ...]


@dataclass
class _OutputBatch:
    outputs: tuple[_Output, ...]


class _Scheduler:
    def add_request(self, request: _Request) -> None:
        del request

    def _enqueue_waiting_request(self, request: _Request) -> None:
        del request

    def schedule(self) -> object:
        return SimpleNamespace(scheduled_new_reqs=("request-0",))

    def update_from_output(self) -> dict[str, _OutputBatch]:
        return {"request-0": _OutputBatch((_Output((1, 2)),))}

    def make_stats(self) -> object:
        return SimpleNamespace(num_running_reqs=1, num_waiting_reqs=0)


class _View:
    runtime_version = "0.23.0"
    block_size_tokens = 16
    device = ""

    def __init__(self) -> None:
        self.runner = _Runner()
        self.engine_core = _Core(self.runner)
        self.scheduler = _Scheduler()
        self.manager = _Manager()


def _recorder(
    view: _View,
    *,
    level: MetadataInstrumentationLevel = MetadataInstrumentationLevel.MINIMAL,
    optimization: MetadataOptimization = MetadataOptimization.BASELINE,
    request_scope: Any = None,
) -> VllmMetadataRecorder0230:
    return VllmMetadataRecorder0230(
        view,
        level=level,
        optimization=optimization,
        request_scope=request_scope,
        clock_ns=_StepClock(start=1_000, step=10),
        process_cpu_ns=_StepClock(start=2_000, step=3),
        thread_cpu_ns=_StepClock(start=3_000, step=2),
    )


def test_exact_hook_counters_generator_safety_and_native_metric_capture() -> None:
    view = _View()
    recorder = _recorder(view)
    shared = (_Block(1, 8), _Block(2, 8))
    computed = _BlockGroups((shared,))
    releasable = (_Block(3, 1), _Block(4, 2))

    recorder.begin_post_root_ready(branch_count=1, prefix_block_count=2)
    recorder.record_branch_session_allocation()
    recorder.record_prompt_token_writes(33)
    request = view.engine_core.preprocess_add_request(_Request("request-0"))
    view.scheduler.add_request(request)
    view.scheduler._enqueue_waiting_request(request)
    view.manager.get_computed_blocks(_Request("lookup-request", ("hit", "miss")))
    assert view.manager.block_pool.get_cached_block("hit") is not None
    assert view.manager.block_pool.get_cached_block("miss") is None
    view.manager.coordinator.allocate_new_computed_blocks(
        request_id="request-0",
        new_computed_blocks=computed.blocks,
        num_local_computed_tokens=32,
        num_external_computed_tokens=0,
    )
    view.manager.block_pool.touch(shared)
    view.manager.coordinator.allocate_new_blocks("request-0", 1, 1)
    view.manager.block_pool.get_new_blocks(1)
    view.manager.block_pool.free_blocks((block for block in releasable), prepend=True)
    assert view.manager.block_pool.freed == releasable
    view.runner.input_batch.block_table.add_row(([1, 2],), 0)
    view.runner.input_batch.block_table.append_row(([3],), 0)
    recorder.mark_scheduler_wait_start()
    view.scheduler.schedule()
    view.engine_core.model_executor.execute_model(object())
    view.engine_core.model_executor.sample_tokens(object())
    view.scheduler.update_from_output()
    view.scheduler.make_stats()
    recorder.end_post_root_ready()

    observation = recorder.snapshot()
    assert observation["post_root_ready_operation_counters"] == {
        "block_allocations": 1,
        "block_frees": 1,
        "block_table_writes": 3,
        "branch_session_metadata_allocations": 1,
        "output_token_commits": 2,
        "prefix_block_candidates": 2,
        "prefix_blocks_bound": 2,
        "prefix_hash_calls": 1,
        "prefix_hash_hits": 1,
        "prefix_hash_lookups": 2,
        "prefix_hash_misses": 1,
        "prefix_hash_template_hits": 0,
        "prefix_lookup_calls": 1,
        "private_suffix_allocations": 1,
        "prompt_token_writes": 33,
        "refcount_decrements": 2,
        "refcount_increments": 3,
        "request_block_hashes_computed": 2,
        "runtime_request_allocations": 1,
        "scheduler_candidate_evaluations": 1,
        "scheduler_queue_inserts": 1,
        "scheduler_queue_removals": 1,
        "scheduler_scans": 1,
    }
    assert observation["post_root_ready_native_metric_count"] == 1
    assert len(observation["post_root_ready_runtime_native_metrics"]) == 1
    gpu_execution_names = {
        row["name"] for row in observation["spans"] if row["category"] == "GPU_EXECUTION"
    }
    assert gpu_execution_names == {
        "gpu_worker_execution",
        "gpu_sampling_bookkeeping_and_output_sync",
    }
    assert observation["stage_measurement_semantics"]["SCHEDULER_WAIT"]["evidence"] == (
        "direct_boundaries_attribution_inferred"
    )
    assert set(observation["stage_measurement_semantics"]) == {
        stage.name for stage in ReadinessStage
    }
    assert (
        "MINIMAL derived-exact" in observation["operation_counter_semantics"]["prefix_hash_lookups"]
    )


def test_readiness_counters_freeze_while_lifecycle_teardown_counters_persist() -> None:
    view = _View()
    recorder = _recorder(view)
    block = _Block(7, 1)

    recorder.begin_post_root_ready(branch_count=1, prefix_block_count=2)
    view.manager.block_pool.get_new_blocks(1)
    recorder.end_post_root_ready()
    readiness = recorder.snapshot()["post_root_ready_operation_counters"]
    view.manager.block_pool.free_blocks((block,))
    lifecycle = recorder.snapshot()

    assert readiness["block_allocations"] == 1
    assert readiness["refcount_decrements"] == 0
    assert lifecycle["post_root_ready_operation_counters"] == readiness
    assert lifecycle["operation_counters"]["refcount_decrements"] == 1
    assert lifecycle["operation_counters"]["block_frees"] == 1


def test_independent_prefix_allocations_are_not_labeled_private_suffix() -> None:
    view = _View()

    def independent_scope(request_id: str) -> RequestScope0230:
        return RequestScope0230(
            request_id=request_id,
            root_reference_id=None,
            source_session_id=None,
            prefix_token_count=None,
            eligible_for_root_hash_template=False,
        )

    recorder = _recorder(view, request_scope=independent_scope)
    recorder.begin_post_root_ready(branch_count=1, prefix_block_count=2)
    view.manager.coordinator.allocate_new_blocks("independent-request", 2, 2)
    recorder.end_post_root_ready()

    counters = recorder.snapshot()["post_root_ready_operation_counters"]
    assert counters["private_suffix_allocations"] == 0


def test_scheduler_wait_is_a_direct_non_overlapping_span_with_absolute_clocks() -> None:
    view = _View()
    recorder = _recorder(view)
    recorder.begin_post_root_ready(branch_count=8, prefix_block_count=1024)
    with recorder.span("helix", "HELIX_ORCHESTRATION"):
        pass
    recorder.mark_scheduler_wait_start()
    view.scheduler.schedule()
    recorder.end_post_root_ready()

    observation = recorder.snapshot()
    wait = next(row for row in observation["spans"] if row["category"] == "SCHEDULER_WAIT")
    selection = next(row for row in observation["spans"] if row["category"] == "SCHEDULER_SELECT")
    assert wait["wall_end_ns"] <= selection["wall_start_ns"]
    assert wait["process_cpu_end_ns"] <= selection["process_cpu_start_ns"]
    assert wait["thread_cpu_end_ns"] <= selection["thread_cpu_start_ns"]
    for row in observation["spans"]:
        assert row["wall_time_ns"] == row["wall_end_ns"] - row["wall_start_ns"]
        assert row["process_cpu_time_ns"] == (
            row["process_cpu_end_ns"] - row["process_cpu_start_ns"]
        )
        assert row["thread_cpu_time_ns"] == (row["thread_cpu_end_ns"] - row["thread_cpu_start_ns"])

    hierarchy = hierarchy_from_vllm_observation(observation)
    decomposition = hierarchy.decomposition()
    assert (
        dict(decomposition.stages)[ReadinessStage.SCHEDULER_WAIT].wall_ns == (wait["wall_time_ns"])
    )
    assert sum(duration.wall_ns for _, duration in decomposition.stages) == (
        decomposition.parent.wall_ns
    )
    assert sum(duration.process_cpu_ns for _, duration in decomposition.stages) == (
        decomposition.parent.process_cpu_ns
    )
    assert (
        sum(duration.thread_cpu_ns or 0 for _, duration in decomposition.stages)
        == decomposition.parent.thread_cpu_ns
    )


def test_minimal_counts_prefix_probes_without_per_probe_timing_spans() -> None:
    minimal_view = _View()
    minimal = _recorder(minimal_view)
    minimal.begin_post_root_ready(branch_count=1, prefix_block_count=2)
    minimal_view.manager.get_computed_blocks(_Request("lookup", ("hit", "miss")))
    minimal.end_post_root_ready()

    minimal_observation = minimal.snapshot()
    assert minimal_observation["post_root_ready_operation_counters"]["prefix_hash_lookups"] == 2
    assert not any(row["name"] == "prefix_hash_table_probe" for row in minimal_observation["spans"])

    full_view = _View()
    full = _recorder(full_view, level=MetadataInstrumentationLevel.FULL)
    full.begin_post_root_ready(branch_count=1, prefix_block_count=2)
    full_view.manager.block_pool.get_cached_block("hit", [0])
    full.end_post_root_ready()
    assert any(row["name"] == "prefix_hash_table_probe" for row in full.snapshot()["spans"])


def test_first_model_forward_uses_cuda_events_resolved_after_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Event:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True
            self.recorded = False

        def record(self) -> None:
            self.recorded = True

        def elapsed_time(self, end: _Event) -> float:
            assert self.recorded and end.recorded
            return 1.25

    real_import = importlib.import_module

    def fake_import(name: str) -> object:
        if name == "torch":
            return SimpleNamespace(cuda=SimpleNamespace(Event=_Event))
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    view = _View()
    view.device = "cuda:0"
    recorder = _recorder(view)
    recorder.begin_post_root_ready(branch_count=1, prefix_block_count=2)
    assert view.runner._model_forward(object()) == "gpu-output"
    recorder.end_post_root_ready()

    assert recorder.snapshot()["gpu_event_measurements"] == [
        {
            "index": 0,
            "scope": "first_GPUModelRunner._model_forward",
            "elapsed_ns": 1_250_000,
            "synchronization": "resolved_after_CPU-visible_sample_output",
        }
    ]


def test_cached_immutable_root_hashes_bypass_per_branch_hasher_without_aliasing() -> None:
    view = _View()

    def scope(request_id: str) -> RequestScope0230 | None:
        if request_id != "branch-request":
            return None
        return RequestScope0230(
            request_id=request_id,
            root_reference_id="root-reference",
            source_session_id="source-request",
            prefix_token_count=32,
            eligible_for_root_hash_template=True,
        )

    recorder = _recorder(
        view,
        optimization=MetadataOptimization.CACHED_IMMUTABLE_ROOT_HASHES,
        request_scope=scope,
    )
    source_hashes = view.engine_core.request_block_hasher(_Request("source-request", num_tokens=32))
    recorder.publish_root_hash_template(
        root_reference_id="root-reference",
        source_request_id="source-request",
        prefix_block_count=2,
    )

    recorder.begin_post_root_ready(branch_count=1, prefix_block_count=2)
    branch_hashes = view.engine_core.request_block_hasher(_Request("branch-request", num_tokens=33))
    branch_hashes.append("mutation-must-not-touch-template")
    second = view.engine_core.request_block_hasher(_Request("branch-request", num_tokens=33))
    recorder.end_post_root_ready()

    assert source_hashes == ["root-hash-0", "root-hash-1"]
    assert second == source_hashes
    counters = recorder.snapshot()["post_root_ready_operation_counters"]
    assert counters["prefix_hash_calls"] == 2
    assert counters["prefix_hash_template_hits"] == 2
    assert counters["request_block_hashes_computed"] == 0


def test_cached_root_hash_optimization_operates_with_instrumentation_disabled() -> None:
    view = _View()

    def scope(request_id: str) -> RequestScope0230 | None:
        if request_id != "branch-request":
            return None
        return RequestScope0230(
            request_id=request_id,
            root_reference_id="root-reference",
            source_session_id="source-request",
            prefix_token_count=32,
            eligible_for_root_hash_template=True,
        )

    recorder = _recorder(
        view,
        level=MetadataInstrumentationLevel.DISABLED,
        optimization=MetadataOptimization.CACHED_IMMUTABLE_ROOT_HASHES,
        request_scope=scope,
    )
    source_hashes = view.engine_core.request_block_hasher(_Request("source-request", num_tokens=32))
    recorder.publish_root_hash_template(
        root_reference_id="root-reference",
        source_request_id="source-request",
        prefix_block_count=2,
    )
    branch_hashes = view.engine_core.request_block_hasher(_Request("branch-request", num_tokens=33))

    assert branch_hashes == source_hashes
    assert all(value == 0 for value in recorder.snapshot()["operation_counters"].values())


def test_optimized_independent_control_accepts_vllm_hasher_disabled() -> None:
    view = _View()
    view.engine_core.request_block_hasher = None  # type: ignore[assignment]
    recorder = _recorder(
        view,
        level=MetadataInstrumentationLevel.DISABLED,
        optimization=MetadataOptimization.CACHED_IMMUTABLE_ROOT_HASHES,
    )

    assert recorder.snapshot()["optimization"] == "cached_immutable_root_hashes"
    recorder.close()


def test_close_restores_every_patched_instance_method_and_is_idempotent() -> None:
    view = _View()
    originals = {
        "hasher": view.engine_core.request_block_hasher,
        "preprocess": view.engine_core.preprocess_add_request,
        "schedule": view.scheduler.schedule,
        "free_blocks": view.manager.block_pool.free_blocks,
        "runner": view.runner.execute_model,
        "runner_sample": view.runner.sample_tokens,
    }
    recorder = _recorder(view)
    assert view.engine_core.request_block_hasher is not originals["hasher"]

    recorder.close()
    recorder.close()

    assert view.engine_core.request_block_hasher == originals["hasher"]
    assert view.engine_core.preprocess_add_request == originals["preprocess"]
    assert view.scheduler.schedule == originals["schedule"]
    assert view.manager.block_pool.free_blocks == originals["free_blocks"]
    assert view.runner.execute_model == originals["runner"]
    assert view.runner.sample_tokens == originals["runner_sample"]
    assert view.engine_core.request_block_hasher(_Request("plain")) == [
        "root-hash-0",
        "root-hash-1",
    ]


def test_version_scope_active_span_and_template_shape_fail_closed() -> None:
    wrong = _View()
    wrong.runtime_version = "0.24.0"
    with pytest.raises(ValueError, match=r"requires exactly vLLM 0\.23\.0"):
        _recorder(wrong)

    view = _View()
    recorder = _recorder(view)
    recorder.begin_post_root_ready(branch_count=1, prefix_block_count=2)
    with pytest.raises(RuntimeError, match="POST_ROOT_READY remains active"):
        recorder.snapshot()
    with pytest.raises(RuntimeError, match="active spans"):
        recorder.close()
    recorder.end_post_root_ready()
    recorder.close()
