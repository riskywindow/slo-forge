"""Version-scoped vLLM 0.23.0 metadata instrumentation for Experiment 003.

This module intentionally depends on vLLM V1 internals.  It never exports
vLLM objects through Continuum IR: artifacts contain only scalar timestamps,
operation counts, request identifiers, and JSON-compatible native metrics.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, cast

VLLM_METADATA_RUNTIME_VERSION: Final = "0.23.0"
VLLM_METADATA_SOURCE_TAG: Final = "v0.23.0"
VLLM_METADATA_SOURCE_COMMIT: Final = "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"


class MetadataInstrumentationLevel(StrEnum):
    DISABLED = "disabled"
    MINIMAL = "minimal"
    FULL = "full"


class MetadataOptimization(StrEnum):
    BASELINE = "baseline"
    CACHED_IMMUTABLE_ROOT_HASHES = "cached_immutable_root_hashes"


@dataclass(frozen=True, slots=True)
class RequestScope0230:
    request_id: str
    root_reference_id: str | None
    source_session_id: str | None
    prefix_token_count: int | None
    eligible_for_root_hash_template: bool


@dataclass(slots=True)
class _OpenSpan:
    span_id: str
    parent_span_id: str | None
    name: str
    category: str
    wall_start_ns: int
    process_cpu_start_ns: int
    thread_cpu_start_ns: int
    attributes: dict[str, Any]


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    return repr(value)


def _block_count(value: object) -> int:
    groups = getattr(value, "blocks", value)
    if not isinstance(groups, (tuple, list)):
        return 0
    total = 0
    for group in groups:
        if isinstance(group, Sequence):
            total += len(group)
    return total


class VllmMetadataRecorder0230:
    """Bounded in-process probes for the exact vLLM 0.23.0 V1 path.

    The recorder is single-owner but thread-aware.  Sibling overlap is allowed
    in raw observations and rejected later when a critical-path decomposition
    is requested.  Primary Experiment 003 uses synchronous InprocClient/TP=1,
    so the POST_ROOT_READY stack is expected to remain on the owner thread.
    """

    _COUNTER_NAMES: Final = (
        "block_table_writes",
        "refcount_increments",
        "refcount_decrements",
        "block_allocations",
        "block_frees",
        "branch_session_metadata_allocations",
        "runtime_request_allocations",
        "scheduler_queue_inserts",
        "scheduler_queue_removals",
        "scheduler_scans",
        "scheduler_candidate_evaluations",
        "private_suffix_allocations",
        "prefix_lookup_calls",
        "prefix_block_candidates",
        "prefix_hash_calls",
        "prefix_hash_lookups",
        "prefix_hash_hits",
        "prefix_hash_misses",
        "prefix_blocks_bound",
        "request_block_hashes_computed",
        "prefix_hash_template_hits",
        "prompt_token_writes",
        "output_token_commits",
    )

    _SPAN_MEASUREMENT_SEMANTICS: Final = {
        "POST_ROOT_READY": {
            "evidence": "direct_wall",
            "scope": "continuous runner boundary from root-available branch work to all first outputs",
        },
        "HELIX_ORCHESTRATION": {
            "evidence": "direct_wall",
            "scope": "explicit SLOForge branch/session orchestration",
        },
        "REQUEST_BUILD": {
            "evidence": "direct_wall",
            "scope": "adapter submission plus nested vLLM request construction and hashing",
        },
        "PREFIX_LOOKUP": {
            "evidence": "direct_wall",
            "scope": "KVCacheManager.get_computed_blocks and optional FULL per-probe spans",
        },
        "BLOCK_TABLE_BUILD": {
            "evidence": "direct_wall",
            "scope": "scheduler prefix binding and GPU-worker CPU block-table writes",
        },
        "REFCOUNT_UPDATE": {
            "evidence": "direct_wall",
            "scope": "BlockPool.touch and BlockPool.free_blocks",
        },
        "PREFIX_STATE_BIND": {
            "evidence": "direct_wall_grouping",
            "scope": "KVCacheManager allocate/free inclusive grouping; exclusive time is prefix_metadata_other",
        },
        "PREFIX_METADATA_OTHER": {
            "evidence": "derived_exact_exclusive",
            "scope": "PREFIX_STATE_BIND wall minus nested lookup/table/refcount/allocation spans",
        },
        "PHYSICAL_KV_METADATA": {
            "evidence": "unpopulated_reserved_stage",
            "scope": "no standalone span; physical handling is decomposed into prefix_metadata_other, table, refcount, and allocation stages",
        },
        "SCHEDULER_ADMISSION": {
            "evidence": "direct_wall",
            "scope": "Scheduler.add_request and _enqueue_waiting_request",
        },
        "SCHEDULER_WAIT": {
            "evidence": "direct_boundaries_attribution_inferred",
            "scope": "last successful submit end to first Scheduler.schedule entry",
        },
        "SCHEDULER_SELECT": {
            "evidence": "direct_wall",
            "scope": "Scheduler.schedule inclusive; decomposition uses exclusive time",
        },
        "PRIVATE_STATE_PREP": {
            "evidence": "direct_wall",
            "scope": "coordinator and BlockPool allocation calls",
        },
        "GPU_SUBMISSION": {
            "evidence": "direct_cpu_wall",
            "scope": "UniProcExecutor execute_model/sample_tokens dispatch",
        },
        "GPU_EXECUTION": {
            "evidence": "direct_cpu_wall_gpu_inclusive",
            "scope": "GPUModelRunner execute_model plus sample_tokens/bookkeeping/output synchronization; not pure device time",
        },
        "OUTPUT_TOKEN_COMMIT": {
            "evidence": "direct_wall",
            "scope": "Scheduler.update_from_output plus frontend output processing",
        },
        "RESIDUAL": {
            "evidence": "derived_exact_difference",
            "scope": "POST_ROOT_READY minus all non-overlapping classified children",
        },
    }

    _COUNTER_SEMANTICS: Final = {
        "block_table_writes": "direct CPU integer slots written by MultiGroupBlockTable add/append",
        "refcount_increments": "direct blocks passed to successful BlockPool.touch plus blocks returned by get_new_blocks",
        "refcount_decrements": "direct blocks passed to successful-path BlockPool.free_blocks",
        "block_allocations": "direct non-null blocks returned by BlockPool.get_new_blocks",
        "block_frees": "derived-exact from pre-call ref_cnt==1 non-null blocks on successful free_blocks calls",
        "branch_session_metadata_allocations": "direct SLOForge logical branch/session objects created",
        "runtime_request_allocations": "direct successful EngineCore.preprocess_add_request results",
        "scheduler_queue_inserts": "direct _enqueue_waiting_request calls",
        "scheduler_queue_removals": "derived-exact scheduled_new_reqs count for the validated new-request/no-preemption path",
        "scheduler_scans": "direct Scheduler.schedule calls, not queue entries inspected",
        "scheduler_candidate_evaluations": "direct get_computed_blocks calls; equals candidates only for the validated unblocked FCFS path",
        "private_suffix_allocations": "direct coordinator blocks allocated for shared-root requests only",
        "prefix_lookup_calls": "direct KVCacheManager.get_computed_blocks calls",
        "prefix_block_candidates": "direct available request.block_hashes length, not probes executed",
        "prefix_hash_calls": "direct EngineCore.request_block_hasher calls",
        "prefix_hash_lookups": "FULL direct probes; MINIMAL derived-exact from aligned returned hits and pinned early-miss loop",
        "prefix_hash_hits": "FULL direct probe results; MINIMAL derived-exact aligned hit blocks",
        "prefix_hash_misses": "FULL direct probe results; MINIMAL derived-exact early-miss indicator",
        "prefix_blocks_bound": "direct cached blocks passed to coordinator allocation",
        "request_block_hashes_computed": "direct hashes returned by the original request hasher",
        "prefix_hash_template_hits": "direct optimized immutable-root template uses",
        "prompt_token_writes": "direct prompt token IDs copied into frontend request lists",
        "output_token_commits": "direct new token IDs returned by Scheduler.update_from_output",
    }

    def __init__(
        self,
        view: object,
        *,
        level: MetadataInstrumentationLevel,
        optimization: MetadataOptimization = MetadataOptimization.BASELINE,
        request_scope: Callable[[str], RequestScope0230 | None] | None = None,
        maximum_spans: int = 262_144,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        process_cpu_ns: Callable[[], int] = time.process_time_ns,
        thread_cpu_ns: Callable[[], int] = time.thread_time_ns,
    ) -> None:
        runtime_version = getattr(view, "runtime_version", None)
        if runtime_version != VLLM_METADATA_RUNTIME_VERSION:
            raise ValueError(
                "metadata instrumentation requires exactly vLLM "
                f"{VLLM_METADATA_RUNTIME_VERSION}, got {runtime_version!r}"
            )
        if not 1 <= maximum_spans <= 1_000_000:
            raise ValueError("maximum_spans must be in 1..1000000")
        self.level = level
        self.optimization = optimization
        self._view = cast(Any, view)
        self._request_scope = request_scope
        self._maximum_spans = maximum_spans
        self._clock_ns = clock_ns
        self._process_cpu_ns = process_cpu_ns
        self._thread_cpu_ns = thread_cpu_ns
        self._owner_thread_id = threading.get_ident()
        self._local = threading.local()
        self._sequence = 0
        self._spans: list[dict[str, Any]] = []
        self._counters: Counter[str] = Counter({name: 0 for name in self._COUNTER_NAMES})
        self._native_metrics: list[dict[str, Any]] = []
        self._request_hashes: dict[str, tuple[object, ...]] = {}
        self._root_hash_templates: dict[str, tuple[object, ...]] = {}
        self._first_gpu_execution_events: list[tuple[Any, Any]] = []
        self._patches: list[tuple[object, str, object]] = []
        self._post_root_span: _OpenSpan | None = None
        self._post_root_span_start_index: int | None = None
        self._post_root_native_metric_start_index: int | None = None
        self._post_root_counter_snapshot: dict[str, int] | None = None
        self._post_root_native_metric_count: int | None = None
        self._post_root_span_count: int | None = None
        self._last_admission_end: tuple[int, int, int] | None = None
        self._closed = False
        if level is not MetadataInstrumentationLevel.DISABLED:
            self._install()
        elif optimization is MetadataOptimization.CACHED_IMMUTABLE_ROOT_HASHES:
            # The optimized implementation must remain usable in the
            # tracing-disabled overhead control. Install only the semantic hash
            # template hook; do not install any timing/counter probes.
            self._install_request_hasher()

    def _stack(self) -> list[_OpenSpan]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def _start(
        self, name: str, category: str, attributes: Mapping[str, Any] | None = None
    ) -> _OpenSpan:
        if self._closed:
            raise RuntimeError("metadata recorder is closed")
        if len(self._spans) + sum(1 for _ in self._stack()) >= self._maximum_spans:
            raise RuntimeError("metadata span bound exceeded")
        self._sequence += 1
        stack = self._stack()
        frame = _OpenSpan(
            span_id=f"vllm0230-span-{self._sequence:08d}",
            parent_span_id=stack[-1].span_id if stack else None,
            name=name,
            category=category,
            wall_start_ns=self._clock_ns(),
            process_cpu_start_ns=self._process_cpu_ns(),
            thread_cpu_start_ns=self._thread_cpu_ns(),
            attributes=dict(attributes or {}),
        )
        stack.append(frame)
        return frame

    def _finish(self, frame: _OpenSpan) -> None:
        wall_end = self._clock_ns()
        process_end = self._process_cpu_ns()
        thread_end = self._thread_cpu_ns()
        stack = self._stack()
        if not stack or stack[-1] is not frame:
            raise RuntimeError("metadata span stack is not properly nested")
        stack.pop()
        row = {
            "span_id": frame.span_id,
            "parent_span_id": frame.parent_span_id,
            "name": frame.name,
            "category": frame.category,
            "wall_start_ns": frame.wall_start_ns,
            "wall_end_ns": wall_end,
            "process_cpu_start_ns": frame.process_cpu_start_ns,
            "process_cpu_end_ns": process_end,
            "thread_cpu_start_ns": frame.thread_cpu_start_ns,
            "thread_cpu_end_ns": thread_end,
            "wall_time_ns": wall_end - frame.wall_start_ns,
            "process_cpu_time_ns": process_end - frame.process_cpu_start_ns,
            "thread_cpu_time_ns": thread_end - frame.thread_cpu_start_ns,
            "thread_id": threading.get_ident(),
            "attributes": _json_value(frame.attributes),
        }
        if (
            row["wall_time_ns"] < 0
            or row["process_cpu_time_ns"] < 0
            or row["thread_cpu_time_ns"] < 0
        ):
            raise RuntimeError("metadata recorder observed a negative duration")
        self._spans.append(row)

    @contextmanager
    def span(
        self, name: str, category: str, attributes: Mapping[str, Any] | None = None
    ) -> Iterator[None]:
        if self.level is MetadataInstrumentationLevel.DISABLED:
            yield
            return
        frame = self._start(name, category, attributes)
        try:
            yield
        finally:
            self._finish(frame)

    def begin_post_root_ready(self, *, branch_count: int, prefix_block_count: int) -> None:
        if self.level is MetadataInstrumentationLevel.DISABLED:
            return
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("POST_ROOT_READY must begin on the recorder owner thread")
        if self._post_root_span is not None or self._stack():
            raise RuntimeError("POST_ROOT_READY is already active or another span is open")
        self._counters = Counter({name: 0 for name in self._COUNTER_NAMES})
        self._post_root_span_start_index = len(self._spans)
        self._post_root_native_metric_start_index = len(self._native_metrics)
        self._post_root_span = self._start(
            "POST_ROOT_READY",
            "POST_ROOT_READY",
            {"branch_count": branch_count, "prefix_block_count": prefix_block_count},
        )
        self._last_admission_end = None

    def end_post_root_ready(self) -> None:
        if self.level is MetadataInstrumentationLevel.DISABLED:
            return
        frame = self._post_root_span
        if frame is None:
            raise RuntimeError("POST_ROOT_READY is not active")
        if self._stack() != [frame]:
            raise RuntimeError("a child span remained open at POST_ROOT_READY end")
        self._finish(frame)
        self._post_root_span = None
        self._post_root_counter_snapshot = dict(sorted(self._counters.items()))
        assert self._post_root_native_metric_start_index is not None
        assert self._post_root_span_start_index is not None
        self._post_root_native_metric_count = (
            len(self._native_metrics) - self._post_root_native_metric_start_index
        )
        self._post_root_span_count = len(self._spans) - self._post_root_span_start_index

    def increment(self, name: str, amount: int = 1) -> None:
        if name not in self._counters:
            raise KeyError(f"unknown metadata operation counter {name!r}")
        if not isinstance(amount, int) or amount < 0:
            raise ValueError("metadata operation increments must be non-negative integers")
        if self.level is MetadataInstrumentationLevel.DISABLED:
            return
        self._counters[name] += amount

    def record_branch_session_allocation(self, amount: int = 1) -> None:
        if self.level is not MetadataInstrumentationLevel.DISABLED:
            self.increment("branch_session_metadata_allocations", amount)

    def record_prompt_token_writes(self, amount: int) -> None:
        if self.level is not MetadataInstrumentationLevel.DISABLED:
            self.increment("prompt_token_writes", amount)

    def mark_scheduler_wait_start(self) -> None:
        if self.level is MetadataInstrumentationLevel.DISABLED or self._post_root_span is None:
            return
        self._last_admission_end = (
            self._clock_ns(),
            self._process_cpu_ns(),
            self._thread_cpu_ns(),
        )

    def _append_scheduler_wait(self) -> None:
        """Materialize the uncovered admission-to-first-selection queue interval."""

        start = self._last_admission_end
        if start is None or self._post_root_span is None:
            return
        wall_end = self._clock_ns()
        process_end = self._process_cpu_ns()
        thread_end = self._thread_cpu_ns()
        self._last_admission_end = None
        if wall_end <= start[0]:
            return
        self._sequence += 1
        self._spans.append(
            {
                "span_id": f"vllm0230-span-{self._sequence:08d}",
                "parent_span_id": self._post_root_span.span_id,
                "name": "scheduler_waiting_queue_residency",
                "category": "SCHEDULER_WAIT",
                "wall_start_ns": start[0],
                "wall_end_ns": wall_end,
                "process_cpu_start_ns": start[1],
                "process_cpu_end_ns": process_end,
                "thread_cpu_start_ns": start[2],
                "thread_cpu_end_ns": thread_end,
                "wall_time_ns": wall_end - start[0],
                "process_cpu_time_ns": process_end - start[1],
                "thread_cpu_time_ns": thread_end - start[2],
                "thread_id": threading.get_ident(),
                "attributes": {"direct": True, "definition": "last_enqueue_end_to_schedule_start"},
            }
        )

    def publish_root_hash_template(
        self, *, root_reference_id: str, source_request_id: str, prefix_block_count: int
    ) -> None:
        if self.optimization is not MetadataOptimization.CACHED_IMMUTABLE_ROOT_HASHES:
            return
        hashes = self._request_hashes.get(source_request_id)
        if hashes is None or len(hashes) < prefix_block_count:
            raise RuntimeError("source request hashes do not cover the immutable shared root")
        self._root_hash_templates[root_reference_id] = hashes[:prefix_block_count]

    def _patch(self, owner: object, name: str, replacement: object) -> None:
        original = getattr(owner, name, None)
        if not callable(original):
            raise RuntimeError(
                f"required vLLM 0.23.0 callable {type(owner).__name__}.{name} is absent"
            )
        self._patches.append((owner, name, original))
        setattr(owner, name, replacement)

    def _timed_wrapper(
        self,
        original: Callable[..., Any],
        *,
        name: str,
        category: str,
        before: Callable[[tuple[Any, ...], dict[str, Any]], None] | None = None,
        after: Callable[[tuple[Any, ...], dict[str, Any], Any], None] | None = None,
    ) -> Callable[..., Any]:
        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if before is not None:
                before(args, kwargs)
            with self.span(name, category):
                result = original(*args, **kwargs)
            if after is not None:
                after(args, kwargs, result)
            return result

        return wrapped

    def _install_request_hasher(self) -> None:
        core = self._view.engine_core
        original_hasher = getattr(core, "request_block_hasher", None)
        if not callable(original_hasher):
            # Independent Experiment 003 trials intentionally disable prefix
            # caching, so vLLM sets this field to None. The implementation tag
            # remains "optimized" for the paired campaign, but there is no
            # shared-root optimization to apply on that control path.
            return

        def measured_hasher(request: object) -> list[object]:
            request_id = str(getattr(request, "request_id", ""))
            scope = self._request_scope(request_id) if self._request_scope else None
            existing = getattr(request, "block_hashes", ())
            if (
                self.optimization is MetadataOptimization.CACHED_IMMUTABLE_ROOT_HASHES
                and not existing
                and scope is not None
                and scope.eligible_for_root_hash_template
                and scope.root_reference_id in self._root_hash_templates
            ):
                template = self._root_hash_templates[scope.root_reference_id]
                expected = scope.prefix_token_count
                num_tokens = int(cast(Any, request).num_tokens)
                if expected is None or num_tokens != expected + 1:
                    raise RuntimeError(
                        "optimized branch request does not match root template shape"
                    )
                if (
                    getattr(request, "cache_salt", None) is not None
                    or getattr(request, "lora_request", None) is not None
                    or bool(getattr(request, "mm_features", ()))
                    or getattr(request, "prompt_embeds", None) is not None
                ):
                    raise RuntimeError(
                        "optimized root hashes require the validated pure-token request"
                    )
                self.increment("prefix_hash_calls")
                self.increment("prefix_hash_template_hits")
                # BlockHash is bytes in vLLM 0.23.0. Copy the list container so
                # Request.update_block_hashes cannot alias the cached template.
                return list(template)
            with self.span("request_block_hash_construction", "REQUEST_BUILD"):
                result = list(original_hasher(request))
            self.increment("prefix_hash_calls")
            self.increment("request_block_hashes_computed", len(result))
            if not existing and request_id:
                self._request_hashes[request_id] = tuple(result)
            return result

        self._patch(core, "request_block_hasher", measured_hasher)

    def _install(self) -> None:
        core = self._view.engine_core
        scheduler = self._view.scheduler
        manager = self._view.manager
        coordinator = manager.coordinator
        block_pool = manager.block_pool
        executor = core.model_executor
        runner = self._view.runner
        self._install_request_hasher()

        def request_after(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
            del args, kwargs, result
            self.increment("runtime_request_allocations")

        self._patch(
            core,
            "preprocess_add_request",
            self._timed_wrapper(
                core.preprocess_add_request,
                name="runtime_request_construction",
                category="REQUEST_BUILD",
                after=request_after,
            ),
        )

        self._patch(
            scheduler,
            "add_request",
            self._timed_wrapper(
                scheduler.add_request,
                name="scheduler_admission",
                category="SCHEDULER_ADMISSION",
            ),
        )

        def enqueue_before(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            del args, kwargs
            self.increment("scheduler_queue_inserts")

        self._patch(
            scheduler,
            "_enqueue_waiting_request",
            self._timed_wrapper(
                scheduler._enqueue_waiting_request,
                name="scheduler_queue_insert",
                category="SCHEDULER_ADMISSION",
                before=enqueue_before,
            ),
        )

        def schedule_before(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            del args, kwargs
            self._append_scheduler_wait()
            self.increment("scheduler_scans")

        def schedule_after(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
            del args, kwargs
            scheduled = getattr(result, "scheduled_new_reqs", ())
            self.increment("scheduler_queue_removals", len(scheduled))

        self._patch(
            scheduler,
            "schedule",
            self._timed_wrapper(
                scheduler.schedule,
                name="scheduler_selection",
                category="SCHEDULER_SELECT",
                before=schedule_before,
                after=schedule_after,
            ),
        )

        def lookup_before(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            del kwargs
            request = args[0] if args else None
            self.increment("prefix_lookup_calls")
            self.increment("scheduler_candidate_evaluations")
            self.increment("prefix_block_candidates", len(getattr(request, "block_hashes", ())))

        def lookup_after(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
            del kwargs
            if self.level is MetadataInstrumentationLevel.FULL:
                return
            request = args[0] if args else None
            if request is None or not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("vLLM prefix lookup result changed shape")
            if not bool(getattr(manager, "enable_caching", False)) or bool(
                getattr(request, "skip_reading_prefix_cache", False)
            ):
                return
            block_size = int(self._view.block_size_tokens)
            num_tokens = int(getattr(request, "num_tokens", 0))
            hit_tokens = int(result[1])
            if block_size <= 0 or hit_tokens < 0 or hit_tokens % block_size:
                raise RuntimeError("vLLM prefix lookup returned an invalid aligned hit count")
            max_probes = min(
                len(getattr(request, "block_hashes", ())),
                max(0, num_tokens - 1) // block_size,
            )
            hits = hit_tokens // block_size
            if hits > max_probes:
                raise RuntimeError("vLLM prefix lookup hit count exceeds its candidates")
            misses = int(hits < max_probes)
            self.increment("prefix_hash_lookups", hits + misses)
            self.increment("prefix_hash_hits", hits)
            self.increment("prefix_hash_misses", misses)

        self._patch(
            manager,
            "get_computed_blocks",
            self._timed_wrapper(
                manager.get_computed_blocks,
                name="prefix_shared_block_lookup",
                category="PREFIX_LOOKUP",
                before=lookup_before,
                after=lookup_after,
            ),
        )

        def cached_after(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
            del args, kwargs
            self.increment("prefix_hash_lookups")
            self.increment("prefix_hash_hits" if result is not None else "prefix_hash_misses")

        if self.level is MetadataInstrumentationLevel.FULL:
            original_get_cached_block = block_pool.get_cached_block

            @functools.wraps(original_get_cached_block)
            def measured_get_cached_block(*args: Any, **kwargs: Any) -> Any:
                with self.span("prefix_hash_table_probe", "PREFIX_LOOKUP"):
                    result = original_get_cached_block(*args, **kwargs)
                cached_after(args, kwargs, result)
                return result

            self._patch(block_pool, "get_cached_block", measured_get_cached_block)

        def computed_after(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
            del result
            # KVCacheManager.allocate_slots calls the coordinator with keyword
            # arguments in v0.23.0 (kv_cache_manager.py:406-411).
            blocks = kwargs.get("new_computed_blocks", args[1] if len(args) > 1 else ())
            count = _block_count(blocks)
            self.increment("prefix_blocks_bound", count)

        self._patch(
            coordinator,
            "allocate_new_computed_blocks",
            self._timed_wrapper(
                coordinator.allocate_new_computed_blocks,
                name="shared_block_table_bind",
                category="BLOCK_TABLE_BUILD",
                after=computed_after,
            ),
        )

        def new_blocks_after(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
            count = _block_count(result)
            request_id = str(kwargs.get("request_id", args[0] if args else ""))
            scope = self._request_scope(request_id) if self._request_scope else None
            # Independent-prefill blocks are private physical allocations, but
            # they are not private *suffix* allocations. The adapter always
            # supplies a request scope; retain the old generic-fixture behavior
            # only when no scope resolver is installed.
            if self._request_scope is None or (
                scope is not None and scope.root_reference_id is not None
            ):
                self.increment("private_suffix_allocations", count)

        self._patch(
            coordinator,
            "allocate_new_blocks",
            self._timed_wrapper(
                coordinator.allocate_new_blocks,
                name="private_suffix_block_table_extend",
                category="PRIVATE_STATE_PREP",
                after=new_blocks_after,
            ),
        )

        def touch_before(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            del kwargs
            blocks = args[0] if args else ()
            self.increment("refcount_increments", len(blocks))

        self._patch(
            block_pool,
            "touch",
            self._timed_wrapper(
                block_pool.touch,
                name="shared_block_refcount_update",
                category="REFCOUNT_UPDATE",
                before=touch_before,
            ),
        )

        def allocate_after(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
            del args, kwargs
            count = len(result)
            self.increment("block_allocations", count)
            self.increment("refcount_increments", count)

        self._patch(
            block_pool,
            "get_new_blocks",
            self._timed_wrapper(
                block_pool.get_new_blocks,
                name="physical_kv_block_allocation",
                category="PRIVATE_STATE_PREP",
                after=allocate_after,
            ),
        )

        original_free_blocks = block_pool.free_blocks

        @functools.wraps(original_free_blocks)
        def measured_free_blocks(ordered_blocks: Any, prepend: bool = False) -> Any:
            materialized = tuple(ordered_blocks)
            self.increment("refcount_decrements", len(materialized))
            self.increment(
                "block_frees",
                sum(
                    int(getattr(block, "ref_cnt", 0)) == 1
                    and not bool(getattr(block, "is_null", False))
                    for block in materialized
                ),
            )
            with self.span("physical_kv_block_release", "REFCOUNT_UPDATE"):
                return original_free_blocks(materialized, prepend=prepend)

        self._patch(block_pool, "free_blocks", measured_free_blocks)

        self._patch(
            manager,
            "allocate_slots",
            self._timed_wrapper(
                manager.allocate_slots,
                name="physical_kv_metadata_handling",
                category="PREFIX_STATE_BIND",
            ),
        )
        self._patch(
            manager,
            "free",
            self._timed_wrapper(
                manager.free,
                name="physical_kv_metadata_release",
                category="PREFIX_STATE_BIND",
            ),
        )

        input_batch = getattr(runner, "input_batch", None)
        worker_block_table = getattr(input_batch, "block_table", None)
        block_tables = getattr(worker_block_table, "block_tables", None)
        if not isinstance(block_tables, list) or not block_tables:
            raise RuntimeError("vLLM 0.23.0 GPU worker MultiGroupBlockTable is unavailable")
        worker_block_table = cast(Any, worker_block_table)

        def worker_write_count(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            grouped_ids = kwargs.get("block_ids", args[0] if args else ())
            if not isinstance(grouped_ids, (tuple, list)):
                raise RuntimeError("vLLM worker block IDs changed shape")
            if len(grouped_ids) != len(block_tables):
                raise RuntimeError("vLLM worker block-table group count changed")
            count = 0
            for ids, table in zip(grouped_ids, block_tables, strict=True):
                if not isinstance(ids, Sequence):
                    raise RuntimeError("vLLM worker block-ID row changed shape")
                blocks_per_kv_block = int(getattr(table, "blocks_per_kv_block", 1))
                if blocks_per_kv_block <= 0:
                    raise RuntimeError("vLLM worker block expansion factor is invalid")
                count += len(ids) * blocks_per_kv_block
            self.increment("block_table_writes", count)

        def worker_write_after(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
            del result
            worker_write_count(args, kwargs)

        self._patch(
            worker_block_table,
            "add_row",
            self._timed_wrapper(
                worker_block_table.add_row,
                name="worker_block_table_initial_write",
                category="BLOCK_TABLE_BUILD",
                after=worker_write_after,
            ),
        )
        self._patch(
            worker_block_table,
            "append_row",
            self._timed_wrapper(
                worker_block_table.append_row,
                name="worker_block_table_append",
                category="BLOCK_TABLE_BUILD",
                after=worker_write_after,
            ),
        )

        self._patch(
            executor,
            "execute_model",
            self._timed_wrapper(
                executor.execute_model,
                name="model_execution_submission",
                category="GPU_SUBMISSION",
            ),
        )
        self._patch(
            executor,
            "sample_tokens",
            self._timed_wrapper(
                executor.sample_tokens,
                name="sampling_execution_submission",
                category="GPU_SUBMISSION",
            ),
        )
        original_model_forward = runner._model_forward

        @functools.wraps(original_model_forward)
        def measured_model_forward(*args: Any, **kwargs: Any) -> Any:
            events: tuple[Any, Any] | None = None
            if (
                self._post_root_span is not None
                and not self._first_gpu_execution_events
                and str(getattr(self._view, "device", "")).startswith("cuda")
            ):
                torch = importlib.import_module("torch")
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                events = (start, end)
            result = original_model_forward(*args, **kwargs)
            if events is not None:
                events[1].record()
                self._first_gpu_execution_events.append(events)
            return result

        self._patch(runner, "_model_forward", measured_model_forward)
        self._patch(
            runner,
            "execute_model",
            self._timed_wrapper(
                runner.execute_model,
                name="gpu_worker_execution",
                category="GPU_EXECUTION",
            ),
        )
        self._patch(
            runner,
            "sample_tokens",
            self._timed_wrapper(
                runner.sample_tokens,
                name="gpu_sampling_bookkeeping_and_output_sync",
                category="GPU_EXECUTION",
            ),
        )

        def output_after(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
            del args, kwargs
            outputs = result.values() if isinstance(result, dict) else ()
            committed = 0
            for batch in outputs:
                for output in getattr(batch, "outputs", ()):
                    committed += len(getattr(output, "new_token_ids", ()))
            self.increment("output_token_commits", committed)

        self._patch(
            scheduler,
            "update_from_output",
            self._timed_wrapper(
                scheduler.update_from_output,
                name="runtime_output_token_commit",
                category="OUTPUT_TOKEN_COMMIT",
                after=output_after,
            ),
        )

        original_make_stats = scheduler.make_stats

        @functools.wraps(original_make_stats)
        def measured_make_stats(*args: Any, **kwargs: Any) -> Any:
            result = original_make_stats(*args, **kwargs)
            if result is not None:
                self._native_metrics.append(
                    {
                        "observed_at_monotonic_ns": self._clock_ns(),
                        "source": "vllm.v1.core.sched.scheduler.Scheduler.make_stats",
                        "metrics": _json_value(result),
                    }
                )
            return result

        self._patch(scheduler, "make_stats", measured_make_stats)

    def snapshot(self) -> dict[str, Any]:
        if self._post_root_span is not None:
            raise RuntimeError("cannot snapshot while POST_ROOT_READY remains active")
        if self._stack():
            raise RuntimeError("cannot snapshot while a metadata span remains active")
        span_start = self._post_root_span_start_index or 0
        span_count = self._post_root_span_count or 0
        post_root_spans = self._spans[span_start : span_start + span_count]
        native_start = self._post_root_native_metric_start_index or 0
        native_count = self._post_root_native_metric_count or 0
        gpu_events: list[dict[str, Any]] = []
        for index, (start, end) in enumerate(self._first_gpu_execution_events):
            elapsed_time = getattr(start, "elapsed_time", None)
            if not callable(elapsed_time):
                raise RuntimeError("CUDA event lacks elapsed_time")
            gpu_events.append(
                {
                    "index": index,
                    "scope": "first_GPUModelRunner._model_forward",
                    "elapsed_ns": round(float(elapsed_time(end)) * 1_000_000),
                    "synchronization": "resolved_after_CPU-visible_sample_output",
                }
            )
        return {
            "schema_version": "sloforge.branchfabric.vllm-metadata-observation-0230/v1",
            "runtime": "vllm",
            "runtime_version": VLLM_METADATA_RUNTIME_VERSION,
            "runtime_source_tag": VLLM_METADATA_SOURCE_TAG,
            "runtime_source_commit": VLLM_METADATA_SOURCE_COMMIT,
            "instrumentation_level": self.level.value,
            "optimization": self.optimization.value,
            "spans": sorted(
                post_root_spans,
                key=lambda row: (row["wall_start_ns"], row["span_id"]),
            ),
            "operation_counters": dict(sorted(self._counters.items())),
            "post_root_ready_operation_counters": self._post_root_counter_snapshot,
            "post_root_ready_span_count": self._post_root_span_count,
            "post_root_ready_native_metric_count": self._post_root_native_metric_count,
            "runtime_native_metrics": list(self._native_metrics),
            "post_root_ready_runtime_native_metrics": self._native_metrics[
                native_start : native_start + native_count
            ],
            "gpu_event_measurements": gpu_events,
            "stage_measurement_semantics": self._SPAN_MEASUREMENT_SEMANTICS,
            "operation_counter_semantics": self._COUNTER_SEMANTICS,
        }

    def close(self) -> None:
        if self._closed:
            return
        if self._post_root_span is not None or self._stack():
            raise RuntimeError("cannot close metadata recorder with active spans")
        for owner, name, original in reversed(self._patches):
            setattr(owner, name, original)
        self._patches.clear()
        self._closed = True


__all__ = [
    "VLLM_METADATA_RUNTIME_VERSION",
    "VLLM_METADATA_SOURCE_COMMIT",
    "VLLM_METADATA_SOURCE_TAG",
    "MetadataInstrumentationLevel",
    "MetadataOptimization",
    "RequestScope0230",
    "VllmMetadataRecorder0230",
]
