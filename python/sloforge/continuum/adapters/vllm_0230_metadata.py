"""Version-scoped timing and operation models for vLLM 0.23.0 studies.

This module deliberately contains no vLLM imports.  The live adapter installs
the version-specific observation hooks; these types validate their pointer-free
output and keep vLLM internals out of the generic Continuum IR.

Wall-time decomposition is based on a single continuous ``POST_ROOT_READY``
span.  Nested spans may be useful diagnostics, but only their exclusive time is
added to a stage.  Uncovered parent time is always assigned to ``RESIDUAL``.
Consequently, inclusive child spans are never summed and cannot double count.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Any, Final, overload


class ReadinessStage(StrEnum):
    """Experiment 003 stages; grouping stages are not additive outputs."""

    POST_ROOT_READY = "post_root_ready"
    HELIX_ORCHESTRATION = "helix_orchestration"
    REQUEST_BUILD = "request_build"
    PREFIX_STATE_BIND = "prefix_state_bind"
    PREFIX_LOOKUP = "prefix_lookup"
    BLOCK_TABLE_BUILD = "block_table_build"
    REFCOUNT_UPDATE = "refcount_update"
    PREFIX_METADATA_OTHER = "prefix_metadata_other"
    PHYSICAL_KV_METADATA = "physical_kv_metadata"
    SCHEDULER_ADMISSION = "scheduler_admission"
    SCHEDULER_WAIT = "scheduler_wait"
    SCHEDULER_SELECT = "scheduler_select"
    PRIVATE_STATE_PREP = "private_state_prep"
    GPU_SUBMISSION = "gpu_submission"
    GPU_EXECUTION = "gpu_execution"
    OUTPUT_TOKEN_COMMIT = "output_token_commit"
    RESIDUAL = "residual"


ADDITIVE_STAGES: Final[tuple[ReadinessStage, ...]] = tuple(
    stage
    for stage in ReadinessStage
    if stage not in {ReadinessStage.POST_ROOT_READY, ReadinessStage.PREFIX_STATE_BIND}
)


class MetadataOperation(StrEnum):
    """Directly countable operations on the pinned vLLM 0.23.0 path."""

    BLOCK_TABLE_WRITES = "block_table_writes"
    REFCOUNT_INCREMENTS = "refcount_increments"
    REFCOUNT_DECREMENTS = "refcount_decrements"
    BLOCK_ALLOCATIONS = "block_allocations"
    BLOCK_FREES = "block_frees"
    BRANCH_SESSION_METADATA_ALLOCATIONS = "branch_session_metadata_allocations"
    RUNTIME_REQUEST_ALLOCATIONS = "runtime_request_allocations"
    SCHEDULER_QUEUE_INSERTS = "scheduler_queue_inserts"
    SCHEDULER_QUEUE_REMOVALS = "scheduler_queue_removals"
    SCHEDULER_SCANS = "scheduler_scans"
    SCHEDULER_CANDIDATE_EVALUATIONS = "scheduler_candidate_evaluations"
    PRIVATE_SUFFIX_ALLOCATIONS = "private_suffix_allocations"
    PREFIX_LOOKUP_CALLS = "prefix_lookup_calls"
    PREFIX_BLOCK_CANDIDATES = "prefix_block_candidates"
    PREFIX_HASH_CALLS = "prefix_hash_calls"
    PREFIX_HASH_LOOKUPS = "prefix_hash_lookups"
    PREFIX_HASH_HITS = "prefix_hash_hits"
    PREFIX_HASH_MISSES = "prefix_hash_misses"
    PREFIX_BLOCKS_BOUND = "prefix_blocks_bound"
    REQUEST_BLOCK_HASHES_COMPUTED = "request_block_hashes_computed"
    PREFIX_HASH_TEMPLATE_HITS = "prefix_hash_template_hits"
    PROMPT_TOKEN_WRITES = "prompt_token_writes"
    OUTPUT_TOKEN_COMMITS = "output_token_commits"


@dataclass(frozen=True, slots=True)
class TimingPoint:
    """One co-sampled wall/process/thread timestamp."""

    wall_ns: int
    process_cpu_ns: int
    thread_cpu_ns: int | None

    def __post_init__(self) -> None:
        for name, value in (
            ("wall_ns", self.wall_ns),
            ("process_cpu_ns", self.process_cpu_ns),
            ("thread_cpu_ns", self.thread_cpu_ns),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")

    def minus(self, earlier: TimingPoint) -> TimingDuration:
        if (self.thread_cpu_ns is None) != (earlier.thread_cpu_ns is None):
            raise ValueError("thread CPU availability changed within a span")
        duration = TimingDuration(
            wall_ns=self.wall_ns - earlier.wall_ns,
            process_cpu_ns=self.process_cpu_ns - earlier.process_cpu_ns,
            thread_cpu_ns=(
                None if self.thread_cpu_ns is None else self.thread_cpu_ns - earlier.thread_cpu_ns  # type: ignore[operator]
            ),
        )
        return duration


@dataclass(frozen=True, slots=True)
class TimingDuration:
    wall_ns: int = 0
    process_cpu_ns: int = 0
    thread_cpu_ns: int | None = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("wall_ns", self.wall_ns),
            ("process_cpu_ns", self.process_cpu_ns),
            ("thread_cpu_ns", self.thread_cpu_ns),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")

    def __add__(self, other: TimingDuration) -> TimingDuration:
        if (self.thread_cpu_ns is None) != (other.thread_cpu_ns is None):
            raise ValueError("cannot add durations with different thread CPU availability")
        return TimingDuration(
            wall_ns=self.wall_ns + other.wall_ns,
            process_cpu_ns=self.process_cpu_ns + other.process_cpu_ns,
            thread_cpu_ns=(
                None if self.thread_cpu_ns is None else self.thread_cpu_ns + other.thread_cpu_ns  # type: ignore[operator]
            ),
        )

    def subtract(self, other: TimingDuration) -> TimingDuration:
        if (self.thread_cpu_ns is None) != (other.thread_cpu_ns is None):
            raise ValueError("cannot subtract durations with different thread CPU availability")
        return TimingDuration(
            wall_ns=self.wall_ns - other.wall_ns,
            process_cpu_ns=self.process_cpu_ns - other.process_cpu_ns,
            thread_cpu_ns=(
                None if self.thread_cpu_ns is None else self.thread_cpu_ns - other.thread_cpu_ns  # type: ignore[operator]
            ),
        )

    def as_dict(self) -> dict[str, int | None]:
        return {
            "wall_ns": self.wall_ns,
            "process_cpu_ns": self.process_cpu_ns,
            "thread_cpu_ns": self.thread_cpu_ns,
        }


@dataclass(frozen=True, slots=True)
class TimingTolerance:
    """Tolerance for an independent cross-check, never for internal overlap."""

    absolute_ns: int = 100_000
    relative_fraction: float = 0.001

    def __post_init__(self) -> None:
        if self.absolute_ns < 0:
            raise ValueError("absolute_ns must be non-negative")
        if not math.isfinite(self.relative_fraction) or self.relative_fraction < 0:
            raise ValueError("relative_fraction must be finite and non-negative")

    def allowed_ns(self, reference_ns: int) -> int:
        return max(self.absolute_ns, math.ceil(reference_ns * self.relative_fraction))


@dataclass(frozen=True, slots=True)
class OperationCounters:
    """Sorted, unique, non-negative operation counts."""

    values: tuple[tuple[MetadataOperation, int], ...] = ()

    def __post_init__(self) -> None:
        names = [name for name, _ in self.values]
        if names != sorted(names, key=str) or len(names) != len(set(names)):
            raise ValueError("operation counters must be unique and sorted by name")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for _, value in self.values
        ):
            raise ValueError("operation counters must be non-negative integers")

    @classmethod
    @overload
    def from_mapping(cls, values: Mapping[MetadataOperation, int]) -> OperationCounters: ...

    @classmethod
    @overload
    def from_mapping(cls, values: Mapping[str, int]) -> OperationCounters: ...

    @classmethod
    def from_mapping(
        cls, values: Mapping[MetadataOperation, int] | Mapping[str, int]
    ) -> OperationCounters:
        normalized: dict[MetadataOperation, int] = {}
        for raw_name, value in values.items():
            name = MetadataOperation(raw_name)
            if name in normalized:
                raise ValueError(f"duplicate normalized counter {name.value}")
            normalized[name] = value
        return cls(tuple(sorted(normalized.items(), key=lambda item: item[0].value)))

    def as_dict(self) -> dict[str, int]:
        return {name.value: value for name, value in self.values}

    def delta(self, earlier: OperationCounters) -> OperationCounters:
        current = dict(self.values)
        prior = dict(earlier.values)
        delta: dict[MetadataOperation, int] = {}
        for name in MetadataOperation:
            value = current.get(name, 0) - prior.get(name, 0)
            if value < 0:
                raise ValueError(f"counter {name.value} decreased")
            if value:
                delta[name] = value
        return OperationCounters.from_mapping(delta)

    def normalized(
        self,
        *,
        branch_count: int,
        prefix_block_count: int,
        token_count: int,
        request_count: int,
        fanout: int,
    ) -> dict[str, object]:
        denominators = {
            "per_branch": branch_count,
            "per_prefix_block": prefix_block_count,
            "per_token": token_count,
            "per_request": request_count,
            "per_fanout_member": fanout,
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in denominators.values()
        ):
            raise ValueError("normalization denominators must be positive integers")
        totals = self.as_dict()
        return {
            "fanout": fanout,
            "totals": totals,
            **{
                label: {name: value / denominator for name, value in totals.items()}
                for label, denominator in denominators.items()
            },
        }


_EMPTY_COUNTERS: Final = OperationCounters()


@dataclass(frozen=True, slots=True)
class StageSpan:
    span_id: str
    parent_span_id: str | None
    stage: ReadinessStage
    start: TimingPoint
    end: TimingPoint
    branch_count: int
    prefix_block_count: int
    operations: OperationCounters = _EMPTY_COUNTERS
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.span_id:
            raise ValueError("span_id must be non-empty")
        if self.branch_count <= 0:
            raise ValueError("branch_count must be positive")
        if self.prefix_block_count < 0:
            raise ValueError("prefix_block_count must be non-negative")
        self.end.minus(self.start)

    @property
    def duration(self) -> TimingDuration:
        return self.end.minus(self.start)

    def as_dict(self) -> dict[str, object]:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "stage": self.stage.value,
            "start": {
                "wall_ns": self.start.wall_ns,
                "process_cpu_ns": self.start.process_cpu_ns,
                "thread_cpu_ns": self.start.thread_cpu_ns,
            },
            "end": {
                "wall_ns": self.end.wall_ns,
                "process_cpu_ns": self.end.process_cpu_ns,
                "thread_cpu_ns": self.end.thread_cpu_ns,
            },
            "duration": self.duration.as_dict(),
            "branch_count": self.branch_count,
            "prefix_block_count": self.prefix_block_count,
            "operations": self.operations.as_dict(),
            "detail": self.detail,
        }


def _exclusive_stage(stage: ReadinessStage) -> ReadinessStage:
    if stage is ReadinessStage.POST_ROOT_READY:
        return ReadinessStage.RESIDUAL
    if stage is ReadinessStage.PREFIX_STATE_BIND:
        return ReadinessStage.PREFIX_METADATA_OTHER
    return stage


@dataclass(frozen=True, slots=True)
class TimingDecomposition:
    parent: TimingDuration
    stages: tuple[tuple[ReadinessStage, TimingDuration], ...]

    def __post_init__(self) -> None:
        names = [name for name, _ in self.stages]
        if tuple(names) != ADDITIVE_STAGES:
            raise ValueError("decomposition must contain every additive stage in canonical order")
        total = TimingDuration(thread_cpu_ns=None if self.parent.thread_cpu_ns is None else 0)
        for _, duration in self.stages:
            total = total + duration
        if total != self.parent:
            raise ValueError("non-overlapping stage sum does not equal POST_ROOT_READY")

    def as_dict(self) -> dict[str, object]:
        return {
            "parent": self.parent.as_dict(),
            "stages": {stage.value: duration.as_dict() for stage, duration in self.stages},
            "invariant": {
                "sum_equals_parent": True,
                "wall_error_ns": 0,
                "process_cpu_error_ns": 0,
                "thread_cpu_error_ns": None if self.parent.thread_cpu_ns is None else 0,
            },
        }

    def assert_parent_cross_check(
        self, expected: TimingDuration, tolerance: TimingTolerance | None = None
    ) -> None:
        tolerance = tolerance or TimingTolerance()
        dimensions = (
            ("wall", self.parent.wall_ns, expected.wall_ns),
            ("process CPU", self.parent.process_cpu_ns, expected.process_cpu_ns),
        )
        for label, observed, reference in dimensions:
            if abs(observed - reference) > tolerance.allowed_ns(reference):
                raise ValueError(f"{label} parent cross-check exceeds tolerance")
        if (self.parent.thread_cpu_ns is None) != (expected.thread_cpu_ns is None):
            raise ValueError("thread CPU parent cross-check availability differs")
        if (
            self.parent.thread_cpu_ns is not None
            and expected.thread_cpu_ns is not None
            and abs(self.parent.thread_cpu_ns - expected.thread_cpu_ns)
            > tolerance.allowed_ns(expected.thread_cpu_ns)
        ):
            raise ValueError("thread CPU parent cross-check exceeds tolerance")


@dataclass(frozen=True, slots=True)
class SpanHierarchy:
    spans: tuple[StageSpan, ...]
    maximum_spans: int = 100_000

    def __post_init__(self) -> None:
        if not 1 <= len(self.spans) <= self.maximum_spans:
            raise ValueError("span hierarchy is empty or exceeds its bound")
        by_id = {span.span_id: span for span in self.spans}
        if len(by_id) != len(self.spans):
            raise ValueError("span IDs must be unique")
        roots = [span for span in self.spans if span.parent_span_id is None]
        if len(roots) != 1 or roots[0].stage is not ReadinessStage.POST_ROOT_READY:
            raise ValueError("exactly one POST_ROOT_READY root span is required")
        root = roots[0]
        children: dict[str, list[StageSpan]] = defaultdict(list)
        for span in self.spans:
            if span is root:
                continue
            if span.parent_span_id not in by_id:
                raise ValueError(f"span {span.span_id} has an unknown parent")
            parent = by_id[span.parent_span_id]
            if (
                span.start.wall_ns < parent.start.wall_ns
                or span.end.wall_ns > parent.end.wall_ns
                or span.start.process_cpu_ns < parent.start.process_cpu_ns
                or span.end.process_cpu_ns > parent.end.process_cpu_ns
            ):
                raise ValueError(f"span {span.span_id} is not contained by its parent")
            if (span.start.thread_cpu_ns is None) != (parent.start.thread_cpu_ns is None):
                raise ValueError("thread CPU availability differs within the hierarchy")
            if span.start.thread_cpu_ns is not None and (
                span.start.thread_cpu_ns < parent.start.thread_cpu_ns  # type: ignore[operator]
                or span.end.thread_cpu_ns > parent.end.thread_cpu_ns  # type: ignore[operator]
            ):
                raise ValueError(f"span {span.span_id} thread CPU is not contained by its parent")
            children[parent.span_id].append(span)

        visited: set[str] = set()
        active: set[str] = set()

        def visit(span_id: str) -> None:
            if span_id in active:
                raise ValueError("span hierarchy contains a cycle")
            if span_id in visited:
                return
            active.add(span_id)
            for child in children.get(span_id, ()):
                visit(child.span_id)
            active.remove(span_id)
            visited.add(span_id)

        visit(root.span_id)
        if len(visited) != len(self.spans):
            raise ValueError("every span must descend from POST_ROOT_READY")

        for parent_id, siblings in children.items():
            ordered = sorted(siblings, key=lambda span: (span.start.wall_ns, span.end.wall_ns))
            for left, right in pairwise(ordered):
                if right.start.wall_ns < left.end.wall_ns:
                    raise ValueError(f"sibling spans {left.span_id} and {right.span_id} overlap")
            parent_duration = by_id[parent_id].duration
            child_total = TimingDuration(
                thread_cpu_ns=None if parent_duration.thread_cpu_ns is None else 0
            )
            for child in siblings:
                child_total = child_total + child.duration
            try:
                parent_duration.subtract(child_total)
            except ValueError as error:
                raise ValueError(f"children exceed parent {parent_id}") from error

    @property
    def root(self) -> StageSpan:
        return next(span for span in self.spans if span.parent_span_id is None)

    def decomposition(self) -> TimingDecomposition:
        children: dict[str, list[StageSpan]] = defaultdict(list)
        for span in self.spans:
            if span.parent_span_id is not None:
                children[span.parent_span_id].append(span)
        totals = {
            stage: TimingDuration(
                thread_cpu_ns=None if self.root.duration.thread_cpu_ns is None else 0
            )
            for stage in ADDITIVE_STAGES
        }
        for span in self.spans:
            child_total = TimingDuration(
                thread_cpu_ns=None if span.duration.thread_cpu_ns is None else 0
            )
            for child in children.get(span.span_id, ()):
                child_total = child_total + child.duration
            exclusive = span.duration.subtract(child_total)
            target = _exclusive_stage(span.stage)
            totals[target] = totals[target] + exclusive
        return TimingDecomposition(
            parent=self.root.duration,
            stages=tuple((stage, totals[stage]) for stage in ADDITIVE_STAGES),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "sloforge.branchfabric.vllm-0230-metadata-timing/v1",
            "spans": [span.as_dict() for span in self.spans],
            "decomposition": self.decomposition().as_dict(),
        }


def hierarchy_from_vllm_observation(observation: Mapping[str, Any]) -> SpanHierarchy:
    """Validate and convert the pointer-free raw vLLM 0.23.0 observation."""

    raw_spans = observation.get("spans")
    if not isinstance(raw_spans, list) or not raw_spans:
        raise ValueError("vLLM observation contains no POST_ROOT_READY spans")
    stage_by_category = {stage.value.upper(): stage for stage in ReadinessStage}
    converted: list[StageSpan] = []
    root_rows = [row for row in raw_spans if row.get("category") == "POST_ROOT_READY"]
    if len(root_rows) != 1:
        raise ValueError("vLLM observation must contain exactly one POST_ROOT_READY row")
    root_attributes = root_rows[0].get("attributes", {})
    if not isinstance(root_attributes, Mapping):
        raise ValueError("POST_ROOT_READY attributes are malformed")
    branch_count = int(root_attributes.get("branch_count", 0))
    prefix_block_count = int(root_attributes.get("prefix_block_count", 0))
    strict_counter_names = {operation.value for operation in MetadataOperation}
    raw_counters = observation.get("post_root_ready_operation_counters", {})
    if not isinstance(raw_counters, Mapping):
        raise ValueError("post-root operation counters are malformed")
    root_operations = OperationCounters.from_mapping(
        {
            str(name): int(value)
            for name, value in raw_counters.items()
            if str(name) in strict_counter_names
        }
    )
    for row in raw_spans:
        if not isinstance(row, Mapping):
            raise ValueError("vLLM observation span row is malformed")
        category = str(row.get("category", ""))
        try:
            stage = stage_by_category[category]
        except KeyError as error:
            raise ValueError(f"unknown vLLM observation category {category!r}") from error
        attributes = row.get("attributes", {})
        detail = str(row.get("name", category))
        if attributes:
            detail += ":" + str(attributes)
        converted.append(
            StageSpan(
                span_id=str(row["span_id"]),
                parent_span_id=(
                    None if row.get("parent_span_id") is None else str(row["parent_span_id"])
                ),
                stage=stage,
                start=TimingPoint(
                    wall_ns=int(row["wall_start_ns"]),
                    process_cpu_ns=int(row["process_cpu_start_ns"]),
                    thread_cpu_ns=int(row["thread_cpu_start_ns"]),
                ),
                end=TimingPoint(
                    wall_ns=int(row["wall_end_ns"]),
                    process_cpu_ns=int(row["process_cpu_end_ns"]),
                    thread_cpu_ns=int(row["thread_cpu_end_ns"]),
                ),
                branch_count=branch_count,
                prefix_block_count=prefix_block_count,
                operations=(
                    root_operations if stage is ReadinessStage.POST_ROOT_READY else _EMPTY_COUNTERS
                ),
                detail=detail,
            )
        )
    return SpanHierarchy(tuple(converted), maximum_spans=max(100_000, len(converted)))


class SystemTimingClock:
    """Clock used by the live probe; tests inject a deterministic replacement."""

    def __call__(self) -> TimingPoint:
        thread_clock = getattr(time, "thread_time_ns", None)
        return TimingPoint(
            wall_ns=time.monotonic_ns(),
            process_cpu_ns=time.process_time_ns(),
            thread_cpu_ns=thread_clock() if callable(thread_clock) else None,
        )


@dataclass(slots=True)
class _OpenSpan:
    span_id: str
    parent_span_id: str | None
    stage: ReadinessStage
    start: TimingPoint
    branch_count: int
    prefix_block_count: int
    operations: OperationCounters
    detail: str | None


@dataclass(slots=True)
class ExclusiveSpanRecorder:
    """Deterministic, bounded stack recorder for one synchronous critical path."""

    clock: Callable[[], TimingPoint] = field(default_factory=SystemTimingClock)
    maximum_spans: int = 100_000
    _stack: list[_OpenSpan] = field(default_factory=list, init=False)
    _closed: list[StageSpan] = field(default_factory=list, init=False)
    _next_id: int = field(default=0, init=False)

    def begin(
        self,
        stage: ReadinessStage,
        *,
        branch_count: int,
        prefix_block_count: int,
        operations: OperationCounters = _EMPTY_COUNTERS,
        detail: str | None = None,
    ) -> str:
        if len(self._stack) + len(self._closed) >= self.maximum_spans:
            raise OverflowError("span recorder reached maximum_spans")
        if not self._stack and stage is not ReadinessStage.POST_ROOT_READY:
            raise ValueError("the first span must be POST_ROOT_READY")
        if self._stack and stage is ReadinessStage.POST_ROOT_READY:
            raise ValueError("POST_ROOT_READY cannot be nested")
        span_id = f"span-{self._next_id:08d}"
        self._next_id += 1
        self._stack.append(
            _OpenSpan(
                span_id=span_id,
                parent_span_id=self._stack[-1].span_id if self._stack else None,
                stage=stage,
                start=self.clock(),
                branch_count=branch_count,
                prefix_block_count=prefix_block_count,
                operations=operations,
                detail=detail,
            )
        )
        return span_id

    def end(self, span_id: str) -> StageSpan:
        if not self._stack or self._stack[-1].span_id != span_id:
            raise ValueError("spans must close in LIFO order")
        opened = self._stack.pop()
        closed = StageSpan(
            span_id=opened.span_id,
            parent_span_id=opened.parent_span_id,
            stage=opened.stage,
            start=opened.start,
            end=self.clock(),
            branch_count=opened.branch_count,
            prefix_block_count=opened.prefix_block_count,
            operations=opened.operations,
            detail=opened.detail,
        )
        self._closed.append(closed)
        return closed

    @contextmanager
    def span(
        self,
        stage: ReadinessStage,
        *,
        branch_count: int,
        prefix_block_count: int,
        operations: OperationCounters = _EMPTY_COUNTERS,
        detail: str | None = None,
    ) -> Iterator[str]:
        span_id = self.begin(
            stage,
            branch_count=branch_count,
            prefix_block_count=prefix_block_count,
            operations=operations,
            detail=detail,
        )
        try:
            yield span_id
        finally:
            self.end(span_id)

    def freeze(self) -> SpanHierarchy:
        if self._stack:
            raise ValueError("cannot freeze a recorder with open spans")
        return SpanHierarchy(tuple(sorted(self._closed, key=lambda span: span.start.wall_ns)))


__all__ = [
    "ADDITIVE_STAGES",
    "ExclusiveSpanRecorder",
    "MetadataOperation",
    "OperationCounters",
    "ReadinessStage",
    "SpanHierarchy",
    "StageSpan",
    "SystemTimingClock",
    "TimingDecomposition",
    "TimingDuration",
    "TimingPoint",
    "TimingTolerance",
    "hierarchy_from_vllm_observation",
]
