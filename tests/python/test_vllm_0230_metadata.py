from __future__ import annotations

import pytest

from sloforge.continuum.adapters.vllm_0230_metadata import (
    ADDITIVE_STAGES,
    ExclusiveSpanRecorder,
    MetadataOperation,
    OperationCounters,
    ReadinessStage,
    SpanHierarchy,
    StageSpan,
    TimingDuration,
    TimingPoint,
    TimingTolerance,
)


class _Clock:
    def __init__(self, points: list[tuple[int, int, int | None]]) -> None:
        self._points = iter(TimingPoint(*point) for point in points)

    def __call__(self) -> TimingPoint:
        return next(self._points)


def _span(
    span_id: str,
    parent: str | None,
    stage: ReadinessStage,
    start: int,
    end: int,
    *,
    process_start: int | None = None,
    process_end: int | None = None,
) -> StageSpan:
    return StageSpan(
        span_id=span_id,
        parent_span_id=parent,
        stage=stage,
        start=TimingPoint(start, start if process_start is None else process_start, start),
        end=TimingPoint(end, end if process_end is None else process_end, end),
        branch_count=8,
        prefix_block_count=1024,
    )


def test_exclusive_decomposition_partitions_parent_and_bills_every_gap_to_residual() -> None:
    hierarchy = SpanHierarchy(
        (
            _span("root", None, ReadinessStage.POST_ROOT_READY, 0, 100),
            _span("helix", "root", ReadinessStage.HELIX_ORCHESTRATION, 0, 10),
            _span("bind", "root", ReadinessStage.PREFIX_STATE_BIND, 20, 60),
            _span("lookup", "bind", ReadinessStage.PREFIX_LOOKUP, 20, 35),
            _span("refs", "bind", ReadinessStage.REFCOUNT_UPDATE, 40, 50),
            _span("gpu", "root", ReadinessStage.GPU_EXECUTION, 70, 90),
        )
    )

    decomposition = hierarchy.decomposition()
    values = dict(decomposition.stages)
    assert tuple(values) == ADDITIVE_STAGES
    assert values[ReadinessStage.HELIX_ORCHESTRATION].wall_ns == 10
    assert values[ReadinessStage.PREFIX_LOOKUP].wall_ns == 15
    assert values[ReadinessStage.REFCOUNT_UPDATE].wall_ns == 10
    assert values[ReadinessStage.PREFIX_METADATA_OTHER].wall_ns == 15
    assert values[ReadinessStage.GPU_EXECUTION].wall_ns == 20
    # Root gaps: [10,20), [60,70), and [90,100).
    assert values[ReadinessStage.RESIDUAL].wall_ns == 30
    assert sum(duration.wall_ns for duration in values.values()) == 100
    assert decomposition.as_dict()["invariant"]["wall_error_ns"] == 0


def test_sibling_overlap_is_rejected_instead_of_double_counted() -> None:
    with pytest.raises(ValueError, match=r"sibling spans .* overlap"):
        SpanHierarchy(
            (
                _span("root", None, ReadinessStage.POST_ROOT_READY, 0, 20),
                _span("left", "root", ReadinessStage.REQUEST_BUILD, 1, 10),
                _span("right", "root", ReadinessStage.SCHEDULER_ADMISSION, 9, 15),
            )
        )


def test_nested_inclusive_span_is_made_exclusive_before_summing() -> None:
    hierarchy = SpanHierarchy(
        (
            _span("root", None, ReadinessStage.POST_ROOT_READY, 0, 20),
            _span("select", "root", ReadinessStage.SCHEDULER_SELECT, 2, 18),
            _span("lookup", "select", ReadinessStage.PREFIX_LOOKUP, 5, 15),
        )
    )
    values = dict(hierarchy.decomposition().stages)
    assert values[ReadinessStage.SCHEDULER_SELECT].wall_ns == 6
    assert values[ReadinessStage.PREFIX_LOOKUP].wall_ns == 10
    assert values[ReadinessStage.RESIDUAL].wall_ns == 4


def test_recorder_uses_shared_boundaries_and_closes_spans_after_exceptions() -> None:
    recorder = ExclusiveSpanRecorder(
        clock=_Clock(
            [
                (0, 0, 0),
                (10, 4, 3),
                (30, 14, 11),
                (50, 20, 16),
            ]
        )
    )
    with (
        recorder.span(ReadinessStage.POST_ROOT_READY, branch_count=8, prefix_block_count=1024),
        pytest.raises(RuntimeError, match="diagnostic"),
        recorder.span(
            ReadinessStage.REQUEST_BUILD,
            branch_count=8,
            prefix_block_count=1024,
        ),
    ):
        raise RuntimeError("diagnostic")

    hierarchy = recorder.freeze()
    decomposition = hierarchy.decomposition()
    assert decomposition.parent == TimingDuration(50, 20, 16)
    values = dict(decomposition.stages)
    assert values[ReadinessStage.REQUEST_BUILD] == TimingDuration(20, 10, 8)
    assert values[ReadinessStage.RESIDUAL] == TimingDuration(30, 10, 8)


def test_recorder_is_bounded_and_requires_lifo_close() -> None:
    recorder = ExclusiveSpanRecorder(
        clock=_Clock([(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]),
        maximum_spans=2,
    )
    root = recorder.begin(ReadinessStage.POST_ROOT_READY, branch_count=1, prefix_block_count=1)
    child = recorder.begin(ReadinessStage.REQUEST_BUILD, branch_count=1, prefix_block_count=1)
    with pytest.raises(ValueError, match="LIFO"):
        recorder.end(root)
    recorder.end(child)
    recorder.end(root)
    with pytest.raises(OverflowError, match="maximum_spans"):
        recorder.begin(ReadinessStage.POST_ROOT_READY, branch_count=1, prefix_block_count=1)


def test_operation_counters_are_canonical_delta_checked_and_normalized() -> None:
    before = OperationCounters.from_mapping(
        {
            MetadataOperation.BLOCK_TABLE_WRITES: 5,
            MetadataOperation.REFCOUNT_INCREMENTS: 4,
        }
    )
    after = OperationCounters.from_mapping(
        {
            MetadataOperation.REFCOUNT_INCREMENTS: 8_200,
            MetadataOperation.BLOCK_TABLE_WRITES: 8_205,
            MetadataOperation.PRIVATE_SUFFIX_ALLOCATIONS: 8,
        }
    )
    delta = after.delta(before)
    assert list(delta.as_dict()) == sorted(delta.as_dict())
    assert delta.as_dict() == {
        "block_table_writes": 8_200,
        "private_suffix_allocations": 8,
        "refcount_increments": 8_196,
    }
    normalized = delta.normalized(
        branch_count=8,
        prefix_block_count=1024,
        token_count=16_384,
        request_count=8,
        fanout=8,
    )
    assert normalized["fanout"] == 8
    assert normalized["per_branch"]["block_table_writes"] == 1025.0
    assert normalized["per_prefix_block"]["private_suffix_allocations"] == 8 / 1024

    with pytest.raises(ValueError, match="decreased"):
        before.delta(after)


def test_parent_cross_check_uses_documented_absolute_or_relative_tolerance() -> None:
    hierarchy = SpanHierarchy((_span("root", None, ReadinessStage.POST_ROOT_READY, 0, 1_000),))
    decomposition = hierarchy.decomposition()
    tolerance = TimingTolerance(absolute_ns=10, relative_fraction=0.01)
    decomposition.assert_parent_cross_check(TimingDuration(1_009, 1_009, 1_009), tolerance)
    with pytest.raises(ValueError, match="wall parent cross-check"):
        decomposition.assert_parent_cross_check(TimingDuration(1_012, 1_012, 1_012), tolerance)


def test_thread_cpu_unavailable_is_preserved_without_fabrication() -> None:
    hierarchy = SpanHierarchy(
        (
            StageSpan(
                span_id="root",
                parent_span_id=None,
                stage=ReadinessStage.POST_ROOT_READY,
                start=TimingPoint(0, 0, None),
                end=TimingPoint(10, 5, None),
                branch_count=1,
                prefix_block_count=1,
            ),
        )
    )
    decomposition = hierarchy.decomposition()
    assert decomposition.parent.thread_cpu_ns is None
    assert decomposition.as_dict()["invariant"]["thread_cpu_error_ns"] is None
