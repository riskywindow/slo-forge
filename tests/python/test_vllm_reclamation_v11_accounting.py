from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import sloforge.continuum.adapters.vllm_reclamation_v11_accounting as accounting
from sloforge.continuum.adapters.vllm_reclamation_v11_accounting import (
    V11_DEFAULT_DIAGNOSTIC_EMERGENCY_RESERVE,
    V11_DEFAULT_MAXIMUM_PENDING_CUDA_PASSES,
    V11_DEFAULT_MAXIMUM_TOTAL_PASSES,
    V11PassSpec,
    V11StatePassRecord,
    V11StatePassRecorder,
    canonical_pass_totals,
    canonical_physical_bytes_by_tier,
    validate_authoritative_pass_records,
)


class _FakeCudaEvent:
    def __init__(self) -> None:
        self.fail_synchronize = False
        self.synchronize_calls = 0

    def record(self, _stream: object) -> None:
        pass

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        if self.fail_synchronize:
            raise RuntimeError("synthetic CUDA event synchronization failure")

    def elapsed_time(self, _other: object) -> float:
        return 0.005


def _install_fake_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, list[_FakeCudaEvent]]:
    stream = SimpleNamespace(cuda_stream=17)
    events: list[_FakeCudaEvent] = []

    def event_factory(*, enable_timing: bool) -> _FakeCudaEvent:
        assert enable_timing
        event = _FakeCudaEvent()
        events.append(event)
        return event

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            Event=event_factory,
            current_stream=lambda: stream,
        )
    )
    monkeypatch.setattr(accounting.importlib, "import_module", lambda _name: fake_torch)
    return stream, events


def _spec(pass_id: str = "capture:chunk-0000:host-hash") -> V11PassSpec:
    return V11PassSpec(
        pass_id=pass_id,
        operation="checksum",
        source_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
        destination_representation="SHA256_DIGEST",
        source_memory_tier="HOST_PINNED",
        destination_memory_tier="HOST_PAGEABLE_METADATA",
        logical_bytes=128,
        physical_bytes_read=128,
        physical_bytes_written=32,
        temporary_bytes=64,
        chunk_id="chunk-0000",
        branch_group="branch-group-004",
        required_avoidable="REQUIRED",
    )


def test_cpu_recorder_emits_complete_canonical_record_at_execution_site() -> None:
    recorder = V11StatePassRecorder(device="cuda:0@GPU-test", branch_group="branch-group-004")
    with recorder.cpu_pass(_spec()):
        bytes(range(32))
    records = recorder.resolve()
    validate_authoritative_pass_records(
        records,
        expected_device="cuda:0@GPU-test",
        expected_branch_group="branch-group-004",
    )
    record = records[0]
    assert record.device == "cpu"
    assert record.cuda_stream is None
    assert record.wall_end_ns >= record.wall_start_ns
    assert record.as_dict()["physical_touch_bytes"] == 160
    assert canonical_pass_totals(records) == {
        "logical_bytes": 128,
        "physical_bytes_read": 128,
        "physical_bytes_written": 32,
        "external_transfer_bytes": 0,
        "full_physical_bytes": 160,
        "required_physical_bytes": 160,
        "avoidable_physical_bytes": 0,
        "diagnostic_physical_bytes": 0,
        "temporary_bytes_sum_not_peak": 64,
        "cuda_time_ns_timing_owners": 0,
        "synchronization_wait_ns": 0,
    }
    assert canonical_physical_bytes_by_tier(records) == {
        "HOST_PAGEABLE_METADATA": 32,
        "HOST_PINNED": 128,
    }


def test_recorder_rejects_duplicate_pass_id_and_post_resolution_append() -> None:
    recorder = V11StatePassRecorder(device="cpu", branch_group="branch-group-004")
    with recorder.cpu_pass(_spec()):
        pass
    with pytest.raises(ValueError, match="duplicate"), recorder.cpu_pass(_spec()):
        pass
    recorder.resolve()
    with (
        pytest.raises(RuntimeError, match="already resolved"),
        recorder.cpu_pass(_spec("capture:chunk-0001:host-hash")),
    ):
        pass


def test_cuda_record_requires_events_stream_and_elapsed_time() -> None:
    kwargs = dict(
        pass_id="restore:chunk-0000:h2d",
        operation="h2d",
        source_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
        destination_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
        source_memory_tier="HOST_PINNED",
        destination_memory_tier="GPU_HBM_TEMPORARY",
        logical_bytes=128,
        physical_bytes_read=128,
        physical_bytes_written=128,
        temporary_bytes=128,
        device="cuda:0@GPU-test",
        cuda_stream=None,
        start_cuda_event=None,
        end_cuda_event=None,
        wall_start_ns=1,
        wall_end_ns=2,
        synchronization_dependency=(),
        chunk_id="chunk-0000",
        branch_group="branch-group-004",
        required_avoidable="REQUIRED",
        external_transfer_bytes=128,
        cuda_elapsed_ns=None,
    )
    with pytest.raises(ValueError, match="requires stream"):
        V11StatePassRecord(**kwargs)


def test_timing_group_has_exactly_one_owner_and_shared_interval() -> None:
    recorder = V11StatePassRecorder(device="cpu", branch_group="branch-group-004")
    with recorder.cpu_pass(_spec("a")):
        pass
    with recorder.cpu_pass(_spec("b")):
        pass
    first, second = recorder.resolve()
    invalid = (
        replace(first, timing_group_id="shared", timing_owner=True),
        replace(second, timing_group_id="shared", timing_owner=True),
    )
    with pytest.raises(ValueError, match="exactly one timing owner"):
        validate_authoritative_pass_records(
            invalid,
            expected_device="cpu",
            expected_branch_group="branch-group-004",
        )


def test_profiler_and_copy_counter_bytes_cannot_be_added_to_authoritative_total() -> None:
    record = V11StatePassRecord(
        pass_id="restore:chunk-0000:h2d",
        operation="h2d",
        source_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
        destination_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
        source_memory_tier="HOST_PINNED",
        destination_memory_tier="GPU_HBM_TEMPORARY",
        logical_bytes=128,
        physical_bytes_read=128,
        physical_bytes_written=128,
        temporary_bytes=128,
        device="cuda:0@GPU-test",
        cuda_stream="cuda-stream-7",
        start_cuda_event="start",
        end_cuda_event="end",
        wall_start_ns=1,
        wall_end_ns=3,
        synchronization_dependency=(),
        chunk_id="chunk-0000",
        branch_group="branch-group-004",
        required_avoidable="REQUIRED",
        external_transfer_bytes=128,
        cuda_elapsed_ns=1,
        timing_group_id="restore-h2d-0000",
        timing_owner=True,
    )
    validate_authoritative_pass_records(
        (record,),
        expected_device="cuda:0@GPU-test",
        expected_branch_group="branch-group-004",
    )
    profiler_observation = {"pass_id": record.pass_id, "physical_bytes": 384}
    copy_counter_observation = {"pass_id": record.pass_id, "bytes": 128}
    totals = canonical_pass_totals((record,))
    assert totals["full_physical_bytes"] == 384
    assert profiler_observation["physical_bytes"] == totals["full_physical_bytes"]
    assert copy_counter_observation["bytes"] == totals["external_transfer_bytes"]


def test_authoritative_ledger_rejects_unknown_and_cyclic_dependencies() -> None:
    recorder = V11StatePassRecorder(device="cpu", branch_group="branch-group-004")
    with recorder.cpu_pass(_spec("a")):
        pass
    with recorder.cpu_pass(replace(_spec("b"), synchronization_dependency=("a",))):
        pass
    first, second = recorder.resolve()
    validate_authoritative_pass_records(
        (first, second),
        expected_device="cpu",
        expected_branch_group="branch-group-004",
    )
    with pytest.raises(ValueError, match="unknown synchronization"):
        validate_authoritative_pass_records(
            (replace(first, synchronization_dependency=("missing",)),),
            expected_device="cpu",
            expected_branch_group="branch-group-004",
        )
    cyclic = (
        replace(first, synchronization_dependency=("b",)),
        replace(second, synchronization_dependency=("a",)),
    )
    with pytest.raises(ValueError, match="cycle"):
        validate_authoritative_pass_records(
            cyclic,
            expected_device="cpu",
            expected_branch_group="branch-group-004",
        )


def test_mixed_required_avoidable_and_tier_bytes_conserve_exactly() -> None:
    recorder = V11StatePassRecorder(device="cpu", branch_group="branch-group-004")
    mixed = replace(
        _spec("native-gather"),
        source_memory_tier="GPU_HBM_NATIVE_POOL",
        destination_memory_tier="GPU_HBM_TEMPORARY",
        physical_bytes_read=128,
        physical_bytes_written=128,
        required_avoidable="MIXED",
        required_physical_bytes=128,
        avoidable_physical_bytes=128,
        diagnostic_physical_bytes=0,
        physical_bytes_by_tier=(
            ("GPU_HBM_NATIVE_POOL", 128),
            ("GPU_HBM_TEMPORARY", 128),
        ),
    )
    with recorder.cpu_pass(mixed):
        pass
    records = recorder.resolve()
    totals = canonical_pass_totals(records)
    assert totals["full_physical_bytes"] == 256
    assert totals["required_physical_bytes"] == 128
    assert totals["avoidable_physical_bytes"] == 128
    assert totals["diagnostic_physical_bytes"] == 0
    assert canonical_physical_bytes_by_tier(records) == {
        "GPU_HBM_NATIVE_POOL": 128,
        "GPU_HBM_TEMPORARY": 128,
    }
    with pytest.raises(ValueError, match="tier bytes"):
        replace(mixed, physical_bytes_by_tier=(("GPU_HBM_NATIVE_POOL", 127),))


def test_cuda_record_separates_launch_wall_interval_event_time_and_resolve_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Event:
        def record(self, _stream: object) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, _other: object) -> float:
            return 0.005

    stream = SimpleNamespace(cuda_stream=17)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            Event=lambda enable_timing: _Event(),
            current_stream=lambda: stream,
        )
    )
    monkeypatch.setattr(accounting.importlib, "import_module", lambda _name: fake_torch)
    timestamps = iter((100, 110, 200, 230))
    monkeypatch.setattr(accounting.time, "monotonic_ns", lambda: next(timestamps))
    recorder = V11StatePassRecorder(device="cuda:0@GPU-test", branch_group="branch-group-004")
    spec = replace(
        _spec("h2d"),
        operation="h2d",
        source_memory_tier="HOST_PINNED",
        destination_memory_tier="GPU_HBM_TEMPORARY",
        physical_bytes_written=128,
        external_transfer_bytes=128,
    )
    with recorder.cuda_pass(spec, stream=stream):
        pass
    record = recorder.resolve()[0]
    assert (record.wall_start_ns, record.wall_end_ns) == (100, 110)
    assert record.cuda_elapsed_ns == 5_000
    assert record.synchronization_wait_ns == 30
    assert record.start_cuda_event is not None and record.end_cuda_event is not None


def test_incremental_cuda_materialization_is_exact_once_and_final_resolve_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, events = _install_fake_cuda(monkeypatch)
    recorder = V11StatePassRecorder(
        device="cuda:0@GPU-test",
        branch_group="branch-group-004",
        maximum_pending_cuda_passes=4,
        maximum_total_passes=32,
        diagnostic_emergency_reserve=1,
    )
    specs = tuple(
        replace(
            _spec(f"cuda-{index}"),
            synchronization_dependency=((f"cuda-{index - 1}",) if index else ()),
        )
        for index in range(8)
    )
    for spec in specs:
        with recorder.cuda_pass(spec, stream=stream):
            pass

    # Two non-final drains occurred at the three-pass normal-lane threshold,
    # but the recorder remains appendable and retains cross-batch identities.
    assert len(recorder._records) == 6
    assert len(recorder._pending) == 2
    assert not recorder._resolved
    with pytest.raises(RuntimeError, match="must be resolved"):
        _ = recorder.records
    with pytest.raises(ValueError, match="duplicate"), recorder.cuda_pass(specs[0], stream=stream):
        pass

    records = recorder.resolve()
    assert tuple(item.pass_id for item in records) == tuple(item.pass_id for item in specs)
    assert len(records) == len({item.pass_id for item in records}) == 8
    assert recorder.resolve() == records
    assert all(event.synchronize_calls == 1 for event in events[1::2])
    validate_authoritative_pass_records(
        records,
        expected_device="cuda:0@GPU-test",
        expected_branch_group="branch-group-004",
    )
    assert canonical_pass_totals(records)["full_physical_bytes"] == 8 * 160


def test_incremental_sync_failure_retains_only_unmaterialized_suffix_and_retries_exact_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, _events = _install_fake_cuda(monkeypatch)
    recorder = V11StatePassRecorder(
        device="cuda:0@GPU-test",
        branch_group="branch-group-004",
        maximum_pending_cuda_passes=3,
        maximum_total_passes=16,
        diagnostic_emergency_reserve=1,
    )
    first = _spec("first")
    second = replace(_spec("second"), synchronization_dependency=("first",))
    third = replace(_spec("third"), synchronization_dependency=("second",))
    with recorder.cuda_pass(first, stream=stream):
        pass
    with recorder.cuda_pass(second, stream=stream):
        pass
    first_end = recorder._pending[0].end_event
    second_end = recorder._pending[1].end_event
    second_end.fail_synchronize = True

    with (
        pytest.raises(RuntimeError, match="synthetic CUDA event"),
        recorder.cuda_pass(third, stream=stream),
    ):
        pass
    assert tuple(item.pass_id for item in recorder._records) == ("first",)
    assert tuple(item.spec.pass_id for item in recorder._pending) == ("second",)
    assert "third" not in recorder._pass_ids
    assert first_end.synchronize_calls == 1
    assert second_end.synchronize_calls == 1

    second_end.fail_synchronize = False
    with recorder.cuda_pass(third, stream=stream):
        pass
    records = recorder.resolve()
    assert tuple(item.pass_id for item in records) == ("first", "second", "third")
    assert first_end.synchronize_calls == 1
    assert second_end.synchronize_calls == 2


def test_diagnostic_emergency_lane_is_nonborrowable_and_fails_closed_when_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, _events = _install_fake_cuda(monkeypatch)
    recorder = V11StatePassRecorder(
        device="cuda:0@GPU-test",
        branch_group="branch-group-004",
        maximum_pending_cuda_passes=4,
        maximum_total_passes=8,
        diagnostic_emergency_reserve=2,
    )
    with recorder.cuda_pass(_spec("normal-0"), stream=stream):
        pass
    with recorder.cuda_pass(_spec("normal-1"), stream=stream):
        pass
    blocked_event = recorder._pending[0].end_event
    blocked_event.fail_synchronize = True

    with (
        pytest.raises(RuntimeError, match="synthetic CUDA event"),
        recorder.cuda_pass(_spec("normal-blocked"), stream=stream),
    ):
        pass
    diagnostic = replace(_spec("scrub-0"), required_avoidable="DIAGNOSTIC")
    with recorder.cuda_pass(diagnostic, stream=stream):
        pass
    with recorder.cuda_pass(replace(diagnostic, pass_id="scrub-1"), stream=stream):
        pass
    assert len(recorder._pending) == recorder.maximum_pending_cuda_passes

    with (
        pytest.raises(RuntimeError, match="emergency CUDA-pass reserve exhausted"),
        recorder.cuda_pass(replace(diagnostic, pass_id="scrub-exhausted"), stream=stream),
    ):
        pass
    assert "scrub-exhausted" not in recorder._pass_ids
    with pytest.raises(RuntimeError, match="synthetic CUDA event"):
        recorder.resolve()
    assert tuple(item.spec.pass_id for item in recorder._pending) == (
        "normal-0",
        "normal-1",
        "scrub-0",
        "scrub-1",
    )


def test_total_record_bound_preserves_diagnostic_reserve_for_cpu_or_cuda_cleanup() -> None:
    recorder = V11StatePassRecorder(
        device="cpu",
        branch_group="branch-group-004",
        maximum_pending_cuda_passes=4,
        maximum_total_passes=4,
        diagnostic_emergency_reserve=2,
    )
    with recorder.cpu_pass(_spec("normal-0")):
        pass
    with recorder.cpu_pass(_spec("normal-1")):
        pass
    with (
        pytest.raises(RuntimeError, match="diagnostic reserve is protected"),
        recorder.cpu_pass(_spec("normal-blocked")),
    ):
        pass
    diagnostic = replace(_spec("scrub-0"), required_avoidable="DIAGNOSTIC")
    with recorder.cpu_pass(diagnostic):
        pass
    with recorder.cpu_pass(replace(diagnostic, pass_id="scrub-1")):
        pass
    with (
        pytest.raises(RuntimeError, match="emergency total-pass reserve exhausted"),
        recorder.cpu_pass(replace(diagnostic, pass_id="scrub-exhausted")),
    ):
        pass
    assert len(recorder.resolve()) == 4


def test_production_bounds_cover_attempt_c_and_singleton_run_restore_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layers = 28
    first_subset_cuda_passes = 48 * (1 + layers * 3)
    remaining_subset_cuda_passes = 16 * (1 + layers * (2 + 7))
    assert first_subset_cuda_passes == 4080
    assert remaining_subset_cuda_passes == 4048
    assert first_subset_cuda_passes + remaining_subset_cuda_passes == 8128

    pages = 1152
    subset_chunks = 64
    singleton_run_restore_cuda_passes = pages * layers + subset_chunks * (1 + 2 * layers)
    assert singleton_run_restore_cuda_passes == 35904
    assert singleton_run_restore_cuda_passes > V11_DEFAULT_MAXIMUM_PENDING_CUDA_PASSES
    assert (
        singleton_run_restore_cuda_passes
        <= V11_DEFAULT_MAXIMUM_TOTAL_PASSES - V11_DEFAULT_DIAGNOSTIC_EMERGENCY_RESERVE
    )
    # One scrub has one index upload, one zero and one verify per layer, plus a
    # CPU fence in the total ledger.  Even the historical nested path fits the
    # protected lane; the production stager now owns one authoritative scrub.
    cuda_passes_per_scrub = 1 + layers + layers
    total_passes_per_scrub = cuda_passes_per_scrub + 1
    assert 2 * total_passes_per_scrub <= V11_DEFAULT_DIAGNOSTIC_EMERGENCY_RESERVE

    stream, events = _install_fake_cuda(monkeypatch)
    recorder = V11StatePassRecorder(
        device="cuda:0@GPU-test",
        branch_group="branch-group-004",
    )
    for index in range(singleton_run_restore_cuda_passes):
        with recorder.cuda_pass(_spec(f"restore-pass-{index:05d}"), stream=stream):
            pass
    records = recorder.resolve()
    assert len(records) == singleton_run_restore_cuda_passes
    assert len({item.pass_id for item in records}) == singleton_run_restore_cuda_passes
    assert all(event.synchronize_calls == 1 for event in events[1::2])


def test_cuda_event_setup_failure_rolls_back_pass_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = SimpleNamespace(cuda_stream=17)
    event_calls = 0

    def event_factory(*, enable_timing: bool) -> _FakeCudaEvent:
        nonlocal event_calls
        assert enable_timing
        event_calls += 1
        if event_calls == 2:
            raise RuntimeError("synthetic CUDA event construction failure")
        return _FakeCudaEvent()

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(Event=event_factory, current_stream=lambda: stream)
    )
    monkeypatch.setattr(accounting.importlib, "import_module", lambda _name: fake_torch)
    recorder = V11StatePassRecorder(device="cuda:0@GPU-test", branch_group="branch-group-004")
    spec = _spec("event-setup-failure")
    with (
        pytest.raises(RuntimeError, match="event construction failure"),
        recorder.cuda_pass(spec, stream=stream),
    ):
        pass
    assert spec.pass_id not in recorder._pass_ids
    assert not recorder._pending
