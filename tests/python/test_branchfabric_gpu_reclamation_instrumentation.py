from __future__ import annotations

import subprocess
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_reclamation_instrumentation import (
    CommandStatus,
    CudaEventRecorder,
    CudaOperationKind,
    CudaOperationRecord,
    GpuDescriptor,
    HostAllocationLedger,
    HostMemoryKind,
    InstrumentationUnavailable,
    MetricName,
    NvmlProbe,
    SamplingConfig,
    TopologyCommand,
    TopologyCommandName,
    TopologyValidationError,
    TraceCollectionLevel,
    TraceOverheadTrial,
    assess_trace_overhead,
    capture_cuda_peer_access,
    capture_topology_commands,
    summarize_cuda_operations,
    topology_command_plan,
    validate_topology_capture,
)


@dataclass
class _Memory:
    total: int
    used: int
    free: int


class _FakeNvml:
    NVML_PCIE_UTIL_RX_BYTES = 0
    NVML_PCIE_UTIL_TX_BYTES = 1
    NVML_VALUE_NOT_AVAILABLE = 2**64 - 1

    def __init__(self, *, pcie_available: bool = True) -> None:
        self.initialized = False
        self.shutdown = False
        self.pcie_available = pcie_available

    def nvmlInit(self) -> None:
        self.initialized = True

    def nvmlShutdown(self) -> None:
        self.shutdown = True

    def nvmlDeviceGetCount(self) -> int:
        return 2

    def nvmlSystemGetDriverVersion(self) -> bytes:
        return b"570.86.15"

    def nvmlSystemGetCudaDriverVersion_v2(self) -> int:
        return 12080

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetUUID(self, handle: int) -> bytes:
        return f"GPU-fixture-{handle}".encode()

    def nvmlDeviceGetName(self, _handle: int) -> bytes:
        return b"NVIDIA A100 80GB PCIe"

    def nvmlDeviceGetPciInfo(self, handle: int) -> SimpleNamespace:
        return SimpleNamespace(busId=f"00000000:{handle + 1:02x}:00.0".encode())

    def nvmlDeviceGetMemoryInfo(self, _handle: int) -> _Memory:
        gib = 1024**3
        return _Memory(total=80 * gib, used=20 * gib, free=60 * gib)

    def nvmlDeviceGetUtilizationRates(self, _handle: int) -> SimpleNamespace:
        return SimpleNamespace(gpu=73, memory=42)

    def nvmlDeviceGetComputeRunningProcesses(self, handle: int) -> list[SimpleNamespace]:
        if handle == 0:
            return [SimpleNamespace(pid=100, usedGpuMemory=4096)]
        return []

    def nvmlDeviceGetPcieThroughput(self, _handle: int, counter: int) -> int:
        if not self.pcie_available:
            raise RuntimeError("Not Supported")
        return 125 if counter == self.NVML_PCIE_UTIL_RX_BYTES else 250


def test_nvml_probe_captures_two_gpu_identity_and_samples_without_zero_fallback() -> None:
    fake = _FakeNvml()
    ticks = iter(range(100, 1000))
    with NvmlProbe(nvml_module=fake, clock_ns=lambda: next(ticks)) as probe:
        inventory = probe.inventory()
        inventory.require_exact_gpu_count(2)
        samples = probe.sample(sequence=0, descriptors=inventory.devices, require_pcie=True)

    assert fake.initialized and fake.shutdown
    assert inventory.driver_version == "570.86.15"
    assert inventory.cuda_driver_version == "12.8"
    assert [device.uuid for device in inventory.devices] == [
        "GPU-fixture-0",
        "GPU-fixture-1",
    ]
    assert samples[0].pcie_rx_bytes_per_second == 125 * 1024
    assert samples[0].pcie_tx_bytes_per_second == 250 * 1024
    assert samples[0].compute_processes[0].used_gpu_memory_bytes == 4096
    assert not samples[0].unavailable_metrics


def test_nvml_probe_records_unavailable_pcie_and_required_policy_fails() -> None:
    fake = _FakeNvml(pcie_available=False)
    with NvmlProbe(nvml_module=fake) as probe:
        inventory = probe.inventory()
        samples = probe.sample(sequence=0, descriptors=inventory.devices)
        with pytest.raises(InstrumentationUnavailable, match="required NVML PCIe"):
            probe.sample(sequence=1, descriptors=inventory.devices, require_pcie=True)

    assert samples[0].pcie_rx_bytes_per_second is None
    assert {item.metric for item in samples[0].unavailable_metrics} == {
        MetricName.PCIE_RX_BYTES_PER_SECOND,
        MetricName.PCIE_TX_BYTES_PER_SECOND,
    }


def test_nvml_inventory_gpu_count_is_exact() -> None:
    with NvmlProbe(nvml_module=_FakeNvml()) as probe:
        inventory = probe.inventory()
    with pytest.raises(InstrumentationUnavailable, match="exactly 1"):
        inventory.require_exact_gpu_count(1)


def test_topology_plan_is_bounded_and_captures_p2p_both_directions() -> None:
    plan = topology_command_plan(timeout_seconds=7.0)
    assert {command.name for command in plan if command.required} == {
        TopologyCommandName.GPU_INVENTORY,
        TopologyCommandName.TOPOLOGY_MATRIX,
        TopologyCommandName.P2P_READ,
        TopologyCommandName.P2P_WRITE,
    }
    assert all(command.timeout_seconds == 7.0 for command in plan)
    assert all(command.argv[0] == "nvidia-smi" for command in plan)


def test_topology_capture_preserves_raw_output_and_validates_exactly_two_gpus() -> None:
    plan = topology_command_plan()
    inventory = (
        "0, GPU-fixture-0, NVIDIA A100 80GB PCIe, 00000000:01:00.0, 81920, 570.86\n"
        "1, GPU-fixture-1, NVIDIA A100 80GB PCIe, 00000000:02:00.0, 81920, 570.86\n"
    )

    def runner(argv, **_kwargs):
        stdout = inventory if "--query-gpu" in argv[1] else "fixture raw topology\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    ticks = iter(range(10, 100))
    results = capture_topology_commands(plan=plan, runner=runner, clock_ns=lambda: next(ticks))
    validate_topology_capture(results, expected_gpu_count=2)

    assert all(result.status is CommandStatus.SUCCESS for result in results)
    assert results[1].stdout == "fixture raw topology\n"
    with pytest.raises(TopologyValidationError, match="expected exactly 3"):
        validate_topology_capture(results, expected_gpu_count=3)


def test_topology_matrix_failure_accepts_complete_p2p_nvlink_fallback() -> None:
    plan = topology_command_plan()
    inventory = (
        "0, GPU-fixture-0, NVIDIA A100-SXM4-80GB, [Unknown Error], 81920, 580.95\n"
        "1, GPU-fixture-1, NVIDIA A100-SXM4-80GB, [Unknown Error], 81920, 580.95\n"
    )

    def runner(argv, **_kwargs):
        if "--query-gpu" in argv[1]:
            return subprocess.CompletedProcess(argv, 0, stdout=inventory, stderr="")
        if tuple(argv[-2:]) == ("topo", "-m"):
            return subprocess.CompletedProcess(
                argv,
                255,
                stdout="GPU0 GPU1 CPU Affinity\nFailed to run topology matrix\n",
                stderr="",
            )
        if "-p2p" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="GPU0 GPU1\nGPU0 X OK\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="Link 0: 25 GB/s\n", stderr="")

    results = capture_topology_commands(plan=plan, runner=runner)
    validate_topology_capture(results, expected_gpu_count=2)
    assert results[1].status is CommandStatus.NONZERO_EXIT
    assert results[1].returncode == 255


def test_topology_capture_returns_required_command_failure_for_persistence() -> None:
    command = TopologyCommand(
        name=TopologyCommandName.GPU_INVENTORY,
        argv=("missing-nvidia-smi",),
        timeout_seconds=1.0,
        required=True,
    )

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("missing fixture executable")

    results = capture_topology_commands(plan=(command,), runner=missing)
    assert results[0].status is CommandStatus.EXECUTABLE_UNAVAILABLE
    with pytest.raises(TopologyValidationError, match=r"missing=.*topology_matrix"):
        validate_topology_capture(results, expected_gpu_count=2)


class _FakeCudaEvent:
    def __init__(self, elapsed_ms: float) -> None:
        self.elapsed_ms = elapsed_ms
        self.stream = None
        self.synchronized = False

    def record(self, stream: object) -> None:
        self.stream = stream

    def synchronize(self) -> None:
        self.synchronized = True

    def elapsed_time(self, _end: object) -> float:
        return self.elapsed_ms


class _FakeCuda:
    def __init__(self) -> None:
        self.events: list[_FakeCudaEvent] = []

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 2

    def device(self, _index: int):
        return nullcontext()

    def current_stream(self, index: int) -> str:
        return f"stream:{index}"

    def Event(self, *, enable_timing: bool) -> _FakeCudaEvent:
        assert enable_timing
        event = _FakeCudaEvent(1.25)
        self.events.append(event)
        return event

    def get_device_name(self, index: int) -> str:
        return f"A100 fixture {index}"

    def can_device_access_peer(self, source: int, destination: int) -> bool:
        return source != destination


def test_cuda_event_recorder_records_gpu_launch_sync_and_copy_distribution() -> None:
    cuda = _FakeCuda()
    ticks = iter((100, 110, 130, 135, 170))
    recorder = CudaEventRecorder(
        logical_device=1,
        torch_module=SimpleNamespace(cuda=cuda),
        clock_ns=lambda: next(ticks),
    )
    result = recorder.measure(
        operation_id="d2h.chunk-pair",
        kind=CudaOperationKind.D2H,
        operation=lambda: "copied",
        stream_id="copy-stream-1",
        bytes=3072,
        copy_sizes_bytes=(1024, 2048),
    )

    assert result == "copied"
    assert recorder.records[0].cuda_event_elapsed_ns == 1_250_000
    assert recorder.records[0].cpu_launch_ns == 20
    assert recorder.records[0].synchronization_wait_ns == 35
    assert recorder.records[0].copy_count == 2
    assert all(event.stream == "stream:1" for event in cuda.events)


def test_disabled_cuda_event_recorder_executes_without_events_or_clock_reads() -> None:
    cuda = _FakeCuda()
    recorder = CudaEventRecorder(
        logical_device=0,
        torch_module=SimpleNamespace(cuda=cuda),
        clock_ns=lambda: (_ for _ in ()).throw(AssertionError("clock must not be read")),
        enabled=False,
    )

    assert (
        recorder.measure(
            operation_id="disabled-control",
            kind=CudaOperationKind.TRANSFORM,
            operation=lambda: "completed",
            stream_id="cuda:0/default",
        )
        == "completed"
    )
    assert recorder.records == ()
    assert cuda.events == []


def test_cuda_operation_summary_has_exact_counts_sizes_and_directional_bytes() -> None:
    def record(operation_id: str, kind: CudaOperationKind, sizes: tuple[int, ...]):
        return CudaOperationRecord(
            operation_id=operation_id,
            kind=kind,
            logical_device=1,
            stream_id="stream-1",
            cpu_start_monotonic_ns=0,
            cpu_end_monotonic_ns=10,
            cpu_launch_ns=2,
            synchronization_wait_ns=3,
            cuda_event_elapsed_ns=4,
            bytes=sum(sizes),
            copy_sizes_bytes=sizes,
        )

    summary = summarize_cuda_operations(
        (
            record("d2h", CudaOperationKind.D2H, (1024, 2048)),
            record("h2d", CudaOperationKind.H2D, (4096,)),
        )
    )
    assert summary.operation_count == 2
    assert summary.copy_count == 3
    assert summary.bytes_by_kind[CudaOperationKind.D2H] == 3072
    assert summary.bytes_by_kind[CudaOperationKind.H2D] == 4096
    assert summary.copy_size_min_bytes == 1024
    assert summary.copy_size_p50_bytes == 2048
    assert summary.copy_size_max_bytes == 4096


def test_cuda_copy_record_rejects_byte_count_that_does_not_match_sizes() -> None:
    with pytest.raises(ValidationError, match="sum exactly"):
        CudaOperationRecord(
            operation_id="bad-copy",
            kind=CudaOperationKind.D2H,
            logical_device=1,
            stream_id="stream-1",
            cpu_start_monotonic_ns=0,
            cpu_end_monotonic_ns=10,
            cpu_launch_ns=2,
            synchronization_wait_ns=3,
            cuda_event_elapsed_ns=4,
            bytes=100,
            copy_sizes_bytes=(99,),
        )


def test_cuda_peer_access_requires_exactly_two_visible_devices() -> None:
    cuda = _FakeCuda()
    matrix = capture_cuda_peer_access(
        expected_gpu_count=2,
        torch_module=SimpleNamespace(cuda=cuda),
        clock_ns=lambda: 123,
    )
    assert matrix.device_count == 2
    assert len(matrix.access) == 2
    assert all(record.can_access_peer for record in matrix.access)
    with pytest.raises(InstrumentationUnavailable, match="expected exactly 1"):
        capture_cuda_peer_access(
            expected_gpu_count=1,
            torch_module=SimpleNamespace(cuda=cuda),
        )


def test_host_allocation_ledger_tracks_pinned_pageable_peak_and_cleanup() -> None:
    ledger = HostAllocationLedger()
    ledger.allocate(
        allocation_id="pinned-a",
        kind=HostMemoryKind.PINNED,
        purpose="transport",
        bytes=1024,
        timestamp_ns=10,
    )
    ledger.allocate(
        allocation_id="pinned-b",
        kind=HostMemoryKind.PINNED,
        purpose="double-buffer",
        bytes=2048,
        timestamp_ns=20,
    )
    ledger.allocate(
        allocation_id="pageable-a",
        kind=HostMemoryKind.PAGEABLE,
        purpose="checksum-metadata",
        bytes=512,
        timestamp_ns=25,
    )
    ledger.free("pinned-a", timestamp_ns=30)
    ledger.free("pinned-b", timestamp_ns=40)
    ledger.free("pageable-a", timestamp_ns=41)
    summary = ledger.summarize(require_all_freed=True)

    assert summary.allocated_bytes_by_kind[HostMemoryKind.PINNED] == 3072
    assert summary.peak_live_bytes_by_kind[HostMemoryKind.PINNED] == 3072
    assert summary.peak_live_bytes_by_kind[HostMemoryKind.PAGEABLE] == 512
    assert summary.active_bytes_by_kind == {
        HostMemoryKind.PAGEABLE: 0,
        HostMemoryKind.PINNED: 0,
    }


def test_host_allocation_cleanup_fails_explicitly_when_buffer_remains_live() -> None:
    ledger = HostAllocationLedger()
    ledger.allocate(
        allocation_id="leak",
        kind=HostMemoryKind.PINNED,
        purpose="transport",
        bytes=1024,
        timestamp_ns=10,
    )
    with pytest.raises(InstrumentationUnavailable, match="remain live"):
        ledger.summarize(require_all_freed=True)


def _overhead_trial(
    trial_id: str,
    level: TraceCollectionLevel,
    *,
    latency_ns: int,
    throughput: float,
    copy_count: int = 2,
    copy_bytes: int = 4096,
) -> TraceOverheadTrial:
    return TraceOverheadTrial(
        trial_id=trial_id,
        seed=41,
        repetition=0,
        collection_level=level,
        workload_fingerprint="a" * 64,
        reclamation_interruption_ns=latency_ns,
        serving_throughput_tokens_per_second=throughput,
        state_copy_count=copy_count,
        state_copy_bytes=copy_bytes,
    )


def test_trace_overhead_assessment_rejects_material_timing_or_behavior_change() -> None:
    disabled = _overhead_trial(
        "disabled", TraceCollectionLevel.DISABLED, latency_ns=100, throughput=100.0
    )
    minimal = _overhead_trial(
        "minimal", TraceCollectionLevel.MINIMAL, latency_ns=104, throughput=97.0
    )
    full = _overhead_trial(
        "full",
        TraceCollectionLevel.FULL,
        latency_ns=106,
        throughput=100.0,
        copy_count=3,
    )

    minimal_result = assess_trace_overhead(disabled, minimal)
    full_result = assess_trace_overhead(disabled, full)
    assert minimal_result.reclamation_latency_overhead_fraction == pytest.approx(0.04)
    assert minimal_result.serving_throughput_degradation_fraction == pytest.approx(0.03)
    assert not minimal_result.materially_changes_trial
    assert full_result.copy_behavior_changed
    assert full_result.materially_changes_trial


def test_sampling_config_is_bounded() -> None:
    with pytest.raises(ValidationError):
        SamplingConfig(interval_ms=1, max_samples=1)


def test_gpu_descriptor_rejects_non_gpu_uuid() -> None:
    with pytest.raises(ValidationError):
        GpuDescriptor(
            nvml_index=0,
            uuid="fixture-0",
            model="A100",
            pci_bus_id="0000:01:00.0",
            memory_total_bytes=1024,
        )
