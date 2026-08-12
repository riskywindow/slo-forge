from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sloforge.helix.characterization.gpu_reclamation_instrumentation import (
    GpuDescriptor,
    GpuSample,
    HostResourceSample,
    MetricName,
    NvmlInventory,
    ProcessResourceSample,
    ResourceSample,
    ResourceSamplingResult,
    UnavailabilityKind,
    UnavailableMetric,
)

_ROOT = Path(__file__).resolve().parents[2]
_WORKER_PATH = _ROOT / "experiments/branchfabric/gpu_reclamation_worker.py"
_SPEC = importlib.util.spec_from_file_location(
    "sloforge_experiment_004_worker_telemetry_tests",
    _WORKER_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_WORKER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _WORKER
_SPEC.loader.exec_module(_WORKER)


def _config() -> dict[str, object]:
    return {
        "schema_version": "sloforge.branchfabric.experiment-004-modal-config/v1",
        "attempt_id": "exp004-fixture",
        "seed": 41,
        "initialization_timeout_seconds": 600,
        "maximum_wall_seconds": 600,
        "cleanup_timeout_seconds": 60,
    }


def _inventory() -> NvmlInventory:
    gib = 1024**3
    return NvmlInventory(
        observed_at_monotonic_ns=1,
        driver_version="570.86.15",
        cuda_driver_version_raw=13_000,
        cuda_driver_version="13.0",
        cuda_visible_devices="GPU-fixture-0",
        devices=(
            GpuDescriptor(
                nvml_index=0,
                uuid="GPU-fixture-0",
                model="NVIDIA A100 80GB PCIe",
                pci_bus_id="00000000:01:00.0",
                memory_total_bytes=80 * gib,
            ),
            GpuDescriptor(
                nvml_index=1,
                uuid="GPU-fixture-1",
                model="NVIDIA A100 80GB PCIe",
                pci_bus_id="00000000:02:00.0",
                memory_total_bytes=80 * gib,
            ),
        ),
    )


def _resource_result(*, pid: int, uuid: str, config) -> ResourceSamplingResult:
    bandwidth_unavailable = (
        UnavailableMetric(
            metric=MetricName.HOST_MEMORY_READ_BYTES_PER_SECOND,
            provider="fixture",
            kind=UnavailabilityKind.UNSUPPORTED,
            reason="fixture has no DRAM controller counter",
        ),
        UnavailableMetric(
            metric=MetricName.HOST_MEMORY_WRITE_BYTES_PER_SECOND,
            provider="fixture",
            kind=UnavailabilityKind.UNSUPPORTED,
            reason="fixture has no DRAM controller counter",
        ),
    )
    sample = ResourceSample(
        sequence=0,
        sample_trigger_monotonic_ns=10,
        gpu_samples=(
            GpuSample(
                sequence=0,
                query_start_monotonic_ns=11,
                query_end_monotonic_ns=12,
                nvml_index=0 if uuid.endswith("0") else 1,
                uuid=uuid,
                gpu_utilization_percent=75,
                memory_utilization_percent=60,
                memory_used_bytes=20 * 1024**3,
                memory_free_bytes=60 * 1024**3,
                memory_total_bytes=80 * 1024**3,
                pcie_rx_bytes_per_second=1000,
                pcie_tx_bytes_per_second=2000,
                compute_processes=(),
                unavailable_metrics=(),
            ),
        ),
        host_sample=HostResourceSample(
            sequence=0,
            query_start_monotonic_ns=13,
            query_end_monotonic_ns=14,
            logical_cpu_count=16,
            system_cpu_user_ns=100,
            system_cpu_system_ns=50,
            system_cpu_idle_ns=1000,
            host_memory_total_bytes=512 * 1024**3,
            host_memory_available_bytes=400 * 1024**3,
            processes=(
                ProcessResourceSample(
                    pid=pid,
                    rss_bytes=1_000_000,
                    peak_rss_bytes=2_000_000,
                    cpu_user_ns=20,
                    cpu_system_ns=10,
                    thread_count=4,
                    unavailable_metrics=(),
                ),
            ),
            unavailable_metrics=bandwidth_unavailable,
        ),
    )
    return ResourceSamplingResult(config=config, samples=(sample,), reached_sample_bound=False)


class _FakeGpuProbe:
    def __init__(self, inventory: NvmlInventory) -> None:
        self._inventory = inventory
        self.closed = False

    def inventory(self) -> NvmlInventory:
        return self._inventory

    def close(self) -> None:
        self.closed = True


class _FakeSampler:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.is_running = False
        self.stop_timeout = None

    def start(self) -> None:
        self.started = True
        self.is_running = True

    def stop(self, *, timeout_seconds: float) -> ResourceSamplingResult:
        self.stop_timeout = timeout_seconds
        self.is_running = False
        pid = self.kwargs["host_probe"].pid
        uuid = self.kwargs["descriptors"][0].uuid
        return _resource_result(pid=pid, uuid=uuid, config=self.kwargs["config"])


@pytest.mark.parametrize("role", ("serving", "rollout"))
def test_both_worker_roles_write_bounded_raw_resource_telemetry(tmp_path: Path, role: str) -> None:
    gpu_probe = _FakeGpuProbe(_inventory())
    created: dict[str, object] = {}

    def sampler_factory(**kwargs):
        sampler = _FakeSampler(**kwargs)
        created["sampler"] = sampler
        return sampler

    def host_factory(*, process_ids):
        assert process_ids == (123,)
        return SimpleNamespace(pid=123)

    telemetry = _WORKER._start_worker_resource_telemetry(
        role=role,
        physical_gpu_uuid="GPU-fixture-1" if role == "rollout" else "GPU-fixture-0",
        worker_pid=123,
        work_root=tmp_path,
        config=_config(),
        gpu_probe_factory=lambda: gpu_probe,
        host_probe_factory=host_factory,
        sampler_factory=sampler_factory,
    )
    sampler = created["sampler"]
    assert sampler.started and sampler.is_running
    assert len(sampler.kwargs["descriptors"]) == 1
    assert sampler.kwargs["config"].interval_ms == 100
    assert sampler.kwargs["config"].max_samples == 12_101
    assert not sampler.kwargs["config"].require_pcie_counters

    reference = telemetry.stop(timeout_seconds=3.0)
    raw_path = tmp_path / reference["raw_provenance"]["artifact_reference"]
    payload = json.loads(raw_path.read_text())

    assert not telemetry.is_running
    assert gpu_probe.closed
    assert reference["sample_count"] == 1
    assert reference["raw_provenance"]["sample_selector"] == "$.sampling.samples[*]"
    assert (
        reference["raw_provenance"]["artifact_sha256"]
        == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    )
    assert payload["role"] == role
    assert payload["sampling"]["samples"][0]["gpu_samples"][0]["uuid"] == (
        "GPU-fixture-1" if role == "rollout" else "GPU-fixture-0"
    )
    assert payload["sampling"]["samples"][0]["host_sample"]["processes"][0]["pid"] == 123
    with pytest.raises(RuntimeError, match="already been stopped"):
        telemetry.stop(timeout_seconds=1.0)


def test_missing_assigned_gpu_closes_nvml_before_failing(tmp_path: Path) -> None:
    gpu_probe = _FakeGpuProbe(_inventory())
    with pytest.raises(RuntimeError, match="absent or duplicated"):
        _WORKER._start_worker_resource_telemetry(
            role="rollout",
            physical_gpu_uuid="GPU-not-present",
            worker_pid=123,
            work_root=tmp_path,
            config=_config(),
            gpu_probe_factory=lambda: gpu_probe,
            host_probe_factory=lambda **_kwargs: SimpleNamespace(pid=123),
            sampler_factory=_FakeSampler,
        )
    assert gpu_probe.closed


def test_sampler_timeout_remains_visible_and_does_not_race_nvml_shutdown(
    tmp_path: Path,
) -> None:
    gpu_probe = _FakeGpuProbe(_inventory())

    class StuckSampler(_FakeSampler):
        def stop(self, *, timeout_seconds: float):
            del timeout_seconds
            self.is_running = True
            raise TimeoutError("fixture sampler is stuck")

    telemetry = _WORKER._start_worker_resource_telemetry(
        role="rollout",
        physical_gpu_uuid="GPU-fixture-1",
        worker_pid=123,
        work_root=tmp_path,
        config=_config(),
        gpu_probe_factory=lambda: gpu_probe,
        host_probe_factory=lambda **_kwargs: SimpleNamespace(pid=123),
        sampler_factory=StuckSampler,
    )
    with pytest.raises(TimeoutError, match="fixture sampler is stuck"):
        telemetry.stop(timeout_seconds=0.1)
    assert telemetry.is_running
    assert not gpu_probe.closed


def test_global_serving_arrival_clock_is_positioned_at_reclaim_trigger() -> None:
    trigger_ns = 1_000_000_000
    gpu0 = _WORKER._deterministic_serving_arrivals(
        start_ns=0,
        phases=(
            (1.0, 1.0, "control"),
            (2.0, 4.0, "spike"),
            (1.0, 1.0, "restore-interference"),
        ),
        request_prefix="gpu0",
    )
    gpu1 = _WORKER._deterministic_serving_arrivals(
        start_ns=trigger_ns,
        phases=((2.0, 2.0, "spike"),),
        request_prefix="gpu1",
    )

    assert len([item for item in gpu0 if item[2] == "control"]) == 1
    assert len([item for item in gpu0 if item[2] == "spike"]) == 8
    assert len([item for item in gpu0 if item[2] == "restore-interference"]) == 1
    assert len(gpu1) == 4
    assert gpu1[0][0] == trigger_ns
    assert [item[0] for item in gpu1] == sorted(item[0] for item in gpu1)
    assert len({item[1] for item in (*gpu0, *gpu1)}) == len(gpu0) + len(gpu1)


def test_serving_continuation_is_barrier_bounded_and_extends_past_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Engine:
        def __init__(self) -> None:
            self.requests: list[str] = []

        def add_request(self, request_id: str, _prompt: object, _params: object) -> None:
            self.requests.append(request_id)

        def step(self):  # type: ignore[no-untyped-def]
            request_id = self.requests.pop(0)
            return (
                SimpleNamespace(
                    request_id=request_id,
                    outputs=(SimpleNamespace(token_ids=(7,)),),
                    finished=True,
                ),
            )

    barrier = tmp_path / "restore-complete.json"
    monotonic_value = 0

    def monotonic_ns() -> int:
        nonlocal monotonic_value
        result = monotonic_value
        monotonic_value += 10_000_000
        return result

    monkeypatch.setattr(_WORKER.time, "monotonic_ns", monotonic_ns)
    monkeypatch.setattr(_WORKER, "_sleep_until", lambda _timestamp: barrier.write_text("{}"))
    monkeypatch.setattr(_WORKER, "_sampling_params", lambda **_kwargs: object())

    result = _WORKER._run_raw_serving(
        Engine(),
        prefix=(1,),
        start_ns=0,
        phases=((0.1, 10.0, "control"),),
        output_tokens=1,
        seed=1,
        request_prefix="fixture",
        continuation_barrier=barrier,
        continuation_rate_per_second=10.0,
        continuation_grace_seconds=0.2,
        continuation_timeout_seconds=1.0,
    )

    continuation = [row for row in result["requests"] if row["phase"] == "restore-interference"]
    assert continuation
    assert result["continuation_barrier_observed_ns"] is not None
