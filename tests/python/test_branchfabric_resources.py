from __future__ import annotations

import time
from pathlib import Path

import pytest

from sloforge.helix.characterization.matrix import EvidenceClass, TraceLevel
from sloforge.helix.characterization.resources import (
    CounterCapability,
    PlatformCapabilities,
    ResourceSampler,
    ResourceSamplerConfig,
    ResourceSamplerState,
    TimingProvenance,
    collect_resource_trace,
    detect_platform_capabilities,
    generate_profiler_commands,
)


def _config(
    level: TraceLevel,
    *,
    max_samples: int = 32,
    max_duration_seconds: float | None = None,
) -> ResourceSamplerConfig:
    return ResourceSamplerConfig(
        trace_level=level,
        workload_evidence=EvidenceClass.SYNTHETIC,
        seed=41,
        sample_interval_ms=1,
        max_samples=max_samples,
        max_duration_seconds=max_duration_seconds,
    )


def _available(source: str, *, privileges: bool = False) -> CounterCapability:
    return CounterCapability(
        available=True,
        source=source,
        requires_privileges=privileges,
    )


def _unavailable(reason: str = "not installed") -> CounterCapability:
    return CounterCapability(available=False, reason=reason)


def test_disabled_sampler_is_a_zero_overhead_explicit_trace() -> None:
    trace = ResourceSampler(_config(TraceLevel.DISABLED)).start().stop()

    assert trace.samples == ()
    assert trace.sample_attempts == 0
    assert trace.samples_dropped == 0
    assert trace.started_monotonic_ns == trace.ended_monotonic_ns
    assert trace.config.workload_evidence is EvidenceClass.SYNTHETIC
    assert trace.timing_provenance is TimingProvenance.HOST_MONOTONIC_OBSERVED


def test_minimal_sampler_records_ordered_observed_process_samples() -> None:
    trace = collect_resource_trace(
        _config(TraceLevel.MINIMAL),
        duration_seconds=0.02,
    )

    assert len(trace.samples) >= 1
    assert trace.collection_error_count == 0
    assert [sample.sequence for sample in trace.samples] == list(range(len(trace.samples)))
    timestamps = [sample.monotonic_ns for sample in trace.samples]
    assert timestamps == sorted(timestamps)
    assert all(sample.process_cpu_user_seconds >= 0 for sample in trace.samples)
    assert all(sample.cpu_cycles is None for sample in trace.samples)
    assert all(sample.cache_misses is None for sample in trace.samples)
    assert all(sample.gpu_utilization_percent is None for sample in trace.samples)
    assert all(sample.process_read_bytes is None for sample in trace.samples)


def test_full_sampler_exposes_overflow_instead_of_growing_unbounded() -> None:
    trace = collect_resource_trace(
        _config(TraceLevel.FULL, max_samples=1),
        duration_seconds=0.03,
    )

    assert trace.samples_recorded == 1
    assert trace.samples_dropped > 0
    assert trace.buffer_overflowed
    assert trace.sample_attempts == (
        trace.samples_recorded + trace.samples_dropped + trace.collection_error_count
    )
    sample = trace.samples[0]
    assert sample.process_rss_bytes is None or sample.process_rss_bytes > 0
    assert sample.system_memory_total_bytes is None or sample.system_memory_total_bytes > 0


def test_sampler_stops_at_its_configured_deadline() -> None:
    sampler = ResourceSampler(_config(TraceLevel.MINIMAL, max_duration_seconds=0.015)).start()

    assert sampler.wait(timeout_seconds=1.0)
    assert sampler.state is ResourceSamplerState.STOPPED
    trace = sampler.stop()
    assert trace.samples
    assert trace.ended_monotonic_ns is not None
    assert trace.started_monotonic_ns is not None
    assert trace.ended_monotonic_ns >= trace.started_monotonic_ns


def test_detected_capabilities_never_claim_uncollected_hardware_counters() -> None:
    capabilities = detect_platform_capabilities()

    for counter in (
        "cpu_cycles",
        "cache_misses",
        "gpu_utilization",
        "pcie_bytes",
        "nic_hardware_counters",
    ):
        assert not capabilities.counters[counter].available
        assert capabilities.counters[counter].reason
    assert capabilities.counters["process_cpu_time"].available
    assert capabilities.process_id > 0
    assert capabilities.hostname


def test_profiler_commands_are_capability_gated_and_argument_safe() -> None:
    capabilities = PlatformCapabilities(
        schema_version="sloforge.branchfabric.resource-capabilities/v1",
        hostname="test-host",
        process_id=1234,
        operating_system="test",
        machine="test",
        python_implementation="CPython",
        logical_cpu_count=8,
        physical_cpu_count=4,
        counters={"cpu_cycles": _unavailable()},
        profiler_tools={
            "powermetrics": _available("/usr/bin/powermetrics", privileges=True),
            "xctrace": _available("/usr/bin/xctrace"),
            "nsys": _available("/opt/nvidia/nsight-systems/bin/nsys"),
            "ncu": _available("/opt/nvidia/nsight-compute/ncu"),
        },
    )
    target = ("uv", "run", "sloforge", "helix", "demo", "--seed", "41")

    commands = generate_profiler_commands(
        capabilities,
        output_dir=Path("artifacts/branchfabric/profilers/run-41"),
        target_argv=target,
        sample_interval_ms=250,
        sample_count=20,
    )

    assert [command.tool for command in commands] == ["powermetrics", "xctrace", "nsys", "ncu"]
    assert commands[0].capture_mode == "sidecar"
    assert commands[0].requires_privileges
    assert not commands[0].executes_target
    assert "--sample-count 20" in commands[0].rendered_command
    for command in commands[1:]:
        assert command.capture_mode == "wrap"
        assert command.executes_target
        assert all(argument in command.argv for argument in target)

    unavailable = capabilities.model_copy(
        update={"profiler_tools": {name: _unavailable() for name in capabilities.profiler_tools}}
    )
    assert (
        generate_profiler_commands(
            unavailable,
            output_dir=Path("out"),
            target_argv=target,
        )
        == ()
    )
    with pytest.raises(ValueError, match="NUL-free"):
        generate_profiler_commands(
            capabilities,
            output_dir=Path("out"),
            target_argv=("bad\x00argument",),
        )


def test_context_manager_leaves_no_sampler_thread_running() -> None:
    with ResourceSampler(_config(TraceLevel.MINIMAL)) as sampler:
        time.sleep(0.005)

    assert sampler.state is ResourceSamplerState.STOPPED
