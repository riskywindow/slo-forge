from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.fabric.ir import canonical_hash
from sloforge.fabric.profiling import (
    BenchmarkStatus,
    MeasurementMode,
    Primitive,
    adapter_inventory,
    benchmark_host_memory,
    benchmark_synthetic_fabric,
    build_ibverbs_command,
    build_nccl_tests_command,
    build_nvidia_smi_command,
    load_profile,
    to_canonical_profile,
)
from sloforge.fabric.profiling.models import FabricProfile
from sloforge.fabric.topology import build_canonical_fixture, build_discovery_fixture


def _executable(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o700)
    return path


def test_synthetic_profile_is_deterministic_and_artifact_backed(tmp_path: Path) -> None:
    graph = build_discovery_fixture("two_node_infiniband")
    output = tmp_path / "profile"
    first = benchmark_synthetic_fabric(
        graph, seed=17, suite="quick", warmup_count=2, sample_count=7, output_dir=output
    )
    second = benchmark_synthetic_fabric(
        graph, seed=17, suite="quick", warmup_count=2, sample_count=7
    )
    assert first.profile_hash == second.profile_hash
    assert first.results == second.results
    loaded = load_profile(output)
    assert loaded == first
    assert all(result.mode is MeasurementMode.SYNTHETIC_CALIBRATED for result in first.results)
    assert all(result.status is BenchmarkStatus.SUCCESS for result in first.results)
    assert all(
        result.raw_artifact and (output / result.raw_artifact).is_file() for result in first.results
    )


def test_profile_loader_rejects_tampered_raw_sample(tmp_path: Path) -> None:
    output = tmp_path / "profile"
    profile = benchmark_synthetic_fabric(
        build_discovery_fixture("single_gpu_workstation"),
        seed=19,
        suite="quick",
        sample_count=3,
        output_dir=output,
    )
    first = profile.results[0]
    assert first.raw_artifact is not None
    raw_path = output / first.raw_artifact
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    payload["samples"][0]["duration_microseconds"] += 1.0
    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="raw artifact does not match"):
        load_profile(output)


def test_full_suite_covers_required_categories_and_curves() -> None:
    profile = benchmark_synthetic_fabric(
        build_discovery_fixture("eight_gpu_nvlink"), seed=3, suite="full", sample_count=3
    )
    assert {result.case.primitive for result in profile.results} == set(Primitive)
    all_reduce = [
        result for result in profile.results if result.case.primitive is Primitive.ALL_REDUCE
    ]
    assert {result.case.rank_count for result in all_reduce} == {2, 4, 8}
    assert len({result.case.message_bytes for result in all_reduce}) >= 4
    assert {result.case.concurrency for result in all_reduce} == {1, 2}
    assert all(result.summary and result.summary.sample_count == 3 for result in profile.results)


def test_synthetic_profile_converts_to_canonical_fabric_ir() -> None:
    topology = build_canonical_fixture("two_node_infiniband")
    raw = benchmark_synthetic_fabric(
        build_discovery_fixture("two_node_infiniband"), seed=31, suite="quick", sample_count=3
    )
    canonical = to_canonical_profile(raw, topology=topology)
    assert canonical.kind == "FabricProfile"
    assert len(canonical.measurements) == len(raw.results)
    assert len(canonical.raw_artifacts) == len(raw.results)
    assert canonical.extensions.root["sloforge.io/measurement-modes"] == ["synthetic_calibrated"]
    assert canonical.topology_fingerprint.value == canonical_hash(topology)


def test_canonical_topology_input_is_compile_compatible() -> None:
    topology = build_canonical_fixture("two_node_infiniband")
    raw = benchmark_synthetic_fabric(topology, seed=37, suite="quick", sample_count=3)
    canonical = to_canonical_profile(raw, topology=topology)
    assert raw.topology_fingerprint == canonical_hash(topology)
    assert canonical.topology_fingerprint.value == canonical_hash(topology)


def test_canonical_profile_rejects_mismatched_topology() -> None:
    raw = benchmark_synthetic_fabric(
        build_discovery_fixture("single_gpu_workstation"), seed=41, suite="quick", sample_count=3
    )
    with pytest.raises(ValueError, match="does not correspond"):
        to_canonical_profile(raw, topology=build_canonical_fixture("eight_gpu_nvlink"))


def test_degraded_link_increases_synthetic_collective_latency() -> None:
    healthy = benchmark_synthetic_fabric(
        build_discovery_fixture("two_node_infiniband"), seed=5, suite="quick", sample_count=5
    )
    degraded = benchmark_synthetic_fabric(
        build_discovery_fixture("degraded_topology"), seed=5, suite="quick", sample_count=5
    )
    healthy_case = next(
        result
        for result in healthy.results
        if result.case.primitive is Primitive.ALL_REDUCE
        and result.case.message_bytes == 16_777_216
        and result.case.rank_count == 8
        and result.case.concurrency == 1
    )
    degraded_case = next(
        result for result in degraded.results if result.case.case_id == healthy_case.case.case_id
    )
    assert degraded_case.summary is not None and healthy_case.summary is not None
    assert degraded_case.summary.median_microseconds > healthy_case.summary.median_microseconds


def test_limited_visibility_is_unavailable_not_a_rank_one_fallback() -> None:
    profile = benchmark_synthetic_fabric(
        build_discovery_fixture("limited_container"), seed=7, suite="quick", sample_count=3
    )
    collectives = [
        result
        for result in profile.results
        if result.case.primitive in {Primitive.ALL_REDUCE, Primitive.ALL_TO_ALL}
    ]
    assert collectives
    assert all(result.case.rank_count == 2 for result in collectives)
    assert all(result.status is BenchmarkStatus.UNAVAILABLE for result in collectives)
    assert all(not result.raw_samples and result.failure_reason for result in collectives)


def test_actual_host_memory_benchmark_is_labeled_measured() -> None:
    result = benchmark_host_memory(message_bytes=64 * 1024, warmup_count=1, sample_count=5, seed=9)
    assert result.status is BenchmarkStatus.SUCCESS
    assert result.mode is MeasurementMode.MEASURED
    assert len(result.raw_samples) == 5
    assert all(not sample.synthetic and sample.seed is None for sample in result.raw_samples)
    assert result.summary is not None


def test_profile_hash_rejects_tampering() -> None:
    profile = benchmark_synthetic_fabric(
        build_discovery_fixture("single_gpu_workstation"), seed=2, suite="quick", sample_count=3
    )
    payload = profile.model_dump(mode="json")
    payload["seed"] = 99
    with pytest.raises(ValidationError, match="profile hash mismatch"):
        FabricProfile.model_validate_json(json.dumps(payload))


def test_adapter_inventory_never_claims_missing_tool_is_available() -> None:
    inventory = adapter_inventory()
    assert inventory
    assert len({adapter.name for adapter in inventory}) == len(inventory)
    assert all(adapter.version is not None or adapter.reason is not None for adapter in inventory)


def test_nccl_builder_is_version_isolated_and_does_not_execute(tmp_path: Path) -> None:
    binary = _executable(tmp_path, "all_reduce_perf")
    command = build_nccl_tests_command(
        executable=binary,
        operation="all_reduce",
        minimum_bytes=1024,
        maximum_bytes=1024 * 1024,
        step_factor=2,
        gpus_per_process=4,
        iterations=20,
        warmups=5,
        algorithm="Ring",
        protocol="LL128",
        channels=4,
    )
    assert command.argv[0] == str(binary.resolve())
    assert ("NCCL_ALGO", "Ring") in command.environment
    assert command.requires_gpu
    with pytest.raises(ValueError, match="all_gather_perf"):
        build_nccl_tests_command(
            executable=binary,
            operation="all_gather",
            minimum_bytes=1,
            maximum_bytes=1,
            step_factor=2,
            gpus_per_process=1,
            iterations=1,
            warmups=0,
        )


def test_ibverbs_and_nvml_builders_are_read_only_and_explicit(tmp_path: Path) -> None:
    ib = _executable(tmp_path, "ib_write_bw")
    perftest = build_ibverbs_command(
        executable=ib,
        role="client",
        server_address="192.0.2.1",
        device="mlx5_0",
        port=18515,
        message_bytes=4096,
        iterations=10,
    )
    assert perftest.expected_transport == "ibverbs"
    with pytest.raises(ValueError, match="server address"):
        build_ibverbs_command(
            executable=ib,
            role="client",
            server_address=None,
            device="mlx5_0",
            port=18515,
            message_bytes=4096,
            iterations=10,
        )
    smi = _executable(tmp_path, "nvidia-smi")
    query = build_nvidia_smi_command(
        executable=smi, gpu_id="GPU-test", fields=("uuid", "clocks.sm")
    )
    assert not any("-lgc" in argument for argument in query.argv)
    with pytest.raises(ValueError, match="read-only"):
        build_nvidia_smi_command(executable=smi, gpu_id="0", fields=("clocks.applications",))


def test_command_builder_requires_executable(tmp_path: Path) -> None:
    missing = tmp_path / "all_reduce_perf"
    with pytest.raises(FileNotFoundError):
        build_nccl_tests_command(
            executable=missing,
            operation="all_reduce",
            minimum_bytes=1,
            maximum_bytes=1,
            step_factor=2,
            gpus_per_process=1,
            iterations=1,
            warmups=0,
        )


def test_adapter_builders_do_not_mutate_environment(tmp_path: Path) -> None:
    before = dict(os.environ)
    binary = _executable(tmp_path, "all_reduce_perf")
    build_nccl_tests_command(
        executable=binary,
        operation="all_reduce",
        minimum_bytes=1,
        maximum_bytes=1,
        step_factor=2,
        gpus_per_process=1,
        iterations=1,
        warmups=0,
        algorithm="Tree",
    )
    assert dict(os.environ) == before
