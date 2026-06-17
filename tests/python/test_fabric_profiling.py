from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.fabric.compiler import (
    CompilerAssumptions,
    CompilerConstraints,
    CompilerObjective,
    CompilerRequest,
    OptimizationStrategy,
    compile_physical_plan,
)
from sloforge.fabric.ir import DocumentReference, canonical_hash, load_model_graph
from sloforge.fabric.profiling import (
    AdapterExecutionError,
    BenchmarkStatus,
    MeasurementMode,
    Primitive,
    adapter_inventory,
    benchmark_host_memory,
    benchmark_synthetic_fabric,
    build_ibverbs_command,
    build_nccl_tests_command,
    build_nvidia_smi_command,
    execute_bounded,
    load_profile,
    parse_nccl_tests_output,
    read_nvidia_inventory,
    run_nccl_tests_profile,
    to_canonical_profile,
)
from sloforge.fabric.profiling.models import FabricProfile
from sloforge.fabric.topology import build_canonical_fixture, build_discovery_fixture
from sloforge.ir import ArtifactDigest

FABRIC_FIXTURES = Path(__file__).parents[1] / "fixtures" / "fabric"


def _executable(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o700)
    return path


def _script(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\nset -eu\n{body}\n", encoding="utf-8")
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
    transports = {measurement.transport for measurement in canonical.measurements}
    assert "sloforge-synthetic-calibrated-v1" not in transports
    assert "infiniband" in transports


def test_canonical_topology_input_is_compile_compatible() -> None:
    topology = build_canonical_fixture("two_node_infiniband")
    raw = benchmark_synthetic_fabric(topology, seed=37, suite="quick", sample_count=3)
    canonical = to_canonical_profile(raw, topology=topology)
    assert raw.topology_fingerprint == canonical_hash(topology)
    assert canonical.topology_fingerprint.value == canonical_hash(topology)

    digest = ArtifactDigest(value="a" * 64)
    request = CompilerRequest(
        logical_deployment_plan=DocumentReference(
            kind="DeploymentPlan",
            api_version="sloforge.io/v1",
            uri="artifacts/plans/logical.json",
            digest=digest,
        ),
        model=load_model_graph(FABRIC_FIXTURES / "model-graph-v1.json"),
        topology=topology,
        fabric_profile=canonical,
        constraints=CompilerConstraints(
            prompt_tokens_p95=512,
            output_tokens_p95=64,
            maximum_concurrent_requests=4,
            p95_ttft_ms=1_500.0,
            p99_tpot_ms=150.0,
            maximum_ranks=2,
        ),
        assumptions=CompilerAssumptions(
            prefill_tokens_per_second_per_gpu=8_000.0,
            decode_tokens_per_second_per_gpu=120.0,
            gpu_hourly_price_usd=2.0,
            base_availability=0.999,
            cold_start_ms=2_000.0,
            measurement_relative_uncertainty=0.10,
        ),
        objective=CompilerObjective.ROBUST_BALANCED,
        strategy=OptimizationStrategy.HIERARCHICAL,
        generated_at=datetime(2026, 8, 1, tzinfo=UTC),
        seed=37,
        git_commit="fixture",
        environment_digest=ArtifactDigest(value="b" * 64),
    )
    compiled = compile_physical_plan(request)
    assert compiled.selected.topology_fingerprint == canonical.topology_fingerprint


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
        visible_devices=("GPU-0", "GPU-1", "GPU-2", "GPU-3"),
        iterations=20,
        warmups=5,
        algorithm="Ring",
        protocol="LL128",
        channels=4,
    )
    assert command.argv[0] == str(binary.resolve())
    assert ("CUDA_VISIBLE_DEVICES", "GPU-0,GPU-1,GPU-2,GPU-3") in command.environment
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
            visible_devices=("GPU-0",),
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
            visible_devices=("GPU-0",),
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
        visible_devices=("GPU-0",),
        iterations=1,
        warmups=0,
        algorithm="Tree",
    )
    assert dict(os.environ) == before


def test_nccl_builder_rejects_implicit_or_mismatched_device_sets(tmp_path: Path) -> None:
    binary = _executable(tmp_path, "all_reduce_perf")
    common = {
        "executable": binary,
        "operation": "all_reduce",
        "minimum_bytes": 1,
        "maximum_bytes": 1,
        "step_factor": 2,
        "gpus_per_process": 2,
        "iterations": 1,
        "warmups": 0,
    }
    with pytest.raises(ValueError, match="one unique explicit identifier"):
        build_nccl_tests_command(**common, visible_devices=())
    with pytest.raises(ValueError, match="one unique explicit identifier"):
        build_nccl_tests_command(**common, visible_devices=("GPU-0", "GPU-0"))
    with pytest.raises(ValueError, match="one unique explicit identifier"):
        build_nccl_tests_command(**common, visible_devices=("GPU-0,GPU-1", "GPU-2"))


NCCL_TABLE = """# size count type redop root time algbw busbw #wrong time algbw busbw #wrong
1024 256 float sum -1 5.00 0.205 0.360 0 4.90 0.209 0.366 0
2048 512 float sum -1 6.00 0.341 0.597 0 5.80 0.353 0.617 0
4096 1024 float sum -1 7.00 0.585 1.024 0 6.90 0.594 1.039 0
"""


def test_nccl_parser_retains_standard_out_of_place_and_in_place_columns() -> None:
    rows = parse_nccl_tests_output(NCCL_TABLE)
    assert [row.message_bytes for row in rows] == [1024, 2048, 4096]
    assert rows[0].out_of_place_time_us == 5.0
    assert rows[0].in_place_bus_gbps == 0.366
    with pytest.raises(AdapterExecutionError, match="no recognized"):
        parse_nccl_tests_output("# only diagnostic output\n")
    with pytest.raises(AdapterExecutionError, match="repeated message size"):
        parse_nccl_tests_output(NCCL_TABLE + NCCL_TABLE.splitlines()[1] + "\n")


def test_measured_nccl_runner_executes_bounded_fixture_and_preserves_raw_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = _script(
        tmp_path,
        "all_reduce_perf",
        'test -z "${NCCL_UNSAFE_AMBIENT:-}"\n' + f"cat <<'EOF'\n{NCCL_TABLE}EOF",
    )
    monkeypatch.setenv("NCCL_UNSAFE_AMBIENT", "must-not-be-inherited")
    command = build_nccl_tests_command(
        executable=binary,
        operation="all_reduce",
        minimum_bytes=1024,
        maximum_bytes=4096,
        step_factor=2,
        gpus_per_process=2,
        visible_devices=("GPU-0", "GPU-1"),
        iterations=5,
        warmups=2,
        timeout_seconds=2.0,
    )
    output = tmp_path / "evidence"
    profile = run_nccl_tests_profile(
        command=command,
        operation="all_reduce",
        topology_fingerprint="a" * 64,
        suite="quick",
        minimum_bytes=1024,
        maximum_bytes=4096,
        step_factor=2,
        repetitions=3,
        warmup_count=2,
        seed=17,
        output_dir=output,
    )
    assert len(profile.results) == 3
    assert all(result.status is BenchmarkStatus.SUCCESS for result in profile.results)
    assert all(result.mode is MeasurementMode.MEASURED for result in profile.results)
    assert all(len(result.raw_samples) == 3 for result in profile.results)
    assert all(not sample.synthetic for result in profile.results for sample in result.raw_samples)
    assert profile.results[0].raw_samples[0].throughput_bytes_per_second == 205_000_000.0
    assert len(tuple((output / "captures").glob("*.json"))) == 3


def test_measured_nccl_runner_records_adapter_failure_without_partial_measurement(
    tmp_path: Path,
) -> None:
    binary = _script(tmp_path, "all_reduce_perf", "echo adapter-failed >&2\nexit 7")
    command = build_nccl_tests_command(
        executable=binary,
        operation="all_reduce",
        minimum_bytes=1024,
        maximum_bytes=1024,
        step_factor=2,
        gpus_per_process=1,
        visible_devices=("GPU-0",),
        iterations=1,
        warmups=0,
    )
    profile = run_nccl_tests_profile(
        command=command,
        operation="all_reduce",
        topology_fingerprint="b" * 64,
        suite="quick",
        minimum_bytes=1024,
        maximum_bytes=1024,
        step_factor=2,
        repetitions=3,
        warmup_count=0,
        seed=3,
        output_dir=tmp_path / "failed",
    )
    assert len(profile.results) == 1
    result = profile.results[0]
    assert result.status is BenchmarkStatus.FAILED
    assert not result.raw_samples
    assert result.failure_reason and "return_code=7" in result.failure_reason


def test_bounded_executor_caps_output_and_terminates_process(tmp_path: Path) -> None:
    binary = _script(
        tmp_path,
        "all_reduce_perf",
        "i=0\nwhile test $i -lt 6000; do printf x; i=$((i + 1)); done\nsleep 20",
    )
    command = build_nccl_tests_command(
        executable=binary,
        operation="all_reduce",
        minimum_bytes=1,
        maximum_bytes=1,
        step_factor=2,
        gpus_per_process=1,
        visible_devices=("GPU-0",),
        iterations=1,
        warmups=0,
        timeout_seconds=2.0,
    )
    capture = execute_bounded(command, repetition=0, maximum_output_bytes=4096)
    assert capture.output_limited
    assert len(capture.stdout.encode()) == 4096
    assert capture.duration_seconds < command.timeout_seconds


def test_read_only_nvidia_inventory_executes_allowlisted_query(tmp_path: Path) -> None:
    binary = _script(
        tmp_path,
        "nvidia-smi",
        "printf '%s\\n' 'GPU-test, NVIDIA H100 80GB HBM3, 81559'",
    )
    fields = ("uuid", "name", "memory.total")
    command = build_nvidia_smi_command(executable=binary, gpu_id="GPU-test", fields=fields)
    record = read_nvidia_inventory(command, gpu_id="GPU-test", fields=fields)
    assert dict(record.fields) == {
        "uuid": "GPU-test",
        "name": "NVIDIA H100 80GB HBM3",
        "memory.total": "81559",
    }
