from __future__ import annotations

import hashlib
import json
from pathlib import Path

from click import unstyle
from typer.testing import CliRunner

from sloforge.autopsy import (
    AutopsyEvent,
    AutopsyRun,
    BottleneckKind,
    EventType,
    EvidenceRef,
    ResourceRef,
    SourceClock,
    compare_runs,
    diagnose,
)
from sloforge.cli.main import app
from sloforge.fabric.ir import (
    load_fabric_profile,
    load_physical_execution_plan,
    load_topology_graph,
)
from sloforge.fabric.profiling import BenchmarkStatus, MeasurementMode, load_profile
from sloforge.warmpath import (
    ArtifactGraph,
    ArtifactKind,
    ArtifactNode,
    HostEnvironment,
    StorageKind,
    StorageTierSpec,
    load_plan,
    save_graph,
)

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
runner = CliRunner()


def _invoke(arguments: list[str]) -> str:
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output or repr(result.exception)
    return result.output


def test_extension_command_groups_preserve_original_cli() -> None:
    output = _invoke(["--help"])
    for command in ("trace", "fabric", "autopsy", "recovery", "forgeci", "warmpath"):
        assert command in output
    expected = {
        "fabric": ("discover", "benchmark", "model", "compile", "explain", "simulate", "validate"),
        "autopsy": ("capture", "compare", "diagnose", "replay", "minimize", "report"),
        "recovery": ("plan", "apply"),
        "forgeci": ("run", "bisect"),
        "warmpath": ("profile", "compile"),
    }
    for group, commands in expected.items():
        help_text = _invoke([group, "--help"])
        assert all(command in help_text for command in commands)


def test_fabric_discover_benchmark_model_compile_and_explain(tmp_path: Path) -> None:
    topology = tmp_path / "topology.json"
    profile_dir = tmp_path / "profile"
    model = tmp_path / "model.json"
    plan = tmp_path / "physical.json"
    _invoke(
        [
            "fabric",
            "discover",
            "--fixture",
            "two_node_infiniband",
            "--output",
            str(topology),
        ]
    )
    graph = load_topology_graph(topology)
    assert graph.topology_id == "two_node_infiniband"
    _invoke(
        [
            "fabric",
            "benchmark",
            "--topology",
            str(topology),
            "--suite",
            "quick",
            "--synthetic",
            "--samples",
            "3",
            "--warmups",
            "0",
            "--seed",
            "29",
            "--output",
            str(profile_dir),
        ]
    )
    profile = load_fabric_profile(profile_dir / "fabric-profile.json")
    assert profile.measurements
    _invoke(
        [
            "fabric",
            "model",
            "inspect",
            "--model",
            "synthetic",
            "--synthetic-moe",
            "--output",
            str(model),
        ]
    )
    _invoke(
        [
            "fabric",
            "compile",
            "--deployment-plan",
            str(FIXTURES / "ir" / "deployment-plan-v1.json"),
            "--topology",
            str(topology),
            "--fabric-profile",
            str(profile_dir),
            "--model-graph",
            str(model),
            "--slo",
            "p95_ttft_ms<=1500,p99_tpot_ms<=150",
            "--maximum-ranks",
            "2",
            "--seed",
            "29",
            "--output",
            str(plan),
        ]
    )
    physical = load_physical_execution_plan(plan)
    assert physical.reproducibility.seed == 29
    assert plan.with_suffix(".optimization.json").is_file()
    explanation = _invoke(["fabric", "explain", str(plan)])
    assert physical.plan_id in explanation
    assert "Parallelism" in explanation


def test_fabric_benchmark_never_hides_synthetic_fallback(tmp_path: Path) -> None:
    topology = tmp_path / "topology.json"
    _invoke(
        [
            "fabric",
            "discover",
            "--fixture",
            "two_node_infiniband",
            "--output",
            str(topology),
        ]
    )
    result = runner.invoke(
        app,
        [
            "fabric",
            "benchmark",
            "--topology",
            str(topology),
            "--suite",
            "quick",
            "--output",
            str(tmp_path / "profile"),
        ],
    )
    assert result.exit_code == 2
    assert "--synthetic for fixtures" in unstyle(result.output)


def _fake_nccl_executable(tmp_path: Path, *, exit_code: int = 0) -> Path:
    executable = tmp_path / "all_reduce_perf"
    if exit_code:
        body = f"#!/bin/sh\necho intentional-adapter-failure >&2\nexit {exit_code}\n"
    else:
        rows = []
        message_bytes = 1_024
        while message_bytes <= 1 << 20:
            count = message_bytes // 4
            duration = 5.0 + message_bytes / (1 << 20)
            algorithm_gbps = message_bytes / (duration / 1_000_000.0) / 1_000_000_000.0
            rows.append(
                f"{message_bytes} {count} float sum -1 {duration:.6f} "
                f"{algorithm_gbps:.9f} {algorithm_gbps:.9f} 0 {duration:.6f} "
                f"{algorithm_gbps:.9f} {algorithm_gbps:.9f} 0"
            )
            message_bytes *= 2
        body = (
            "#!/bin/sh\ncat <<'EOF'\n"
            "# size count type redop root time algbw busbw #wrong time algbw busbw #wrong\n"
            + "\n".join(rows)
            + "\nEOF\n"
        )
    executable.write_text(body, encoding="utf-8")
    executable.chmod(executable.stat().st_mode | 0o700)
    return executable


def test_fabric_measured_nccl_cli_executes_only_explicit_adapter(tmp_path: Path) -> None:
    topology = tmp_path / "topology.json"
    profile_dir = tmp_path / "profile"
    _invoke(
        [
            "fabric",
            "discover",
            "--fixture",
            "two_node_infiniband",
            "--output",
            str(topology),
        ]
    )
    executable = _fake_nccl_executable(tmp_path)
    _invoke(
        [
            "fabric",
            "benchmark",
            "--topology",
            str(topology),
            "--suite",
            "quick",
            "--measured",
            "--adapter",
            "nccl-tests",
            "--transport",
            "nccl-local",
            "--adapter-executable",
            str(executable),
            "--visible-device",
            "GPU-00-00",
            "--samples",
            "3",
            "--warmups",
            "0",
            "--inner-iterations",
            "1",
            "--timeout-seconds",
            "2",
            "--output",
            str(profile_dir),
        ]
    )
    canonical = load_fabric_profile(profile_dir / "fabric-profile.json")
    raw = load_profile(profile_dir / "raw")
    assert len(canonical.measurements) == 11
    assert all(measurement.transport == "nccl-local" for measurement in canonical.measurements)
    assert all(result.mode is MeasurementMode.MEASURED for result in raw.results)
    assert all(result.status is BenchmarkStatus.SUCCESS for result in raw.results)


def test_fabric_measured_nccl_cli_preserves_failure_artifact(tmp_path: Path) -> None:
    topology = tmp_path / "topology.json"
    profile_dir = tmp_path / "profile"
    _invoke(
        [
            "fabric",
            "discover",
            "--fixture",
            "two_node_infiniband",
            "--output",
            str(topology),
        ]
    )
    executable = _fake_nccl_executable(tmp_path, exit_code=9)
    result = runner.invoke(
        app,
        [
            "fabric",
            "benchmark",
            "--topology",
            str(topology),
            "--suite",
            "quick",
            "--measured",
            "--adapter",
            "nccl-tests",
            "--transport",
            "nccl-local",
            "--adapter-executable",
            str(executable),
            "--visible-device",
            "GPU-00-00",
            "--samples",
            "3",
            "--warmups",
            "0",
            "--timeout-seconds",
            "2",
            "--output",
            str(profile_dir),
        ],
    )
    assert result.exit_code == 2
    assert "failed closed" in result.output
    raw = load_profile(profile_dir / "raw")
    assert raw.results
    assert all(item.status is BenchmarkStatus.FAILED for item in raw.results)
    assert not (profile_dir / "fabric-profile.json").exists()


def _autopsy_run(run_id: str, *, degraded: bool, directory: Path) -> AutopsyRun:
    artifact = directory / f"{run_id}-raw.json"
    artifact.write_text("{}\n", encoding="utf-8")
    evidence = EvidenceRef(
        source="test-fixture",
        artifact_uri=str(artifact),
        sha256=hashlib.sha256(b"{}\n").hexdigest(),
    )
    events = tuple(
        AutopsyEvent(
            event_id=f"{run_id}-rank-{rank}",
            event_type=EventType.DECODE,
            host="host-0",
            rank=rank,
            gpu=f"gpu-{rank}",
            request_id="request-0",
            operation="decode",
            start_ns=0,
            end_ns=(5_000_000 if degraded and rank == 3 else 1_000_000),
            source_clock=SourceClock.SYNTHETIC,
            normalized_start_ns=0,
            normalized_end_ns=(5_000_000 if degraded and rank == 3 else 1_000_000),
            alignment_confidence=1.0,
            alignment_uncertainty_ns=0,
            resource=ResourceRef(
                resource_id=f"compute:gpu-{rank}",
                resource_type="gpu_compute",
            ),
            evidence=evidence.model_copy(update={"record_index": rank}),
        )
        for rank in range(4)
    )
    return AutopsyRun(
        run_id=run_id,
        source="synthetic_fixture",
        topology_fingerprint="a" * 64,
        physical_plan_hash="b" * 64,
        workload_fingerprint="c" * 64,
        reference_host="host-0",
        events=events,
        artifacts=(evidence,),
    )


def test_autopsy_compare_diagnose_minimize_and_report(tmp_path: Path) -> None:
    healthy_path = tmp_path / "healthy.json"
    degraded_path = tmp_path / "degraded.json"
    healthy_run = _autopsy_run("healthy", degraded=False, directory=tmp_path)
    degraded_run = _autopsy_run("degraded", degraded=True, directory=tmp_path)
    healthy_path.write_text(healthy_run.model_dump_json())
    degraded_path.write_text(degraded_run.model_dump_json())
    comparison = tmp_path / "comparison.json"
    diagnosis = tmp_path / "diagnosis.json"
    minimized = tmp_path / "minimized.json"
    report = tmp_path / "report.md"
    _invoke(
        [
            "autopsy",
            "compare",
            "--healthy",
            str(healthy_path),
            "--degraded",
            str(degraded_path),
            "--output",
            str(comparison),
        ]
    )
    diagnostic_output = _invoke(
        [
            "autopsy",
            "diagnose",
            str(degraded_path),
            "--baseline",
            str(healthy_path),
            "--output",
            str(diagnosis),
        ]
    )
    assert "rank_straggler" in diagnostic_output
    _invoke(
        [
            "autopsy",
            "minimize",
            str(degraded_path),
            "--baseline",
            str(healthy_path),
            "--output",
            str(minimized),
        ]
    )
    minimized_payload = json.loads(minimized.read_text(encoding="utf-8"))
    minimized_run = minimized_payload["minimized_run"]
    assert minimized_payload["minimized_event_count"] >= 2
    parsed_minimized_run = AutopsyRun.model_validate_json(json.dumps(minimized_run))
    minimized_diagnosis = diagnose(
        parsed_minimized_run,
        comparison=compare_runs(healthy_run, parsed_minimized_run),
        baseline=healthy_run,
    )
    assert minimized_diagnosis.top_hypothesis is BottleneckKind.RANK_STRAGGLER
    assert minimized_diagnosis.hypotheses[0].rejected_reason is None
    assert minimized_diagnosis.hypotheses[0].supporting_evidence
    rendered = _invoke(["autopsy", "report", str(diagnosis), "--output", str(report)])
    assert "SLOForge Autopsy" in rendered
    assert report.is_file()


def test_recovery_apply_is_non_mutating_and_completes_shadow_canary(tmp_path: Path) -> None:
    output = tmp_path / "execution.json"
    rendered = _invoke(
        [
            "recovery",
            "apply",
            "--proposal",
            str(FIXTURES / "fabric" / "recovery-plan-v1.json"),
            "--mode",
            "shadow-canary",
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["state"] == "COMPLETED"
    assert '"external_mutations": 0' in rendered


def test_warmpath_profile_and_compile_use_measured_local_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = b"typed-warm-path-fixture"
    (source / "weights.bin").write_bytes(payload)
    graph = ArtifactGraph(
        graph_id="cli-warmpath",
        artifacts=(
            ArtifactNode(
                artifact_id="weights",
                kind=ArtifactKind.MODEL_WEIGHTS,
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                source_relative_path="weights.bin",
            ),
        ),
    )
    graph_path = tmp_path / "graph.json"
    save_graph(graph, graph_path)
    host = HostEnvironment(
        operating_system="test",
        architecture="cpu",
        runtime="mock-runtime",
        runtime_version="1.0.0",
        host_fingerprint="d" * 64,
    )
    host_path = tmp_path / "host.json"
    host_path.write_text(host.model_dump_json(), encoding="utf-8")
    tier = StorageTierSpec(
        tier_id="local",
        kind=StorageKind.LOCAL_NVME,
        capacity_bytes=1 << 20,
        read_bandwidth_bytes_per_second=1_000_000_000.0,
        base_read_latency_ms=0.1,
        local_path=str(tmp_path / "cache"),
    )
    tiers_path = tmp_path / "tiers.json"
    tiers_path.write_text(json.dumps([tier.model_dump(mode="json")]), encoding="utf-8")
    profile = tmp_path / "profile"
    plan = tmp_path / "warmpath-plan.json"
    _invoke(
        [
            "warmpath",
            "profile",
            "--graph",
            str(graph_path),
            "--source",
            str(source),
            "--host",
            str(host_path),
            "--tiers",
            str(tiers_path),
            "--warmups",
            "0",
            "--samples",
            "3",
            "--seed",
            "13",
            "--output",
            str(profile),
        ]
    )
    _invoke(
        [
            "warmpath",
            "compile",
            "--graph",
            str(graph_path),
            "--profile",
            str(profile),
            "--simulation-trials",
            "5",
            "--seed",
            "13",
            "--output",
            str(plan),
        ]
    )
    compiled = load_plan(plan)
    assert compiled.optimizer_seed == 13
    assert compiled.evidence_references
