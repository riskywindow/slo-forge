from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.util import canonical_json, sha256_bytes
from sloforge.warmpath import (
    ArtifactGraph,
    ArtifactKind,
    ArtifactNode,
    CompatibilityConstraint,
    HostEnvironment,
    LocalWarmPathExecutor,
    MaterializationMode,
    StartupProfile,
    StorageKind,
    StorageTierSpec,
    WarmPathObjective,
    compile_warmpath,
    create_mock_snapshot_artifact,
    profile_local_startup,
    simulate_cold_start,
)
from sloforge.warmpath.demo import WarmPathDemoManifest, _reset
from sloforge.warmpath.statistics import robust_summary


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _host(*, fingerprint: str = "a" * 64) -> HostEnvironment:
    return HostEnvironment(
        operating_system="linux",
        architecture="x86_64",
        runtime="vllm",
        runtime_version="0.11.0",
        gpu_architecture=None,
        driver_major_version=None,
        topology_fingerprint=None,
        host_fingerprint=fingerprint,
        gpu_count=0,
    )


def _fixture(tmp_path: Path) -> tuple[ArtifactGraph, Path, tuple[StorageTierSpec, ...]]:
    source = tmp_path / "source"
    source.mkdir()
    config = b'{"model":"fixture"}'
    weights = bytes(range(64)) * 4
    optional = b"optional-runtime-cache"
    (source / "config.json").write_bytes(config)
    (source / "weights.bin").write_bytes(weights)
    (source / "runtime.cache").write_bytes(optional)
    graph = ArtifactGraph(
        graph_id="fixture-startup",
        artifacts=(
            ArtifactNode(
                artifact_id="model-config",
                kind=ArtifactKind.MODEL_CONFIG,
                size_bytes=len(config),
                sha256=_sha(config),
                source_relative_path="config.json",
            ),
            ArtifactNode(
                artifact_id="model-weights",
                kind=ArtifactKind.MODEL_WEIGHTS,
                size_bytes=len(weights),
                sha256=_sha(weights),
                dependencies=("model-config",),
                source_relative_path="weights.bin",
            ),
            ArtifactNode(
                artifact_id="runtime-cache",
                kind=ArtifactKind.RUNTIME_CACHE,
                size_bytes=len(optional),
                sha256=_sha(optional),
                dependencies=("model-config",),
                required_for_readiness=False,
                lazy_restore_allowed=True,
                rebuild_time_ms=3.0,
                source_relative_path="runtime.cache",
            ),
        ),
    )
    tiers = (
        StorageTierSpec(
            tier_id="fast-local",
            kind=StorageKind.LOCAL_NVME,
            capacity_bytes=1 << 20,
            read_bandwidth_bytes_per_second=1_000_000_000.0,
            base_read_latency_ms=0.1,
            maximum_parallel_reads=2,
            hourly_cost_per_gib=0.01,
            local_path=str(tmp_path / "fast-cache"),
        ),
        StorageTierSpec(
            tier_id="slow-local",
            kind=StorageKind.LOCAL_NVME,
            capacity_bytes=1 << 20,
            read_bandwidth_bytes_per_second=1_000.0,
            base_read_latency_ms=25.0,
            maximum_parallel_reads=1,
            hourly_cost_per_gib=0.0,
            local_path=str(tmp_path / "slow-cache"),
        ),
    )
    return graph, source, tiers


def test_checked_in_warmpath_demo_is_measured_and_hash_verified() -> None:
    root = Path(__file__).parents[2]
    artifact_root = root / "artifacts" / "warmpath"
    manifest = WarmPathDemoManifest.model_validate_json(
        (artifact_root / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest.profile_source == "measured"
    assert manifest.synthetic_snapshot is True
    assert manifest.restore_success and manifest.checksum_verified
    assert manifest.deferred_artifact_count > 0
    for artifact in manifest.artifacts:
        payload = (artifact_root / artifact.path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == artifact.sha256


def test_demo_reset_preserves_sibling_evaluation_artifacts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "warmpath"
    evaluation = artifact_root / "evaluation" / "result.json"
    evaluation.parent.mkdir(parents=True)
    evaluation.write_text("retained-evidence", encoding="utf-8")
    (artifact_root / "profile").mkdir()
    (artifact_root / "profile" / "stale.json").write_text("stale", encoding="utf-8")
    (artifact_root / "manifest.json").write_text("stale", encoding="utf-8")

    _reset(artifact_root, enabled=True)

    assert evaluation.read_text(encoding="utf-8") == "retained-evidence"
    assert not (artifact_root / "profile").exists()
    assert not (artifact_root / "manifest.json").exists()


def test_artifact_graph_rejects_cycles_unknown_edges_and_path_traversal() -> None:
    payload = b"x"
    with pytest.raises(ValidationError, match="cycle"):
        ArtifactGraph(
            graph_id="cycle",
            artifacts=(
                ArtifactNode(
                    artifact_id="a",
                    kind=ArtifactKind.MODEL_CONFIG,
                    size_bytes=1,
                    sha256=_sha(payload),
                    dependencies=("b",),
                    source_relative_path="a",
                ),
                ArtifactNode(
                    artifact_id="b",
                    kind=ArtifactKind.MODEL_CONFIG,
                    size_bytes=1,
                    sha256=_sha(payload),
                    dependencies=("a",),
                    source_relative_path="b",
                ),
            ),
        )
    with pytest.raises(ValidationError, match="relative"):
        ArtifactNode(
            artifact_id="escape",
            kind=ArtifactKind.MODEL_CONFIG,
            size_bytes=1,
            sha256=_sha(payload),
            source_relative_path="../secret",
        )
    with pytest.raises(ValidationError, match="relative"):
        ArtifactNode(
            artifact_id="windows-escape",
            kind=ArtifactKind.MODEL_CONFIG,
            size_bytes=1,
            sha256=_sha(payload),
            source_relative_path=r"cache\..\secret",
        )
    with pytest.raises(ValidationError, match="relative"):
        ArtifactNode(
            artifact_id="windows-drive",
            kind=ArtifactKind.MODEL_CONFIG,
            size_bytes=1,
            sha256=_sha(payload),
            source_relative_path=r"C:\secret",
        )
    with pytest.raises(ValidationError, match="GPU memory images"):
        ArtifactNode(
            artifact_id="gpu-snapshot",
            kind=ArtifactKind.GPU_MEMORY_IMAGE,
            size_bytes=1,
            sha256=_sha(payload),
            source_relative_path="snapshot.bin",
        )


def test_local_profiler_preserves_raw_samples_and_environment(tmp_path: Path) -> None:
    graph, source, tiers = _fixture(tmp_path)
    profile = profile_local_startup(
        profile_id="local-profile",
        graph=graph,
        host=_host(),
        tiers=(tiers[0],),
        source_directory=source,
        output_directory=tmp_path / "profile",
        warmups=1,
        sample_count=3,
        seed=2026,
    )
    assert len(profile.measurements) == len(graph.artifacts) * 2
    assert all(item.warmup_count == 1 for item in profile.measurements)
    assert all(len(item.raw_samples_ms) == 3 for item in profile.measurements)
    assert all(item.source == "measured" for item in profile.measurements)
    assert Path(profile.environment_manifest_path).is_file()
    raw_files = sorted(Path(profile.raw_artifact_directory).glob("*.json"))
    assert len(raw_files) == len(profile.measurements)
    raw = json.loads(raw_files[0].read_text(encoding="utf-8"))
    assert raw["raw_samples_ms"]
    assert raw["artifact_hash"] == profile.measurements[0].artifact_hash or any(
        raw["artifact_hash"] == item.artifact_hash for item in profile.measurements
    )


def test_planner_uses_measurements_and_defers_noncritical_artifacts(tmp_path: Path) -> None:
    graph, source, tiers = _fixture(tmp_path)
    measured = profile_local_startup(
        profile_id="optimizer-profile",
        graph=graph,
        host=_host(),
        tiers=(tiers[0],),
        source_directory=source,
        output_directory=tmp_path / "profile",
        warmups=0,
        sample_count=3,
        seed=31,
    )
    profile = StartupProfile(
        profile_id=measured.profile_id,
        graph_id=measured.graph_id,
        host=measured.host,
        tiers=tiers,
        measurements=measured.measurements,
        raw_artifact_directory=measured.raw_artifact_directory,
        environment_manifest_path=measured.environment_manifest_path,
        warnings=measured.warnings,
    )
    objective = WarmPathObjective(
        ready_time_weight=1.0,
        hourly_cost_weight=0.001,
        failure_risk_weight=1_000.0,
    )
    first = compile_warmpath(graph=graph, profile=profile, objective=objective, seed=91)
    second = compile_warmpath(graph=graph, profile=profile, objective=objective, seed=91)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    placements = {item.artifact_id: item for item in first.placements}
    assert placements["model-weights"].tier_id == "fast-local"
    assert placements["runtime-cache"].mode == MaterializationMode.LAZY_RESTORE
    predictions = {item.artifact_id: item for item in first.stage_predictions}
    required_finish = max(
        predictions[item.artifact_id].finish_ms
        for item in graph.artifacts
        if item.required_for_readiness
    )
    assert predictions["runtime-cache"].start_ms >= required_finish
    assert first.evaluated_candidate_count > 1
    assert first.evidence_references == (
        profile.raw_artifact_directory,
        profile.environment_manifest_path,
    )
    simulation = simulate_cold_start(
        graph=graph,
        placements=first.placements,
        profile=profile,
        seed=first.optimizer_seed,
        trial_count=11,
    )
    assert simulation.p95_ready_time_ms >= simulation.p50_ready_time_ms
    assert "measured" in simulation.estimate_sources


def test_incompatible_snapshot_is_rebuilt_and_reason_is_preserved(tmp_path: Path) -> None:
    payload = b"captured-process-state"
    source = tmp_path / "source"
    source.mkdir()
    (source / "state.bin").write_bytes(payload)
    graph = ArtifactGraph(
        graph_id="process-restore",
        artifacts=(
            ArtifactNode(
                artifact_id="process-state",
                kind=ArtifactKind.PROCESS_CHECKPOINT,
                size_bytes=len(payload),
                sha256=_sha(payload),
                rebuild_time_ms=12.0,
                compatibility=CompatibilityConstraint(
                    portable=False,
                    captured_host_fingerprint="b" * 64,
                ),
                source_relative_path="state.bin",
            ),
        ),
    )
    tier = StorageTierSpec(
        tier_id="local",
        kind=StorageKind.LOCAL_NVME,
        capacity_bytes=1024,
        read_bandwidth_bytes_per_second=1_000_000.0,
        base_read_latency_ms=1.0,
        local_path=str(tmp_path / "cache"),
    )
    profile = profile_local_startup(
        profile_id="restore-profile",
        graph=graph,
        host=_host(),
        tiers=(tier,),
        source_directory=source,
        output_directory=tmp_path / "profile",
        warmups=0,
        sample_count=3,
    )
    plan = compile_warmpath(
        graph=graph,
        profile=profile,
        objective=WarmPathObjective(),
    )
    assert plan.placements[0].mode == MaterializationMode.REBUILD
    assert plan.rejected_candidates[0].reason_code == "compatibility"
    assert "host fingerprint differs" in plan.rejected_candidates[0].explanation


def test_local_executor_restores_checksums_and_records_failures(tmp_path: Path) -> None:
    graph, source, tiers = _fixture(tmp_path)
    profile = profile_local_startup(
        profile_id="executor-profile",
        graph=graph,
        host=_host(),
        tiers=(tiers[0],),
        source_directory=source,
        output_directory=tmp_path / "profile",
        warmups=0,
        sample_count=3,
    )
    plan = compile_warmpath(
        graph=graph,
        profile=profile,
        objective=WarmPathObjective(),
        seed=7,
    )
    executor = LocalWarmPathExecutor(maximum_operation_seconds=2.0)
    result = executor.execute(
        execution_id="restore-success",
        plan=plan,
        graph=graph,
        host=_host(),
        tiers=(tiers[0],),
        source_directory=source,
        output_directory=tmp_path / "restored",
        seed=7,
    )
    assert result.success
    assert (tmp_path / "restored" / "weights.bin").read_bytes() == (
        source / "weights.bin"
    ).read_bytes()
    assert all(record.checksum_verified for record in result.records if record.status != "deferred")
    record_body = result.model_dump(mode="json", exclude={"artifact_hash"})
    assert result.artifact_hash == sha256_bytes(canonical_json(record_body).encode())

    failed = executor.execute(
        execution_id="restore-failure",
        plan=plan,
        graph=graph,
        host=_host(),
        tiers=(tiers[0],),
        source_directory=source,
        output_directory=tmp_path / "failed",
        seed=7,
        injected_failures=frozenset({"model-weights"}),
    )
    assert not failed.success
    assert failed.failure_reason == "OSError: deterministic injected restore failure"
    assert failed.records[-1].status == "failed"
    assert (tmp_path / "failed" / "warmpath-execution.json").is_file()

    bounded = LocalWarmPathExecutor(maximum_artifact_bytes=128)
    too_large = bounded.execute(
        execution_id="bounded-failure",
        plan=plan,
        graph=graph,
        host=_host(),
        tiers=(tiers[0],),
        source_directory=source,
        output_directory=tmp_path / "bounded",
        seed=7,
    )
    assert not too_large.success
    assert "exceeds executor byte limit" in (too_large.failure_reason or "")


def test_restore_failure_probability_and_rebuild_fallback_are_simulated(tmp_path: Path) -> None:
    graph, source, tiers = _fixture(tmp_path)
    failing_tier = tiers[0].model_copy(update={"restore_failure_probability": 1.0})
    profile = profile_local_startup(
        profile_id="failure-profile",
        graph=graph,
        host=_host(),
        tiers=(failing_tier,),
        source_directory=source,
        output_directory=tmp_path / "profile",
        warmups=0,
        sample_count=3,
    )
    placements = tuple(
        {
            "artifact_id": artifact.artifact_id,
            "tier_id": failing_tier.tier_id,
            "mode": MaterializationMode.EAGER_RESTORE,
            "prefetch_order": index,
            "expected_duration_ms": 1.0,
            "estimate_source": "measured",
        }
        for index, artifact in enumerate(graph.topological_order())
    )
    from sloforge.warmpath import ArtifactPlacement

    typed = tuple(ArtifactPlacement.model_validate(item) for item in placements)
    with pytest.raises(RuntimeError, match="every simulated"):
        simulate_cold_start(
            graph=graph,
            placements=typed,
            profile=profile,
            seed=11,
            trial_count=5,
        )


def test_host_memory_is_measured_and_pinned_memory_never_silently_falls_back(
    tmp_path: Path,
) -> None:
    graph, source, _ = _fixture(tmp_path)
    host_memory = StorageTierSpec(
        tier_id="host-memory",
        kind=StorageKind.HOST_MEMORY,
        capacity_bytes=1 << 20,
        read_bandwidth_bytes_per_second=10_000_000_000.0,
        base_read_latency_ms=0.01,
        local_path=str(tmp_path / "host-memory"),
    )
    profile = profile_local_startup(
        profile_id="host-memory-profile",
        graph=graph,
        host=_host(),
        tiers=(host_memory,),
        source_directory=source,
        output_directory=tmp_path / "profile",
        warmups=1,
        sample_count=3,
    )
    assert all(item.invocation.startswith("host-memory-copy:") for item in profile.measurements)
    pinned = host_memory.model_copy(
        update={"tier_id": "pinned-memory", "kind": StorageKind.PINNED_HOST_MEMORY}
    )
    with pytest.raises(ValueError, match="cannot measure"):
        profile_local_startup(
            profile_id="pinned-profile",
            graph=graph,
            host=_host(),
            tiers=(pinned,),
            source_directory=source,
            output_directory=tmp_path / "pinned-profile",
            sample_count=3,
        )


def test_executor_evicts_lru_entries_across_plans_and_mock_snapshots_are_deterministic(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "source-one"
    second_source = tmp_path / "source-two"
    first_checksum = create_mock_snapshot_artifact(
        first_source / "snapshot.bin", seed=101, size_bytes=256
    )
    second_checksum = create_mock_snapshot_artifact(
        second_source / "snapshot.bin", seed=202, size_bytes=256
    )
    duplicate_checksum = create_mock_snapshot_artifact(
        tmp_path / "duplicate.bin", seed=101, size_bytes=256
    )
    assert first_checksum == duplicate_checksum
    assert first_checksum != second_checksum
    tier = StorageTierSpec(
        tier_id="bounded-cache",
        kind=StorageKind.LOCAL_NVME,
        capacity_bytes=300,
        read_bandwidth_bytes_per_second=1_000_000.0,
        base_read_latency_ms=0.1,
        local_path=str(tmp_path / "bounded-cache"),
    )

    def prepare(
        identifier: str, checksum: str, source: Path
    ) -> tuple[ArtifactGraph, StartupProfile]:
        graph = ArtifactGraph(
            graph_id=f"graph-{identifier}",
            artifacts=(
                ArtifactNode(
                    artifact_id=identifier,
                    kind=ArtifactKind.PROCESS_CHECKPOINT,
                    size_bytes=256,
                    sha256=checksum,
                    source_relative_path="snapshot.bin",
                ),
            ),
        )
        profile = profile_local_startup(
            profile_id=f"profile-{identifier}",
            graph=graph,
            host=_host(),
            tiers=(tier,),
            source_directory=source,
            output_directory=tmp_path / f"profile-{identifier}",
            sample_count=3,
            warmups=0,
        )
        return graph, profile

    first_graph, first_profile = prepare("snapshot-one", first_checksum, first_source)
    second_graph, second_profile = prepare("snapshot-two", second_checksum, second_source)
    first_plan = compile_warmpath(
        graph=first_graph, profile=first_profile, objective=WarmPathObjective()
    )
    second_plan = compile_warmpath(
        graph=second_graph, profile=second_profile, objective=WarmPathObjective()
    )
    executor = LocalWarmPathExecutor()
    first = executor.execute(
        execution_id="first-restore",
        plan=first_plan,
        graph=first_graph,
        host=_host(),
        tiers=(tier,),
        source_directory=first_source,
        output_directory=tmp_path / "output-one",
    )
    second = executor.execute(
        execution_id="second-restore",
        plan=second_plan,
        graph=second_graph,
        host=_host(),
        tiers=(tier,),
        source_directory=second_source,
        output_directory=tmp_path / "output-two",
    )
    assert first.success and second.success
    assert second.cache_evictions == ("snapshot-one",)
    assert not (tmp_path / "bounded-cache" / f"snapshot-one-{first_checksum}").exists()
    assert (tmp_path / "bounded-cache" / f"snapshot-two-{second_checksum}").is_file()


def test_benchmark_template_contains_no_result_values() -> None:
    root = Path(__file__).parents[2]
    template = (root / "benchmarks/warmpath/local-profile.yaml").read_text(encoding="utf-8")
    assert "raw_samples:" in template
    assert "p95_ready_time_ms:" not in template
    assert "benchmark result" not in template.lower()


def test_robust_summary_interval_contains_median_under_ulp_interpolation() -> None:
    samples = (0.000833, 0.000667, 0.000667)
    median, _, _, low, high = robust_summary(samples, seed=2029)
    assert low <= median <= high
