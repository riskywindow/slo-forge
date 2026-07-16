"""Portable CPU-only WarmPath demonstration with measured local artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from sloforge.util import (
    canonical_json,
    environment_manifest,
    sha256_bytes,
    sha256_file,
    write_json,
)
from sloforge.warmpath import (
    ArtifactGraph,
    ArtifactKind,
    ArtifactNode,
    HostEnvironment,
    LocalWarmPathExecutor,
    SecurityClass,
    StorageKind,
    StorageTierSpec,
    WarmPathObjective,
    compile_warmpath,
    create_mock_snapshot_artifact,
    profile_local_startup,
    save_graph,
    save_plan,
    save_profile,
    simulate_cold_start,
)


class _DemoModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class WarmPathDemoArtifact(_DemoModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class WarmPathDemoManifest(_DemoModel):
    schema_version: Literal["sloforge.warmpath.demo/v1"] = "sloforge.warmpath.demo/v1"
    synthetic_snapshot: bool
    profile_source: Literal["measured"]
    plan_id: str
    optimizer: str
    candidates_evaluated: Annotated[int, Field(gt=0)]
    predicted_p50_ready_ms: Annotated[float, Field(ge=0.0)]
    predicted_p95_ready_ms: Annotated[float, Field(ge=0.0)]
    measured_ready_ms: Annotated[float, Field(ge=0.0)]
    restore_success: bool
    checksum_verified: bool
    artifact_count: Annotated[int, Field(gt=0)]
    deferred_artifact_count: Annotated[int, Field(ge=0)]
    artifacts: tuple[WarmPathDemoArtifact, ...]


_DEMO_OWNED_PATHS = (
    "input",
    "cache",
    "profile",
    "execution",
    "plan.json",
    "simulation.json",
    "manifest.json",
)


def _reset(path: Path, *, enabled: bool) -> None:
    if path.exists() and enabled:
        resolved = path.resolve()
        if resolved in {Path("/").resolve(), Path.home().resolve()}:
            raise ValueError("refusing to reset a broad directory")
        # The checked-in H6 evaluation is a sibling under artifacts/warmpath.
        # Reset only files owned by the local demo so running `warmpath-demo`
        # cannot invalidate the subsequent repository test gate.
        for relative in _DEMO_OWNED_PATHS:
            target = path / relative
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
    path.mkdir(parents=True, exist_ok=True)


def _host() -> HostEnvironment:
    manifest = environment_manifest(include_packages=True)
    return HostEnvironment(
        operating_system=platform.system().lower(),
        architecture=platform.machine().lower(),
        runtime="sloforge-local-reference",
        runtime_version="0.1.0",
        host_fingerprint=sha256_bytes(canonical_json(manifest).encode()),
        gpu_count=0,
    )


def _fixture(source: Path) -> ArtifactGraph:
    config = source / "config.json"
    tokenizer = source / "tokenizer.json"
    weights = source / "weights.bin"
    runtime_cache = source / "runtime.cache"
    source.mkdir(parents=True, exist_ok=True)
    config.write_text('{"architectures":["SyntheticMoE"]}\n', encoding="utf-8")
    tokenizer.write_text('{"type":"deterministic-byte-tokenizer"}\n', encoding="utf-8")
    create_mock_snapshot_artifact(weights, seed=2026, size_bytes=2 * 1024 * 1024)
    create_mock_snapshot_artifact(runtime_cache, seed=2027, size_bytes=256 * 1024)
    return ArtifactGraph(
        graph_id="warmpath-local-demo",
        artifacts=(
            ArtifactNode(
                artifact_id="model-config",
                kind=ArtifactKind.MODEL_CONFIG,
                size_bytes=config.stat().st_size,
                sha256=sha256_file(config),
                source_relative_path=config.name,
            ),
            ArtifactNode(
                artifact_id="tokenizer",
                kind=ArtifactKind.TOKENIZER,
                size_bytes=tokenizer.stat().st_size,
                sha256=sha256_file(tokenizer),
                dependencies=("model-config",),
                source_relative_path=tokenizer.name,
            ),
            ArtifactNode(
                artifact_id="model-weights",
                kind=ArtifactKind.MODEL_WEIGHTS,
                size_bytes=weights.stat().st_size,
                sha256=sha256_file(weights),
                dependencies=("model-config",),
                security_class=SecurityClass.RESTRICTED,
                source_relative_path=weights.name,
            ),
            ArtifactNode(
                artifact_id="runtime-cache",
                kind=ArtifactKind.RUNTIME_CACHE,
                size_bytes=runtime_cache.stat().st_size,
                sha256=sha256_file(runtime_cache),
                dependencies=("model-config",),
                required_for_readiness=False,
                lazy_restore_allowed=True,
                rebuild_time_ms=4.0,
                source_relative_path=runtime_cache.name,
            ),
        ),
    )


def _tiers(root: Path) -> tuple[StorageTierSpec, ...]:
    return (
        StorageTierSpec(
            tier_id="local-nvme",
            kind=StorageKind.LOCAL_NVME,
            capacity_bytes=16 * 1024 * 1024,
            read_bandwidth_bytes_per_second=1_000_000_000.0,
            base_read_latency_ms=0.10,
            maximum_parallel_reads=2,
            hourly_cost_per_gib=0.001,
            maximum_security_class=SecurityClass.RESTRICTED,
            local_path=str(root / "cache" / "nvme"),
        ),
        StorageTierSpec(
            tier_id="page-cache",
            kind=StorageKind.PAGE_CACHE,
            capacity_bytes=8 * 1024 * 1024,
            read_bandwidth_bytes_per_second=4_000_000_000.0,
            base_read_latency_ms=0.02,
            maximum_parallel_reads=4,
            maximum_security_class=SecurityClass.RESTRICTED,
            local_path=str(root / "cache" / "page"),
        ),
        StorageTierSpec(
            tier_id="host-memory",
            kind=StorageKind.HOST_MEMORY,
            capacity_bytes=4 * 1024 * 1024,
            read_bandwidth_bytes_per_second=12_000_000_000.0,
            base_read_latency_ms=0.005,
            maximum_parallel_reads=4,
            hourly_cost_per_gib=0.005,
            maximum_security_class=SecurityClass.RESTRICTED,
            local_path=str(root / "cache" / "memory"),
        ),
    )


def _artifact(path: Path, root: Path) -> WarmPathDemoArtifact:
    return WarmPathDemoArtifact(path=str(path.relative_to(root)), sha256=sha256_file(path))


def run_warmpath_demo(
    *, artifact_dir: Path, report_path: Path, reset: bool = False, seed: int = 2026
) -> WarmPathDemoManifest:
    _reset(artifact_dir, enabled=reset)
    source = artifact_dir / "input" / "files"
    graph = _fixture(source)
    graph_path = artifact_dir / "input" / "artifact-graph.json"
    save_graph(graph, graph_path)
    tiers = _tiers(artifact_dir)
    profile = profile_local_startup(
        profile_id="warmpath-local-demo",
        graph=graph,
        host=_host(),
        tiers=tiers,
        source_directory=source,
        output_directory=artifact_dir / "profile",
        warmups=2,
        sample_count=7,
        seed=seed,
    )
    profile_path = artifact_dir / "profile" / "profile.json"
    save_profile(profile, profile_path)
    plan = compile_warmpath(
        graph=graph,
        profile=profile,
        objective=WarmPathObjective(
            ready_time_weight=1.0,
            hourly_cost_weight=0.05,
            failure_risk_weight=1_000.0,
        ),
        seed=seed,
    )
    plan_path = artifact_dir / "plan.json"
    save_plan(plan, plan_path)
    simulation = simulate_cold_start(
        graph=graph,
        placements=plan.placements,
        profile=profile,
        seed=seed,
        trial_count=31,
    )
    simulation_path = artifact_dir / "simulation.json"
    write_json(simulation_path, simulation.model_dump(mode="json"))
    execution = LocalWarmPathExecutor(maximum_operation_seconds=10.0).execute(
        execution_id="warmpath-local-demo",
        plan=plan,
        graph=graph,
        host=profile.host,
        tiers=tiers,
        source_directory=source,
        output_directory=artifact_dir / "execution",
        seed=seed,
    )
    execution_path = artifact_dir / "execution" / "warmpath-execution.json"
    verified = all(
        record.checksum_verified
        for record in execution.records
        if record.status not in {"deferred", "kept_warm"}
    )
    tracked = (graph_path, profile_path, plan_path, simulation_path, execution_path)
    manifest = WarmPathDemoManifest(
        synthetic_snapshot=True,
        profile_source="measured",
        plan_id=plan.plan_id,
        optimizer=plan.optimizer,
        candidates_evaluated=plan.evaluated_candidate_count,
        predicted_p50_ready_ms=simulation.p50_ready_time_ms,
        predicted_p95_ready_ms=simulation.p95_ready_time_ms,
        measured_ready_ms=execution.ready_time_ms,
        restore_success=execution.success,
        checksum_verified=verified,
        artifact_count=len(graph.artifacts),
        deferred_artifact_count=sum(record.status == "deferred" for record in execution.records),
        artifacts=tuple(_artifact(path, artifact_dir) for path in tracked),
    )
    manifest_path = artifact_dir / "manifest.json"
    write_json(manifest_path, manifest.model_dump(mode="json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# WarmPath local demonstration\n\n"
        "The profile uses measured local reads and checksum verification. The snapshot payload "
        "is an explicitly synthetic deterministic fixture.\n\n"
        f"- Plan: `{manifest.plan_id}` ({manifest.optimizer}; "
        f"{manifest.candidates_evaluated} candidates)\n"
        f"- Predicted p50/p95 readiness: {manifest.predicted_p50_ready_ms:.3f} / "
        f"{manifest.predicted_p95_ready_ms:.3f} ms\n"
        f"- Measured local execution readiness: {manifest.measured_ready_ms:.3f} ms\n"
        f"- Restore/checksums: {'pass' if manifest.restore_success and verified else 'fail'}\n"
        f"- Deferred non-critical artifacts: {manifest.deferred_artifact_count}\n\n"
        "All reported values are loaded from `artifacts/warmpath/manifest.json`; raw stage "
        "samples are retained under `artifacts/warmpath/profile/raw/`.\n",
        encoding="utf-8",
    )
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if f"{float(loaded['measured_ready_ms']):.3f}" not in report_path.read_text(encoding="utf-8"):
        raise RuntimeError("WarmPath report is not derived from its manifest")
    if not execution.success or not verified:
        raise RuntimeError("WarmPath local restore or checksum verification failed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/warmpath"))
    parser.add_argument("--report", type=Path, default=Path("reports/warmpath-evaluation.md"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    result = run_warmpath_demo(
        artifact_dir=args.artifact_dir,
        report_path=args.report,
        reset=args.reset,
        seed=args.seed,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
