"""End-to-end deterministic CPU demonstration for SLOForge Genesis."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import random
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import CodeType

from pydantic import BaseModel, ConfigDict

from sloforge.genesis.capsule import (
    CapsuleValidationReport,
    GenesisCapsule,
    ValidationContext,
    build_local_capsule,
    load_capsule,
    validate_capsule,
)
from sloforge.genesis.compiler import initialize_genesis_run
from sloforge.genesis.evolution import (
    CapsuleReference,
    ChallengerSpec,
    EvolutionConfig,
    EvolutionController,
    EvolutionPhase,
    EvolutionStore,
    ExecutionTarget,
    IsolationContract,
    IsolationMode,
    TransitionCategory,
    TriggerObservation,
    local_gate_evidence_validator,
    run_local_evolution_fixture,
)
from sloforge.genesis.frontend import inspect_reference_package, load_reference_package
from sloforge.genesis.ir import canonical_json, write_canonical
from sloforge.genesis.kernel_lab import (
    AttributionScope,
    BottleneckEvidence,
    EvidenceSource,
    KernelBenchmarkConfig,
    RawBottleneckRecord,
    run_kernel_lab_demo,
)
from sloforge.genesis.reports import export_genesis_ui_bundle
from sloforge.genesis.runtime.adapter import ReferenceRuntimeAdapter
from sloforge.genesis.synthesis import synthesize_local_run
from sloforge.lineage.demo import run_lineage_transfer_demo
from sloforge.redteam import run_demo as run_redteam_demo
from sloforge.util import sha256_file


class GenesisDemoResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = "1.0.0"
    seed: int
    output_directory: str
    package_id: str
    operator_count: int
    state_field_count: int
    baseline_genome_hash: str
    accepted_candidate_id: str
    accepted_genome_hash: str
    cross_layer_accepted: bool
    rejected_candidate_ids: tuple[str, ...]
    minimized_counterexample_ids: tuple[str, ...]
    learned_constraint_ids: tuple[str, ...]
    runtime_differential_passed: bool
    capsule_path: str
    capsule_digest: str
    capsule_promotion_eligible: bool
    capsule_local_evolution_eligible: bool
    capsule_external_production_eligible: bool
    redteam_finding_count: int
    redteam_replayed_count: int
    kernel_candidate_count: int
    kernel_speedup_claim_count: int
    kernel_measurement_scope: str
    kernel_causal_attribution: bool
    evolution_promoted: bool
    active_stream_preserved: bool
    physical_degradation_triggered: bool
    hardware_backed: bool
    report_path: str
    ui_bundle_path: str


def _write(path: Path, value: BaseModel | dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite demo artifact: {path}")
    path.write_bytes(canonical_json(value) + b"\n")


def _safe_reset(output: Path, repository: Path) -> None:
    if not output.exists():
        return
    if output.is_symlink():
        raise ValueError(f"refusing to reset symlinked output path: {output}")
    resolved = output.resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), repository.resolve()}
    if resolved in forbidden or len(resolved.parts) < 4:
        raise ValueError(f"refusing to reset unsafe output path: {resolved}")
    shutil.rmtree(resolved)


def _local_hardware_identity() -> dict[str, object]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }


def _local_hardware_fingerprint() -> str:
    return hashlib.sha256(canonical_json(_local_hardware_identity())).hexdigest()


def _profile_seed(seed: int, request_id: str, phase: str, position: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\0{request_id}\0{phase}\0{position}".encode()).digest()[:8],
        "big",
    )


def _measure_bottleneck(path: Path, *, seed: int, package_root: Path) -> BottleneckEvidence:
    """Profile the focused operator inside a retained reference workload trace."""

    package = load_reference_package(package_root)
    corpus_path = package.resolve(package.manifest.quality_contract.search_corpus)
    corpus = tuple(
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not corpus or not all(isinstance(item, dict) for item in corpus):
        raise ValueError("kernel target profiling requires a non-empty object corpus")
    generator = random.Random(seed)
    requests: list[dict[str, object]] = []
    for index in range(6):
        sample = corpus[index % len(corpus)]
        maximum = sample.get("maximum_new_tokens")
        text = sample.get("text")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or not isinstance(text, str):
            raise TypeError("kernel target corpus row has invalid text or token bound")
        requests.append(
            {
                "request_id": f"kernel-profile-{index:02d}",
                "text": text + chr(ord("a") + generator.randrange(26)) * (index % 3),
                "maximum_new_tokens": max(1, min(8, maximum + index % 2)),
            }
        )
    workload_path = path.with_name("reference-workload-trace.json")
    workload_payload = (
        canonical_json(
            {
                "schema_version": "sloforge.genesis.kernel-target-trace/v1",
                "seed": seed,
                "requests": requests,
            }
        )
        + b"\n"
    )
    workload_path.parent.mkdir(parents=True, exist_ok=True)
    if workload_path.exists():
        raise FileExistsError(f"refusing to overwrite kernel workload trace: {workload_path}")
    workload_path.write_bytes(workload_payload)
    workload_fingerprint = hashlib.sha256(workload_payload).hexdigest()
    samples: list[int] = []
    other_samples: list[int] = []
    reference_path = package.resolve(package.manifest.reference_module).resolve(strict=True)
    for trial in range(7):
        adapter = ReferenceRuntimeAdapter(
            reference_path=reference_path,
            tokenizer_path=package.resolve(package.manifest.tokenizer_module),
            entry_points=package.manifest.entry_points,
            identity=f"kernel_target_profile_{seed}_{trial}",
            seed=seed,
        )
        profiler = cProfile.Profile()
        profiler.enable()
        for request in requests:
            request_id = str(request["request_id"])
            maximum_new_tokens = request["maximum_new_tokens"]
            if not isinstance(maximum_new_tokens, int) or isinstance(maximum_new_tokens, bool):
                raise TypeError("kernel target request token bound must be an integer")
            prompt = adapter.tokenize(str(request["text"]))
            state = adapter.allocate_state(
                request_id,
                prompt,
                _profile_seed(seed, request_id, "allocate", 0),
            )
            state = adapter.prefill(
                prompt,
                state,
                _profile_seed(seed, request_id, "prefill", 0),
            )
            previous = prompt[-1]
            for position in range(maximum_new_tokens):
                result = adapter.decode_step(
                    previous,
                    state,
                    position,
                    _profile_seed(seed, request_id, "decode", position),
                )
                state = result.state
                previous = adapter.sample(
                    result.logits,
                    _profile_seed(seed, request_id, "sample", position),
                )
        profiler.disable()
        operator_seconds = 0.0
        advance_seconds = 0.0
        for entry in profiler.getstats():
            code = entry.code
            if not isinstance(code, CodeType) or Path(code.co_filename).resolve() != reference_path:
                continue
            if code.co_name == "quantized_state_update":
                operator_seconds += entry.totaltime
            elif code.co_name == "_advance":
                advance_seconds += entry.totaltime
        operator_ns = round(operator_seconds * 1_000_000_000)
        comparison_ns = round((advance_seconds - operator_seconds) * 1_000_000_000)
        if operator_ns <= 0 or comparison_ns <= 0:
            raise RuntimeError(
                "reference workload profile did not resolve positive operator timing"
            )
        samples.append(operator_ns)
        other_samples.append(comparison_ns)
    fraction = sum(samples) / (sum(samples) + sum(other_samples))
    raw = RawBottleneckRecord(
        operator_id="quantized-state-update",
        inclusive_cpu_time_ns=tuple(samples),
        comparison_work_time_ns=tuple(other_samples),
        operator_probe_fraction=fraction,
        attribution_scope=AttributionScope.REFERENCE_WORKLOAD_TRACE_PROFILE.value,
        workload_fingerprint=workload_fingerprint,
        workload_trace_path=str(workload_path.resolve()),
        workload_trace_sha256=sha256_file(workload_path),
        seed=seed,
    )
    _write(path, raw)
    hardware_fingerprint = _local_hardware_fingerprint()
    return BottleneckEvidence(
        evidence_id="cpu-trace-profile-hybrid-quantized-state",
        source=EvidenceSource.CPU_PROFILE_MEASURED,
        attribution_scope=AttributionScope.REFERENCE_WORKLOAD_TRACE_PROFILE,
        causal_attribution=False,
        operator_id="quantized-state-update",
        genome_region="state.quantized_state",
        observed_fraction=fraction,
        sample_count=len(samples),
        deterministic_seed=seed,
        raw_evidence_path=str(path.resolve()),
        raw_evidence_sha256=sha256_file(path),
        hardware_fingerprint=f"cpu-{hardware_fingerprint[:12]}",
        workload_fingerprint=workload_fingerprint,
        synthetic=False,
    )


def run_genesis_demo(
    output: Path,
    *,
    seed: int = 73129,
    runtime_seed: int | None = None,
    reset: bool = False,
    observed_at: datetime | None = None,
) -> GenesisDemoResult:
    """Run the CPU vertical slice, real rejection, kernel experiment, and local evolution."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if output.exists() and output.is_symlink():
        raise ValueError(f"refusing to use symlinked output path: {output}")
    repository = Path(__file__).resolve().parents[3]
    if reset:
        _safe_reset(output, repository)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"demo output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    package_root = repository / "models/reference_tasks/hybrid_decoder"
    package = load_reference_package(package_root)
    inputs = output / "inputs"
    inputs.mkdir()
    workload = inputs / "workload.jsonl"
    hardware = inputs / "hardware.json"
    shutil.copyfile(package_root / "search_samples.jsonl", workload)
    _write(
        hardware,
        {
            "schema_version": "1.0.0",
            "architecture": "cpu",
            "memory_bytes": 8 * 1024**3,
            "device_count": 1,
            "measurement_mode": "local-cpu",
            "measured_identity": _local_hardware_identity(),
            "measured_fingerprint": _local_hardware_fingerprint(),
        },
    )
    inspection_directory = output / "inspection"
    inspection = inspect_reference_package(
        package_root,
        use_torch_export=False,
        output_path=inspection_directory / "inspection.json",
    )
    _write(
        inspection_directory / "reference_package.json",
        {
            "schema_version": "1.0.0",
            "package_root": str(package_root.resolve()),
            "package_hash": package.package_hash,
            "manifest_hash": package.manifest_hash,
            "inspection_seed": seed,
        },
    )
    run_directory = output / "run"
    effective_runtime_seed = seed if runtime_seed is None else runtime_seed
    initialized = initialize_genesis_run(
        package_root,
        inspection,
        run_directory,
        seed=effective_runtime_seed,
        workload_contract_hash=sha256_file(workload),
        hardware_contract_hash=sha256_file(hardware),
    )
    _write(
        run_directory / "run_manifest.json",
        {
            "schema_version": "1.0.0",
            "seed": seed,
            "runtime_seed": effective_runtime_seed,
            "genome_hash": initialized.genome_hash,
            "runtime_id": initialized.runtime.runtime_id,
            "package_hash": initialized.runtime.package_hash,
            "workload_contract": {
                "path": str(workload.resolve()),
                "sha256": sha256_file(workload),
            },
            "hardware_contract": {
                "path": str(hardware.resolve()),
                "sha256": sha256_file(hardware),
            },
        },
    )
    synthesis = synthesize_local_run(run_directory, seed=seed, budget_usd=0.0)
    if synthesis.accepted_candidate_id is None or synthesis.accepted_genome_hash is None:
        raise RuntimeError(
            "local synthesis produced no candidate accepted by all downstream runtime, "
            "model-check, and simulation gates"
        )
    candidate_directory = run_directory / "candidates" / synthesis.accepted_candidate_id
    default_timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(
        seconds=seed % (366 * 24 * 60 * 60)
    )
    timestamp = (observed_at or default_timestamp).astimezone(UTC)
    capsule_directory = output / "capsule"
    capsule_result = build_local_capsule(
        candidate_directory,
        capsule_directory,
        observed_at=timestamp,
    )

    capsule = load_capsule(Path(capsule_result.capsule_path))
    context = ValidationContext.model_validate_json(
        Path(capsule_result.context_path).read_bytes(), strict=True
    )
    challenger_validation: tuple[GenesisCapsule, Path, Path, ValidationContext] | None = None

    def validator(path: Path) -> CapsuleValidationReport:
        resolved = path.resolve()
        if resolved == Path(capsule_result.capsule_path).resolve():
            return validate_capsule(capsule, capsule_directory, context)
        if challenger_validation is not None:
            (
                challenger_capsule,
                challenger_manifest,
                challenger_directory,
                challenger_context,
            ) = challenger_validation
            if resolved == challenger_manifest.resolve():
                return validate_capsule(
                    challenger_capsule,
                    challenger_directory,
                    challenger_context,
                )
        raise ValueError("controller requested validation of an unknown capsule")

    champion = CapsuleReference(
        capsule_id="genesis-champion",
        capsule_digest=capsule_result.capsule_digest,
        genome_hash=synthesis.accepted_genome_hash,
        path=capsule_result.capsule_path,
    )
    gate_evidence_root = output / "evolution/runtime-gates"
    controller = EvolutionController.initialize(
        store=EvolutionStore(output / "evolution/controller-state.json"),
        capsule_validator=validator,
        gate_evidence_validator=local_gate_evidence_validator(gate_evidence_root),
        config=EvolutionConfig(
            execution_target=ExecutionTarget.LOCAL,
            minimum_shadow_samples=2,
            minimum_canary_samples=3,
            maximum_p95_ttft_ratio=10.0,
            maximum_p99_tpot_ratio=10.0,
        ),
        deployment_id="genesis-demo",
        champion=champion,
        seed=seed,
        now_ms=0,
    )
    controller.open_stream(
        "active-stream-before-promotion",
        event_id="demo-open-active-stream",
        observed_at_ms=1,
        externally_visible_output=True,
    )
    triggered = controller.observe_trigger(
        TriggerObservation(
            event_id="fixture-workload-drift",
            observed_at_ms=10,
            workload_js_divergence=0.40,
            detail="fixture prompt distribution moved from short to bimodal",
        )
    )
    write_canonical(triggered, output / "evolution/triggered-snapshot.json")

    # Evolve from one independently validated capsule to another.  The
    # workload trigger above starts evolution before this separate immutable
    # challenger run. The search seed is distinct while the fixed reference-
    # model seed and content-addressed contracts remain unchanged.
    challenger_run = output / "evolution/challenger-run"
    challenger_initialized = initialize_genesis_run(
        package_root,
        inspection,
        challenger_run,
        seed=effective_runtime_seed,
        workload_contract_hash=sha256_file(workload),
        hardware_contract_hash=sha256_file(hardware),
    )
    _write(
        challenger_run / "run_manifest.json",
        {
            "schema_version": "1.0.0",
            "seed": seed + 1,
            "runtime_seed": effective_runtime_seed,
            "genome_hash": challenger_initialized.genome_hash,
            "runtime_id": challenger_initialized.runtime.runtime_id,
            "package_hash": challenger_initialized.runtime.package_hash,
            "workload_contract": {
                "path": str(workload.resolve()),
                "sha256": sha256_file(workload),
            },
            "hardware_contract": {
                "path": str(hardware.resolve()),
                "sha256": sha256_file(hardware),
            },
        },
    )
    challenger_synthesis = synthesize_local_run(challenger_run, seed=seed + 1, budget_usd=0.0)
    if (
        challenger_synthesis.accepted_candidate_id is None
        or challenger_synthesis.accepted_genome_hash is None
    ):
        raise RuntimeError("evolution synthesis produced no accepted challenger")
    challenger_candidate_directory = (
        challenger_run / "candidates" / challenger_synthesis.accepted_candidate_id
    )
    challenger_capsule_directory = output / "evolution/challenger-capsule"
    challenger_capsule_result = build_local_capsule(
        challenger_candidate_directory,
        challenger_capsule_directory,
        observed_at=timestamp + timedelta(seconds=1),
    )
    challenger_capsule = load_capsule(Path(challenger_capsule_result.capsule_path))
    challenger_context = ValidationContext.model_validate_json(
        Path(challenger_capsule_result.context_path).read_bytes(), strict=True
    )
    challenger_validation = (
        challenger_capsule,
        Path(challenger_capsule_result.capsule_path),
        challenger_capsule_directory,
        challenger_context,
    )
    synthesis_event_path = output / "evolution/challenger-synthesis-event.json"
    synthesis_event = {
        "schema_version": "sloforge.genesis.evolution.synthesis-event/v1",
        "event_id": "demo-synthesize-challenger",
        "observed_at_ms": 15,
        "trigger_event_id": "fixture-workload-drift",
        "trigger_snapshot_sha256": sha256_file(output / "evolution/triggered-snapshot.json"),
        "search_seed": seed + 1,
        "runtime_seed": effective_runtime_seed,
        "candidate_id": challenger_synthesis.accepted_candidate_id,
        "genome_hash": challenger_synthesis.accepted_genome_hash,
        "capsule_digest": challenger_capsule_result.capsule_digest,
    }
    _write(synthesis_event_path, synthesis_event)

    redteam = run_redteam_demo(output / "redteam", seed=seed)
    bottleneck = _measure_bottleneck(
        output / "kernel/raw-bottleneck.json",
        seed=seed,
        package_root=package_root,
    )
    kernel = run_kernel_lab_demo(
        evidence=bottleneck,
        output_root=output / "kernel/lab",
        seed=seed,
        candidate_limit=2,
        reference_package_root=package_root,
        benchmark_config=KernelBenchmarkConfig(
            deterministic_seed=seed,
            warmup_count=1,
            repetitions=7,
            micro_iterations=8,
            token_loop_iterations=2,
            token_steps=3,
            bootstrap_rounds=100,
            wall_time_seconds=10.0,
        ),
    )

    challenger = CapsuleReference(
        capsule_id="genesis-corrected",
        capsule_digest=challenger_capsule_result.capsule_digest,
        genome_hash=challenger_synthesis.accepted_genome_hash,
        path=challenger_capsule_result.capsule_path,
    )
    isolation = output / "evolution/challenger-isolation"
    isolation.mkdir(parents=True)
    spec = ChallengerSpec(
        candidate_id=challenger_synthesis.accepted_candidate_id,
        capsule=challenger,
        transition_category=TransitionCategory.REQUEST_BOUNDARY_SWAP,
        isolation=IsolationContract(
            mode=IsolationMode.SANDBOX,
            writable_artifact_root=str(isolation.resolve()),
        ),
        active_stream_compatible=False,
    )
    promoted = run_local_evolution_fixture(
        controller,
        spec,
        start_at_ms=10,
        evidence_directory=gate_evidence_root,
    )
    active_preserved = any(
        stream.stream_id == "active-stream-before-promotion"
        and stream.capsule_id == champion.capsule_id
        for stream in promoted.active_streams
    )
    write_canonical(promoted, output / "evolution/promoted-snapshot.json")
    degraded = controller.observe_trigger(
        TriggerObservation(
            event_id="demo-fabric-degradation",
            observed_at_ms=100,
            fabric_health_ratio=0.70,
            detail="simulated selected-link degradation after promotion",
        )
    )
    write_canonical(degraded, output / "evolution/degraded-snapshot.json")
    timeline_events: list[dict[str, object]] = []
    for audit in degraded.audit:
        timeline_events.append({"source": "controller_audit", **audit.model_dump(mode="json")})
        if audit.event_id == "fixture-workload-drift":
            timeline_events.append(
                {
                    "source": "artifact_bound_synthesis",
                    "action": "synthesize_challenger",
                    "artifact_path": str(synthesis_event_path.resolve()),
                    "artifact_sha256": sha256_file(synthesis_event_path),
                    **synthesis_event,
                }
            )
    _write(
        output / "evolution/timeline.json",
        {
            "schema_version": "1.0.0",
            "source": "controller_audit_records_and_artifact_bound_synthesis",
            "events": timeline_events,
        },
    )
    run_lineage_transfer_demo(output / "lineage", seed=seed)

    report_path = output / "GENESIS_DEMO_REPORT.json"
    ui_bundle_path = output / "genesis-ui-bundle.json"
    result = GenesisDemoResult(
        seed=seed,
        output_directory=str(output.resolve()),
        package_id=inspection.package_id,
        operator_count=len(inspection.graph.operators),
        state_field_count=len(inspection.graph.state_fields),
        baseline_genome_hash=synthesis.baseline_genome_hash,
        accepted_candidate_id=synthesis.accepted_candidate_id,
        accepted_genome_hash=synthesis.accepted_genome_hash,
        cross_layer_accepted=synthesis.cross_layer_accepted,
        rejected_candidate_ids=synthesis.rejected_candidate_ids,
        minimized_counterexample_ids=synthesis.counterexample_ids,
        learned_constraint_ids=synthesis.constraint_ids,
        runtime_differential_passed=synthesis.runtime_differential_passed,
        capsule_path=capsule_result.capsule_path,
        capsule_digest=capsule_result.capsule_digest,
        capsule_promotion_eligible=capsule_result.promotion_eligible,
        capsule_local_evolution_eligible=capsule_result.local_evolution_eligible,
        capsule_external_production_eligible=capsule_result.external_production_eligible,
        redteam_finding_count=redteam.finding_count,
        redteam_replayed_count=redteam.reproduced_regressions,
        kernel_candidate_count=len(kernel.candidates),
        kernel_speedup_claim_count=sum(
            decision.status.value == "accepted" for decision in kernel.decisions
        ),
        kernel_measurement_scope="cpu_generated_runtime_end_to_end_serving",
        kernel_causal_attribution=bottleneck.causal_attribution,
        evolution_promoted=promoted.phase is EvolutionPhase.PROMOTED,
        active_stream_preserved=active_preserved,
        physical_degradation_triggered=degraded.active_trigger is not None,
        hardware_backed=False,
        report_path=str(report_path.resolve()),
        ui_bundle_path=str(ui_bundle_path.resolve()),
    )
    _write(report_path, result)
    export_genesis_ui_bundle(output, ui_bundle_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73129)
    parser.add_argument(
        "--runtime-seed",
        type=int,
        default=73129,
        help="reference-model initialization seed bound to the fixed quality corpus",
    )
    parser.add_argument("--reset", action="store_true")
    arguments = parser.parse_args(argv)
    result = run_genesis_demo(
        arguments.output,
        seed=arguments.seed,
        runtime_seed=arguments.runtime_seed,
        reset=arguments.reset,
    )
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["GenesisDemoResult", "main", "run_genesis_demo"]
