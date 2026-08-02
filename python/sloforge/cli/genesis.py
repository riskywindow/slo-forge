"""CLI for fail-closed zero-day inspection and baseline runtime synthesis."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import statistics
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import typer
import yaml
from rich.console import Console

from sloforge.genesis.capsule import (
    ArtifactRole,
    CapsuleValidationReport,
    Digest,
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
    GateStage,
    IsolationContract,
    IsolationMode,
    TransitionCategory,
    TriggerObservation,
    collect_local_gate_evidence,
    local_gate_evidence_validator,
)
from sloforge.genesis.frontend import InspectionResult, inspect_reference_package
from sloforge.genesis.frontend.package import load_reference_package
from sloforge.genesis.ir import (
    Candidate,
    CandidateSuccessState,
    canonical_hash,
    load_candidate,
    load_inference_genome,
)
from sloforge.genesis.sandbox import SandboxLimits, SandboxRequest, execute_sandboxed
from sloforge.genesis.search import CandidateDesign
from sloforge.genesis.synthesis import CancellationPolicyVerifier, synthesize_local_run
from sloforge.util import sha256_file

genesis_app = typer.Typer(
    help="Synthesize proof-carrying inference implementations from reference packages.",
    no_args_is_help=True,
)
capsule_app = typer.Typer(
    help="Build and independently validate hash-addressed Genesis capsules.",
    no_args_is_help=True,
)
genesis_app.add_typer(capsule_app, name="capsule")
console = Console()

_REPLAY_RUNNER = '''"""Trusted bounded adapter for one validated capsule replay."""
import argparse
import json
from pathlib import Path

from sloforge.genesis.runtime import load_generated_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    runtime = load_generated_runtime(Path("runtime_config.json"), seed=args.seed)
    results = []
    runtime.start()
    try:
        with args.trace.open("rb") as handle:
            for index, encoded in enumerate(handle):
                if index >= 128:
                    raise ValueError("trace exceeds 128-request replay bound")
                if len(encoded) > 65536:
                    raise ValueError("trace request exceeds 65536-byte bound")
                value = json.loads(encoded)
                if not isinstance(value, dict):
                    raise TypeError("trace records must be JSON objects")
                request_id = str(value.get("request_id", f"replay-{index}"))
                text = str(value.get("text", ""))
                maximum_new_tokens = int(value.get("maximum_new_tokens", 1))
                timeout_seconds = float(value.get("timeout_seconds", 3.0))
                if not request_id or len(request_id.encode()) > 256:
                    raise ValueError("request_id must contain at most 256 bytes")
                if len(text.encode()) > 65536:
                    raise ValueError("request text exceeds 65536-byte bound")
                if not 1 <= maximum_new_tokens <= 256:
                    raise ValueError("maximum_new_tokens must be between 1 and 256")
                if not 0.1 <= timeout_seconds <= 3.0:
                    raise ValueError("timeout_seconds must be between 0.1 and 3.0")
                request_seed = value.get("seed", args.seed + index)
                handle_result = runtime.submit_text(
                    request_id=request_id,
                    text=text,
                    maximum_new_tokens=maximum_new_tokens,
                    seed=int(request_seed),
                    timeout_seconds=timeout_seconds,
                )
                events = [
                    {
                        "kind": event.kind.value,
                        "sequence": event.sequence,
                        "token_id": event.token_id,
                    }
                    for event in handle_result.events(3.0)
                ]
                if not events or events[-1]["kind"] not in {"completed", "cancelled"}:
                    raise RuntimeError("replay request did not reach a terminal event")
                results.append({"request_id": handle_result.request_id, "events": events})
    finally:
        runtime.shutdown()
    print(json.dumps({"passed": bool(results), "requests": results}, sort_keys=True))
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _json_result(value: object) -> None:
    console.print_json(json.dumps(value, default=str))


def _atomic_json(path: Path, value: object) -> None:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise typer.BadParameter(f"refusing to overwrite evidence artifact: {path}") from error


def _package_path(reference: Path) -> Path:
    if reference.is_dir():
        return reference
    if reference.is_file():
        if reference.name in {
            "reference_package.json",
            "reference_package.yaml",
            "reference_package.yml",
        }:
            return reference
        return reference.parent
    raise typer.BadParameter(f"reference package does not exist: {reference}")


@genesis_app.command("inspect")
def inspect_command(
    reference: Annotated[Path, typer.Option("--reference", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    tokenizer: Annotated[Path | None, typer.Option("--tokenizer", exists=True)] = None,
    contract: Annotated[
        Path | None, typer.Option("--contract", exists=True, dir_okay=False)
    ] = None,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 73129,
    torch_export: Annotated[bool, typer.Option("--torch-export/--no-torch-export")] = False,
) -> None:
    """Inspect a reference package without importing its model source by default."""

    package_path = _package_path(reference)
    package = load_reference_package(package_path)
    if reference.is_file() and reference.name == package.manifest.reference_module:
        expected = package.resolve(package.manifest.reference_module)
        if reference.resolve() != expected:
            raise typer.BadParameter("--reference does not match the package manifest")
    if tokenizer is not None:
        expected_tokenizer = package.resolve(package.manifest.tokenizer_module)
        resolved_tokenizer = tokenizer.resolve()
        if resolved_tokenizer not in {expected_tokenizer, expected_tokenizer.parent}:
            raise typer.BadParameter("--tokenizer does not match the package manifest")
    if contract is not None and contract.resolve() != package.manifest_path:
        raise typer.BadParameter("--contract must be the package manifest")

    inspection = inspect_reference_package(
        package_path,
        use_torch_export=torch_export,
        output_path=output / "inspection.json",
    )
    _atomic_json(
        output / "reference_package.json",
        {
            "schema_version": "1.0.0",
            "package_root": str(package.root),
            "package_hash": package.package_hash,
            "manifest_hash": package.manifest_hash,
            "inspection_seed": seed,
        },
    )
    _json_result(
        {
            "output": str(output),
            "package_id": inspection.package_id,
            "package_hash": inspection.package_hash,
            "operators": len(inspection.graph.operators),
            "state_fields": len(inspection.graph.state_fields),
            "diagnostics": len(inspection.diagnostics),
            "unsupported": inspection.has_unsupported_behavior,
            "torch_export_exercised": inspection.torch_export is not None,
            "seed": seed,
        }
    )


def _load_inspection(path: Path) -> tuple[InspectionResult, Path]:
    if path.is_dir():
        inspection_path = path / "inspection.json"
        package_reference_path = path / "reference_package.json"
    else:
        inspection_path = path
        package_reference_path = path.with_name("reference_package.json")
    if not inspection_path.is_file():
        raise typer.BadParameter(f"inspection artifact is missing: {inspection_path}")
    if not package_reference_path.is_file():
        raise typer.BadParameter(
            f"inspection package provenance is missing: {package_reference_path}"
        )
    inspection = InspectionResult.model_validate_json(
        inspection_path.read_text(encoding="utf-8"), strict=True
    )
    package_reference = json.loads(package_reference_path.read_text(encoding="utf-8"))
    if not isinstance(package_reference, dict):
        raise typer.BadParameter("reference_package.json must be an object")
    package_root = package_reference.get("package_root")
    if not isinstance(package_root, str) or not package_root:
        raise typer.BadParameter("reference_package.json is missing package_root")
    package_path = Path(package_root)
    package = load_reference_package(package_path)
    if package.package_hash != inspection.package_hash:
        raise typer.BadParameter("reference package changed after inspection")
    return inspection, package_path


@genesis_app.command("initialize")
def initialize_command(
    inspection: Annotated[Path, typer.Option("--inspection", exists=True)],
    workload: Annotated[Path, typer.Option("--workload", exists=True, dir_okay=False)],
    hardware: Annotated[Path, typer.Option("--hardware", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    seed: Annotated[int, typer.Option("--seed", min=0)] = 73129,
) -> None:
    """Compile inspection evidence into a genome and generated baseline runtime."""

    inspection_result, package_path = _load_inspection(inspection)
    workload_hash = sha256_file(workload)
    hardware_hash = sha256_file(hardware)
    result = initialize_genesis_run(
        package_path,
        inspection_result,
        output,
        seed=seed,
        workload_contract_hash=workload_hash,
        hardware_contract_hash=hardware_hash,
    )
    _atomic_json(
        output / "run_manifest.json",
        {
            "schema_version": "1.0.0",
            "seed": seed,
            "genome_hash": result.genome_hash,
            "runtime_id": result.runtime.runtime_id,
            "package_hash": result.runtime.package_hash,
            "workload_contract": {"path": str(workload.resolve()), "sha256": workload_hash},
            "hardware_contract": {"path": str(hardware.resolve()), "sha256": hardware_hash},
        },
    )
    _json_result(
        {
            "output": str(output),
            "genome_id": result.genome.genome_id,
            "genome_hash": result.genome_hash,
            "runtime_id": result.runtime.runtime_id,
            "runtime": str(result.runtime.output_directory),
            "seed": seed,
        }
    )


def _validated_run(run: Path) -> tuple[dict[str, object], str]:
    manifest_path = run / "run_manifest.json"
    genome_path = run / "inference_genome.json"
    if not manifest_path.is_file() or not genome_path.is_file():
        raise typer.BadParameter("run is missing run_manifest.json or inference_genome.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise typer.BadParameter("run_manifest.json must be an object")
    genome_hash = canonical_hash(load_inference_genome(genome_path))
    if manifest.get("genome_hash") != genome_hash:
        raise typer.BadParameter("run genome hash does not match its manifest")
    return manifest, genome_hash


@genesis_app.command("synthesize")
def synthesize_command(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    budget_usd: Annotated[float, typer.Option("--budget-usd", min=0.0)] = 0.0,
    objectives: Annotated[
        Path | None, typer.Option("--objectives", exists=True, dir_okay=False)
    ] = None,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 73129,
) -> None:
    """Run deterministic local synthesis; paid and external proposal engines remain off."""

    _manifest, genome_hash = _validated_run(run)
    requested_external = os.environ.get("SLOFORGE_GENESIS_ALLOW_EXTERNAL_SYNTHESIS") == "1"
    if requested_external:
        raise typer.BadParameter(
            "external synthesis is not implemented by this deterministic local command"
        )
    request = {
        "schema_version": "1.0.0",
        "seed": seed,
        "baseline_genome_hash": genome_hash,
        "budget_usd_ceiling": budget_usd,
        "spent_usd": 0.0,
        "proposal_engine": "deterministic-local-cegis",
        "external_synthesis": False,
        "hardware_execution": False,
        "objectives": None
        if objectives is None
        else {"path": str(objectives.resolve()), "sha256": sha256_file(objectives)},
    }
    request_path = run / "synthesis/request.json"
    if request_path.exists():
        existing = json.loads(request_path.read_text(encoding="utf-8"))
        if existing != request:
            raise typer.BadParameter("synthesis request already exists with different inputs")
    else:
        _atomic_json(request_path, request)
    try:
        result = synthesize_local_run(run, seed=seed, budget_usd=budget_usd)
    except FileExistsError as error:
        raise typer.BadParameter(str(error)) from error
    payload = result.model_dump(mode="json")
    payload["run"] = str(run.resolve())
    payload["spent_usd"] = 0.0
    _json_result(payload)
    if result.accepted_candidate_id is None:
        raise typer.Exit(code=1)


@genesis_app.command("verify")
def verify_command(
    candidate: Annotated[Path, typer.Option("--candidate", exists=True)],
) -> None:
    """Independently recheck a synthesized candidate's scoped integrity and protocol claim."""

    candidate_ir, design, integrity_errors, passed, evidence_id, failure = _candidate_verification(
        candidate
    )
    _json_result(
        {
            "candidate_id": candidate_ir.candidate_id,
            "passed": passed,
            "verification_level": "level-3-bounded-protocol-fixture",
            "claim": "cancelled requests emit no subsequent committed token",
            "scope": {
                "protocol": "deadline batching cancellation fixture",
                "seed": design.seed,
                "universal_proof": False,
            },
            "integrity_errors": integrity_errors,
            "evidence_id": evidence_id,
            "failure": failure,
        }
    )
    if not passed:
        raise typer.Exit(code=1)


def _candidate_verification(
    candidate: Path,
) -> tuple[Candidate, CandidateDesign, list[str], bool, str, dict[str, object] | None]:
    """Independently recompute candidate integrity and its bounded protocol result."""

    candidate_directory = candidate if candidate.is_dir() else candidate.parent
    candidate_path = candidate_directory / "candidate.json" if candidate.is_dir() else candidate
    design_path = candidate_directory / "candidate_design.json"
    genome_path = candidate_directory / "inference_genome.json"
    if not design_path.is_file() or not genome_path.is_file():
        raise typer.BadParameter("candidate is missing design or genome evidence")
    candidate_ir = load_candidate(candidate_path)
    design = CandidateDesign.model_validate_json(
        design_path.read_text(encoding="utf-8"), strict=True
    )
    genome = load_inference_genome(genome_path)
    integrity_errors: list[str] = []
    if candidate_ir.candidate_id != design.candidate_id:
        integrity_errors.append("candidate identity mismatch")
    if candidate_ir.genome_hash.value != canonical_hash(genome):
        integrity_errors.append("candidate genome hash mismatch")
    if candidate_ir.transformation_ids != tuple(
        mutation.transformation_id for mutation in design.mutations
    ):
        integrity_errors.append("candidate transformation sequence mismatch")
    outcome = CancellationPolicyVerifier().verify(design, None, seed=design.seed)
    return (
        candidate_ir,
        design,
        integrity_errors,
        not integrity_errors and outcome.passed,
        outcome.evidence_id,
        None if outcome.failure is None else outcome.failure.model_dump(mode="json"),
    )


def _parse_timestamp(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise typer.BadParameter("--timestamp must be an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise typer.BadParameter("--timestamp must include an explicit UTC offset")
    return timestamp.astimezone(UTC)


@capsule_app.command("build")
def capsule_build_command(
    candidate: Annotated[Path, typer.Option("--candidate", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    trust_output: Annotated[Path | None, typer.Option("--trust-output")] = None,
    timestamp: Annotated[str | None, typer.Option("--timestamp")] = None,
) -> None:
    """Build a capsule and publish its trust context separately from untrusted files."""

    try:
        result = build_local_capsule(
            candidate,
            output,
            observed_at=_parse_timestamp(timestamp),
            trust_output=trust_output,
        )
    except (FileExistsError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    _json_result(result.model_dump(mode="json"))


def _capsule_manifest(path: Path) -> tuple[Path, Path]:
    if path.is_file():
        root = path.parent.parent if path.parent.name == "manifests" else path.parent
        return path, root
    manifests = tuple(sorted((path / "manifests").glob("*.json")))
    if len(manifests) != 1:
        raise typer.BadParameter("capsule directory must contain exactly one manifest")
    return manifests[0], path


def _validated_capsule(
    capsule: Path,
    *,
    context: Path,
    expected_digest: str,
    hardware: Path | None = None,
) -> tuple[GenesisCapsule, Path, CapsuleValidationReport]:
    manifest_path, root = _capsule_manifest(capsule)
    context_is_inside_capsule = True
    try:
        context.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError:
        context_is_inside_capsule = False
    if context_is_inside_capsule:
        raise typer.BadParameter(
            "--context must be supplied from outside the untrusted capsule directory"
        )
    try:
        trusted_context = ValidationContext.model_validate_json(context.read_bytes(), strict=True)
        expected = Digest(value=expected_digest)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"external validation context is invalid: {error}") from error
    updates: dict[str, object] = {
        "expected_capsule_digest": expected,
        "now": datetime.now(UTC),
    }
    repository = Path(__file__).resolve().parents[3]
    lock_path = repository / "uv.lock"
    if lock_path.is_file():
        updates["dependency_lock_hash"] = Digest(value=sha256_file(lock_path))
    if hardware is not None:
        document = json.loads(hardware.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise typer.BadParameter("hardware contract must be a JSON object")
        digest = Digest(value=sha256_file(hardware))
        measured_fingerprint = document.get("measured_fingerprint")
        fingerprint = (
            Digest(value=measured_fingerprint) if isinstance(measured_fingerprint, str) else digest
        )
        updates.update(
            {
                "hardware_contract_hash": digest,
                "hardware_fingerprint": fingerprint,
                "hardware_architecture": str(document.get("architecture", "unknown")),
            }
        )
    trusted_context = trusted_context.model_copy(update=updates)
    document = load_capsule(manifest_path)
    return document, root, validate_capsule(document, root, trusted_context)


@capsule_app.command("validate")
def capsule_validate_command(
    capsule: Annotated[Path, typer.Argument(exists=True)],
    context: Annotated[Path, typer.Option("--context", exists=True, dir_okay=False, readable=True)],
    expected_digest: Annotated[str, typer.Option("--expected-digest")],
    hardware: Annotated[
        Path | None, typer.Option("--hardware", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Validate a capsule against caller-supplied trust context and expected identity."""

    document, _root, report = _validated_capsule(
        capsule,
        context=context,
        expected_digest=expected_digest,
        hardware=hardware,
    )
    _json_result(
        {
            **report.model_dump(mode="json"),
            "manifest": str(_capsule_manifest(capsule)[0].resolve()),
            "candidate_genome_hash": document.identity.candidate_genome_hash.value,
            "hardware_backed": False,
        }
    )
    if not report.local_evolution_eligible:
        raise typer.Exit(code=1)


@genesis_app.command("benchmark")
def benchmark_command(
    candidate: Annotated[Path, typer.Option("--candidate", exists=True)],
    suite: Annotated[Path, typer.Option("--suite", exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 73129,
) -> None:
    """Measure the accepted generated runtime through the strict local sandbox."""

    candidate_directory = candidate if candidate.is_dir() else candidate.parent
    candidate_ir, _design, integrity_errors, verified, _evidence_id, _failure = (
        _candidate_verification(candidate)
    )
    if not verified:
        detail = "; ".join(integrity_errors) or "bounded protocol verification failed"
        raise typer.BadParameter(f"candidate failed independent verification: {detail}")
    if candidate_ir.state is not CandidateSuccessState.SIMULATED:
        raise typer.BadParameter("benchmarking requires a model-checked and simulated candidate")
    try:
        suite_document = yaml.safe_load(suite.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise typer.BadParameter(f"benchmark suite is invalid: {error}") from error
    if suite_document is None:
        suite_document = {}
    if not isinstance(suite_document, dict):
        raise typer.BadParameter("benchmark suite must be a JSON/YAML object")
    repetitions_value = suite_document.get("repetitions", 3)
    if (
        not isinstance(repetitions_value, int)
        or isinstance(repetitions_value, bool)
        or not 2 <= repetitions_value <= 50
    ):
        raise typer.BadParameter("benchmark suite repetitions must be between 2 and 50")
    run_directory = candidate_directory.parent.parent
    runtime_directory = run_directory / "generated_runtime"
    config_path = runtime_directory / "runtime_config.json"
    if not config_path.is_file():
        raise typer.BadParameter("candidate run is missing its generated runtime")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise typer.BadParameter("generated runtime configuration must be an object")
    package_root = Path(str(config.get("reference_package_root", "")))
    package = load_reference_package(package_root)
    samples = package.resolve(package.manifest.quality_contract.final_evaluation_corpus)
    destination = output or candidate_directory / "benchmark"
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise typer.BadParameter(f"benchmark output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    repository_python = Path(__file__).resolve().parents[2]
    raw: list[dict[str, object]] = []
    durations: list[float] = []
    for repetition in range(repetitions_value):
        sandbox_output = destination / "sandbox" / f"repetition-{repetition:03d}"
        result = execute_sandboxed(
            SandboxRequest(
                argv=(
                    sys.executable,
                    "correctness_harness.py",
                    "--samples",
                    str(samples),
                    "--seed",
                    str(seed),
                    "--timeout-seconds",
                    "3",
                ),
                working_directory=runtime_directory,
                read_only_paths=(runtime_directory, package.root, repository_python),
                artifact_output_directory=sandbox_output,
                seed=seed + repetition,
                limits=SandboxLimits(
                    wall_time_seconds=15.0,
                    cpu_time_seconds=10,
                    memory_bytes=2 * 1024 * 1024 * 1024,
                    process_count=1,
                    output_bytes=128 * 1024,
                    artifact_bytes=1024 * 1024,
                    artifact_entries=16,
                    open_files=64,
                ),
            )
        )
        try:
            harness = json.loads(result.stdout)
        except json.JSONDecodeError:
            harness = None
        passed = bool(
            result.succeeded and isinstance(harness, dict) and harness.get("passed") is True
        )
        durations.append(result.duration_seconds)
        raw.append(
            {
                "repetition": repetition,
                "input_seed": seed,
                "sandbox_environment_seed": seed + repetition,
                "duration_seconds": result.duration_seconds,
                "termination": result.termination.value,
                "passed": passed,
                "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
                "measurement_scope": "sandboxed_generated_runtime_differential_harness",
            }
        )
    raw_payload = b"".join(
        json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        for item in raw
    )
    raw_path = destination / "raw_samples.jsonl"
    _write_new_bytes(raw_path, raw_payload)
    passed = all(bool(item["passed"]) for item in raw)
    summary = {
        "schema_version": "1.0.0",
        "candidate_id": candidate_ir.candidate_id,
        "candidate_genome_hash": candidate_ir.genome_hash.value,
        "seed": seed,
        "suite_path": str(suite.resolve()),
        "suite_sha256": sha256_file(suite),
        "corpus_path": str(samples.resolve()),
        "corpus_sha256": sha256_file(samples),
        "raw_samples_path": str(raw_path.resolve()),
        "raw_samples_sha256": hashlib.sha256(raw_payload).hexdigest(),
        "sample_count": len(raw),
        "median_duration_seconds": statistics.median(durations),
        "all_differential_runs_passed": passed,
        "hardware_backed": False,
        "performance_claim": "none; process-level CPU harness timing only",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }
    _atomic_json(destination / "summary.json", summary)
    _json_result(summary)
    if not passed:
        raise typer.Exit(code=1)


def _capsule_reference(
    document: GenesisCapsule,
    report: CapsuleValidationReport,
    manifest_path: Path,
) -> CapsuleReference:
    if (
        not report.local_evolution_eligible
        or report.capsule_digest is None
        or report.candidate_genome_hash is None
    ):
        raise typer.BadParameter("capsule is not independently eligible for local evolution")
    return CapsuleReference(
        capsule_id=f"capsule-{report.capsule_digest.value[:16]}",
        capsule_digest=report.capsule_digest.value,
        genome_hash=report.candidate_genome_hash.value,
        path=str(manifest_path.resolve()),
    )


def _local_cli_evolution_config() -> EvolutionConfig:
    return EvolutionConfig(
        execution_target=ExecutionTarget.LOCAL,
        minimum_shadow_samples=2,
        minimum_canary_samples=3,
        maximum_p95_ttft_ratio=10.0,
        maximum_p99_tpot_ratio=10.0,
    )


def _default_capsule_context(manifest_path: Path) -> Path:
    root = (
        manifest_path.parent.parent
        if manifest_path.parent.name == "manifests"
        else manifest_path.parent
    )
    return root.with_name(f"{root.name}.validation-context.json")


@genesis_app.command("deploy")
def deploy_command(
    capsule: Annotated[Path, typer.Option("--capsule", exists=True)],
    context: Annotated[Path, typer.Option("--context", exists=True, dir_okay=False)],
    expected_digest: Annotated[str, typer.Option("--expected-digest")],
    mode: Annotated[Literal["shadow"], typer.Option("--mode")] = "shadow",
    deployment: Annotated[str | None, typer.Option("--deployment")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 73129,
) -> None:
    """Create a local shadow-only controller; no live traffic is enabled."""

    del mode
    document, _root, report = _validated_capsule(
        capsule, context=context, expected_digest=expected_digest
    )
    manifest_path = _capsule_manifest(capsule)[0]
    reference = _capsule_reference(document, report, manifest_path)
    deployment_id = deployment or f"genesis-{reference.capsule_digest[:12]}"
    state_path = output or Path("artifacts/genesis/deployments") / deployment_id / "state.json"

    def validator(path: Path) -> CapsuleValidationReport:
        if path.resolve() != manifest_path.resolve():
            raise ValueError("controller requested an unknown capsule")
        return report

    controller = EvolutionController.initialize(
        store=EvolutionStore(state_path),
        capsule_validator=validator,
        config=_local_cli_evolution_config(),
        deployment_id=deployment_id,
        champion=reference,
        seed=seed,
        now_ms=0,
    )
    _json_result(
        {
            "deployment_id": deployment_id,
            "controller_state": str(state_path.resolve()),
            "phase": controller.snapshot.phase.value,
            "mode": "shadow",
            "capsule_digest": reference.capsule_digest,
            "live_traffic": False,
            "external_mutation": False,
            "status": "local_shadow_controller_initialized",
        }
    )


@genesis_app.command("promote")
def promote_command(
    capsule: Annotated[Path, typer.Option("--capsule", exists=True)],
    context: Annotated[Path, typer.Option("--context", exists=True, dir_okay=False)],
    expected_digest: Annotated[str, typer.Option("--expected-digest")],
    controller_state: Annotated[
        Path, typer.Option("--controller-state", exists=True, dir_okay=False)
    ],
    mode: Annotated[Literal["canary"], typer.Option("--mode")] = "canary",
    event_id: Annotated[str, typer.Option("--event-id")] = "cli-promote",
    observed_at_ms: Annotated[int | None, typer.Option("--observed-at-ms", min=0)] = None,
    gate_evidence_root: Annotated[Path | None, typer.Option("--gate-evidence-root")] = None,
) -> None:
    """Run or revalidate bounded local gates before a local-only promotion."""

    del mode
    document, _root, report = _validated_capsule(
        capsule, context=context, expected_digest=expected_digest
    )
    manifest_path = _capsule_manifest(capsule)[0]
    reference = _capsule_reference(document, report, manifest_path)

    store = EvolutionStore(controller_state)
    persisted = store.load()
    known_references = {
        Path(item.path).resolve(): item
        for item in (
            persisted.champion,
            *persisted.retained_capsules,
            *(record.spec.capsule for record in persisted.challengers),
        )
    }
    known_references[manifest_path.resolve()] = reference

    def validator(path: Path) -> CapsuleValidationReport:
        resolved = path.resolve()
        if resolved == manifest_path.resolve():
            return report
        known = known_references.get(resolved)
        if known is None:
            raise ValueError("promotion attempted to validate an untrusted capsule path")
        default_context = _default_capsule_context(resolved)
        _document, _root, known_report = _validated_capsule(
            resolved,
            context=default_context,
            expected_digest=known.capsule_digest,
        )
        return known_report

    evidence_root = gate_evidence_root or controller_state.parent / "runtime-gates"

    controller = EvolutionController.restore(
        store=store,
        capsule_validator=validator,
        gate_evidence_validator=local_gate_evidence_validator(evidence_root),
        config=_local_cli_evolution_config(),
    )
    if controller.snapshot.phase is EvolutionPhase.PROMOTED:
        if controller.snapshot.champion.capsule_digest != reference.capsule_digest:
            raise typer.BadParameter("controller already promoted a different capsule")
        snapshot = controller.snapshot
    else:
        selected = next(
            (
                item
                for item in controller.snapshot.challengers
                if item.spec.candidate_id == controller.snapshot.selected_candidate_id
            ),
            None,
        )
        try:
            timestamp = (
                controller.snapshot.last_observed_at_ms + 1
                if observed_at_ms is None
                else observed_at_ms
            )
            if selected is None:
                if controller.snapshot.phase is EvolutionPhase.IDLE:
                    controller.observe_trigger(
                        TriggerObservation(
                            event_id=f"{event_id}-trigger",
                            observed_at_ms=timestamp,
                            workload_js_divergence=1.0,
                            detail="operator requested bounded local proof-gated promotion",
                        )
                    )
                    timestamp += 1
                if controller.snapshot.phase is not EvolutionPhase.EVOLVING:
                    raise RuntimeError("controller cannot register a challenger in this phase")
                isolation = controller_state.parent / "challenger-isolation"
                isolation.mkdir(parents=True, exist_ok=True)
                controller.register_challenger(
                    ChallengerSpec(
                        candidate_id=f"cli-{reference.capsule_id}",
                        capsule=reference,
                        transition_category=TransitionCategory.REQUEST_BOUNDARY_SWAP,
                        isolation=IsolationContract(
                            mode=IsolationMode.SANDBOX,
                            writable_artifact_root=str(isolation.resolve()),
                        ),
                        active_stream_compatible=False,
                    ),
                    event_id=f"{event_id}-register",
                    observed_at_ms=timestamp,
                )
                timestamp += 1
                selected = next(
                    item
                    for item in controller.snapshot.challengers
                    if item.spec.candidate_id == controller.snapshot.selected_candidate_id
                )
            if selected.spec.capsule.capsule_digest != reference.capsule_digest:
                raise RuntimeError("capsule is not the controller's selected challenger")
            if controller.snapshot.phase is EvolutionPhase.CHALLENGER_READY:
                controller.begin_shadow(event_id=f"{event_id}-shadow", observed_at_ms=timestamp)
                timestamp += 1
            if controller.snapshot.phase is EvolutionPhase.SHADOWING:
                shadow = collect_local_gate_evidence(
                    champion=controller.snapshot.champion,
                    challenger=selected.spec.capsule,
                    candidate_id=selected.spec.candidate_id,
                    stage=GateStage.SHADOW,
                    sample_count=controller.config.minimum_shadow_samples,
                    seed=controller.snapshot.seed,
                    observed_at_ms=timestamp,
                    output_directory=evidence_root / GateStage.SHADOW.value,
                )
                controller.record_gate(shadow)
                timestamp += 1
            if controller.snapshot.phase is EvolutionPhase.SHADOW_VALIDATED:
                controller.begin_canary(event_id=f"{event_id}-canary", observed_at_ms=timestamp)
                timestamp += 1
            if controller.snapshot.phase is EvolutionPhase.CANARYING:
                canary = collect_local_gate_evidence(
                    champion=controller.snapshot.champion,
                    challenger=selected.spec.capsule,
                    candidate_id=selected.spec.candidate_id,
                    stage=GateStage.CANARY,
                    sample_count=controller.config.minimum_canary_samples,
                    seed=controller.snapshot.seed,
                    observed_at_ms=timestamp,
                    output_directory=evidence_root / GateStage.CANARY.value,
                )
                controller.record_gate(canary)
                timestamp += 1
            if controller.snapshot.phase is not EvolutionPhase.READY_TO_PROMOTE:
                raise RuntimeError("challenger did not reach the proof-gated promotion state")
            snapshot = controller.promote(event_id=event_id, observed_at_ms=timestamp)
        except (FileExistsError, RuntimeError, ValueError) as error:
            raise typer.BadParameter(str(error)) from error
    _json_result(
        {
            "deployment_id": snapshot.deployment_id,
            "phase": snapshot.phase.value,
            "champion_capsule_digest": snapshot.champion.capsule_digest,
            "previous_champion_retained": snapshot.previous_champion is not None,
            "external_mutation": False,
            "controller_state": str(controller_state.resolve()),
        }
    )


def _trigger_observation(trigger: str, *, event_id: str, observed_at_ms: int) -> TriggerObservation:
    normalized = trigger.replace("-", "_")
    values: dict[str, object] = {
        "event_id": event_id,
        "observed_at_ms": observed_at_ms,
        "detail": f"offline CLI trigger: {normalized}",
    }
    if normalized == "workload_drift":
        values["workload_js_divergence"] = 1.0
    elif normalized == "fabric_degradation":
        values["fabric_health_ratio"] = 0.0
    elif normalized == "performance_regression":
        values["performance_regression_ratio"] = 1.0
    elif normalized in {
        "model_update",
        "hardware_change",
        "dependency_update",
        "slo_change",
        "cost_change",
        "autopsy_bottleneck_change",
    }:
        values[
            {
                "model_update": "model_updated",
                "hardware_change": "hardware_changed",
                "dependency_update": "dependency_updated",
                "slo_change": "slo_changed",
                "cost_change": "cost_changed",
                "autopsy_bottleneck_change": "autopsy_bottleneck_changed",
            }[normalized]
        ] = True
    else:
        raise typer.BadParameter(f"unsupported evolution trigger: {trigger}")
    return TriggerObservation.model_validate(values)


@genesis_app.command("evolve")
def evolve_command(
    deployment: Annotated[str, typer.Option("--deployment")],
    trigger: Annotated[str, typer.Option("--trigger")],
    context: Annotated[Path, typer.Option("--context", exists=True, dir_okay=False)],
    expected_digest: Annotated[str, typer.Option("--expected-digest")],
    budget_usd: Annotated[float, typer.Option("--budget-usd", min=0.0)] = 0.0,
    controller_state: Annotated[Path | None, typer.Option("--controller-state")] = None,
    event_id: Annotated[str, typer.Option("--event-id")] = "cli-evolve",
) -> None:
    """Persist one deterministic local evolution trigger without spending external budget."""

    if os.environ.get("SLOFORGE_GENESIS_ALLOW_EXTERNAL_SYNTHESIS") == "1":
        raise typer.BadParameter("this command does not implement external synthesis")
    state_path = (
        controller_state or Path("artifacts/genesis/deployments") / deployment / "state.json"
    )
    if not state_path.is_file():
        raise typer.BadParameter(f"local controller state does not exist: {state_path}")

    def validator(path: Path) -> CapsuleValidationReport:
        _document, _root, report = _validated_capsule(
            path,
            context=context,
            expected_digest=expected_digest,
        )
        return report

    controller = EvolutionController.restore(
        store=EvolutionStore(state_path),
        capsule_validator=validator,
        config=_local_cli_evolution_config(),
    )
    if controller.snapshot.deployment_id != deployment:
        raise typer.BadParameter("controller state belongs to another deployment")
    observation = _trigger_observation(
        trigger,
        event_id=event_id,
        observed_at_ms=controller.snapshot.last_observed_at_ms + 1,
    )
    snapshot = controller.observe_trigger(observation)
    _json_result(
        {
            "deployment_id": deployment,
            "phase": snapshot.phase.value,
            "trigger": None if snapshot.active_trigger is None else snapshot.active_trigger.value,
            "budget_usd_ceiling": budget_usd,
            "spent_usd": 0.0,
            "external_synthesis": False,
            "controller_state": str(state_path.resolve()),
        }
    )


@genesis_app.command("compare")
def compare_command(
    champion: Annotated[Path, typer.Option("--champion", exists=True)],
    challenger: Annotated[Path, typer.Option("--challenger", exists=True)],
    champion_context: Annotated[
        Path, typer.Option("--champion-context", exists=True, dir_okay=False)
    ],
    challenger_context: Annotated[
        Path, typer.Option("--challenger-context", exists=True, dir_okay=False)
    ],
    champion_digest: Annotated[str, typer.Option("--champion-digest")],
    challenger_digest: Annotated[str, typer.Option("--challenger-digest")],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Compare two independently validated capsule manifests without executing either."""

    champion_document, _champion_root, champion_report = _validated_capsule(
        champion, context=champion_context, expected_digest=champion_digest
    )
    challenger_document, _challenger_root, challenger_report = _validated_capsule(
        challenger, context=challenger_context, expected_digest=challenger_digest
    )
    if not (
        champion_report.local_evolution_eligible and challenger_report.local_evolution_eligible
    ):
        raise typer.BadParameter("both capsules must pass independent validation")
    champion_evidence = {item.evidence_id for item in champion_document.evidence}
    challenger_evidence = {item.evidence_id for item in challenger_document.evidence}
    result = {
        "schema_version": "1.0.0",
        "champion_digest": champion_digest,
        "challenger_digest": challenger_digest,
        "same_source_model": (
            champion_document.identity.source_model_hash
            == challenger_document.identity.source_model_hash
        ),
        "same_hardware_contract": (
            champion_document.identity.hardware_contract_hash
            == challenger_document.identity.hardware_contract_hash
        ),
        "added_evidence": sorted(challenger_evidence - champion_evidence),
        "removed_evidence": sorted(champion_evidence - challenger_evidence),
        "champion_claim_count": len(champion_document.claims),
        "challenger_claim_count": len(challenger_document.claims),
        "hardware_backed_comparison": False,
    }
    if output is not None:
        if output.exists():
            raise typer.BadParameter(f"refusing to overwrite comparison: {output}")
        _atomic_json(output, result)
    _json_result(result)


def _extract_runtime_bundle(document: GenesisCapsule, root: Path, destination: Path) -> None:
    bundles = tuple(
        artifact
        for artifact in document.artifacts
        if artifact.role is ArtifactRole.GENERATED_RUNTIME
    )
    if len(bundles) != 1:
        raise typer.BadParameter("capsule must contain exactly one generated runtime bundle")
    archive_path = root / bundles[0].path
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if len(entries) > 128 or sum(item.file_size for item in entries) > 64 * 1024 * 1024:
            raise typer.BadParameter("runtime bundle exceeds replay extraction bounds")
        for item in entries:
            path = Path(item.filename)
            mode = item.external_attr >> 16
            if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode):
                raise typer.BadParameter("runtime bundle contains an unsafe archive entry")
        archive.extractall(destination)


@genesis_app.command("replay")
def replay_command(
    capsule: Annotated[Path, typer.Option("--capsule", exists=True)],
    trace: Annotated[Path, typer.Option("--trace", exists=True, dir_okay=False)],
    context: Annotated[Path, typer.Option("--context", exists=True, dir_okay=False)],
    expected_digest: Annotated[str, typer.Option("--expected-digest")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    seed: Annotated[int, typer.Option("--seed", min=0)] = 73129,
) -> None:
    """Replay a trace against a validated capsule only inside the strict sandbox."""

    if output.exists():
        raise typer.BadParameter(f"refusing to overwrite replay evidence: {output}")
    document, root, report = _validated_capsule(
        capsule, context=context, expected_digest=expected_digest
    )
    if not report.local_evolution_eligible:
        raise typer.BadParameter("capsule failed independent validation")
    with tempfile.TemporaryDirectory(prefix="sloforge-genesis-replay-") as temporary:
        # Resolve only the trusted, freshly created temporary root before deriving
        # children. On macOS tempfile may spell /private/var through the /var alias.
        temporary_root = Path(temporary).resolve(strict=True)
        runtime_root = temporary_root / "runtime"
        runtime_root.mkdir()
        _extract_runtime_bundle(document, root, runtime_root)
        runner = runtime_root / "replay_runner.py"
        runner.write_text(_REPLAY_RUNNER, encoding="utf-8")
        repository_python = Path(__file__).resolve().parents[2]
        result = execute_sandboxed(
            SandboxRequest(
                argv=(
                    sys.executable,
                    "replay_runner.py",
                    "--trace",
                    str(trace.resolve()),
                    "--seed",
                    str(seed),
                ),
                working_directory=runtime_root,
                read_only_paths=(runtime_root, trace, repository_python),
                artifact_output_directory=temporary_root / "sandbox-output",
                seed=seed,
                limits=SandboxLimits(
                    wall_time_seconds=30.0,
                    cpu_time_seconds=20,
                    memory_bytes=2 * 1024 * 1024 * 1024,
                    process_count=1,
                    output_bytes=1024 * 1024,
                    artifact_bytes=1024 * 1024,
                    artifact_entries=16,
                    open_files=64,
                ),
            )
        )
        try:
            replay = json.loads(result.stdout)
        except json.JSONDecodeError:
            replay = None
        passed = bool(result.succeeded and isinstance(replay, dict) and replay.get("passed"))
        evidence = {
            "schema_version": "1.0.0",
            "capsule_digest": expected_digest,
            "trace_path": str(trace.resolve()),
            "trace_sha256": sha256_file(trace),
            "seed": seed,
            "sandbox_termination": result.termination.value,
            "sandbox_capabilities": result.capabilities.model_dump(mode="json"),
            "process_group_cleaned": result.process_group_cleaned,
            "passed": passed,
            "replay": replay,
            "hardware_backed": False,
        }
        _atomic_json(output, evidence)
    _json_result(evidence)
    if not passed:
        raise typer.Exit(code=1)


__all__ = ["genesis_app"]
