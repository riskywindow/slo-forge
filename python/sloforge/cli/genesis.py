"""CLI for fail-closed zero-day inspection and baseline runtime synthesis."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from sloforge.genesis.compiler import initialize_genesis_run
from sloforge.genesis.frontend import InspectionResult, inspect_reference_package
from sloforge.genesis.frontend.package import load_reference_package
from sloforge.genesis.ir import canonical_hash, load_candidate, load_inference_genome
from sloforge.genesis.search import CandidateDesign
from sloforge.genesis.synthesis import CancellationPolicyVerifier, synthesize_local_run
from sloforge.util import sha256_file

genesis_app = typer.Typer(
    help="Synthesize proof-carrying inference implementations from reference packages.",
    no_args_is_help=True,
)
console = Console()


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
    passed = not integrity_errors and outcome.passed
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
            "evidence_id": outcome.evidence_id,
            "failure": None if outcome.failure is None else outcome.failure.model_dump(mode="json"),
        }
    )
    if not passed:
        raise typer.Exit(code=1)


__all__ = ["genesis_app"]
