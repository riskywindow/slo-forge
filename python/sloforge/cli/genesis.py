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


__all__ = ["genesis_app"]
