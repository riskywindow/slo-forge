"""Sandboxed opt-in adapter for current ``torch.export`` graph evidence."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from sloforge.genesis.sandbox import (
    SandboxLimits,
    SandboxRequest,
    execute_sandboxed,
    interpreter_read_roots,
)

from .models import TorchExportEvidence
from .package import LoadedReferencePackage


def inspect_with_torch_export(package: LoadedReferencePackage) -> TorchExportEvidence:
    """Execute a declared export fixture in the fail-closed Genesis sandbox.

    The orchestrator never imports the reference module. The worker receives a
    read-only package and may write only one bounded result document.
    """

    if package.manifest.entry_points.torch_export is None:
        raise ValueError("reference package does not declare a torch_export entry point")
    repository_python = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(prefix="sloforge-torch-export-") as temporary:
        output_root = Path(temporary) / "artifacts"
        result_path = output_root / "torch-export-evidence.json"
        result = execute_sandboxed(
            SandboxRequest(
                argv=(
                    sys.executable,
                    "-m",
                    "sloforge.genesis.frontend.torch_export_runner",
                    str(package.root),
                    str(result_path),
                ),
                working_directory=repository_python,
                read_only_paths=(
                    repository_python,
                    package.root,
                    *interpreter_read_roots(),
                ),
                artifact_output_directory=output_root,
                seed=0,
                limits=SandboxLimits(
                    wall_time_seconds=30.0,
                    cpu_time_seconds=25,
                    memory_bytes=4 * 1024 * 1024 * 1024,
                    process_count=1,
                    output_bytes=256 * 1024,
                    artifact_bytes=8 * 1024 * 1024,
                    artifact_entries=8,
                    open_files=128,
                ),
                require_network_isolation=True,
                require_filesystem_isolation=True,
            )
        )
        if not result.succeeded:
            raise RuntimeError(
                "torch.export inspection was not executed successfully in the strict sandbox: "
                f"{result.termination.value}"
            )
        try:
            return TorchExportEvidence.model_validate_json(result_path.read_bytes(), strict=True)
        except (OSError, ValueError) as error:
            raise RuntimeError("sandboxed torch.export returned invalid evidence") from error
