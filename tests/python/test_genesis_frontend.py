from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from sloforge.genesis.frontend import (
    DiagnosticSeverity,
    inspect_reference_package,
    load_reference_package,
    torch_adapter,
    unsupported_obligations,
)
from sloforge.genesis.runtime import generate_baseline_runtime, load_generated_runtime
from sloforge.genesis.sandbox import SandboxRequest

ROOT = Path(__file__).resolve().parents[2]
HYBRID = ROOT / "models" / "reference_tasks" / "hybrid_decoder"


def test_reference_package_static_inspection_recovers_cross_layer_contract(tmp_path: Path) -> None:
    output = tmp_path / "inspection.json"
    first = inspect_reference_package(HYBRID, output_path=output)
    second = inspect_reference_package(HYBRID)

    assert first.package_hash == second.package_hash
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["package_hash"] == first.package_hash
    assert {operator.category for operator in first.graph.operators} >= {
        "tensor",
        "state",
        "sampling",
        "custom",
    }
    assert "quantized-state-update" in first.graph.custom_operator_ids
    assert any("quantized_state:write" in edge for edge in first.graph.state_dependencies)
    assert any("kv_window:write" in edge for edge in first.graph.state_dependencies)
    assert {dimension.name for dimension in first.graph.symbolic_dimensions} == {
        "batch",
        "sequence",
    }
    assert first.graph.legal_batching_axes == ("batch",)
    assert not first.has_unsupported_behavior
    assert all(item.proof_obligation for item in unsupported_obligations(first))


def test_default_inspection_does_not_execute_reference_source(tmp_path: Path) -> None:
    package = tmp_path / "package"
    shutil.copytree(HYBRID, package)
    source = ROOT / "tests" / "fixtures" / "genesis_reference" / "hostile_import.py"
    shutil.copyfile(source, package / "reference.py")

    result = inspect_reference_package(package)

    assert result.package_id == "hybrid-decoder-zero-day"
    assert result.torch_export is None


def test_torch_export_dispatches_hostile_reference_only_to_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    shutil.copytree(HYBRID, package)
    source = ROOT / "tests" / "fixtures" / "genesis_reference" / "hostile_import.py"
    shutil.copyfile(source, package / "reference.py")

    class SandboxCalled(RuntimeError):
        pass

    def intercept(request: SandboxRequest) -> None:
        assert request.require_network_isolation
        assert request.require_filesystem_isolation
        assert package.resolve() in request.read_only_paths
        assert request.environment == ()
        assert "sloforge.genesis.frontend.torch_export_runner" in request.argv
        raise SandboxCalled

    monkeypatch.setattr(torch_adapter, "execute_sandboxed", intercept)
    with pytest.raises(SandboxCalled):
        inspect_reference_package(package, use_torch_export=True)


def test_undeclared_control_flow_becomes_unsupported_diagnostic(tmp_path: Path) -> None:
    package = tmp_path / "package"
    shutil.copytree(HYBRID, package)
    reference = package / "reference.py"
    reference.write_text(
        reference.read_text(encoding="utf-8")
        + "\n\ndef unsupported_loop(value: int) -> int:\n"
        + "    while value > 0:\n"
        + "        value -= 1\n"
        + "    return value\n",
        encoding="utf-8",
    )

    result = inspect_reference_package(package)

    assert result.has_unsupported_behavior
    diagnostic = next(
        item for item in result.diagnostics if item.severity == DiagnosticSeverity.UNSUPPORTED
    )
    assert diagnostic.category == "dynamic_control_flow"
    assert diagnostic.proof_obligation is not None


def test_missing_declared_entry_point_is_explicitly_unsupported(tmp_path: Path) -> None:
    package = tmp_path / "package"
    shutil.copytree(HYBRID, package)
    manifest_path = package / "reference_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entry_points"]["decode_step"] = "missing_decode"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = inspect_reference_package(package)

    assert any(
        item.category == "contract" and "missing_decode" in item.message
        for item in result.diagnostics
    )


def test_package_rejects_path_traversal(tmp_path: Path) -> None:
    package = tmp_path / "package"
    shutil.copytree(HYBRID, package)
    manifest_path = package / "reference_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reference_module"] = "../reference.py"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="normalized and relative"):
        load_reference_package(package)


def test_package_identity_binds_declared_auxiliary_module_and_runtime(tmp_path: Path) -> None:
    package = tmp_path / "package"
    shutil.copytree(HYBRID, package)
    helper = package / "helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    reference = package / "reference.py"
    reference.write_text(
        reference.read_text(encoding="utf-8") + "\nimport helper\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="undeclared local dependency"):
        load_reference_package(package)

    manifest_path = package / "reference_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["auxiliary_modules"] = ["helper.py"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    inspection = inspect_reference_package(package)
    original_hash = inspection.package_hash
    assert dict(inspection.source_hashes)["helper.py"]
    bundle = generate_baseline_runtime(package, inspection, tmp_path / "runtime", seed=73129)
    runtime = load_generated_runtime(
        bundle.output_directory / "runtime_config.json",
        seed=73129,
        allow_untrusted_in_process=True,
    )
    runtime.start()
    runtime.shutdown()

    helper.write_text("VALUE = 2\n", encoding="utf-8")
    assert load_reference_package(package).package_hash != original_hash
    with pytest.raises(ValueError, match="inspection no longer matches"):
        generate_baseline_runtime(package, inspection, tmp_path / "runtime", seed=73129)


def test_package_rejects_symlink_and_dynamic_loading(tmp_path: Path) -> None:
    package = tmp_path / "package"
    shutil.copytree(HYBRID, package)
    (package / "linked.py").symlink_to(package / "reference.py")
    with pytest.raises(ValueError, match="contains a symlink"):
        load_reference_package(package)
    (package / "linked.py").unlink()
    reference = package / "reference.py"
    reference.write_text(
        reference.read_text(encoding="utf-8") + "\nDYNAMIC = __import__('math')\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dynamic execution"):
        load_reference_package(package)


def test_opaque_imported_call_creates_semantic_obligation(tmp_path: Path) -> None:
    package = tmp_path / "package"
    shutil.copytree(HYBRID, package)
    reference = package / "reference.py"
    reference.write_text(
        reference.read_text(encoding="utf-8")
        + "\nimport secrets\n\ndef opaque_call() -> bytes:\n    return secrets.token_bytes(1)\n",
        encoding="utf-8",
    )
    result = inspect_reference_package(package)
    assert any(
        diagnostic.category == "unknown_semantics" and "secrets.token_bytes" in diagnostic.message
        for diagnostic in result.diagnostics
    )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_explicit_torch_export_recovers_current_graph_metadata() -> None:
    result = inspect_reference_package(HYBRID, use_torch_export=True)

    assert result.torch_export is not None
    assert result.torch_export.graph_nodes
    assert result.torch_export.tensor_metadata
    assert any("sequence" in constraint for constraint in result.torch_export.range_constraints)
