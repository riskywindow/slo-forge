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
    unsupported_obligations,
)

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


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_explicit_torch_export_recovers_current_graph_metadata() -> None:
    result = inspect_reference_package(HYBRID, use_torch_export=True)

    assert result.torch_export is not None
    assert result.torch_export.graph_nodes
    assert result.torch_export.tensor_metadata
    assert any("sequence" in constraint for constraint in result.torch_export.range_constraints)
