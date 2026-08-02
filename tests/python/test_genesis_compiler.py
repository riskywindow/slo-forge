from __future__ import annotations

import json
from pathlib import Path

import pytest

from sloforge.genesis.compiler import (
    GenomeCompilationError,
    compile_inference_genome,
    initialize_genesis_run,
)
from sloforge.genesis.frontend import inspect_reference_package
from sloforge.genesis.frontend.models import DiagnosticSeverity, InspectionDiagnostic
from sloforge.genesis.ir import Precision, canonical_hash, load_inference_genome

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "models/reference_tasks/hybrid_decoder"


def test_compiler_recovers_complete_deterministic_genome() -> None:
    inspection = inspect_reference_package(PACKAGE)

    first = compile_inference_genome(PACKAGE, inspection, seed=73129)
    second = compile_inference_genome(PACKAGE, inspection, seed=73129)

    assert canonical_hash(first) == canonical_hash(second)
    assert first.source_model.value == inspection.package_hash
    assert not first.tensor.operators
    recovered = first.tensor.node.extensions.root["sloforge.dev/recovered-graph"]
    assert recovered["algebraic_graph_status"] == "unresolved_static_call_inventory"
    assert len(recovered["unresolved_operators"]) == len(inspection.graph.operators)
    assert {state.precision for state in first.state.states} == {
        Precision.FLOAT64,
        Precision.INT64,
        Precision.INT32,
        Precision.INT8,
    }
    assert first.request.maximum_queue_depth == 32
    assert first.recovery.transitions[0].active_stream_behavior == "drain"
    obligation_ids = {
        obligation.obligation_id for obligation in first.tensor.node.proof_obligations
    }
    assert any("unknown-call" in obligation_id for obligation_id in obligation_ids)


def test_compiler_identity_changes_with_seed() -> None:
    inspection = inspect_reference_package(PACKAGE)
    first = compile_inference_genome(PACKAGE, inspection, seed=1)
    second = compile_inference_genome(PACKAGE, inspection, seed=2)

    assert first.genome_id != second.genome_id
    assert canonical_hash(first) != canonical_hash(second)


def test_compiler_fails_closed_on_unsupported_semantics() -> None:
    inspection = inspect_reference_package(PACKAGE)
    invalid = inspection.model_copy(
        update={
            "diagnostics": (
                *inspection.diagnostics,
                InspectionDiagnostic(
                    diagnostic_id="unsupported-test",
                    severity=DiagnosticSeverity.UNSUPPORTED,
                    category="unknown_semantics",
                    message="undeclared dynamic dispatch",
                    proof_obligation="declare dynamic dispatch semantics",
                ),
            )
        }
    )

    with pytest.raises(GenomeCompilationError, match="unsupported reference behavior"):
        compile_inference_genome(PACKAGE, invalid, seed=7)


def test_compiler_recomputes_inspection_instead_of_trusting_package_hash() -> None:
    inspection = inspect_reference_package(PACKAGE)
    altered = inspection.model_copy(
        update={
            "graph": inspection.graph.model_copy(update={"legal_batching_axes": ()}),
        }
    )
    assert altered.package_hash == inspection.package_hash

    with pytest.raises(GenomeCompilationError, match="independent static inspection"):
        compile_inference_genome(PACKAGE, altered, seed=7)


def test_initialize_generates_hash_bound_runtime(tmp_path: Path) -> None:
    inspection = inspect_reference_package(PACKAGE)
    result = initialize_genesis_run(PACKAGE, inspection, tmp_path / "run", seed=73)

    loaded = load_inference_genome(result.output_directory / "inference_genome.json")
    runtime_config = json.loads(
        (result.runtime.output_directory / "runtime_config.json").read_text(encoding="utf-8")
    )
    assert canonical_hash(loaded) == result.genome_hash
    assert runtime_config["package_hash"] == inspection.package_hash
    assert (result.runtime.output_directory / "correctness_harness.py").is_file()
