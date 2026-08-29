from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
COORDINATOR_PATH = ROOT / "tools/branchfabric-experiment-004-v11-kill-recompute.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coordinator() -> Any:
    name = "_kill_recompute_g_admission_coordinator"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, COORDINATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_agent13_control_gate_review_exact_binding_and_semantics_pass() -> None:
    coordinator = _coordinator()
    review = coordinator._validate_control_gate_policy_review(ROOT)
    assert review["status"] == "PASS"
    assert review["attempt_id"] == coordinator.ATTEMPT_F_ID
    assert review["authorized_successor_attempt_id"] == coordinator.ATTEMPT_ID
    assert _sha(ROOT / coordinator.CONTROL_GATE_POLICY_REVIEW_REFERENCE) == (
        coordinator.CONTROL_GATE_POLICY_REVIEW_SHA256
    )


def test_agent13_control_gate_review_hash_tamper_rejects() -> None:
    coordinator = _coordinator()
    coordinator.CONTROL_GATE_POLICY_REVIEW_SHA256 = "0" * 64
    with pytest.raises(RuntimeError, match="bound evidence changed"):
        coordinator._validate_control_gate_policy_review(ROOT)


@pytest.mark.parametrize(
    ("group", "field", "value"),
    (
        ("methodology_decision", "corrected_predicate_scope", "INTEGRATED"),
        ("attempt_f_exact_replay", "corrected_control_interval_pass", False),
        ("d_independent_runtime_and_test_bindings", "kill_worker", {}),
        ("consumer_checkpoint_and_cycle_free_trust", "review_alone_authorizes_gpu", True),
    ),
)
def test_agent13_control_gate_review_resealed_semantic_tamper_rejects(
    group: str, field: str, value: Any
) -> None:
    coordinator = _coordinator()
    temporary_root = Path(tempfile.mkdtemp(prefix=".agent13-review-tamper-", dir=ROOT))
    try:
        review = json.loads((ROOT / coordinator.CONTROL_GATE_POLICY_REVIEW_REFERENCE).read_text())
        review[group][field] = value
        path = temporary_root / "review.json"
        path.write_text(json.dumps(review, indent=2) + "\n")
        coordinator.CONTROL_GATE_POLICY_REVIEW_REFERENCE = str(path.relative_to(ROOT))
        coordinator.CONTROL_GATE_POLICY_REVIEW_SHA256 = _sha(path)
        with pytest.raises(RuntimeError, match="Agent13 control-gate review"):
            coordinator._validate_control_gate_policy_review(ROOT)
    finally:
        shutil.rmtree(temporary_root)


def test_g_seal_source_projection_independently_binds_review_coordinator_and_tests() -> None:
    coordinator = _coordinator()
    expected = {
        "policy_review": coordinator.CONTROL_GATE_POLICY_REVIEW_REFERENCE,
        "coordinator": "tools/branchfabric-experiment-004-v11-kill-recompute.py",
        "tests": "tests/python/test_gpu_reclamation_kill_recompute_v11.py",
        "admission_tests": "tests/python/test_gpu_reclamation_kill_recompute_g_admission.py",
    }
    assert all(
        coordinator.SOURCE_BINDINGS[label] == reference for label, reference in expected.items()
    )
    projected = {
        label: {"artifact": reference, "sha256": _sha(ROOT / reference)}
        for label, reference in coordinator.SOURCE_BINDINGS.items()
    }
    assert projected["policy_review"]["sha256"] == (
        "b6828316d33d0fa2349b37c7d94abe5b22fd9e5d4a983593126940cd69ff6aa8"
    )
    assert projected["coordinator"]["sha256"] == _sha(COORDINATOR_PATH)
    assert projected["tests"]["sha256"] == (
        "ed5b76c2e8074a8cd6a33408532a0a4d60ca9fb6457dd37586b16516f3e50c2b"
    )
