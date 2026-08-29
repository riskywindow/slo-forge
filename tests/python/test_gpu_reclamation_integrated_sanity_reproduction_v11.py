from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiments/branchfabric"
sys.path.insert(0, str(EXPERIMENT_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "gpu_reclamation_integrated_controller_v11_sanity_reproduction_test",
    EXPERIMENT_ROOT / "gpu_reclamation_integrated_controller_v11.py",
)
assert SPEC is not None and SPEC.loader is not None
CONTROLLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLLER)
INTEGRATED_WORKER = sys.modules["gpu_reclamation_integrated_worker_v11"]


def _assessment(passed: bool) -> dict[str, Any]:
    return {
        "schema_version": "sloforge.branchfabric.v10-sanity-guard-assessment/v1",
        "guard": "sanity_12rps_stable",
        "expected_rate_rps": 12.0,
        "passed": passed,
        "targeted_probe_evidence_commitment_sha256": "a" * 64,
    }


@pytest.mark.parametrize(
    ("pattern", "classification"),
    [
        ((True, True), "CURRENT_RETAINED_ENVELOPE_STABLE"),
        ((False, False), "SUSTAINED_WITHIN_ALLOCATION_FAILURE"),
        ((False, True), "ORDER_DEPENDENT_RECOVERY_SIGNAL"),
        ((True, False), "UNSTABLE_ENVELOPE"),
    ],
)
def test_exact_two_probe_classifications(pattern: tuple[bool, bool], classification: str) -> None:
    result = CONTROLLER._classify_targeted_serving_sanity(
        [_assessment(value) for value in pattern],
        readiness_comparability={"readiness_comparable": True},
    )

    assert result["classification"] == classification
    assert result["probe_pass_pattern"] == list(pattern)
    assert result["integrated_retry_authorized"] is (pattern == (True, True))
    assert result["adaptive_retry_performed"] is False
    assert result["broad_capacity_calibration_performed"] is False
    assert result["full_export_authorized"] is False
    assert result["source_release_authorized"] is False


def test_pass_pass_cannot_override_slow_readiness() -> None:
    result = CONTROLLER._classify_targeted_serving_sanity(
        [_assessment(True), _assessment(True)],
        readiness_comparability={"readiness_comparable": False},
    )

    assert result["classification"] == "CURRENT_RETAINED_ENVELOPE_STABLE"
    assert result["integrated_retry_authorized"] is False


@pytest.mark.parametrize(
    "assessments",
    [
        [],
        [_assessment(True)],
        [_assessment(True), _assessment(True), _assessment(True)],
        [{**_assessment(True), "expected_rate_rps": 15.0}, _assessment(True)],
        [{**_assessment(True), "passed": 1}, _assessment(True)],
        [
            {**_assessment(True), "targeted_probe_evidence_commitment_sha256": None},
            _assessment(True),
        ],
    ],
)
def test_two_probe_classifier_fails_closed(assessments: list[dict[str, Any]]) -> None:
    with pytest.raises(RuntimeError, match=r"exactly two|exact gate|evidence commitment"):
        CONTROLLER._classify_targeted_serving_sanity(
            assessments,
            readiness_comparability={"readiness_comparable": True},
        )


def _readiness() -> tuple[dict[str, Any], dict[str, Any]]:
    common = {
        "pid": 101,
        "engine_nonce": "a" * 64,
        "engine_started_ns": 10,
        "engine_object_identity": 201,
        "adapter_object_identity": 301,
        "physical_gpu_uuid": "GPU-serving",
        "engine_reloaded": False,
    }
    serving = {**common, "device": "gpu0", "role": "serving"}
    rollout = {
        **common,
        "pid": 102,
        "engine_nonce": "b" * 64,
        "engine_object_identity": 202,
        "adapter_object_identity": 302,
        "physical_gpu_uuid": "GPU-rollout",
        "device": "gpu1",
        "role": "rollout",
    }
    return serving, rollout


def _runtime_shards(
    *, probe_id: str, request_count: int, cumulative_salt_count: int
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    readiness = _readiness()
    raw: list[dict[str, Any]] = []
    resets: list[dict[str, Any]] = []
    for ready in readiness:
        count = request_count if ready["device"] == "gpu0" else 0
        raw.append(
            {
                "schema_version": "sloforge.branchfabric.capacity-worker-probe/v1",
                "probe_id": probe_id,
                "device": ready["device"],
                "physical_gpu_uuid": ready["physical_gpu_uuid"],
                "observations": [{} for _ in range(count)],
                "engine_continuity": {
                    key: ready[key]
                    for key in (
                        "pid",
                        "engine_nonce",
                        "engine_started_ns",
                        "engine_object_identity",
                        "adapter_object_identity",
                        "physical_gpu_uuid",
                        "engine_reloaded",
                    )
                },
                "compilation_observation": {
                    "events": [],
                    "no_deferred_compilation_event": True,
                },
                "cache_salt_evidence": {
                    "passed": True,
                    "request_count": cumulative_salt_count if count else 0,
                    "unique_request_count": cumulative_salt_count if count else 0,
                    "unique_salt_count": cumulative_salt_count if count else 0,
                },
                "prefix_cache_policy": {
                    "calibration_reuse_prevented": True,
                    "method": "per-request-vllm-cache-salt-sha256",
                },
            }
        )
        resets.append(
            {
                **{
                    key: ready[key]
                    for key in (
                        "pid",
                        "engine_nonce",
                        "engine_started_ns",
                        "engine_object_identity",
                        "adapter_object_identity",
                        "physical_gpu_uuid",
                        "engine_reloaded",
                    )
                },
                "device": ready["device"],
                "probe_id": probe_id,
            }
        )
    return tuple(raw), tuple(resets)


def test_probe_runtime_evidence_binds_reset_and_cumulative_cache_salts() -> None:
    raw, resets = _runtime_shards(
        probe_id="v11-targeted-sanity-12rps-b",
        request_count=48,
        cumulative_salt_count=96,
    )

    CONTROLLER._validate_targeted_probe_runtime_evidence(
        raw_payloads=raw,
        resets=resets,
        readiness=_readiness(),
        probe_id="v11-targeted-sanity-12rps-b",
        expected_request_count=48,
        expected_cumulative_salt_count=96,
    )


def test_15rps_probe_requires_exact_156_request_cumulative_salt_chain() -> None:
    raw, resets = _runtime_shards(
        probe_id="v11-sanity-15rps",
        request_count=60,
        cumulative_salt_count=156,
    )
    CONTROLLER._validate_targeted_probe_runtime_evidence(
        raw_payloads=raw,
        resets=resets,
        readiness=_readiness(),
        probe_id="v11-sanity-15rps",
        expected_request_count=60,
        expected_cumulative_salt_count=156,
    )
    with pytest.raises(RuntimeError, match="cache-salt cohort"):
        CONTROLLER._validate_targeted_probe_runtime_evidence(
            raw_payloads=raw,
            resets=resets,
            readiness=_readiness(),
            probe_id="v11-sanity-15rps",
            expected_request_count=60,
            expected_cumulative_salt_count=144,
        )


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        ("raw-payload", "schema_version", "wrong"),
        ("raw-payload", "probe_id", "wrong"),
        ("raw", "engine_nonce", "c" * 64),
        ("reset", "pid", 999),
        ("raw-top", "events", ["deferred compile"]),
        ("salt", "unique_salt_count", 95),
    ],
)
def test_probe_runtime_evidence_rejects_continuity_compile_and_salt_tamper(
    location: str, field: str, value: Any
) -> None:
    raw_tuple, reset_tuple = _runtime_shards(
        probe_id="v11-targeted-sanity-12rps-b",
        request_count=48,
        cumulative_salt_count=96,
    )
    raw = [json.loads(json.dumps(item)) for item in raw_tuple]
    resets = [dict(item) for item in reset_tuple]
    if location == "raw-payload":
        raw[0][field] = value
    elif location == "raw":
        raw[0]["engine_continuity"][field] = value
    elif location == "reset":
        resets[0][field] = value
    elif location == "raw-top":
        raw[0]["compilation_observation"][field] = value
    else:
        raw[0]["cache_salt_evidence"][field] = value

    with pytest.raises(RuntimeError, match=r"retained-engine evidence|cache-salt"):
        CONTROLLER._validate_targeted_probe_runtime_evidence(
            raw_payloads=raw,
            resets=resets,
            readiness=_readiness(),
            probe_id="v11-targeted-sanity-12rps-b",
            expected_request_count=48,
            expected_cumulative_salt_count=96,
        )


def test_v11_wrapper_scopes_three_probe_count_and_restores_on_success() -> None:
    observed: list[int] = []
    frozen = SimpleNamespace(
        _V10_SANITY_GUARD_COUNT=2,
        _validate_integrated_transaction_command=lambda **_kwargs: ({}, {}),
    )

    def run_phase(**_kwargs: Any) -> tuple[dict[str, Any], object, dict[str, Any]]:
        observed.append(frozen._V10_SANITY_GUARD_COUNT)
        return {}, object(), {}

    frozen._run_integrated_calibration_phase = run_phase
    INTEGRATED_WORKER._run_v11_integrated_calibration_phase(
        frozen_worker=frozen,
        adapter=object(),
        config={},
        inputs={},
        role="serving",
        physical_gpu_uuid="GPU-test",
        barrier_root=Path("unused"),
        model_load_started_ns=1,
        model_ready_ns=2,
    )

    assert observed == [3]
    assert frozen._V10_SANITY_GUARD_COUNT == 2


def test_v11_wrapper_restores_probe_count_when_calibration_fails() -> None:
    def original_validator(**_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {}

    frozen = SimpleNamespace(
        _V10_SANITY_GUARD_COUNT=2,
        _validate_integrated_transaction_command=original_validator,
    )

    def fail(**_kwargs: Any) -> None:
        raise RuntimeError("probe failed")

    frozen._run_integrated_calibration_phase = fail
    with pytest.raises(RuntimeError, match="probe failed"):
        INTEGRATED_WORKER._run_v11_integrated_calibration_phase(
            frozen_worker=frozen,
            adapter=object(),
            config={},
            inputs={},
            role="serving",
            physical_gpu_uuid="GPU-test",
            barrier_root=Path("unused"),
            model_load_started_ns=1,
            model_ready_ns=2,
        )
    assert frozen._V10_SANITY_GUARD_COUNT == 2
    assert frozen._validate_integrated_transaction_command is original_validator


def test_integrated_controller_orders_ab_before_unchanged_15_and_transaction() -> None:
    source = (EXPERIMENT_ROOT / "gpu_reclamation_integrated_controller_v11.py").read_text()
    tree = ast.parse(source)
    controller = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_integrated_v11_controller"
    )
    controller_source = ast.get_source_segment(source, controller)
    assert controller_source is not None
    readiness_assessment = controller_source.index(
        "readiness_comparability = _assess_retained_engine_readiness_comparability("
    )
    readiness_persist = controller_source.index(
        "_write_new(readiness_comparability_path, readiness_comparability)"
    )
    readiness_gate = controller_source.index(
        "_validate_retained_engine_readiness_comparability(ready_payloads)"
    )
    probe_a_b = controller_source.index('for probe_index, probe_name in enumerate(("A", "B"))')
    classification = controller_source.index(
        "targeted_sanity_classification = _classify_targeted_serving_sanity("
    )
    unchanged_15 = controller_source.index("enumerate(SANITY_RATES_RPS[1:], start=2)")
    transaction = controller_source.index('barrier_root / "transaction.command.json"')

    assert (
        readiness_assessment
        < readiness_persist
        < readiness_gate
        < probe_a_b
        < classification
        < unchanged_15
        < transaction
    )
    assert source.count("_run_targeted_serving_sanity_probe(") == 2
    assert "targeted_probe_evidence_commitment_sha256" in source
    assert "sum(reproduction_request_counts) + len(plan.arrivals)" in source
    assert '"full_export_authorized": False' in source
    assert '"source_release_authorized": False' in source


def _selected_gate_fixture(tmp_path: Path) -> dict[str, Any]:
    readiness_path = tmp_path / "readiness.json"
    classification_path = tmp_path / "classification.json"
    CONTROLLER._write_new(readiness_path, {"readiness_comparable": True})
    probe_rows: list[dict[str, Any]] = []
    probe_roots: list[Path] = []
    artifact_names = (
        "plan.json",
        "raw.json",
        "worker-shards.json",
        "result.json",
        "assessment.json",
        "request-state-reset.json",
    )
    for probe in ("A", "B"):
        probe_root = tmp_path / f"12-rps-probe-{probe.lower()}"
        for artifact_name in artifact_names:
            CONTROLLER._write_new(
                probe_root / artifact_name,
                {"probe": probe, "artifact": artifact_name},
            )
        commitment_path = probe_root / "evidence-commitment.json"
        CONTROLLER._write_new(
            commitment_path,
            {
                "schema_version": ("sloforge.branchfabric.targeted-sanity-probe-commitment/v1"),
                "probe": probe,
                "probe_id": f"v11-sanity-12rps-{probe.lower()}",
                "artifacts": {
                    name: CONTROLLER._sha256(probe_root / name) for name in artifact_names
                },
            },
        )
        probe_rows.append(
            {
                "probe": probe,
                "passed": True,
                "evidence_commitment_sha256": CONTROLLER._sha256(commitment_path),
            }
        )
        probe_roots.append(probe_root)
    CONTROLLER._write_new(
        classification_path,
        {
            "schema_version": ("sloforge.branchfabric.targeted-serving-sanity-classification/v1"),
            "classification": "CURRENT_RETAINED_ENVELOPE_STABLE",
            "probe_results": probe_rows,
        },
    )
    selected = {
        "schema_version": "sloforge.branchfabric.v11-stale-calibration-selection/v1",
        "lambda_1_rps": 12.0,
        "lambda_spike_rps": 15.0,
        "lambda_2_rps": 9.0,
        "sanity_result_sha256": "3" * 64,
        "readiness_comparability_sha256": CONTROLLER._sha256(readiness_path),
        "targeted_12rps_reproduction_sha256": CONTROLLER._sha256(classification_path),
        "broad_capacity_calibration_performed": False,
    }
    selected_path = tmp_path / "selected-load.json"
    CONTROLLER._write_new(selected_path, selected)
    return {
        "readiness_path": readiness_path,
        "classification_path": classification_path,
        "probe_roots": tuple(probe_roots),
        "selected": selected,
        "selected_path": selected_path,
        "selected_sha256": CONTROLLER._sha256(selected_path),
    }


@pytest.mark.parametrize(
    "tampered_gate",
    ("readiness_comparability_sha256", "targeted_12rps_reproduction_sha256"),
)
def test_selected_load_transitively_commits_both_gates_and_handoff_rejects_tamper(
    tmp_path: Path,
    tampered_gate: str,
) -> None:
    fixture = _selected_gate_fixture(tmp_path)
    committed_sha256 = fixture["selected_sha256"]
    CONTROLLER._validate_selected_load_gate_commitment(
        selection_path=fixture["selected_path"],
        expected_selection_sha256=committed_sha256,
        readiness_comparability_path=fixture["readiness_path"],
        targeted_sanity_path=fixture["classification_path"],
        targeted_probe_roots=fixture["probe_roots"],
    )

    continuity = {
        "gpu0": {"pid": 101, "engine_nonce": "a" * 64},
        "gpu1": {"pid": 102, "engine_nonce": "b" * 64},
    }
    handoffs = tuple(
        {
            "schema_version": "sloforge.branchfabric.v11-transaction-ready/v1",
            "passed": True,
            "device": device,
            **identity,
            "selected_load_sha256": committed_sha256,
            "authorization_artifact_hash": "5" * 64,
            "pre_gpu_evidence_hashes": {
                "sanity_result_sha256": "3" * 64,
                "authorization_artifact_hash": "5" * 64,
            },
        }
        for device, identity in continuity.items()
    )
    CONTROLLER._validate_handoffs(
        handoffs,
        continuity=continuity,
        selection_sha256=committed_sha256,
        sanity_result_sha256="3" * 64,
        authorization_sha256="5" * 64,
    )

    tampered = dict(fixture["selected"])
    tampered[tampered_gate] = "4" * 64
    tampered_path = tmp_path / "selected-load-tampered.json"
    CONTROLLER._write_new(tampered_path, tampered)
    tampered_sha256 = CONTROLLER._sha256(tampered_path)
    assert tampered_sha256 != committed_sha256
    with pytest.raises(RuntimeError, match="handoff identity/commitment"):
        CONTROLLER._validate_handoffs(
            handoffs,
            continuity=continuity,
            selection_sha256=tampered_sha256,
            sanity_result_sha256="3" * 64,
            authorization_sha256="5" * 64,
        )


@pytest.mark.parametrize(
    "tampered_gate",
    ("readiness", "classification", "commitment", "probe-artifact"),
)
def test_post_gate_artifact_tamper_is_rejected_before_worker_handoff(
    tmp_path: Path,
    tampered_gate: str,
) -> None:
    fixture = _selected_gate_fixture(tmp_path)
    if tampered_gate == "readiness":
        tampered_path = fixture["readiness_path"]
    elif tampered_gate == "classification":
        tampered_path = fixture["classification_path"]
    elif tampered_gate == "commitment":
        tampered_path = fixture["probe_roots"][0] / "evidence-commitment.json"
    else:
        tampered_path = fixture["probe_roots"][0] / "raw.json"
    tampered_path.write_text('{"tampered":true}\n')

    with pytest.raises(RuntimeError, match=r"changed before worker handoff|commitment|changed"):
        CONTROLLER._validate_selected_load_gate_commitment(
            selection_path=fixture["selected_path"],
            expected_selection_sha256=fixture["selected_sha256"],
            readiness_comparability_path=fixture["readiness_path"],
            targeted_sanity_path=fixture["classification_path"],
            targeted_probe_roots=fixture["probe_roots"],
        )
