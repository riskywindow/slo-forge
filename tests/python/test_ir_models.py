from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from sloforge.ir import (
    ArtifactDigest,
    DeploymentPlan,
    EvidenceBundle,
    Extensions,
    IRValidationError,
    canonical_hash,
    canonical_json,
    load_deployment_plan,
    load_evidence_bundle,
    migrate_document,
    save_deployment_plan,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "ir"


def test_golden_deployment_plan_round_trips_canonically() -> None:
    plan = load_deployment_plan(FIXTURES / "deployment-plan-v1.json")
    reparsed = DeploymentPlan.model_validate_json(canonical_json(plan))
    assert reparsed == plan
    assert canonical_hash(reparsed) == canonical_hash(plan)
    assert canonical_hash(plan) == (
        "d578ce29d3a2c5f026c581575fa43bfb3763914197d09eea03edd6d14ee42151"
    )
    assert plan.model.model_id == "Qwen/Qwen3-0.6B"


def test_golden_evidence_bundle_round_trips_canonically() -> None:
    bundle = load_evidence_bundle(FIXTURES / "evidence-bundle-v1.json")
    reparsed = EvidenceBundle.model_validate_json(canonical_json(bundle))
    assert reparsed == bundle
    assert len(bundle.measurements) == 1
    assert bundle.measurements[0].sample_count == 600
    assert canonical_hash(bundle) == (
        "06407e562984e50d04a4b289c0520462c0bb80c54e34e9ef4e7fb2d2831471dc"
    )


def test_core_objects_reject_unknown_fields() -> None:
    raw = json.loads((FIXTURES / "deployment-plan-v1.json").read_text())
    raw["engine"]["surprise"] = True
    with pytest.raises(IRValidationError, match="extra_forbidden"):
        load_deployment_plan(raw)


@pytest.mark.parametrize(
    "key",
    ["unqualified", "/missing", "UPPER.example/key", "example.org/bad key", "a//b"],
)
def test_extensions_require_namespace(key: str) -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        Extensions(root={key: True})


def test_namespaced_extensions_preserve_nested_json() -> None:
    extensions = Extensions(root={"example.org/tuning": {"levels": [1, 2, 3], "safe": True}})
    assert extensions.model_dump() == {"example.org/tuning": {"levels": [1, 2, 3], "safe": True}}


def test_cross_field_batch_limits_are_enforced() -> None:
    raw = json.loads((FIXTURES / "deployment-plan-v1.json").read_text())
    raw["batching"]["maximum_active_sequences"] = 17
    with pytest.raises(IRValidationError, match="maximum_active_sequences must match"):
        load_deployment_plan(raw)


def test_unknown_version_fails_closed() -> None:
    with pytest.raises(IRValidationError, match="unsupported IR schema version"):
        migrate_document({"schema_version": "2.0.0", "kind": "DeploymentPlan"})


@given(st.binary(min_size=32, max_size=32))
def test_digest_round_trip_for_arbitrary_sha256_bytes(value: bytes) -> None:
    digest = ArtifactDigest(value=value.hex())
    assert bytes.fromhex(digest.value) == value


def test_engine_version_requires_semver() -> None:
    raw = json.loads((FIXTURES / "deployment-plan-v1.json").read_text())
    raw["engine"]["version"] = "latest"
    with pytest.raises(IRValidationError, match="string_pattern_mismatch"):
        load_deployment_plan(raw)


def test_canonical_plan_save_is_atomic_and_reloadable(tmp_path: Path) -> None:
    plan = load_deployment_plan(FIXTURES / "deployment-plan-v1.json")
    output = tmp_path / "nested" / "plan.json"
    save_deployment_plan(output, plan)
    assert load_deployment_plan(output) == plan
    assert output.read_bytes().endswith(b"\n")
    assert list(output.parent.glob("*.tmp")) == []


def test_canonical_edge_number_hash_matches_rust_golden() -> None:
    document = json.loads((FIXTURES / "canonical-edge-cases.json").read_text())
    assert canonical_hash(document) == (
        "82e4b7f7fc6f923704946afe34437a6f5ed6f927addfa46bad6858b551e0b01a"
    )
