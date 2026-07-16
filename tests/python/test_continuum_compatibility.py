from __future__ import annotations

import pytest

from sloforge.continuum.compatibility import (
    CompatibilityRequest,
    ExactnessClass,
    ModelSemantics,
    QualityEvidence,
    RuntimeCapabilities,
    StateDependencyEvidence,
    analyze_compatibility,
    to_canonical_report,
)
from sloforge.continuum.ir import (
    CompatibilityReport as CanonicalCompatibilityReport,
)
from sloforge.continuum.ir import (
    Digest,
    RuntimeIdentity,
)


def _model(**updates: object) -> ModelSemantics:
    model = ModelSemantics(
        model_id="hybrid-decoder-v1",
        architecture="hybrid_decoder",
        weights_hash="weights-a",
        state_producing_weights_hash="state-weights-a",
        output_head_hash="head-a",
        tokenizer_hash="tokenizer-a",
        special_tokens_hash="special-a",
        positional_encoding="rope",
        rope_fingerprint="rope-theta-10000",
        attention_mask_semantics="causal",
        sliding_window=None,
        layer_count=4,
        head_count=8,
        kv_head_count=4,
        head_dim=16,
        recurrent_update_fingerprint="recurrent-v1",
        adapter_hash=None,
        state_dtype="float32",
        quantization="none",
        sampler_algorithm="counter_rng_v1",
    )
    return model.model_copy(update=updates)


def _runtime(name: str = "reference", **updates: object) -> RuntimeCapabilities:
    runtime = RuntimeCapabilities(
        runtime_name=name,
        runtime_version="1.0.0",
        adapter_version="1.0.0",
        supported_state_types=("attention.kv", "recurrent", "sampler", "guided_decoding"),
        supported_dtypes=("float32", "float16"),
        supported_quantizations=("none", "fp8_e4m3"),
        can_recompute_from_token_history=True,
    )
    return runtime.model_copy(update=updates)


def _request(**updates: object) -> CompatibilityRequest:
    request = CompatibilityRequest(
        source=_model(),
        destination=_model(),
        source_runtime=_runtime(),
        destination_runtime=_runtime(),
        source_layout_fingerprint="layout-a",
        destination_layout_fingerprint="layout-a",
        required_state_types=("attention.kv", "recurrent", "sampler", "guided_decoding"),
        required_exactness=ExactnessClass.EXACT_BITWISE,
    )
    return request.model_copy(update=updates)


def test_identical_compatibility_domain_is_bitwise() -> None:
    report = analyze_compatibility(_request())

    assert report.safe
    assert report.compatibility_class is ExactnessClass.EXACT_BITWISE
    assert report.required_conversion == ("copy",)
    assert report.verification_obligations[0].kind.value == "structural"


def test_layout_and_runtime_change_is_exact_semantic() -> None:
    report = analyze_compatibility(
        _request(
            destination_runtime=_runtime("genesis"),
            destination_layout_fingerprint="layout-b",
            required_exactness=ExactnessClass.EXACT_SEMANTIC,
        )
    )

    assert report.safe
    assert report.compatibility_class is ExactnessClass.EXACT_SEMANTIC
    assert "layout_convert" in report.required_conversion
    assert {reason.code for reason in report.reasons} == {
        "LAYOUT_ONLY_EXACT",
        "RUNTIME_BOUNDARY_SEMANTIC",
    }


def test_engine_decision_maps_to_the_single_canonical_wire_report() -> None:
    decision = analyze_compatibility(
        _request(
            destination_runtime=_runtime("genesis"),
            destination_layout_fingerprint="layout-b",
            required_exactness=ExactnessClass.EXACT_SEMANTIC,
        )
    )
    report = to_canonical_report(
        decision,
        source_capsule_id="a" * 64,
        destination_runtime=RuntimeIdentity(
            runtime_name="genesis",
            runtime_version="1.0.0",
            adapter_version="1.0.0",
            build_hash=Digest(value="b" * 64),
            dependency_versions=("python=3.12",),
            target_hardware=("cpu",),
        ),
        destination_physical_plan=Digest(value="c" * 64),
    )

    assert isinstance(report, CanonicalCompatibilityReport)
    assert report.compatibility_class is ExactnessClass.EXACT_SEMANTIC
    assert report.kind == "CompatibilityReport"
    assert report.required_conversions[0].operation == "layout_convert"
    assert report.report_id.startswith("continuum-compatibility-")


@pytest.mark.parametrize(
    ("field", "replacement", "reason"),
    [
        ("tokenizer_hash", "tokenizer-b", "TOKENIZER_MISMATCH"),
        ("rope_fingerprint", "rope-theta-500000", "ROPE_MISMATCH"),
        ("recurrent_update_fingerprint", "recurrent-v2", "RECURRENT_UPDATE_MISMATCH"),
        ("attention_mask_semantics", "prefix-lm", "ATTENTION_MASK_MISMATCH"),
    ],
)
def test_semantic_mismatch_is_rejected(field: str, replacement: str, reason: str) -> None:
    report = analyze_compatibility(
        _request(
            destination=_model(**{field: replacement}),
            required_exactness=ExactnessClass.EXACT_SEMANTIC,
        )
    )

    assert not report.safe
    assert report.compatibility_class is ExactnessClass.INCOMPATIBLE
    assert reason in {item.code for item in report.reasons}
    assert report.migration_restrictions == ("destination activation prohibited",)


def test_matching_shapes_do_not_allow_changed_state_producing_weights() -> None:
    report = analyze_compatibility(
        _request(
            destination=_model(
                model_id="hybrid-decoder-v2",
                weights_hash="weights-b",
                state_producing_weights_hash="state-weights-b",
            ),
            required_exactness=ExactnessClass.EXACT_SEMANTIC,
        )
    )

    assert not report.safe
    assert {reason.code for reason in report.reasons} == {"STATE_PRODUCING_DEPENDENCY_CHANGED"}


def test_state_weight_change_can_only_be_recomputation_assisted_with_complete_evidence() -> None:
    evidence = StateDependencyEvidence(
        dependency_graph_hash="dependency-graph-a",
        changed_components=("attention",),
        state_producing_components=("attention", "recurrent_update"),
        affected_state_components=("attention.kv",),
        recomputable_state_components=("attention.kv",),
        output_head_is_state_sink=True,
        token_history_available=True,
    )
    report = analyze_compatibility(
        _request(
            destination=_model(
                model_id="hybrid-decoder-v2",
                weights_hash="weights-b",
                state_producing_weights_hash="state-weights-b",
            ),
            dependency_evidence=evidence,
            allow_recomputation=True,
            required_exactness=ExactnessClass.RECOMPUTATION_ASSISTED,
        )
    )

    assert report.safe
    assert report.compatibility_class is ExactnessClass.RECOMPUTATION_ASSISTED
    assert report.required_recomputation[0].state_components == ("attention.kv",)
    assert "direct reuse" in report.migration_restrictions[0]


def test_output_head_only_change_requires_dependency_proof() -> None:
    destination = _model(
        model_id="hybrid-decoder-v1-head-b",
        weights_hash="weights-head-b",
        output_head_hash="head-b",
    )
    without_evidence = analyze_compatibility(
        _request(destination=destination, required_exactness=ExactnessClass.EXACT_SEMANTIC)
    )
    assert not without_evidence.safe

    evidence = StateDependencyEvidence(
        dependency_graph_hash="dependency-graph-output-sink",
        changed_components=("output_head",),
        state_producing_components=("attention", "recurrent_update"),
        affected_state_components=(),
        output_head_is_state_sink=True,
        token_history_available=True,
    )
    with_evidence = analyze_compatibility(
        _request(
            destination=destination,
            dependency_evidence=evidence,
            destination_layout_fingerprint="layout-b",
            required_exactness=ExactnessClass.EXACT_SEMANTIC,
        )
    )
    assert with_evidence.safe
    assert with_evidence.compatibility_class is ExactnessClass.EXACT_SEMANTIC
    assert "layout_convert" in with_evidence.required_conversion
    assert "OUTPUT_HEAD_ONLY_SAFE" in {reason.code for reason in with_evidence.reasons}


def test_adapter_change_uses_dependency_graph_instead_of_shape_heuristic() -> None:
    destination = _model(adapter_hash="lora-b")
    no_evidence = analyze_compatibility(
        _request(destination=destination, required_exactness=ExactnessClass.EXACT_SEMANTIC)
    )
    assert not no_evidence.safe

    evidence = StateDependencyEvidence(
        dependency_graph_hash="dependency-graph-adapter-output-only",
        changed_components=("adapter",),
        state_producing_components=("attention", "recurrent_update"),
        affected_state_components=(),
        output_head_is_state_sink=True,
        token_history_available=True,
    )
    report = analyze_compatibility(
        _request(
            destination=destination,
            dependency_evidence=evidence,
            required_exactness=ExactnessClass.EXACT_SEMANTIC,
        )
    )
    assert report.safe
    assert "DEPENDENCY_PROVEN_STATE_UNAFFECTED" in {reason.code for reason in report.reasons}


def test_quantization_requires_measured_quality_budget() -> None:
    destination = _model(quantization="fp8_e4m3")
    unproven = analyze_compatibility(
        _request(destination=destination, required_exactness=ExactnessClass.QUALITY_BOUNDED)
    )
    assert not unproven.safe
    assert unproven.reasons[0].code == "QUANTIZATION_BUDGET_UNPROVEN"

    evidence = QualityEvidence(
        metric="continuation_kl_divergence",
        observed_loss=0.002,
        maximum_loss=0.01,
        artifact_hash="sha256:quality-samples",
        sample_count=128,
    )
    proven = analyze_compatibility(
        _request(
            destination=destination,
            quality_evidence=evidence,
            required_exactness=ExactnessClass.QUALITY_BOUNDED,
        )
    )
    assert proven.safe
    assert proven.compatibility_class is ExactnessClass.QUALITY_BOUNDED
    assert "128 samples" in proven.quality_implications[0]


def test_float_dtype_conversion_is_numerical_not_exact() -> None:
    destination = _model(state_dtype="float16")
    numerical = analyze_compatibility(
        _request(
            destination=destination,
            required_exactness=ExactnessClass.NUMERICALLY_EQUIVALENT,
        )
    )
    assert numerical.safe
    assert numerical.compatibility_class is ExactnessClass.NUMERICALLY_EQUIVALENT

    exact = analyze_compatibility(
        _request(destination=destination, required_exactness=ExactnessClass.EXACT_SEMANTIC)
    )
    assert not exact.safe
    assert "EXACTNESS_REQUIREMENT_UNSATISFIED" in {reason.code for reason in exact.reasons}


def test_missing_non_kv_state_capability_is_explicit() -> None:
    destination_runtime = _runtime(
        supported_state_types=("attention.kv", "sampler", "guided_decoding")
    )
    report = analyze_compatibility(_request(destination_runtime=destination_runtime))
    assert not report.safe
    assert report.unsupported_state == ("recurrent",)
