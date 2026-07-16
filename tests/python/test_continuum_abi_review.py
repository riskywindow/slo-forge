from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from sloforge.continuum.adapters import (
    ReferenceHeadMajorAdapter,
    ReferenceTokenMajorAdapter,
    SnapshotConsistencyError,
)
from sloforge.continuum.adapters.external import (
    IntegrationStatus,
    OptionalRuntimeNotInstalledError,
    SemanticVersion,
)
from sloforge.continuum.adapters.pytorch import probe_pytorch
from sloforge.continuum.compatibility import (
    CompatibilityRequest,
    ExactnessClass,
    ModelSemantics,
    RuntimeCapabilities,
    StateDependencyEvidence,
    analyze_compatibility,
)
from sloforge.continuum.conversion import (
    ConversionCompilationError,
    KVLayout,
    KVLayoutKind,
    canonical_convert,
    compile_conversion,
    decode_logical,
    direct_convert,
    direct_convert_capture,
    make_random_state,
    pytorch_convert,
    verify_direct_against_canonical,
)
from sloforge.continuum.ir import (
    ByteRange,
    ContinuumValidationError,
    ExecutionStateCapsule,
    ShardDescriptor,
    load_capsule,
    migrate_document,
)

ROOT = Path(__file__).parents[2]
GOLDEN = ROOT / "schemas/continuum/golden-execution-state-capsule-v1.json"


def _model(**updates: object) -> ModelSemantics:
    source = ModelSemantics(
        model_id="review-hybrid-v1",
        architecture="hybrid_decoder",
        weights_hash="weights-a",
        state_producing_weights_hash="state-weights-a",
        output_head_hash="head-a",
        tokenizer_hash="tokenizer-a",
        special_tokens_hash="special-a",
        positional_encoding="rope",
        rope_fingerprint="rope-10000",
        attention_mask_semantics="causal",
        layer_count=2,
        head_count=4,
        kv_head_count=4,
        head_dim=4,
        recurrent_update_fingerprint="recurrent-v1",
        state_dtype="float32",
        quantization="none",
        sampler_algorithm="counter-v1",
    )
    return source.model_copy(update=updates)


def _runtime(**updates: object) -> RuntimeCapabilities:
    source = RuntimeCapabilities(
        runtime_name="reference-a",
        runtime_version="1.0.0",
        adapter_version="1.0.0",
        supported_state_types=("attention.kv", "recurrent", "sampler"),
        supported_dtypes=("float32", "float16", "int32", "int8"),
        supported_quantizations=("none",),
        can_recompute_from_token_history=True,
    )
    return source.model_copy(update=updates)


def _request(**updates: object) -> CompatibilityRequest:
    source = CompatibilityRequest(
        source=_model(),
        destination=_model(),
        source_runtime=_runtime(),
        destination_runtime=_runtime(),
        source_layout_fingerprint="layout-a",
        destination_layout_fingerprint="layout-a",
        required_state_types=("attention.kv", "recurrent", "sampler"),
        required_exactness=ExactnessClass.EXACT_SEMANTIC,
    )
    return source.model_copy(update=updates)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("identity", "capture_timestamp"), "2099-01-01T00:00:00Z"),
        (("identity", "git_commit"), "attacker-controlled"),
        (("extensions", "sloforge.io/fixture-seed"), 1),
    ],
)
def test_capsule_identity_and_extensions_are_content_addressed(
    path: tuple[str, ...], value: object
) -> None:
    raw = json.loads(GOLDEN.read_bytes())
    cursor = raw
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value
    with pytest.raises(ContinuumValidationError, match="Merkle integrity"):
        load_capsule(raw)


def test_capsule_physical_components_must_exactly_match_logical_components() -> None:
    raw = json.loads(GOLDEN.read_bytes())
    raw["physical_state"]["logical_component_sizes"][0]["component_id"] = "state.rogue"
    raw["physical_state"]["segments"][0]["logical_state_reference"] = "state.rogue"
    with pytest.raises(ValidationError, match="exactly cover logical components"):
        ExecutionStateCapsule.model_validate_json(json.dumps(raw))


def test_physical_shard_rank_is_bounded_by_tensor_parallel_degree() -> None:
    with pytest.raises(ValidationError, match="smaller than tensor_parallel_degree"):
        ShardDescriptor(
            shard_id="invalid-rank",
            tensor_parallel_degree=2,
            pipeline_stage=0,
            expert_parallel_group=0,
            data_parallel_replica=0,
            rank=2,
            source_logical_slice=ByteRange(offset=0, length=4),
            destination_logical_slice=ByteRange(offset=0, length=4),
            shard_order=0,
        )


def test_unknown_versions_and_ambiguous_migrations_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported Continuum schema version"):
        migrate_document({"schema_version": "2.0.0", "kind": "LogicalStateSchema"})
    with pytest.raises(ValueError, match="both"):
        migrate_document(
            {
                "version": "v1alpha1",
                "kind": "physical_state",
                "runtime_identity": {},
                "runtime": {},
            }
        )


def test_equal_full_weights_cannot_hide_changed_state_fingerprint() -> None:
    report = analyze_compatibility(
        _request(destination=_model(state_producing_weights_hash="state-weights-b"))
    )
    assert not report.safe
    assert report.reasons[0].code == "MODEL_FINGERPRINT_INCONSISTENT"


def test_runtime_semantics_contract_mismatch_is_rejected() -> None:
    report = analyze_compatibility(
        _request(
            destination_runtime=_runtime(
                runtime_name="reference-b",
                logical_state_contract="different-logical-state-contract",
            )
        )
    )
    assert not report.safe
    assert report.reasons[0].code == "RUNTIME_LOGICAL_STATE_CONTRACT_MISMATCH"


def test_adapter_change_requires_evidence_that_names_the_adapter() -> None:
    misleading = StateDependencyEvidence(
        dependency_graph_hash="graph-a",
        changed_components=("unrelated",),
        state_producing_components=("attention", "recurrent_update"),
        affected_state_components=(),
        output_head_is_state_sink=True,
    )
    report = analyze_compatibility(
        _request(destination=_model(adapter_hash="adapter-b"), dependency_evidence=misleading)
    )
    assert not report.safe
    assert report.reasons[0].code == "ADAPTER_DEPENDENCY_EVIDENCE_INCOMPLETE"


def test_state_producer_change_with_complete_recomputation_proof_remains_legal() -> None:
    evidence = StateDependencyEvidence(
        dependency_graph_hash="graph-attention-v1",
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
                weights_hash="weights-b",
                state_producing_weights_hash="state-weights-b",
                # Some adapters conservatively bind this to the full-model hash.
                output_head_hash="weights-b",
            ),
            dependency_evidence=evidence,
            allow_recomputation=True,
            required_exactness=ExactnessClass.RECOMPUTATION_ASSISTED,
        )
    )
    assert report.safe
    assert report.compatibility_class is ExactnessClass.RECOMPUTATION_ASSISTED
    assert report.required_recomputation[0].state_components == ("attention.kv",)


def test_float_conversion_carries_declared_tolerance_and_measures_source_loss() -> None:
    source_layout = KVLayout(
        kind=KVLayoutKind.TOKEN_MAJOR_SEPARATE,
        tensor_parallel_degree=2,
        page_size_tokens=3,
        layer_count=2,
        token_count=7,
        kv_head_count=4,
        head_dim=3,
        dtype="float32",
    )
    destination_layout = replace(
        source_layout,
        kind=KVLayoutKind.HEAD_MAJOR_PACKED,
        page_size_tokens=5,
        dtype="float16",
    )
    state = make_random_state(source_layout, seed=104)
    evidence = verify_direct_against_canonical(
        state,
        destination_layout,
        maximum_temporary_bytes=2_048,
        numeric_tolerance=0.01,
    )
    assert evidence.exact  # exact direct-versus-canonical destination comparison
    assert evidence.declared_exactness is ExactnessClass.NUMERICALLY_EQUIVALENT
    assert evidence.source_to_destination_maximum_absolute_error > 0
    assert evidence.numeric_contract_satisfied


def test_converter_rejects_unbounded_lossy_integer_conversion() -> None:
    source = KVLayout(
        kind=KVLayoutKind.TOKEN_MAJOR_SEPARATE,
        tensor_parallel_degree=1,
        page_size_tokens=2,
        layer_count=1,
        token_count=3,
        kv_head_count=2,
        head_dim=2,
        dtype="int32",
    )
    destination = replace(source, dtype="int8")
    with pytest.raises(ConversionCompilationError, match="quality-bounded backend"):
        compile_conversion(source, destination, maximum_temporary_bytes=1_024)


@pytest.mark.parametrize("seed", range(8))
def test_direct_and_canonical_match_in_both_layout_directions(seed: int) -> None:
    first = KVLayout(
        kind=KVLayoutKind.HEAD_MAJOR_PACKED if seed % 2 else KVLayoutKind.TOKEN_MAJOR_SEPARATE,
        tensor_parallel_degree=2,
        page_size_tokens=2 + seed % 3,
        layer_count=1 + seed % 2,
        token_count=seed,
        kv_head_count=4,
        head_dim=1 + seed % 4,
        dtype="int32",
    )
    second = replace(
        first,
        kind=(
            KVLayoutKind.TOKEN_MAJOR_SEPARATE
            if first.kind is KVLayoutKind.HEAD_MAJOR_PACKED
            else KVLayoutKind.HEAD_MAJOR_PACKED
        ),
        tensor_parallel_degree=4,
        page_size_tokens=1 + (seed * 3) % 5,
    )
    state = make_random_state(first, seed=seed)
    canonical = canonical_convert(state, second)
    direct = direct_convert(state, second, maximum_temporary_bytes=1_024)
    for canonical_tensor, direct_tensor in zip(
        decode_logical(canonical), decode_logical(direct), strict=True
    ):
        assert np.array_equal(canonical_tensor, direct_tensor)


def test_reference_non_kv_state_and_page_table_are_verified_across_resume() -> None:
    source = ReferenceTokenMajorAdapter(page_size_tokens=3)
    destination = ReferenceHeadMajorAdapter(page_size_tokens=5)
    source.create_session(
        session_id="review-non-kv",
        request_id="request-review",
        tenant_id="tenant-review",
        input_token_ids=(2, 3, 5),
        seed=8128,
    )
    for event in source.stream_tokens("review-non-kv", count=7):
        source.acknowledge_gateway(
            "review-non-kv", token_index=event.token_index, owner_epoch=event.owner_epoch
        )
    captured = source.capture_consistent("review-non-kv")
    with pytest.raises(SnapshotConsistencyError, match="every paged segment exactly once"):
        replace(captured, page_table=()).verify()

    converted, evidence = direct_convert_capture(
        captured,
        destination=destination,
        maximum_temporary_bytes=1_024,
    )
    assert evidence.canonical_attention_match
    destination.prepare_destination_session(
        converted,
        destination_session_id="review-non-kv",
        proposed_owner_epoch=2,
    )
    destination.import_captured_state("review-non-kv", converted)
    validation = destination.validate_imported_state("review-non-kv")
    assert validation.continuation_valid
    destination.activate_destination(
        "review-non-kv", committed_owner_epoch=2, fencing_token="review-fence"
    )
    destination_capture = destination.capture_consistent("review-non-kv")
    assert destination_capture.logical.recurrent_state == captured.logical.recurrent_state
    assert destination_capture.logical.sampler == captured.logical.sampler
    assert destination_capture.logical.guided_decoding == captured.logical.guided_decoding
    assert destination.dry_run_next_token("review-non-kv") == source.dry_run_next_token(
        "review-non-kv"
    )


def test_external_version_parser_rejects_prerelease_and_suffix_smuggling() -> None:
    assert SemanticVersion.parse("0.23.0.post1+cu130") == SemanticVersion(0, 23, 0)
    for invalid in ("0.23.0rc1", "0.23.0evil", "0.23.0.dev1", "0.23"):
        with pytest.raises(ValueError):
            SemanticVersion.parse(invalid)


def test_optional_pytorch_backend_fails_closed_or_verifies_cpu() -> None:
    source_layout = KVLayout(
        kind=KVLayoutKind.TOKEN_MAJOR_SEPARATE,
        tensor_parallel_degree=1,
        page_size_tokens=2,
        layer_count=1,
        token_count=2,
        kv_head_count=2,
        head_dim=2,
    )
    destination_layout = replace(source_layout, kind=KVLayoutKind.HEAD_MAJOR_PACKED)
    source = make_random_state(source_layout, seed=7)
    probe = probe_pytorch()
    if probe.status is not IntegrationStatus.READY:
        with pytest.raises(OptionalRuntimeNotInstalledError):
            pytorch_convert(
                source,
                destination_layout,
                maximum_temporary_bytes=1_024,
                device="cpu",
                probe=probe,
            )
        return
    converted, evidence = pytorch_convert(
        source,
        destination_layout,
        maximum_temporary_bytes=1_024,
        device="cpu",
        probe=probe,
    )
    assert evidence.canonical_match
    assert not evidence.gpu_exercised
    assert converted.content_hash == canonical_convert(source, destination_layout).content_hash
