from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from sloforge.continuum.ir import (
    AccessPatternDescriptor,
    AccessPatternKind,
    AttentionLayerState,
    AttentionState,
    ByteRange,
    CapsuleTransactionBinding,
    CapsuleType,
    ClientDeliveryState,
    CompatibilityConstraints,
    CompressionKind,
    ContinuumValidationError,
    ConversionPermission,
    Digest,
    DTypeSemantics,
    EncryptionKind,
    ExactnessClass,
    ExecutionIdentity,
    ExecutionStateCapsule,
    Extensions,
    ExternalChunkReference,
    GuidedDecodingState,
    KVPacking,
    LayoutDescriptor,
    LayoutKind,
    LogicalComponentSize,
    LogicalStateSchema,
    MigrationVerificationEvidence,
    Ordering,
    OwnershipLease,
    OwnershipScope,
    PageTableDescriptor,
    PageTableEntry,
    PhysicalStateLayout,
    PlacementDescriptor,
    Provenance,
    RecomputationPermission,
    RecurrentState,
    RuntimeIdentity,
    SamplerState,
    SegmentManifest,
    ShardDescriptor,
    StateComponentDescriptor,
    StateDependencyEdge,
    StateDependencyGraph,
    StateDependencyNode,
    StateKind,
    StateLifetime,
    StateSegment,
    StorageLocation,
    TerminalStatus,
    TokenHistoryState,
    TokenRange,
    UnknownStateHandling,
    VerificationClaim,
    build_capsule,
    canonical_hash,
    canonical_json,
    load_capsule,
    migrate_document,
    schema_documents,
)

ROOT = Path(__file__).parents[2]
GOLDEN = ROOT / "schemas/continuum/golden-execution-state-capsule-v1.json"
GOLDEN_HASH = "18fbe6c114ec019839072c03213c1bf35efc72d650a668dee2addbb89a7215ee"


def _digest(text: str) -> Digest:
    import hashlib

    return Digest(value=hashlib.sha256(text.encode()).hexdigest())


def _component(
    identifier: str, kind: StateKind, shape: tuple[str, ...]
) -> StateComponentDescriptor:
    return StateComponentDescriptor(
        semantic_id=identifier,
        schema_version="1.0.0",
        kind=kind,
        symbolic_shape=shape,
        dtype_semantics=DTypeSemantics.FLOAT32,
        update_semantics=f"continuum.test/{kind.value}-v1",
        lifetime=StateLifetime.SESSION,
        ownership=OwnershipScope.SESSION_OWNER,
        exactness_requirement=ExactnessClass.EXACT_SEMANTIC,
        conversion_permissions=(ConversionPermission.EXACT_RELAYOUT,),
        recomputation_permission=RecomputationPermission.FROM_TOKEN_HISTORY,
        compatibility_fingerprint=_digest(f"fingerprint:{identifier}"),
        integrity_hash=_digest(f"state:{identifier}"),
        provenance=(
            Provenance(
                producer="test_continuum_ir",
                producer_version="1.0.0",
                captured_at="2026-08-02T12:00:00Z",
                raw_evidence_uri=f"fixture://{identifier}",
            ),
        ),
    )


def capsule_fixture() -> ExecutionStateCapsule:
    model = _digest("model:hybrid-decoder-v1")
    tokenizer = _digest("tokenizer:byte-v1")
    token_component = _component("state.token_history", StateKind.TOKEN_HISTORY, ("tokens",))
    attention_component = _component(
        "state.attention", StateKind.ATTENTION_KV, ("layers", "tokens", "kv_heads", "head_dim")
    )
    recurrent_component = _component("state.recurrent", StateKind.RECURRENT, ("layers", "width"))
    sampler_component = _component("state.sampler", StateKind.SAMPLER, ("rng_words",))
    guided_component = _component("state.guided", StateKind.GUIDED_DECODING, ("automaton",))
    client_component = _component("state.client", StateKind.CLIENT_DELIVERY, ("watermarks",))
    components = (
        token_component,
        attention_component,
        recurrent_component,
        sampler_component,
        guided_component,
        client_component,
    )
    logical = LogicalStateSchema(
        execution=ExecutionIdentity(
            session_id="session-001",
            request_id="request-001",
            workflow_id="workflow-001",
            tenant_id="tenant-fixture",
            model_identity=model,
            tokenizer_identity=tokenizer,
            creation_epoch=1,
            current_owner_epoch=7,
        ),
        token_history=TokenHistoryState(
            component=token_component,
            input_token_ids=(11, 12, 13),
            committed_output_token_ids=(21, 22),
            token_positions=(0, 1, 2, 3, 4),
            attention_mask_semantics="causal-dense-v1",
            tokenizer_fingerprint=tokenizer,
            normalization_contract="identity-token-ids-v1",
        ),
        attention=AttentionState(
            component=attention_component,
            layers=(
                AttentionLayerState(
                    layer_identity="layer.0",
                    logical_k_shape=(5, 4, 8),
                    logical_v_shape=(5, 4, 8),
                    token_range=TokenRange(start=0, end_exclusive=5),
                    head_count=8,
                    kv_head_count=4,
                    head_dimension=8,
                    positional_encoding_semantics="rope/base=10000/scaling=none",
                    attention_window_semantics="dense/full-context",
                    dtype_semantics=DTypeSemantics.FLOAT32,
                ),
            ),
        ),
        recurrent=(
            RecurrentState(
                component=recurrent_component,
                state_identifier="hybrid.recurrent.0",
                layer_identity="layer.0",
                logical_shape=(4,),
                update_semantics="r[t]=tanh(r[t-1]+token/256)-v1",
                dtype=DTypeSemantics.FLOAT32,
                sequence_position=5,
                initialization_contract="zeros-v1",
            ),
        ),
        sampler=SamplerState(
            component=sampler_component,
            sampling_algorithm="counter-prng-categorical-v1",
            seed=73129,
            rng_algorithm="splitmix64-v1",
            rng_counter=2,
            temperature=1.0,
            top_k=32,
            top_p=0.95,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            deterministic_required=True,
            implementation_independent_state="counter=2",
        ),
        guided_decoding=GuidedDecodingState(
            component=guided_component,
            automaton_identity=_digest("automaton:json-mini-v1"),
            current_automaton_state="object:key",
            tokenizer_contract=tokenizer,
            accepted_prefix=(21, 22),
        ),
        client_delivery=ClientDeliveryState(
            component=client_component,
            last_generated_token_index=1,
            last_gateway_committed_token_index=1,
            last_client_acknowledged_token_index=0,
            stream_owner_epoch=7,
            terminal_status=TerminalStatus.OPEN,
        ),
        dependency_graph=StateDependencyGraph(
            nodes=tuple(
                StateDependencyNode(
                    component_id=component.semantic_id,
                    state_producing_fingerprint=component.compatibility_fingerprint,
                )
                for component in components
            ),
            edges=(
                StateDependencyEdge(
                    upstream_component_id="state.token_history",
                    downstream_component_id="state.attention",
                    dependency_semantics="KV derives from tokens and model weights",
                    invalidated_by_weight_change=True,
                ),
                StateDependencyEdge(
                    upstream_component_id="state.token_history",
                    downstream_component_id="state.recurrent",
                    dependency_semantics="recurrent state derives from token sequence",
                    invalidated_by_weight_change=True,
                ),
                StateDependencyEdge(
                    upstream_component_id="state.token_history",
                    downstream_component_id="state.guided",
                    dependency_semantics="automaton consumes emitted tokens",
                    invalidated_by_weight_change=False,
                ),
            ),
        ),
        unknown_state_handling=UnknownStateHandling.REJECT,
        exactness_contract=ExactnessClass.EXACT_SEMANTIC,
    )
    sizes = {
        "state.token_history": 32,
        "state.attention": 64,
        "state.recurrent": 16,
        "state.sampler": 8,
        "state.guided": 8,
        "state.client": 8,
    }
    segments: list[StateSegment] = []
    shards: list[ShardDescriptor] = []
    manifests: list[SegmentManifest] = []
    pages: list[PageTableDescriptor] = []
    for index, (component_id, size) in enumerate(sizes.items()):
        segment_id = f"segment.{index}"
        shard_id = f"shard.{index}"
        chunk_id = f"chunk.{index}"
        checksum = _digest(f"bytes:{component_id}")
        byte_range = ByteRange(offset=0, length=size)
        shards.append(
            ShardDescriptor(
                shard_id=shard_id,
                tensor_parallel_degree=4 if component_id == "state.attention" else 1,
                pipeline_stage=0,
                expert_parallel_group=0,
                data_parallel_replica=0,
                rank=index % 4 if component_id == "state.attention" else 0,
                source_logical_slice=byte_range,
                destination_logical_slice=byte_range,
                shard_order=index,
            )
        )
        page_ids = (f"page.{index}",) if component_id == "state.attention" else ()
        segments.append(
            StateSegment(
                logical_state_reference=component_id,
                segment_id=segment_id,
                logical_byte_range=byte_range,
                physical_byte_range=ByteRange(offset=index * 128, length=size),
                tensor_shape=(size // 4,),
                tensor_strides=(1,),
                storage_offset=index * 128,
                allocation_id="allocation.fixture",
                page_ids=page_ids,
                chunk_ids=(chunk_id,),
                current_version=3,
                dirty_epoch=2,
                checksum=checksum,
                compression=CompressionKind.NONE,
                encryption=EncryptionKind.NONE,
                layout_id="layout.source",
                shard_id=shard_id,
                placement_id=f"placement.{index % 4}",
                access_pattern_id=(
                    "access.append" if component_id == "state.attention" else "access.mutable"
                ),
            )
        )
        manifests.append(
            SegmentManifest(
                segment_id=segment_id,
                segment_hash=checksum,
                chunks=(
                    ExternalChunkReference(
                        chunk_id=chunk_id,
                        content_hash=checksum,
                        size_bytes=size,
                        tenant_security_domain="tenant-fixture",
                        storage_uri=f"cas://tenant-fixture/{checksum.value}",
                        compression=CompressionKind.NONE,
                        encryption=EncryptionKind.NONE,
                    ),
                ),
            )
        )
        if page_ids:
            pages.append(
                PageTableDescriptor(
                    segment_id=segment_id,
                    entries=(
                        PageTableEntry(
                            logical_token_range=TokenRange(start=0, end_exclusive=5),
                            physical_page_id=page_ids[0],
                            page_version=3,
                            owner_epoch=7,
                            dirty=False,
                            copy_on_write_reference_count=1,
                        ),
                    ),
                )
            )
    runtime = RuntimeIdentity(
        runtime_name="continuum-reference-a",
        runtime_version="1.0.0",
        adapter_version="1.0.0",
        build_hash=_digest("runtime-a-build"),
        dependency_versions=("python=3.12.11",),
        target_hardware=(
            "simulated-gpu:0",
            "simulated-gpu:1",
            "simulated-gpu:2",
            "simulated-gpu:3",
        ),
    )
    physical = PhysicalStateLayout(
        layout_id="source-layout-paged-token-major-tp4",
        runtime=runtime,
        physical_plan_hash=_digest("source-physical-plan"),
        owner_epoch=7,
        logical_component_sizes=tuple(
            LogicalComponentSize(component_id=component_id, logical_size_bytes=size)
            for component_id, size in sizes.items()
        ),
        layout_descriptors=(
            LayoutDescriptor(
                layout_id="layout.source",
                kind=LayoutKind.PAGED,
                page_size_bytes=128,
                block_size=16,
                alignment_bytes=64,
                ordering=Ordering.TOKEN_MAJOR,
                k_v_packing=KVPacking.SEPARATE,
            ),
        ),
        shard_descriptors=tuple(shards),
        placement_descriptors=tuple(
            PlacementDescriptor(
                placement_id=f"placement.{index}",
                location=StorageLocation(
                    memory_type="gpu",
                    host_id="source-host",
                    device_id=f"simulated-gpu:{index}",
                    numa_domain=index // 2,
                    memory_tier="device",
                    network_rail="rail.0",
                    fault_domain=f"source-gpu-{index}",
                ),
                nic_id="nic.0",
            )
            for index in range(4)
        ),
        access_patterns=(
            AccessPatternDescriptor(
                access_pattern_id="access.append",
                kind=AccessPatternKind.APPEND_ONLY,
                required_before_resume=True,
                streamable_before_use=False,
                recomputable=True,
            ),
            AccessPatternDescriptor(
                access_pattern_id="access.mutable",
                kind=AccessPatternKind.MUTABLE,
                required_before_resume=True,
                streamable_before_use=False,
                recomputable=True,
            ),
        ),
        segments=tuple(segments),
        page_tables=tuple(pages),
        reconstructible_runtime_state=("scheduler_queue", "cuda_graph", "allocator_handles"),
    )
    evidence = MigrationVerificationEvidence(
        evidence_id="evidence-capture-session-001",
        transaction_id=None,
        generated_at="2026-08-02T12:00:01Z",
        capture_consistency=(
            VerificationClaim(
                claim_id="capture.token-boundary",
                property="capture occurred at a committed token boundary",
                scope="session-001 owner_epoch=7",
                result="pass",
                evidence_digest=_digest("capture-event-journal"),
            ),
        ),
        segment_integrity=tuple(
            VerificationClaim(
                claim_id=f"integrity.{manifest.segment_id}",
                property="segment checksum matches captured bytes",
                scope=manifest.segment_id,
                result="pass",
                evidence_digest=manifest.segment_hash,
            )
            for manifest in manifests
        ),
        model_check_scope="not exercised by ABI fixture",
        known_limitations=("CPU-only deterministic fixture",),
    )
    transaction = CapsuleTransactionBinding(
        ownership_lease=OwnershipLease(
            session_id="session-001",
            owner_runtime="continuum-reference-a",
            owner_epoch=7,
            fencing_token=19,
            expiration="2026-08-02T12:05:00Z",
            coordinator_version=4,
            last_committed_state_version=3,
            last_committed_token_index=1,
        ),
        fencing_token=19,
        source_epoch=7,
        commit_watermark=1,
        rollback_boundary=1,
        transaction_journal_hash=_digest("capture-journal"),
    )
    return build_capsule(
        capsule_type=CapsuleType.COMPLETE,
        logical_state=logical,
        physical_state=physical,
        segment_manifests=tuple(manifests),
        compatibility=CompatibilityConstraints(
            source_compatibility_fingerprint=_digest("compatibility-domain-hybrid-v1"),
            required_destination_capabilities=(
                "attention_kv",
                "recurrent",
                "guided_decoding",
                "owner_epoch_fencing",
            ),
            prohibited_conversions=(ConversionPermission.QUANTIZATION,),
            recomputation_permissions=(RecomputationPermission.FROM_TOKEN_HISTORY,),
            architecture_restrictions=("hybrid-decoder-v1",),
        ),
        transaction=transaction,
        evidence=evidence,
        capture_timestamp="2026-08-02T12:00:01Z",
        git_commit="0123456789abcdef",
        continuum_version="0.1.0",
        extensions=Extensions(root={"sloforge.io/fixture-seed": 73129}),
    )


def test_continuum_ir_golden_capsule_and_stable_hash() -> None:
    capsule = capsule_fixture()
    assert load_capsule(GOLDEN) == capsule
    assert canonical_json(capsule) == GOLDEN.read_bytes().rstrip(b"\n")
    assert canonical_hash(capsule) == GOLDEN_HASH


@given(st.dictionaries(st.text(min_size=1), st.integers(), max_size=30))
def test_continuum_ir_canonical_hash_ignores_mapping_insertion_order(
    values: dict[str, int],
) -> None:
    reversed_values = dict(reversed(tuple(values.items())))
    assert canonical_hash(values) == canonical_hash(reversed_values)


def test_continuum_ir_json_schemas_accept_golden_capsule() -> None:
    schemas = dict(schema_documents())
    for name, schema in schemas.items():
        jsonschema.Draft202012Validator.check_schema(schema)
        assert json.loads((ROOT / "schemas/continuum" / name).read_bytes()) == schema
    jsonschema.validate(
        json.loads(GOLDEN.read_bytes()), schemas["execution-state-capsule-v1.schema.json"]
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("identity", "owner_epoch"), 6),
        (("segment_manifests", 0, "segment_hash", "value"), "f" * 64),
        (("physical_state", "page_tables", 0, "entries", 0, "page_version"), 2),
        (("logical_state", "execution", "model_identity", "value"), "e" * 64),
        (("transaction", "transaction_journal_hash", "value"), "d" * 64),
        (("evidence", "capture_consistency", 0, "property"), "altered claim"),
    ],
)
def test_continuum_ir_rejects_tampering(path: tuple[str | int, ...], value: object) -> None:
    raw = json.loads(GOLDEN.read_bytes())
    cursor = raw
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value
    with pytest.raises(ContinuumValidationError):
        load_capsule(raw)


def test_continuum_ir_rejects_layout_gap_and_unknown_fields() -> None:
    raw = json.loads(GOLDEN.read_bytes())
    raw["physical_state"]["segments"][0]["logical_byte_range"]["offset"] = 1
    raw["physical_state"]["shard_descriptors"][0]["source_logical_slice"]["offset"] = 1
    with pytest.raises(ContinuumValidationError, match="gap or overlap"):
        load_capsule(raw)
    raw = json.loads(GOLDEN.read_bytes())
    raw["logical_state"]["surprise"] = True
    with pytest.raises(ContinuumValidationError, match="Extra inputs are not permitted"):
        load_capsule(raw)


def test_continuum_ir_requires_namespace_qualified_extensions() -> None:
    raw = json.loads(GOLDEN.read_bytes())
    raw["extensions"] = {"unowned": True}
    with pytest.raises(ContinuumValidationError, match="match pattern"):
        load_capsule(raw)


def test_continuum_ir_alpha_migration_is_lossless_and_bounded() -> None:
    alpha = {"version": "v1alpha1", "kind": "migration_plan", "id": "plan-1"}
    migrated = migrate_document(alpha)
    assert migrated["kind"] == "MigrationPlan"
    assert migrated["plan_id"] == "plan-1"
    assert "id" not in migrated
    assert alpha == {"version": "v1alpha1", "kind": "migration_plan", "id": "plan-1"}


def test_continuum_ir_direct_model_rejects_watermark_and_dependency_errors() -> None:
    raw = json.loads(GOLDEN.read_bytes())["logical_state"]
    raw["client_delivery"]["last_gateway_committed_token_index"] = 4
    with pytest.raises(ValidationError, match="gateway watermark"):
        LogicalStateSchema.model_validate_json(json.dumps(raw))
    raw = json.loads(GOLDEN.read_bytes())["logical_state"]
    raw["dependency_graph"]["edges"][0]["downstream_component_id"] = "missing"
    with pytest.raises(ValidationError, match="unknown component"):
        LogicalStateSchema.model_validate_json(json.dumps(raw))


def test_continuum_ir_required_unknown_state_is_explicitly_rejected() -> None:
    raw = json.loads(GOLDEN.read_bytes())["logical_state"]
    component = json.loads(json.dumps(raw["token_history"]["component"]))
    component["semantic_id"] = "state.vendor_required"
    component["kind"] = "unknown"
    raw["unknown_components"] = [
        {
            "component": component,
            "namespace": "vendor.example",
            "type_name": "opaque_scheduler_continuation",
            "type_version": "1.0.0",
            "required_for_resume": True,
            "portable_opaque": False,
            "payload_digest": None,
        }
    ]
    raw["dependency_graph"]["nodes"].append(
        {
            "component_id": "state.vendor_required",
            "state_producing_fingerprint": component["compatibility_fingerprint"],
        }
    )
    with pytest.raises(ValidationError, match="required unknown state"):
        LogicalStateSchema.model_validate_json(json.dumps(raw))


def test_continuum_ir_capsule_fixture_is_deterministic() -> None:
    assert canonical_json(capsule_fixture()) == canonical_json(capsule_fixture())


def _write_golden_for_maintainers() -> None:
    """Intentionally not called by tests; schemas/fixtures change only by review."""

    GOLDEN.write_bytes(canonical_json(capsule_fixture()) + b"\n")


if __name__ == "__main__":
    _write_golden_for_maintainers()
