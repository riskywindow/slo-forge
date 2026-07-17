from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from sloforge.continuum.adapters import (
    ModelContract,
    ReferenceHeadMajorAdapter,
    ReferenceTokenMajorAdapter,
)
from sloforge.continuum.compatibility import (
    CompatibilityRequest,
    ExactnessClass,
    ModelSemantics,
    RuntimeCapabilities,
    StateDependencyEvidence,
)
from sloforge.continuum.operations import checkpoint_full, restore_reference_capture
from sloforge.continuum.storage import MemoryContentStore
from sloforge.continuum.transaction import SessionLease
from sloforge.helix.branching import (
    BranchCleanupError,
    BranchCompatibilityError,
    BranchIntervention,
    BranchStrategy,
    CrossPolicyBranch,
    ExactCowBranch,
    InterventionKind,
    RngMutationBranch,
    build_ir_state_reuse_report,
    create_branch_group,
    minimize_branch_interventions,
)
from sloforge.helix.environments import EnvironmentBackend
from sloforge.helix.ir import (
    Digest,
    LineageReference,
    LineageRelation,
    StateReuseMode,
    load_learning_transaction,
)
from sloforge.helix.ir import (
    StateReuseReport as CanonicalStateReuseReport,
)

_STAMP = "2026-08-03T00:00:00Z"
_COMMIT = "7e51ea7f7338755d23f889820558a4e046d6c42e"


def _checkpoint() -> tuple[ReferenceTokenMajorAdapter, MemoryContentStore, object]:
    runtime = ReferenceTokenMajorAdapter()
    runtime.create_session(
        session_id="helix-parent",
        request_id="helix-branch-request",
        tenant_id="tenant-helix",
        input_token_ids=(2, 3, 5, 7),
        seed=83,
    )
    for event in runtime.stream_tokens("helix-parent", count=5):
        runtime.acknowledge_gateway(
            "helix-parent", token_index=event.token_index, owner_epoch=event.owner_epoch
        )
    metadata = runtime.inspect_session("helix-parent")
    lease = SessionLease(
        session_id="helix-parent",
        owner_runtime=runtime.identity.runtime_name,
        owner_epoch=metadata.owner_epoch,
        fencing_token=metadata.owner_epoch,
        expiration_ms=120_000,
        coordinator_version=1,
        last_committed_state_version=metadata.state_version,
        last_committed_token_index=metadata.committed_output_index,
    )
    store = MemoryContentStore()
    artifact = checkpoint_full(
        runtime,
        "helix-parent",
        store=store,
        lease=lease,
        published_at_ms=1,
        capture_timestamp=_STAMP,
        git_commit=_COMMIT,
        continuum_version="0.1.0",
    )
    return runtime, store, artifact


def _branch_lease(runtime: ReferenceTokenMajorAdapter, branch_id: str) -> SessionLease:
    metadata = runtime.inspect_session("helix-parent")
    return SessionLease(
        session_id=branch_id,
        owner_runtime=runtime.identity.runtime_name,
        owner_epoch=1,
        fencing_token=1,
        expiration_ms=120_000,
        coordinator_version=1,
        last_committed_state_version=metadata.state_version,
        last_committed_token_index=metadata.committed_output_index,
    )


def _model(contract: ModelContract, **updates: object) -> ModelSemantics:
    model = ModelSemantics(
        model_id=contract.model_id,
        architecture="continuum_hybrid_decoder",
        weights_hash=contract.model_hash,
        state_producing_weights_hash=contract.state_producer_hash,
        output_head_hash="continuum-output-head-v1",
        tokenizer_hash=contract.tokenizer_hash,
        special_tokens_hash="continuum-special-tokens-v1",
        positional_encoding="absolute",
        rope_fingerprint=contract.positional_encoding_hash,
        attention_mask_semantics="causal",
        sliding_window=None,
        layer_count=2,
        head_count=8,
        kv_head_count=4,
        head_dim=4,
        recurrent_update_fingerprint=contract.recurrent_update_hash,
        adapter_hash=contract.adapter_hash,
        state_dtype="int32",
        quantization="none",
        sampler_algorithm="continuum-counter-v1",
    )
    return model.model_copy(update=updates)


def _runtime_capability(name: str, *, recompute: bool = True) -> RuntimeCapabilities:
    return RuntimeCapabilities(
        runtime_name=name,
        runtime_version="1.0.0",
        adapter_version="1.0.0",
        supported_state_types=(
            "state/attention-kv",
            "state/recurrent",
            "state/sampler",
            "state/guided-decoding",
            "state/client-delivery",
            "state/token-history",
        ),
        supported_dtypes=("int32",),
        can_recompute_from_token_history=recompute,
    )


def _compatibility(
    source: ModelContract,
    destination: ModelContract | None = None,
    *,
    recompute: bool = False,
    tokenizer_mismatch: bool = False,
) -> CompatibilityRequest:
    target = destination or source
    destination_model = _model(target)
    if tokenizer_mismatch:
        destination_model = destination_model.model_copy(update={"tokenizer_hash": "mismatch"})
    evidence = (
        StateDependencyEvidence(
            dependency_graph_hash="continuum-dependency-graph-v1",
            changed_components=("attention", "recurrent_update"),
            state_producing_components=("attention", "recurrent_update"),
            affected_state_components=("state/attention-kv", "state/recurrent"),
            recomputable_state_components=("state/attention-kv", "state/recurrent"),
            output_head_is_state_sink=True,
            token_history_available=True,
        )
        if recompute
        else None
    )
    return CompatibilityRequest(
        source=_model(source),
        destination=destination_model,
        source_runtime=_runtime_capability("continuum-source"),
        destination_runtime=_runtime_capability("continuum-destination"),
        source_layout_fingerprint="token-major",
        destination_layout_fingerprint="head-major" if recompute else "token-major",
        required_state_types=_runtime_capability("continuum-source").supported_state_types,
        required_exactness=(
            ExactnessClass.RECOMPUTATION_ASSISTED if recompute else ExactnessClass.EXACT_SEMANTIC
        ),
        dependency_evidence=evidence,
        allow_recomputation=recompute,
    )


def _changed_model(source: ModelContract) -> ModelContract:
    return replace(
        source,
        model_id="continuum/hybrid-decoder-v2",
        model_hash="1" * 64,
        state_producer_hash="2" * 64,
    )


def test_same_policy_cow_and_controlled_rng_branch_have_honest_reuse_reports(
    tmp_path: Path,
) -> None:
    runtime, store, parent = _checkpoint()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "state.txt").write_text("base\n")
    environment_backend = EnvironmentBackend(tmp_path / "environment", tenant_id="tenant-helix")
    environment = environment_backend.capture(workspace, seed=83, event_watermark=4)
    plans = (
        ExactCowBranch("branch-exact", "policy-a", _branch_lease(runtime, "branch-exact")),
        RngMutationBranch("branch-rng", "policy-a", _branch_lease(runtime, "branch-rng"), seed=991),
    )
    group = create_branch_group(
        parent,
        branch_point_id="b" * 64,
        source_policy_epoch_id="policy-a",
        plans=plans,
        store=store,
        expected_tenant_id="tenant-helix",
        expected_model=runtime.config.model,
        seed=83,
        published_at_ms=10,
        capture_timestamp=_STAMP,
        git_commit=_COMMIT,
        continuum_version="0.1.0",
        environment_backend=environment_backend,
        environment_capsule=environment,
    )
    assert group.shared_immutable_digests
    exact, mutated = group.members
    exact_state = restore_reference_capture(
        exact.checkpoint,
        store=store,
        expected_tenant_id="tenant-helix",
        expected_model=runtime.config.model,
    )
    assert all(page.copy_on_write_refs == 3 for page in exact_state.page_table)
    assert exact.state_reuse.source_state_exact
    assert exact.state_reuse.strategy is BranchStrategy.EXACT_COW
    assert mutated.rng_override is not None
    assert not mutated.state_reuse.source_state_exact
    assert "state/sampler" in mutated.state_reuse.replaced_components
    assert "state/sampler" not in mutated.state_reuse.directly_reused_components
    assert set(exact.state_reuse.source_components) == set(
        exact.state_reuse.directly_reused_components
    )
    assert set(mutated.state_reuse.source_components) == set(
        mutated.state_reuse.directly_reused_components
    ) | set(mutated.state_reuse.replaced_components)
    incomplete = exact.state_reuse.model_dump()
    incomplete["directly_reused_components"] = incomplete["directly_reused_components"][:-1]
    with pytest.raises(ValueError, match="cover every source component"):
        type(exact.state_reuse).model_validate(incomplete, strict=True)
    assert group.environment_base_capsule_id == environment.capsule_id
    assert exact.environment_branch is not None
    assert mutated.environment_branch is not None
    assert exact.environment_branch.info.seed == 83
    assert mutated.environment_branch.info.seed == 991
    exact.environment_branch.write_text("state.txt", "exact\n")
    assert mutated.environment_branch.read_text("state.txt") == "base\n"
    exact.environment_branch.cleanup()
    mutated.environment_branch.cleanup()


def test_environment_branch_rollback_attempts_every_cleanup_and_reports_leaks(
    tmp_path: Path,
) -> None:
    runtime, store, parent = _checkpoint()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = EnvironmentBackend(
        tmp_path / "environment-source", tenant_id="tenant-helix"
    ).capture(workspace, seed=83)

    class FaultingBackend:
        tenant_id = "tenant-helix"

        def __init__(self) -> None:
            self.created: list[str] = []
            self.cleanup_attempts: list[str] = []

        def fork(self, _capsule: object, *, branch_id: str, seed: int | None) -> object:
            del seed
            if branch_id == "branch-3":
                raise RuntimeError("injected fork failure")
            self.created.append(branch_id)
            return object()

        def cleanup_branch(self, branch_id: str) -> None:
            self.cleanup_attempts.append(branch_id)
            if branch_id == "branch-2":
                raise RuntimeError("injected cleanup failure")
            self.created.remove(branch_id)

    backend = FaultingBackend()
    plans = tuple(
        ExactCowBranch(f"branch-{index}", "policy-a", _branch_lease(runtime, f"branch-{index}"))
        for index in range(1, 4)
    )
    with pytest.raises(BranchCleanupError) as captured:
        create_branch_group(
            parent,
            branch_point_id="b" * 64,
            source_policy_epoch_id="policy-a",
            plans=plans,
            store=store,
            expected_tenant_id="tenant-helix",
            expected_model=runtime.config.model,
            seed=83,
            published_at_ms=10,
            capture_timestamp=_STAMP,
            git_commit=_COMMIT,
            continuum_version="0.1.0",
            environment_backend=backend,
            environment_capsule=environment,
        )
    assert backend.cleanup_attempts == ["branch-2", "branch-1"]
    assert captured.value.leaked_branch_ids == ("branch-2",)
    assert backend.created == ["branch-2"]


def test_cross_policy_incompatible_reuse_is_rejected_before_fork() -> None:
    runtime, store, parent = _checkpoint()
    plans = (
        ExactCowBranch("branch-exact", "policy-a", _branch_lease(runtime, "branch-exact")),
        CrossPolicyBranch(
            "branch-incompatible",
            "policy-b",
            _branch_lease(runtime, "branch-incompatible"),
            compatibility=_compatibility(runtime.config.model, tokenizer_mismatch=True),
        ),
    )
    with pytest.raises(BranchCompatibilityError, match="TOKENIZER_MISMATCH"):
        create_branch_group(
            parent,
            branch_point_id="b" * 64,
            source_policy_epoch_id="policy-a",
            plans=plans,
            store=store,
            expected_tenant_id="tenant-helix",
            expected_model=runtime.config.model,
            seed=83,
            published_at_ms=10,
            capture_timestamp=_STAMP,
            git_commit=_COMMIT,
            continuum_version="0.1.0",
        )


def test_compatible_cross_policy_analysis_reuses_only_declared_state() -> None:
    runtime, store, parent = _checkpoint()
    plans = (
        ExactCowBranch("branch-exact", "policy-a", _branch_lease(runtime, "branch-exact")),
        CrossPolicyBranch(
            "branch-compatible",
            "policy-b",
            _branch_lease(runtime, "branch-compatible"),
            compatibility=_compatibility(runtime.config.model),
        ),
    )
    group = create_branch_group(
        parent,
        branch_point_id="d" * 64,
        source_policy_epoch_id="policy-a",
        plans=plans,
        store=store,
        expected_tenant_id="tenant-helix",
        expected_model=runtime.config.model,
        seed=83,
        published_at_ms=15,
        capture_timestamp=_STAMP,
        git_commit=_COMMIT,
        continuum_version="0.1.0",
    )
    compatible = group.members[1]
    assert compatible.state_reuse.compatibility_class == "exact_semantic"
    assert compatible.state_reuse.recomputed_components == ()
    assert set(compatible.state_reuse.directly_reused_components) == {
        "state/token-history",
        "state/sampler",
        "state/client-delivery",
        "state/attention-kv",
        "state/recurrent",
        "state/guided-decoding",
    }


def test_cross_policy_state_is_explicitly_recomputed_from_history() -> None:
    runtime, store, parent = _checkpoint()
    changed = _changed_model(runtime.config.model)
    destination = ReferenceHeadMajorAdapter(model=changed)
    plans = (
        ExactCowBranch("branch-exact", "policy-a", _branch_lease(runtime, "branch-exact")),
        CrossPolicyBranch(
            "branch-recomputed",
            "policy-b",
            _branch_lease(runtime, "branch-recomputed"),
            compatibility=_compatibility(runtime.config.model, changed, recompute=True),
            destination=destination,
            permit_recomputation=True,
        ),
    )
    group = create_branch_group(
        parent,
        branch_point_id="c" * 64,
        source_policy_epoch_id="policy-a",
        plans=plans,
        store=store,
        expected_tenant_id="tenant-helix",
        expected_model=runtime.config.model,
        seed=83,
        published_at_ms=20,
        capture_timestamp=_STAMP,
        git_commit=_COMMIT,
        continuum_version="0.1.0",
    )
    recomputed = group.members[1]
    assert recomputed.recomputed is not None
    assert recomputed.recomputed.logical.model.model_hash == changed.model_hash
    assert recomputed.state_reuse.strategy is BranchStrategy.RECOMPUTE_FROM_HISTORY
    assert set(recomputed.state_reuse.recomputed_components) == {
        "state/attention-kv",
        "state/recurrent",
    }
    assert recomputed.state_reuse.recomputation_evidence_digest is not None


def test_branch_minimization_finds_cardinality_minimal_deterministic_witness() -> None:
    interventions = tuple(
        BranchIntervention(
            intervention_id=f"intervention-{name}",
            kind=kind,
            target=f"target-{name}",
            value_digest=sha256(name.encode()).hexdigest(),
        )
        for name, kind in (
            ("a", InterventionKind.ENVIRONMENT),
            ("b", InterventionKind.RNG),
            ("c", InterventionKind.POLICY),
        )
    )
    result = minimize_branch_interventions(
        interventions,
        lambda candidate: (
            {item.intervention_id for item in candidate} >= {"intervention-b", "intervention-c"}
        ),
        max_evaluations=16,
    )
    assert result.search_complete
    assert result.minimal_intervention_ids == ("intervention-b", "intervention-c")
    assert result.evaluations <= 16
    invalid = result.model_dump()
    invalid["minimal_intervention_ids"] = ("intervention-unknown",)
    with pytest.raises(ValueError, match="subset"):
        type(result).model_validate(invalid, strict=True)


def test_state_reuse_evidence_late_binds_to_canonical_ir() -> None:
    transaction = load_learning_transaction(
        Path(__file__).parents[2] / "tests/fixtures/helix/learning-transaction-v1.json"
    )
    source = transaction.branch_group.branch_point.environment_state
    policy = source.policy_epoch
    policy_key = f"{policy.policy_id}@{policy.epoch}"
    artifact_ids = (source.capsule_id, policy_key)
    lineage = tuple(
        LineageReference(
            artifact_id=artifact_id,
            artifact_kind="helix.test/bridge",
            relation=LineageRelation.DERIVED_FROM,
            digest=Digest(value=sha256(artifact_id.encode()).hexdigest()),
        )
        for artifact_id in artifact_ids
    )
    report = build_ir_state_reuse_report(
        source_capsule=source,
        target_environment_id=source.environment_id,
        target_policy_epoch=policy,
        target_compatibility_fingerprint=source.compatibility_fingerprint,
        mode=StateReuseMode.EXACT.value,
        compatible=True,
        reused=True,
        conversion_evidence=None,
        reason="policy and environment compatibility fingerprints match",
        assessed_at=_STAMP,
        lineage=lineage,
    )
    assert isinstance(report, CanonicalStateReuseReport)
    assert report.mode is StateReuseMode.EXACT
    assert report.compatible and report.reused
