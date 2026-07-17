"""Continuum-backed Helix branch group construction."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from sloforge.continuum.adapters import ModelContract
from sloforge.continuum.compatibility import (
    CompatibilityDecision,
    ExactnessClass,
    analyze_compatibility,
)
from sloforge.continuum.operations import (
    CheckpointArtifact,
    RecomputeProofError,
    fork_checkpoint,
    recompute_from_token_history,
    verify_checkpoint_artifact,
)
from sloforge.continuum.storage import ContentStore
from sloforge.helix.capture.models import canonical_digest

from .models import (
    BranchGroupExecution,
    BranchMember,
    BranchPlan,
    BranchStrategy,
    CrossPolicyBranch,
    ExactCowBranch,
    RngActivationOverride,
    RngMutationBranch,
    StateReuseReport,
)


class BranchError(RuntimeError):
    code = "helix_branch_error"


class BranchCompatibilityError(BranchError):
    code = "helix_branch_incompatible_state"


class BranchPlanError(BranchError):
    code = "helix_branch_plan_invalid"


class BranchCleanupError(BranchError):
    code = "helix_branch_cleanup_failed"

    def __init__(self, message: str, *, leaked_branch_ids: tuple[str, ...]) -> None:
        super().__init__(message)
        self.leaked_branch_ids = leaked_branch_ids


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,255}$")


def _components(parent: CheckpointArtifact) -> tuple[str, ...]:
    return tuple(
        descriptor.semantic_id
        for descriptor in parent.capsule.logical_state.component_descriptors()
    )


def _report(
    *,
    branch_id: str,
    parent: CheckpointArtifact,
    source_policy_epoch_id: str,
    destination_policy_epoch_id: str,
    strategy: BranchStrategy,
    compatibility_class: str,
    reused: tuple[str, ...],
    recomputed: tuple[str, ...] = (),
    replaced: tuple[str, ...] = (),
    obligations: tuple[str, ...] = (),
    exact: bool,
    recomputation_evidence_digest: str | None = None,
) -> StateReuseReport:
    payload: dict[str, Any] = {
        "schema_version": "sloforge.helix.state-reuse-report/v1",
        "branch_id": branch_id,
        "source_capsule_id": parent.capsule.identity.capsule_id,
        "source_policy_epoch_id": source_policy_epoch_id,
        "destination_policy_epoch_id": destination_policy_epoch_id,
        "strategy": strategy.value,
        "compatibility_class": compatibility_class,
        "source_components": _components(parent),
        "directly_reused_components": reused,
        "recomputed_components": recomputed,
        "replaced_components": replaced,
        "unsupported_components": (),
        "verification_obligations": obligations,
        "source_state_exact": exact,
        "transcript_equivalence_claimed": False,
        "recomputation_evidence_digest": recomputation_evidence_digest,
    }
    report_digest = canonical_digest(payload)
    document = {**payload, "report_id": "0" * 64, "report_digest": report_digest}
    document["report_id"] = canonical_digest(
        {key: value for key, value in document.items() if key != "report_id"}
    )
    return StateReuseReport(
        report_id=document["report_id"],
        branch_id=branch_id,
        source_capsule_id=parent.capsule.identity.capsule_id,
        source_policy_epoch_id=source_policy_epoch_id,
        destination_policy_epoch_id=destination_policy_epoch_id,
        strategy=strategy,
        compatibility_class=compatibility_class,
        source_components=_components(parent),
        directly_reused_components=reused,
        recomputed_components=recomputed,
        replaced_components=replaced,
        unsupported_components=(),
        verification_obligations=obligations,
        source_state_exact=exact,
        recomputation_evidence_digest=recomputation_evidence_digest,
        report_digest=report_digest,
    )


def _preflight(
    parent: CheckpointArtifact,
    source_policy_epoch_id: str,
    plans: tuple[BranchPlan, ...],
) -> dict[str, CompatibilityDecision]:
    if not 2 <= len(plans) <= 64:
        raise BranchPlanError("branch group size must be in 2..64")
    branch_ids = [plan.branch_id for plan in plans]
    if len(branch_ids) != len(set(branch_ids)):
        raise BranchPlanError("branch identifiers must be unique")
    session_ids = [plan.lease.session_id for plan in plans]
    if len(session_ids) != len(set(session_ids)):
        raise BranchPlanError("branch leases require unique session identifiers")
    decisions: dict[str, CompatibilityDecision] = {}
    source_seed = parent.capsule.logical_state.sampler.seed
    actual_components = set(_components(parent))
    for plan in plans:
        if not _IDENTIFIER.fullmatch(plan.branch_id):
            raise BranchPlanError("branch identifier is invalid")
        if not _IDENTIFIER.fullmatch(plan.policy_epoch_id):
            raise BranchPlanError("policy epoch identifier is invalid")
        if plan.branch_id != plan.lease.session_id:
            raise BranchPlanError("branch identifier must equal its Continuum lease session")
        expected_strategy = (
            BranchStrategy.EXACT_COW
            if isinstance(plan, ExactCowBranch)
            else BranchStrategy.CONTROLLED_RNG_MUTATION
            if isinstance(plan, RngMutationBranch)
            else BranchStrategy.COMPATIBLE_CROSS_POLICY
        )
        if plan.strategy is not expected_strategy:
            raise BranchPlanError("branch strategy contradicts its typed branch plan")
        if (
            isinstance(plan, (ExactCowBranch, RngMutationBranch))
            and plan.policy_epoch_id != source_policy_epoch_id
        ):
            raise BranchPlanError("same-policy branch names a different policy epoch")
        if isinstance(plan, RngMutationBranch) and plan.seed == source_seed:
            raise BranchPlanError("controlled RNG mutation cannot reuse the source seed")
        if isinstance(plan, CrossPolicyBranch):
            request = plan.compatibility
            capsule = parent.capsule
            if request.source.weights_hash != capsule.identity.model_hash.value:
                raise BranchCompatibilityError(
                    "compatibility request source model is not the captured Continuum model"
                )
            if request.source.tokenizer_hash != capsule.identity.tokenizer_hash.value:
                raise BranchCompatibilityError(
                    "compatibility request source tokenizer is not the captured tokenizer"
                )
            restrictions = set(capsule.compatibility.architecture_restrictions)
            if f"state_producer={request.source.state_producing_weights_hash}" not in restrictions:
                raise BranchCompatibilityError(
                    "compatibility request source state producer is not bound to the capsule"
                )
            missing_required = actual_components - set(request.required_state_types)
            if missing_required:
                raise BranchCompatibilityError(
                    "compatibility request omits captured state: "
                    + ", ".join(sorted(missing_required))
                )
            if plan.destination is not None:
                destination_model = plan.destination.config.model
                declared = request.destination
                if (
                    declared.weights_hash != destination_model.model_hash
                    or declared.state_producing_weights_hash
                    != destination_model.state_producer_hash
                    or declared.tokenizer_hash != destination_model.tokenizer_hash
                ):
                    raise BranchCompatibilityError(
                        "compatibility request destination is not bound to the runtime model"
                    )
            decision = analyze_compatibility(plan.compatibility)
            if not decision.safe or decision.compatibility_class is ExactnessClass.INCOMPATIBLE:
                unsupported = ",".join(decision.unsupported_state) or "model-derived state"
                codes = ",".join(reason.code for reason in decision.reasons)
                raise BranchCompatibilityError(
                    f"cross-policy state reuse rejected ({codes}); unsupported={unsupported}"
                )
            if decision.compatibility_class in {
                ExactnessClass.NUMERICALLY_EQUIVALENT,
                ExactnessClass.QUALITY_BOUNDED,
            }:
                raise BranchCompatibilityError(
                    "cross-policy branch requires a conversion path this coordinator does not "
                    "silently apply"
                )
            needs_recompute = bool(decision.required_recomputation)
            if needs_recompute and (not plan.permit_recomputation or plan.destination is None):
                raise BranchCompatibilityError(
                    "cross-policy state requires explicit permitted recomputation and a destination"
                )
            decisions[plan.branch_id] = decision
    return decisions


def create_branch_group(
    parent: CheckpointArtifact,
    *,
    branch_point_id: str,
    source_policy_epoch_id: str,
    plans: tuple[BranchPlan, ...],
    store: ContentStore,
    expected_tenant_id: str,
    expected_model: ModelContract,
    seed: int,
    published_at_ms: int,
    capture_timestamp: str,
    git_commit: str,
    continuum_version: str,
    environment_backend: object | None = None,
    environment_capsule: object | None = None,
) -> BranchGroupExecution:
    """Preflight every branch, then make one exact Continuum COW fork publication."""

    if not 0 <= seed < 2**64:
        raise ValueError("branch group seed must be an unsigned 64-bit integer")
    if seed + len(plans) - 1 >= 2**64:
        raise ValueError("derived branch seeds exceed the unsigned 64-bit range")
    if not re.fullmatch(r"[0-9a-f]{64}", branch_point_id):
        raise BranchPlanError("branch point identifier must be a SHA-256 digest")
    if not _IDENTIFIER.fullmatch(source_policy_epoch_id):
        raise BranchPlanError("source policy epoch identifier is invalid")
    if (environment_backend is None) != (environment_capsule is None):
        raise BranchPlanError(
            "environment backend and environment capsule must be supplied together"
        )
    environment_base_capsule_id: str | None = None
    fork_environment: Any = None
    cleanup_environment: Any = None
    if environment_backend is not None and environment_capsule is not None:
        environment_base_capsule_id = getattr(environment_capsule, "capsule_id", None)
        environment_tenant_id = getattr(environment_capsule, "tenant_id", None)
        backend_tenant_id = getattr(environment_backend, "tenant_id", None)
        if not isinstance(environment_base_capsule_id, str) or not environment_base_capsule_id:
            raise BranchPlanError("environment capsule has no stable capsule identifier")
        if environment_tenant_id != expected_tenant_id:
            raise BranchPlanError("environment capsule and Continuum tenant differ")
        if backend_tenant_id is not None and backend_tenant_id != expected_tenant_id:
            raise BranchPlanError("environment backend and Continuum tenant differ")
        fork_environment = getattr(environment_backend, "fork", None)
        cleanup_environment = getattr(environment_backend, "cleanup_branch", None)
        if not callable(fork_environment) or not callable(cleanup_environment):
            raise BranchPlanError("environment backend does not implement fork and cleanup")
    verify_checkpoint_artifact(parent)
    decisions = _preflight(parent, source_policy_epoch_id, plans)
    forked = fork_checkpoint(
        parent,
        store=store,
        expected_tenant_id=expected_tenant_id,
        expected_model=expected_model,
        branch_leases=tuple(plan.lease for plan in plans),
        seed=seed,
        published_at_ms=published_at_ms,
        capture_timestamp=capture_timestamp,
        git_commit=git_commit,
        continuum_version=continuum_version,
    )
    component_ids = _components(parent)
    members: list[BranchMember] = []
    for index, (plan, checkpoint) in enumerate(zip(plans, forked.branches, strict=True)):
        if isinstance(plan, ExactCowBranch):
            report = _report(
                branch_id=plan.branch_id,
                parent=parent,
                source_policy_epoch_id=source_policy_epoch_id,
                destination_policy_epoch_id=plan.policy_epoch_id,
                strategy=plan.strategy,
                compatibility_class=ExactnessClass.EXACT_BITWISE.value,
                reused=component_ids,
                exact=True,
            )
            members.append(BranchMember(plan.branch_id, plan.policy_epoch_id, checkpoint, report))
            continue
        if isinstance(plan, RngMutationBranch):
            sampler = parent.capsule.logical_state.sampler
            override = RngActivationOverride(
                source_seed=sampler.seed,
                source_counter=sampler.rng_counter,
                branch_seed=plan.seed,
                reset_counter_to=sampler.rng_counter,
            )
            reused = tuple(item for item in component_ids if item != "state/sampler")
            report = _report(
                branch_id=plan.branch_id,
                parent=parent,
                source_policy_epoch_id=source_policy_epoch_id,
                destination_policy_epoch_id=plan.policy_epoch_id,
                strategy=plan.strategy,
                compatibility_class="controlled_counterfactual_mutation",
                reused=reused,
                replaced=("state/sampler",),
                obligations=("apply RNG override before the first branch token",),
                exact=False,
            )
            members.append(
                BranchMember(
                    plan.branch_id,
                    plan.policy_epoch_id,
                    checkpoint,
                    report,
                    rng_override=override,
                )
            )
            continue

        decision = decisions[plan.branch_id]
        required = tuple(
            component
            for requirement in decision.required_recomputation
            for component in requirement.state_components
        )
        obligations = tuple(
            f"{item.obligation_id}: {item.method}" for item in decision.verification_obligations
        )
        if required:
            assert plan.destination is not None
            try:
                result = recompute_from_token_history(
                    checkpoint,
                    store=store,
                    destination=plan.destination,
                    expected_tenant_id=expected_tenant_id,
                    seed=seed + index,
                )
            except RecomputeProofError as error:
                raise BranchCompatibilityError(
                    f"explicit cross-policy recomputation failed: {error}"
                ) from error
            recomputed_set = set(required)
            reused = tuple(item for item in component_ids if item not in recomputed_set)
            report = _report(
                branch_id=plan.branch_id,
                parent=parent,
                source_policy_epoch_id=source_policy_epoch_id,
                destination_policy_epoch_id=plan.policy_epoch_id,
                strategy=BranchStrategy.RECOMPUTE_FROM_HISTORY,
                compatibility_class=decision.compatibility_class.value,
                reused=reused,
                recomputed=required,
                obligations=obligations,
                exact=False,
                recomputation_evidence_digest=result.evidence.evidence_digest,
            )
            members.append(
                BranchMember(
                    plan.branch_id,
                    plan.policy_epoch_id,
                    checkpoint,
                    report,
                    recomputed=result.captured,
                )
            )
        else:
            report = _report(
                branch_id=plan.branch_id,
                parent=parent,
                source_policy_epoch_id=source_policy_epoch_id,
                destination_policy_epoch_id=plan.policy_epoch_id,
                strategy=plan.strategy,
                compatibility_class=decision.compatibility_class.value,
                reused=component_ids,
                obligations=obligations,
                exact=(decision.compatibility_class is ExactnessClass.EXACT_BITWISE),
            )
            members.append(BranchMember(plan.branch_id, plan.policy_epoch_id, checkpoint, report))

    if environment_backend is not None and environment_capsule is not None:
        created: list[str] = []
        handles: dict[str, object] = {}
        try:
            for index, plan in enumerate(plans):
                environment_seed = (
                    plan.seed
                    if isinstance(plan, RngMutationBranch)
                    else None
                    if isinstance(plan, ExactCowBranch)
                    else seed + index
                )
                handles[plan.branch_id] = fork_environment(
                    environment_capsule,
                    branch_id=plan.branch_id,
                    seed=environment_seed,
                )
                created.append(plan.branch_id)
        except Exception as error:
            cleanup_failures: list[tuple[str, Exception]] = []
            for branch_id in reversed(created):
                try:
                    cleanup_environment(branch_id)
                except Exception as cleanup_error:
                    cleanup_failures.append((branch_id, cleanup_error))
            if cleanup_failures:
                leaked = tuple(branch_id for branch_id, _error in cleanup_failures)
                detail = "; ".join(
                    f"{branch_id}: {cleanup_error}" for branch_id, cleanup_error in cleanup_failures
                )
                raise BranchCleanupError(
                    "environment branch construction failed and cleanup was incomplete: "
                    f"{error}; cleanup failures: {detail}",
                    leaked_branch_ids=leaked,
                ) from error
            raise BranchError(f"environment COW branch construction failed: {error}") from error
        members = [
            replace(member, environment_branch=handles[member.branch_id]) for member in members
        ]

    group_payload = {
        "schema": "sloforge.helix.branch-group-execution/v1",
        "branch_point_id": branch_point_id,
        "source_capsule_id": parent.capsule.identity.capsule_id,
        "branch_ids": tuple(member.branch_id for member in members),
        "state_reuse_reports": tuple(member.state_reuse.report_id for member in members),
        "environment_base_capsule_id": environment_base_capsule_id,
        "seed": seed,
    }
    return BranchGroupExecution(
        group_id=canonical_digest(group_payload),
        branch_point_id=branch_point_id,
        source_capsule_id=parent.capsule.identity.capsule_id,
        members=tuple(members),
        shared_immutable_digests=forked.shared_immutable_digests,
        seed=seed,
        environment_base_capsule_id=environment_base_capsule_id,
    )
