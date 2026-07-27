"""Transactional checkpoint resume with ownership and gateway epoch cutover."""

from __future__ import annotations

from hashlib import sha256

from sloforge.continuum.adapters import ContinuumRuntimeAdapter, ModelContract
from sloforge.continuum.ir import canonical_hash
from sloforge.continuum.storage import ContentStore
from sloforge.continuum.transaction import (
    CutoverPhase,
    DurableCoordinator,
    GatewayCommitLedger,
)

from .checkpoint import verify_checkpoint_artifact
from .models import AuthorizationError, CheckpointArtifact, ResumeResult
from .restore import restore_reference_capture

_PRECOMMIT_PHASES = (
    CutoverPhase.COMPATIBILITY_VALIDATED,
    CutoverPhase.DESTINATION_PREPARING,
    CutoverPhase.PRECOPYING,
    CutoverPhase.DELTA_SYNCING,
    CutoverPhase.CUTOVER_REQUESTED,
    CutoverPhase.SOURCE_QUIESCING,
    CutoverPhase.SOURCE_FROZEN,
    CutoverPhase.FINAL_DELTA_TRANSFERRING,
    CutoverPhase.DESTINATION_IMPORTING,
    CutoverPhase.DESTINATION_VALIDATING,
)


def _event_hash(label: str, capsule_id: str) -> str:
    return sha256(f"continuum-resume/v1\0{label}\0{capsule_id}".encode()).hexdigest()


def _rollback_precommit_resume(
    *,
    coordinator: DurableCoordinator,
    transaction_id: str,
    destination: ContinuumRuntimeAdapter,
    source: ContinuumRuntimeAdapter | None,
    source_fenced: bool,
    session_id: str,
    source_epoch: int,
    capsule_id: str,
    at_ms: int,
) -> None:
    """Durably abort staged state without changing ownership or gateway state."""

    destination.abort_destination(session_id)
    current = coordinator.transaction(transaction_id)
    if current.phase is not CutoverPhase.ABORTING:
        current = coordinator.transition(
            transaction_id,
            expected=current.phase,
            target=CutoverPhase.ABORTING,
            event_id="resume-precommit-aborting",
            at_ms=at_ms,
            payload_hash=_event_hash("precommit-aborting", capsule_id),
        )
    coordinator.transition(
        transaction_id,
        expected=current.phase,
        target=CutoverPhase.ROLLED_BACK,
        event_id="resume-precommit-rolled-back",
        at_ms=at_ms + 1,
        payload_hash=_event_hash("precommit-rolled-back", capsule_id),
    )
    if source is not None and source_fenced:
        source.release_source_fence(
            session_id,
            expected_owner_epoch=source_epoch,
            coordinator_owner_epoch=coordinator.lease(session_id).owner_epoch,
            ownership_committed=False,
        )


def resume_checkpoint(
    artifact: CheckpointArtifact,
    *,
    store: ContentStore,
    destination: ContinuumRuntimeAdapter,
    source: ContinuumRuntimeAdapter | None = None,
    source_release_confirmed: bool = False,
    expected_tenant_id: str,
    expected_model: ModelContract,
    coordinator: DurableCoordinator,
    gateway: GatewayCommitLedger,
    seed: int,
    now_ms: int,
    timeout_ms: int = 60_000,
) -> ResumeResult:
    """Import, validate, CAS ownership, switch the gateway, then activate output."""

    verify_checkpoint_artifact(artifact)
    capsule = artifact.capsule
    lease = coordinator.lease(capsule.identity.session_id)
    if lease.owner_epoch != capsule.identity.owner_epoch:
        raise AuthorizationError("capsule owner epoch is stale under the coordinator")
    if lease.fencing_token != capsule.transaction.fencing_token:
        raise AuthorizationError("capsule fencing token is stale under the coordinator")
    if lease.owner_runtime != capsule.transaction.ownership_lease.owner_runtime:
        raise AuthorizationError("capsule owner runtime is stale under the coordinator")
    if lease.last_committed_token_index != capsule.transaction.commit_watermark:
        raise AuthorizationError("coordinator and capsule token watermarks differ")
    if source is None and not source_release_confirmed:
        raise AuthorizationError(
            "resume requires source fencing or explicit confirmation that source resources were released"
        )
    source_next_token: int | None = None
    if source is not None:
        source_metadata = source.inspect_session(capsule.identity.session_id)
        if source.identity.runtime_name != lease.owner_runtime:
            raise AuthorizationError("provided source is not the coordinator owner")
        if source_metadata.owner_epoch != lease.owner_epoch:
            raise AuthorizationError("provided source has a stale owner epoch")
        if source_metadata.lifecycle.value != "paused":
            raise AuthorizationError("resident source must be paused before checkpoint resume")
        source_next_token = source.dry_run_next_token(capsule.identity.session_id)
    captured = restore_reference_capture(
        artifact,
        store=store,
        expected_tenant_id=expected_tenant_id,
        expected_model=expected_model,
    )
    plan_hash = canonical_hash(
        {
            "schema": "sloforge.continuum.resume-plan/v1",
            "capsule_id": capsule.identity.capsule_id,
            "destination": destination.identity.runtime_name,
            "destination_adapter": destination.identity.adapter_version,
            "seed": seed,
        }
    )
    transaction = coordinator.begin_transaction(
        session_id=capsule.identity.session_id,
        destination_candidate=destination.identity.runtime_name,
        migration_plan_hash=plan_hash,
        seed=seed,
        now_ms=now_ms,
        timeout_ms=timeout_ms,
    )
    current = CutoverPhase.PROPOSED
    source_fenced = False
    try:
        destination.prepare_destination_session(
            captured,
            destination_session_id=capsule.identity.session_id,
            proposed_owner_epoch=transaction.proposed_destination_epoch,
        )
        for offset, target in enumerate(_PRECOMMIT_PHASES, start=1):
            transaction = coordinator.transition(
                transaction.transaction_id,
                expected=current,
                target=target,
                event_id=f"resume-{target.value.lower()}",
                at_ms=now_ms + offset,
                payload_hash=_event_hash(target.value, capsule.identity.capsule_id),
            )
            current = target
            if target is CutoverPhase.SOURCE_FROZEN and source is not None:
                source.fence_source_writer(
                    capsule.identity.session_id,
                    expected_owner_epoch=lease.owner_epoch,
                )
                source_fenced = True
            if target is CutoverPhase.DESTINATION_IMPORTING:
                destination.import_captured_state(capsule.identity.session_id, captured)
        validation = destination.validate_imported_state(capsule.identity.session_id)
        if not validation.structurally_valid or not validation.continuation_valid:
            raise AuthorizationError("destination continuation validation did not pass")
        if source_next_token is not None and validation.dry_run_next_token != source_next_token:
            raise AuthorizationError("source and destination bounded continuation tokens differ")
        transaction = coordinator.transition(
            transaction.transaction_id,
            expected=CutoverPhase.DESTINATION_VALIDATING,
            target=CutoverPhase.COMMIT_INTENT_RECORDED,
            event_id="resume-commit-intent",
            at_ms=now_ms + len(_PRECOMMIT_PHASES) + 1,
            payload_hash=_event_hash("commit-intent", capsule.identity.capsule_id),
            state_hashes=(captured.logical.continuation_hash,),
            commit_watermark=capsule.transaction.commit_watermark,
            rollback_watermark=capsule.transaction.rollback_boundary,
        )
    except Exception as resume_error:
        try:
            _rollback_precommit_resume(
                coordinator=coordinator,
                transaction_id=transaction.transaction_id,
                destination=destination,
                source=source,
                source_fenced=source_fenced,
                session_id=capsule.identity.session_id,
                source_epoch=lease.owner_epoch,
                capsule_id=capsule.identity.capsule_id,
                at_ms=now_ms + len(_PRECOMMIT_PHASES) + 2,
            )
        except Exception as rollback_error:
            raise AuthorizationError(
                "pre-commit resume failed and durable rollback did not complete "
                f"({type(rollback_error).__name__})"
            ) from resume_error
        raise
    transaction, committed_lease = coordinator.commit_ownership(
        transaction.transaction_id,
        event_id="resume-ownership-committed",
        at_ms=now_ms + len(_PRECOMMIT_PHASES) + 2,
        state_version=captured.logical.state_version,
    )
    transaction = coordinator.transition(
        transaction.transaction_id,
        expected=CutoverPhase.OWNERSHIP_COMMITTED,
        target=CutoverPhase.GATEWAY_SWITCHING,
        event_id="resume-gateway-switching",
        at_ms=now_ms + len(_PRECOMMIT_PHASES) + 3,
        payload_hash=_event_hash("gateway-switch", capsule.identity.capsule_id),
    )
    gateway.switch_owner(
        session_id=capsule.identity.session_id,
        expected_epoch=transaction.source_epoch,
        destination_epoch=committed_lease.owner_epoch,
        expected_watermark=transaction.commit_watermark,
    )
    active = destination.activate_destination(
        capsule.identity.session_id,
        committed_owner_epoch=committed_lease.owner_epoch,
        fencing_token=str(committed_lease.fencing_token),
    )
    for offset, target in enumerate(
        (
            CutoverPhase.DESTINATION_ACTIVE,
            CutoverPhase.SOURCE_DRAINING,
            CutoverPhase.COMPLETED,
        ),
        start=len(_PRECOMMIT_PHASES) + 4,
    ):
        transaction = coordinator.transition(
            transaction.transaction_id,
            expected=transaction.phase,
            target=target,
            event_id=f"resume-{target.value.lower()}",
            at_ms=now_ms + offset,
            payload_hash=_event_hash(target.value, capsule.identity.capsule_id),
        )
    return ResumeResult(
        captured=captured,
        validation=validation,
        transaction=transaction,
        lease=committed_lease,
        active=active,
        source_owner_epoch=capsule.identity.owner_epoch,
        destination_owner_epoch=committed_lease.owner_epoch,
        source_fenced=source_fenced,
    )
