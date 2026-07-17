from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sloforge.continuum.adapters import ReferenceTokenMajorAdapter, SessionLifecycle
from sloforge.continuum.storage import MemoryContentStore
from sloforge.continuum.transaction import SessionLease
from sloforge.helix.capture import (
    CaptureAborted,
    CaptureArtifactIntegrityError,
    CaptureBoundary,
    CaptureBoundaryMismatch,
    CaptureInProgress,
    CapturePhase,
    CaptureSources,
    CoordinatedCaptureCoordinator,
    CoordinatedCaptureRequest,
    VerifiedCaptureArtifact,
    build_ir_branch_point,
)
from sloforge.helix.effects import Effect, EffectClass, EffectLedger
from sloforge.helix.environments import EnvironmentBackend
from sloforge.helix.ir import Digest, LineageReference, LineageRelation, load_learning_transaction

_STAMP = "2026-08-03T00:00:00Z"
_COMMIT = "7e51ea7f7338755d23f889820558a4e046d6c42e"


def _runtime() -> ReferenceTokenMajorAdapter:
    runtime = ReferenceTokenMajorAdapter()
    runtime.create_session(
        session_id="helix-capture-session",
        request_id="helix-capture-request",
        tenant_id="tenant-helix",
        input_token_ids=(2, 3, 5),
        seed=71,
    )
    for event in runtime.stream_tokens("helix-capture-session", count=4):
        runtime.acknowledge_gateway(
            "helix-capture-session",
            token_index=event.token_index,
            owner_epoch=event.owner_epoch,
        )
    return runtime


def _lease(runtime: ReferenceTokenMajorAdapter) -> SessionLease:
    metadata = runtime.inspect_session("helix-capture-session")
    return SessionLease(
        session_id="helix-capture-session",
        owner_runtime=runtime.identity.runtime_name,
        owner_epoch=metadata.owner_epoch,
        fencing_token=metadata.owner_epoch,
        expiration_ms=120_000,
        coordinator_version=1,
        last_committed_state_version=metadata.state_version,
        last_committed_token_index=metadata.committed_output_index,
    )


def _request(boundary: CaptureBoundary) -> CoordinatedCaptureRequest:
    return CoordinatedCaptureRequest(
        capture_id="capture-crash-recovery",
        session_id="helix-capture-session",
        source_trajectory_id="trajectory-source",
        policy_epoch_id="policy-a-epoch-7",
        boundary=boundary,
        seed=71,
        max_quiescence_polls=4,
        published_at_ms=10,
        capture_timestamp=_STAMP,
        git_commit=_COMMIT,
        continuum_version="0.1.0",
        created_at=_STAMP,
        reason="branch before tool choice",
    )


def _sources(
    tmp_path: Path, runtime: ReferenceTokenMajorAdapter
) -> tuple[CaptureSources, CaptureBoundary]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "task.txt").write_text("deterministic task\n")
    environment_backend = EnvironmentBackend(
        tmp_path / "environment-store", tenant_id="tenant-helix"
    )
    environment = environment_backend.capture(
        workspace,
        seed=71,
        event_watermark=2,
        allowed_tools=("read",),
    )
    effects = EffectLedger(tenant_id="tenant-helix")
    effects.record(Effect.build(EffectClass.READ_ONLY, "read", tenant_id="tenant-helix"))
    effects.record(Effect.build(EffectClass.PURE, "parse", tenant_id="tenant-helix"))
    effects.commit(1)
    boundary = CaptureBoundary(
        action_watermark=5,
        model_token_watermark=3,
        environment_event_watermark=2,
        effect_watermark=1,
    )
    sources = CaptureSources(
        model=runtime,
        model_store=MemoryContentStore(),
        lease=_lease(runtime),
        read_boundary=lambda: boundary,
        capture_environment=lambda _seed: environment_backend.artifact_watermark(environment),
        capture_effects=effects.artifact_watermark,
        expected_tenant_id="tenant-helix",
    )
    return sources, boundary


def test_coordinated_capture_is_atomic_idempotent_and_resumes_source(tmp_path: Path) -> None:
    runtime = _runtime()
    sources, boundary = _sources(tmp_path, runtime)
    database = tmp_path / "capture.sqlite"
    artifact_directory = tmp_path / "capture-artifacts"
    request = _request(boundary)
    with CoordinatedCaptureCoordinator(
        database, artifact_directory=artifact_directory
    ) as coordinator:
        proposed = coordinator.propose(request)
        assert proposed.phase is CapturePhase.CAPTURE_PROPOSED
        branch_point = coordinator.execute(request.capture_id, sources)
        assert coordinator.published_branch_point(request.capture_id) == branch_point
        assert coordinator.execute(request.capture_id, sources) == branch_point
        assert runtime.inspect_session(request.session_id).lifecycle is SessionLifecycle.ACTIVE
        phases = tuple(entry.phase for entry in coordinator.journal(request.capture_id))
        assert phases == (
            CapturePhase.CAPTURE_PROPOSED,
            CapturePhase.QUIESCING,
            CapturePhase.QUIESCED,
            CapturePhase.MODEL_CAPTURED,
            CapturePhase.ENVIRONMENT_CAPTURED,
            CapturePhase.EFFECTS_CAPTURED,
            CapturePhase.VALIDATING,
            CapturePhase.COMPLETED,
        )
        assert branch_point.boundary == boundary


def test_completed_capture_retry_does_not_resume_a_session_it_never_paused(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    sources, boundary = _sources(tmp_path, runtime)
    runtime.pause_session("helix-capture-session")
    with CoordinatedCaptureCoordinator(tmp_path / "capture.sqlite") as coordinator:
        coordinator.propose(_request(boundary))
        branch = coordinator.execute("capture-crash-recovery", sources)
        assert runtime.inspect_session("helix-capture-session").lifecycle is SessionLifecycle.PAUSED
        assert coordinator.execute("capture-crash-recovery", sources) == branch
        assert runtime.inspect_session("helix-capture-session").lifecycle is SessionLifecycle.PAUSED


def test_capture_rejects_mismatched_component_without_partial_publication(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    sources, boundary = _sources(tmp_path, runtime)
    bad_environment = sources.capture_environment(71).model_copy(update={"watermark": 1})
    mismatched = CaptureSources(
        model=sources.model,
        model_store=sources.model_store,
        lease=sources.lease,
        read_boundary=sources.read_boundary,
        capture_environment=lambda _seed: bad_environment,
        capture_effects=sources.capture_effects,
    )
    with CoordinatedCaptureCoordinator(
        tmp_path / "capture.sqlite", artifact_directory=tmp_path / "capture-artifacts"
    ) as coordinator:
        coordinator.propose(_request(boundary))
        with pytest.raises(CaptureBoundaryMismatch, match="environment capsule watermark"):
            coordinator.execute("capture-crash-recovery", mismatched)
        assert coordinator.status("capture-crash-recovery").phase is CapturePhase.FAILED
        assert coordinator.published_branch_point("capture-crash-recovery") is None
        assert runtime.inspect_session("helix-capture-session").lifecycle is SessionLifecycle.ACTIVE


def test_capture_abort_is_persisted_and_idempotent(tmp_path: Path) -> None:
    request = _request(
        CaptureBoundary(
            action_watermark=0,
            model_token_watermark=0,
            environment_event_watermark=0,
            effect_watermark=0,
        )
    ).model_copy(update={"capture_id": "capture-aborted"})
    with CoordinatedCaptureCoordinator(tmp_path / "capture.sqlite") as coordinator:
        coordinator.propose(request)
        aborted = coordinator.abort(request.capture_id, reason="operator cancelled branch search")
        assert aborted.phase is CapturePhase.ABORTED
        assert coordinator.abort(request.capture_id, reason="exact retry") == aborted
        assert coordinator.published_branch_point(request.capture_id) is None
        runtime = _runtime()
        sources, _boundary = _sources(tmp_path, runtime)
        with pytest.raises(CaptureAborted):
            coordinator.execute(request.capture_id, sources)


def test_capture_recovers_from_persisted_mid_capture_failure(tmp_path: Path) -> None:
    runtime = _runtime()
    sources, boundary = _sources(tmp_path, runtime)
    database = tmp_path / "capture.sqlite"
    artifact_directory = tmp_path / "capture-artifacts"
    request = _request(boundary)
    with CoordinatedCaptureCoordinator(
        database, artifact_directory=artifact_directory
    ) as coordinator:
        coordinator.propose(request)

        class SimulatedProcessCrash(BaseException):
            pass

        def fail_after_model(phase: CapturePhase) -> None:
            if phase is CapturePhase.MODEL_CAPTURED:
                raise SimulatedProcessCrash("simulated coordinator crash")

        with pytest.raises(SimulatedProcessCrash, match="simulated coordinator crash"):
            coordinator.execute(request.capture_id, sources, phase_hook=fail_after_model)
        assert coordinator.status(request.capture_id).phase is CapturePhase.MODEL_CAPTURED
        assert coordinator.published_branch_point(request.capture_id) is None
    with CoordinatedCaptureCoordinator(
        database, artifact_directory=artifact_directory
    ) as recovered:
        with pytest.raises(CaptureInProgress, match="already executing"):
            recovered.execute(request.capture_id, sources)
        branch_point = recovered.execute(request.capture_id, sources, recover=True)
        assert recovered.status(request.capture_id).attempt == 2
        assert recovered.status(request.capture_id).phase is CapturePhase.COMPLETED
        assert recovered.published_branch_point(request.capture_id) == branch_point
        assert runtime.inspect_session(request.session_id).lifecycle is SessionLifecycle.ACTIVE


def test_completed_capture_late_binds_to_canonical_branch_point(tmp_path: Path) -> None:
    runtime = _runtime()
    sources, boundary = _sources(tmp_path, runtime)
    transaction = load_learning_transaction(
        Path(__file__).parents[2] / "tests/fixtures/helix/learning-transaction-v1.json"
    )
    source = transaction.branch_group.branch_point
    with CoordinatedCaptureCoordinator(tmp_path / "capture.sqlite") as coordinator:
        coordinator.propose(_request(boundary))
        captured = coordinator.execute("capture-crash-recovery", sources)
    policy_key = f"{source.policy_epoch.policy_id}@{source.policy_epoch.epoch}"
    artifact_ids = (
        captured.source_trajectory_id,
        source.environment_state.capsule_id,
        policy_key,
    )
    lineage = tuple(
        LineageReference(
            artifact_id=artifact_id,
            artifact_kind="helix.test/bridge",
            relation=LineageRelation.DERIVED_FROM,
            digest=Digest(value=hashlib.sha256(artifact_id.encode()).hexdigest()),
        )
        for artifact_id in artifact_ids
    )
    canonical = build_ir_branch_point(
        captured,
        environment_state=source.environment_state,
        policy_epoch=source.policy_epoch,
        prefix_digest=Digest(value=hashlib.sha256(b"captured-prefix").hexdigest()),
        candidate_labels=("baseline", "alternate"),
        lineage=lineage,
    )
    assert canonical.branch_point_id == captured.branch_point_id
    assert canonical.event_index == boundary.action_watermark
    assert canonical.token_index == boundary.model_token_watermark


def test_strict_capture_requires_and_hashes_staged_artifact_bytes(tmp_path: Path) -> None:
    runtime = _runtime()
    sources, boundary = _sources(tmp_path, runtime)
    environment_payload = b"authenticated environment artifact"
    effect_payload = b"authenticated effect artifact"
    environment_reference = sources.capture_environment(71)
    effect_reference = sources.capture_effects()
    assert not isinstance(environment_reference, VerifiedCaptureArtifact)
    assert not isinstance(effect_reference, VerifiedCaptureArtifact)
    verified = CaptureSources(
        model=sources.model,
        model_store=sources.model_store,
        lease=sources.lease,
        read_boundary=sources.read_boundary,
        capture_environment=lambda _seed: VerifiedCaptureArtifact(
            reference=environment_reference.model_copy(
                update={"digest": hashlib.sha256(environment_payload).hexdigest()}
            ),
            payload=environment_payload,
        ),
        capture_effects=lambda: VerifiedCaptureArtifact(
            reference=effect_reference.model_copy(
                update={"digest": hashlib.sha256(effect_payload).hexdigest()}
            ),
            payload=effect_payload,
        ),
        expected_tenant_id="tenant-helix",
    )
    with CoordinatedCaptureCoordinator(
        tmp_path / "strict.sqlite",
        artifact_directory=tmp_path / "strict-artifacts",
        require_verified_artifacts=True,
    ) as coordinator:
        coordinator.propose(_request(boundary))
        branch = coordinator.execute("capture-crash-recovery", verified)
        assert branch.environment.digest == hashlib.sha256(environment_payload).hexdigest()


def test_strict_capture_rejects_digest_only_or_tampered_artifacts(tmp_path: Path) -> None:
    runtime = _runtime()
    sources, boundary = _sources(tmp_path, runtime)
    with CoordinatedCaptureCoordinator(
        tmp_path / "digest-only.sqlite",
        artifact_directory=tmp_path / "digest-only-artifacts",
        require_verified_artifacts=True,
    ) as coordinator:
        coordinator.propose(_request(boundary))
        with pytest.raises(CaptureArtifactIntegrityError, match="without authenticated bytes"):
            coordinator.execute("capture-crash-recovery", sources)

    runtime = _runtime()
    second = tmp_path / "second"
    second.mkdir()
    sources, boundary = _sources(second, runtime)
    environment_reference = sources.capture_environment(71)
    assert not isinstance(environment_reference, VerifiedCaptureArtifact)
    tampered = CaptureSources(
        model=sources.model,
        model_store=sources.model_store,
        lease=sources.lease,
        read_boundary=sources.read_boundary,
        capture_environment=lambda _seed: VerifiedCaptureArtifact(
            reference=environment_reference,
            payload=b"tampered",
        ),
        capture_effects=sources.capture_effects,
        expected_tenant_id="tenant-helix",
    )
    with CoordinatedCaptureCoordinator(
        tmp_path / "tampered.sqlite",
        artifact_directory=tmp_path / "tampered-artifacts",
        require_verified_artifacts=True,
    ) as coordinator:
        coordinator.propose(_request(boundary))
        with pytest.raises(CaptureArtifactIntegrityError, match="do not match"):
            coordinator.execute("capture-crash-recovery", tampered)
