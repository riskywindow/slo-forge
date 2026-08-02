"""Deterministic local champion/challenger evolution fixture."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..capsule.models import VerificationLevel
from .controller import EvolutionController
from .evidence import collect_local_gate_evidence
from .models import (
    ChallengerSpec,
    EvolutionSnapshot,
    ExecutionTarget,
    GateObservation,
    GateStage,
    TriggerObservation,
)


def run_local_evolution_fixture(
    controller: EvolutionController,
    challenger: ChallengerSpec,
    *,
    start_at_ms: int = 10,
    evidence_directory: Path | None = None,
) -> EvolutionSnapshot:
    """Exercise evolution; real capsule replay is required when an evidence path is supplied."""

    seed = controller.snapshot.seed
    if (
        evidence_directory is None
        and controller.config.execution_target is not ExecutionTarget.SIMULATED
    ):
        raise ValueError("local or external evolution requires artifact-backed gate evidence")

    def gate_digest(stage: GateStage) -> str:
        payload = (
            f"{seed}:{challenger.candidate_id}:{challenger.capsule.capsule_digest}:{stage.value}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    controller.observe_trigger(
        TriggerObservation(
            event_id="fixture-workload-drift",
            observed_at_ms=start_at_ms,
            workload_js_divergence=0.40,
            detail="fixture prompt distribution moved from short to bimodal",
        )
    )
    controller.register_challenger(
        challenger,
        event_id="fixture-register-challenger",
        observed_at_ms=start_at_ms + 10,
    )
    controller.begin_shadow(event_id="fixture-begin-shadow", observed_at_ms=start_at_ms + 20)
    if evidence_directory is None:
        shadow_observation = GateObservation(
            event_id="fixture-shadow-evidence",
            candidate_id=challenger.candidate_id,
            capsule_digest=challenger.capsule.capsule_digest,
            evidence_digest=gate_digest(GateStage.SHADOW),
            stage=GateStage.SHADOW,
            verification_level=VerificationLevel.PROPERTY,
            observed_at_ms=start_at_ms + 30,
            deterministic_seed=seed,
            sample_count=controller.config.minimum_shadow_samples,
            error_rate=0.0,
            p95_ttft_ratio=1.0,
            p99_tpot_ratio=1.0,
            quality_regression=0.0,
            interrupted_streams=0,
        )
    else:
        shadow_observation = collect_local_gate_evidence(
            champion=controller.snapshot.champion,
            challenger=challenger.capsule,
            candidate_id=challenger.candidate_id,
            stage=GateStage.SHADOW,
            sample_count=controller.config.minimum_shadow_samples,
            seed=seed,
            observed_at_ms=start_at_ms + 30,
            output_directory=evidence_directory / "shadow",
        )
    controller.record_gate(shadow_observation)
    controller.begin_canary(event_id="fixture-begin-canary", observed_at_ms=start_at_ms + 40)
    if evidence_directory is None:
        canary_observation = GateObservation(
            event_id="fixture-canary-evidence",
            candidate_id=challenger.candidate_id,
            capsule_digest=challenger.capsule.capsule_digest,
            evidence_digest=gate_digest(GateStage.CANARY),
            stage=GateStage.CANARY,
            verification_level=VerificationLevel.PROPERTY,
            observed_at_ms=start_at_ms + 50,
            deterministic_seed=seed,
            sample_count=controller.config.minimum_canary_samples,
            error_rate=0.0,
            p95_ttft_ratio=1.0,
            p99_tpot_ratio=1.0,
            quality_regression=0.0,
            interrupted_streams=0,
        )
    else:
        canary_observation = collect_local_gate_evidence(
            champion=controller.snapshot.champion,
            challenger=challenger.capsule,
            candidate_id=challenger.candidate_id,
            stage=GateStage.CANARY,
            sample_count=controller.config.minimum_canary_samples,
            seed=seed,
            observed_at_ms=start_at_ms + 50,
            output_directory=evidence_directory / "canary",
        )
    controller.record_gate(canary_observation)
    return controller.promote(event_id="fixture-promote", observed_at_ms=start_at_ms + 60)
