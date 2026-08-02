"""Deterministic local champion/challenger evolution fixture."""

from __future__ import annotations

from .controller import EvolutionController
from .models import (
    ChallengerSpec,
    EvolutionSnapshot,
    GateObservation,
    GateStage,
    TriggerObservation,
)


def run_local_evolution_fixture(
    controller: EvolutionController,
    challenger: ChallengerSpec,
    *,
    start_at_ms: int = 10,
) -> EvolutionSnapshot:
    """Exercise drift, isolation, gates, and promotion using artifact-backed validation."""

    seed = controller.snapshot.seed
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
    controller.record_gate(
        GateObservation(
            event_id="fixture-shadow-evidence",
            stage=GateStage.SHADOW,
            observed_at_ms=start_at_ms + 30,
            deterministic_seed=seed,
            sample_count=controller.config.minimum_shadow_samples,
            error_rate=0.0,
            p95_ttft_ratio=1.0,
            p99_tpot_ratio=1.0,
            quality_regression=0.0,
            interrupted_streams=0,
        )
    )
    controller.begin_canary(event_id="fixture-begin-canary", observed_at_ms=start_at_ms + 40)
    controller.record_gate(
        GateObservation(
            event_id="fixture-canary-evidence",
            stage=GateStage.CANARY,
            observed_at_ms=start_at_ms + 50,
            deterministic_seed=seed,
            sample_count=controller.config.minimum_canary_samples,
            error_rate=0.0,
            p95_ttft_ratio=1.0,
            p99_tpot_ratio=1.0,
            quality_regression=0.0,
            interrupted_streams=0,
        )
    )
    return controller.promote(event_id="fixture-promote", observed_at_ms=start_at_ms + 60)
