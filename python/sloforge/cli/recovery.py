"""Diagnosis-driven, safe-by-default recovery commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from sloforge.autopsy import DiagnosisRecord
from sloforge.fabric.ir import (
    RecoveryCriterion,
    load_physical_execution_plan,
    load_recovery_plan,
    save_recovery_plan,
)
from sloforge.recovery import (
    DeterministicRecoveryExecutor,
    MetricObservation,
    RecoveryMachineConfig,
    RecoveryObservation,
    RecoveryPolicy,
    RecoveryState,
    plan_recovery,
)

from .common import json_result

recovery_app = typer.Typer(help="Plan and safely exercise diagnosis-driven recovery.")


def _passing_value(criterion: RecoveryCriterion) -> float:
    magnitude = max(abs(criterion.threshold) * 0.01, 1e-6)
    if criterion.comparator == "lt":
        return criterion.threshold - magnitude
    if criterion.comparator == "le":
        return criterion.threshold
    if criterion.comparator == "gt":
        return criterion.threshold + magnitude
    return criterion.threshold


@recovery_app.command("plan")
def plan_command(
    diagnosis: Annotated[Path, typer.Option("--diagnosis", exists=True, dir_okay=False)],
    physical_plan: Annotated[Path, typer.Option("--physical-plan", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    minimum_confidence: Annotated[
        float, typer.Option("--minimum-confidence", min=0.0, max=1.0)
    ] = 0.70,
    shadow_samples: Annotated[int, typer.Option("--shadow-samples", min=1)] = 20,
    canary_samples: Annotated[int, typer.Option("--canary-samples", min=1)] = 40,
) -> None:
    record = DiagnosisRecord.model_validate_json(diagnosis.read_text(encoding="utf-8"), strict=True)
    physical = load_physical_execution_plan(physical_plan)
    proposal = plan_recovery(
        record,
        physical,
        policy=RecoveryPolicy(
            minimum_diagnosis_confidence=minimum_confidence,
            minimum_shadow_samples=shadow_samples,
            minimum_canary_samples=canary_samples,
        ),
    )
    save_recovery_plan(output, proposal)
    json_result(
        {
            "output": str(output),
            "recovery_id": proposal.recovery_id,
            "actions": [item.kind.value for item in proposal.actions],
            "confidence": proposal.confidence,
            "external_mutation_authorized": proposal.external_mutation_authorized,
        }
    )


@recovery_app.command("apply")
def apply_command(
    proposal: Annotated[Path, typer.Option("--proposal", exists=True, dir_okay=False)],
    mode: Annotated[Literal["simulate", "shadow-canary"], typer.Option("--mode")] = (
        "shadow-canary"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/recovery/execution.json"
    ),
) -> None:
    """Exercise the guarded state machine with a non-mutating local driver."""

    plan = load_recovery_plan(proposal)
    if plan.external_mutation_authorized or any(
        action.requires_external_mutation for action in plan.actions
    ):
        raise typer.BadParameter(
            "the portable CLI never applies external mutations; use an explicitly authorized "
            "runtime adapter"
        )
    executor = DeterministicRecoveryExecutor(
        plan,
        now_ms=0,
        config=RecoveryMachineConfig(promotion_cooldown_ms=0),
    )
    metrics = tuple(
        MetricObservation(
            name=criterion.metric,
            value=_passing_value(criterion),
            window_seconds=criterion.window_seconds,
        )
        for criterion in plan.promotion_criteria
    )
    observations = [
        RecoveryObservation(
            observed_at_ms=1_000,
            idempotency_key="cli-simulation-validation",
            simulation_validated=True,
        ),
    ]
    if mode == "shadow-canary":
        observations.extend(
            (
                RecoveryObservation(
                    observed_at_ms=2_000,
                    idempotency_key="cli-build-start",
                ),
                RecoveryObservation(
                    observed_at_ms=3_000,
                    idempotency_key="cli-replacement-ready",
                    replacement_ready=True,
                ),
                RecoveryObservation(
                    observed_at_ms=4_000,
                    idempotency_key="cli-shadow",
                    shadow_samples=plan.traffic_migration.minimum_shadow_samples,
                    metrics=metrics,
                ),
                RecoveryObservation(
                    observed_at_ms=5_000,
                    idempotency_key="cli-canary",
                    canary_samples=plan.traffic_migration.minimum_canary_samples,
                    metrics=metrics,
                ),
                RecoveryObservation(
                    observed_at_ms=6_000,
                    idempotency_key="cli-promotion",
                    traffic_migration_complete=True,
                ),
                RecoveryObservation(
                    observed_at_ms=7_000,
                    idempotency_key="cli-drain-preserve",
                    active_started_streams=1,
                ),
                RecoveryObservation(
                    observed_at_ms=8_000,
                    idempotency_key="cli-drain-complete",
                    active_started_streams=0,
                ),
            )
        )
    for observation in observations:
        executor.tick(observation)
    if mode == "shadow-canary" and executor.snapshot.state is not RecoveryState.COMPLETED:
        raise RuntimeError(f"guarded recovery did not complete: {executor.snapshot.state.value}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(executor.dump_state())
    json_result(
        {
            "output": str(output),
            "recovery_id": plan.recovery_id,
            "mode": mode,
            "state": executor.snapshot.state.value,
            "actions_attempted": len(executor.snapshot.action_attempts),
            "audit_records": len(executor.snapshot.audit),
            "external_mutations": 0,
        }
    )
