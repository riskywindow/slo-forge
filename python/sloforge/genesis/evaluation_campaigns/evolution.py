"""Artifact-backed, multi-seed H7 continuous-evolution campaign.

Only the Genesis arm executes the proof-gated :mod:`sloforge.genesis.evolution`
controller.  The comparison arms are deterministic controller models whose time
unit is an explicitly synthetic logical tick.  Local generated-runtime replay is
preserved as separate gate evidence and is never described as hardware evidence.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from ..capsule import CapsuleValidationReport
from ..evolution import (
    CapsuleReference,
    ChallengerSpec,
    ChallengerStatus,
    EvolutionConfig,
    EvolutionController,
    EvolutionPhase,
    EvolutionSnapshot,
    EvolutionStore,
    EvolutionTrigger,
    ExecutionTarget,
    GateObservation,
    GateStage,
    IsolationContract,
    IsolationMode,
    TransitionCategory,
    TransitionCompatibility,
    TriggerObservation,
    collect_local_gate_evidence,
    local_gate_evidence_validator,
    validate_transition_compatibility,
)
from ..ir import canonical_json

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]

_MAXIMUM_SEEDS = 16
_HORIZON_TICKS = 30


class EvolutionCampaignValidationError(ValueError):
    """The campaign cannot be independently derived from its raw evidence."""


class AdaptationStrategy(StrEnum):
    NO_ADAPTATION = "no_adaptation"
    THRESHOLD_CONTROLLER = "threshold_controller"
    PHYSICAL_REPLAN_ONLY = "physical_replan_only"
    GENESIS_EVOLUTION = "genesis_evolution"


class EventSource(StrEnum):
    SYNTHETIC_COMPARATOR = "deterministic_synthetic_comparator"
    TRUSTED_CONTROLLER = "trusted_evolution_controller"
    LOCAL_RUNTIME = "sandboxed_local_runtime_replay"


class RuntimeEvidenceMode(StrEnum):
    SANDBOXED_GENERATED_RUNTIME = "sandboxed_generated_runtime"
    TEST_FIXTURE = "test_fixture"


class _CampaignModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class EvolutionEvent(_CampaignModel):
    schema_version: Literal["sloforge.genesis.h7-event/v1"] = "sloforge.genesis.h7-event/v1"
    seed: NonNegativeInt
    strategy: AdaptationStrategy
    sequence: NonNegativeInt
    logical_time_tick: NonNegativeInt
    action: NonEmpty
    source: EventSource
    phase: NonEmpty
    trigger: EvolutionTrigger | None = None
    capsule_id: str | None = None
    capsule_digest: str | None = None
    evidence_digest: str | None = None
    slo_violated: bool
    dropped_requests: NonNegativeInt = 0
    interrupted_streams: NonNegativeInt = 0
    adaptation_cost_units: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_digest_pair(self) -> Self:
        for name, digest in (
            ("capsule_digest", self.capsule_digest),
            ("evidence_digest", self.evidence_digest),
        ):
            if digest is not None and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase sha256 digest")
        if self.source is EventSource.LOCAL_RUNTIME and self.evidence_digest is None:
            raise ValueError("local runtime events must bind an evidence digest")
        return self


class StrategyRunMetrics(_CampaignModel):
    seed: NonNegativeInt
    strategy: AdaptationStrategy
    slo_restored: bool
    logical_time_to_slo_restoration_ticks: NonNegativeInt | None
    dropped_requests: NonNegativeInt
    interrupted_streams: NonNegativeInt
    adaptation_cost_units: NonNegativeInt
    rollback_count: NonNegativeInt
    invalid_challengers: NonNegativeInt
    local_runtime_request_observations: NonNegativeInt
    actual_hardware_experiments: Literal[0] = 0
    active_stream_preserved: bool | None

    @model_validator(mode="after")
    def restoration_time_matches_status(self) -> Self:
        if self.slo_restored != (self.logical_time_to_slo_restoration_ticks is not None):
            raise ValueError("restoration status and logical restoration time disagree")
        if self.strategy is AdaptationStrategy.GENESIS_EVOLUTION:
            if self.active_stream_preserved is None:
                raise ValueError("Genesis runs must report active-stream preservation")
        elif self.active_stream_preserved is not None:
            raise ValueError("comparator arms cannot claim an exercised stream transition")
        return self


class GenesisRunEvidence(_CampaignModel):
    seed: NonNegativeInt
    champion: CapsuleReference
    challenger: CapsuleReference
    candidate_id: NonEmpty
    runtime_evidence_mode: RuntimeEvidenceMode
    controller_snapshot_path: NonEmpty
    controller_snapshot_sha256: Sha256
    gate_evidence_root: NonEmpty
    shadow_observation: GateObservation
    canary_observation: GateObservation
    rolled_back: bool


class StrategyAggregate(_CampaignModel):
    strategy: AdaptationStrategy
    run_count: PositiveInt
    restoration_count: NonNegativeInt
    restoration_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    mean_logical_time_to_slo_restoration_ticks: float | None
    total_dropped_requests: NonNegativeInt
    total_interrupted_streams: NonNegativeInt
    mean_adaptation_cost_units: Annotated[float, Field(ge=0.0)]
    total_rollbacks: NonNegativeInt
    total_invalid_challengers: NonNegativeInt
    total_local_runtime_request_observations: NonNegativeInt
    actual_hardware_experiments: Literal[0] = 0


class EvolutionCampaignReport(_CampaignModel):
    schema_version: Literal["sloforge.genesis.h7-campaign/v1"] = "sloforge.genesis.h7-campaign/v1"
    hypothesis_id: Literal["H7"] = "H7"
    statement: Literal[
        "Continuous evolution adapts to workload and hardware drift better than static deployment."
    ] = "Continuous evolution adapts to workload and hardware drift better than static deployment."
    seeds: tuple[NonNegativeInt, ...]
    comparison_horizon_ticks: Literal[30] = 30
    workload_drift_injected: Literal[True] = True
    physical_degradation_injected: Literal[True] = True
    logical_time_scope: Literal["deterministic synthetic controller ticks; not host wall time"] = (
        "deterministic synthetic controller ticks; not host wall time"
    )
    local_runtime_scope: Literal[
        "sandboxed semantic and host-timing observations; not hardware-backed"
    ] = "sandboxed semantic and host-timing observations; not hardware-backed"
    actual_hardware_experiments: Literal[0] = 0
    raw_events_path: NonEmpty
    raw_events_sha256: Sha256
    run_metrics: tuple[StrategyRunMetrics, ...]
    genesis_evidence: tuple[GenesisRunEvidence, ...]
    aggregates: tuple[StrategyAggregate, ...]
    conclusion: Literal["artifact_backed_local_and_deterministic_synthetic_comparison"] = (
        "artifact_backed_local_and_deterministic_synthetic_comparison"
    )
    limitations: tuple[NonEmpty, ...]
    report_path: NonEmpty

    @model_validator(mode="after")
    def validate_complete_matrix(self) -> Self:
        if (
            not self.seeds
            or len(self.seeds) > _MAXIMUM_SEEDS
            or len(self.seeds) != len(set(self.seeds))
        ):
            raise ValueError("campaign seeds must be non-empty and unique")
        expected = tuple((seed, strategy) for seed in self.seeds for strategy in AdaptationStrategy)
        observed = tuple((row.seed, row.strategy) for row in self.run_metrics)
        if observed != expected:
            raise ValueError("campaign metrics do not form the canonical seed/strategy matrix")
        if tuple(item.strategy for item in self.aggregates) != tuple(AdaptationStrategy):
            raise ValueError("campaign aggregates must follow canonical strategy order")
        if tuple(item.seed for item in self.genesis_evidence) != self.seeds:
            raise ValueError("every seed must have one Genesis controller evidence record")
        return self


CapsuleValidator = Callable[[Path], CapsuleValidationReport]
GateCollector = Callable[..., GateObservation]
GateValidator = Callable[[GateObservation, ChallengerSpec, CapsuleReference], bool]
TransitionValidator = Callable[
    [CapsuleReference, CapsuleReference, TransitionCategory], TransitionCompatibility
]


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _synthetic_comparator_events(
    *, seed: int, strategy: AdaptationStrategy
) -> tuple[EvolutionEvent, ...]:
    if strategy is AdaptationStrategy.GENESIS_EVOLUTION:
        raise ValueError("Genesis events come from the trusted controller audit")
    jitter = seed % 3
    base = (
        EvolutionEvent(
            seed=seed,
            strategy=strategy,
            sequence=0,
            logical_time_tick=0,
            action="inject_workload_drift",
            source=EventSource.SYNTHETIC_COMPARATOR,
            phase="degraded",
            trigger=EvolutionTrigger.WORKLOAD_DRIFT,
            slo_violated=True,
        ),
        EvolutionEvent(
            seed=seed,
            strategy=strategy,
            sequence=1,
            logical_time_tick=2,
            action="inject_physical_degradation",
            source=EventSource.SYNTHETIC_COMPARATOR,
            phase="degraded",
            trigger=EvolutionTrigger.FABRIC_DEGRADATION,
            slo_violated=True,
        ),
    )
    if strategy is AdaptationStrategy.NO_ADAPTATION:
        tail = EvolutionEvent(
            seed=seed,
            strategy=strategy,
            sequence=2,
            logical_time_tick=_HORIZON_TICKS,
            action="evaluation_horizon_elapsed",
            source=EventSource.SYNTHETIC_COMPARATOR,
            phase="degraded",
            slo_violated=True,
            dropped_requests=10 + jitter,
            interrupted_streams=2,
        )
    elif strategy is AdaptationStrategy.THRESHOLD_CONTROLLER:
        tail = EvolutionEvent(
            seed=seed,
            strategy=strategy,
            sequence=2,
            logical_time_tick=18 + jitter,
            action="threshold_scale_and_restart_complete",
            source=EventSource.SYNTHETIC_COMPARATOR,
            phase="restored",
            slo_violated=False,
            dropped_requests=4 + jitter,
            interrupted_streams=1,
            adaptation_cost_units=4,
        )
    else:
        tail = EvolutionEvent(
            seed=seed,
            strategy=strategy,
            sequence=2,
            logical_time_tick=_HORIZON_TICKS,
            action="physical_replan_complete_workload_drift_unresolved",
            source=EventSource.SYNTHETIC_COMPARATOR,
            phase="partially_restored",
            slo_violated=True,
            dropped_requests=7 + jitter,
            interrupted_streams=1,
            adaptation_cost_units=3,
        )
    return (*base, tail)


def _metrics_from_events(
    seed: int, strategy: AdaptationStrategy, events: tuple[EvolutionEvent, ...]
) -> StrategyRunMetrics:
    restored = tuple(item for item in events if not item.slo_violated)
    restoration_tick = min((item.logical_time_tick for item in restored), default=None)
    return StrategyRunMetrics(
        seed=seed,
        strategy=strategy,
        slo_restored=restoration_tick is not None,
        logical_time_to_slo_restoration_ticks=restoration_tick,
        dropped_requests=sum(item.dropped_requests for item in events),
        interrupted_streams=sum(item.interrupted_streams for item in events),
        adaptation_cost_units=sum(item.adaptation_cost_units for item in events),
        rollback_count=sum(item.action == "rollback" for item in events),
        invalid_challengers=sum(item.action == "register_rejected_challenger" for item in events),
        local_runtime_request_observations=sum(
            item.adaptation_cost_units
            for item in events
            if item.source is EventSource.LOCAL_RUNTIME
        ),
        active_stream_preserved=(
            any(item.action == "active_stream_preserved" for item in events)
            if strategy is AdaptationStrategy.GENESIS_EVOLUTION
            else None
        ),
    )


def _controller_event(
    *,
    seed: int,
    sequence: int,
    logical_time_tick: int,
    action: str,
    phase: EvolutionPhase,
    trigger: EvolutionTrigger | None,
    capsule: CapsuleReference,
    slo_violated: bool,
    source: EventSource = EventSource.TRUSTED_CONTROLLER,
    evidence_digest: str | None = None,
    adaptation_cost_units: int = 0,
) -> EvolutionEvent:
    return EvolutionEvent(
        seed=seed,
        strategy=AdaptationStrategy.GENESIS_EVOLUTION,
        sequence=sequence,
        logical_time_tick=logical_time_tick,
        action=action,
        source=source,
        phase=phase.value,
        trigger=trigger,
        capsule_id=capsule.capsule_id,
        capsule_digest=capsule.capsule_digest,
        evidence_digest=evidence_digest,
        slo_violated=slo_violated,
        adaptation_cost_units=adaptation_cost_units,
    )


def _expected_genesis_events(
    evidence: GenesisRunEvidence,
) -> tuple[EvolutionEvent, ...]:
    seed = evidence.seed
    rejected_reference = CapsuleReference(
        capsule_id=f"h7-rejected-capsule-{seed}",
        capsule_digest=hashlib.sha256(f"h7-rejected-reference:{seed}".encode()).hexdigest(),
        genome_hash=evidence.challenger.genome_hash,
        path=evidence.challenger.path,
    )
    events = [
        _controller_event(
            seed=seed,
            sequence=0,
            logical_time_tick=0,
            action="inject_workload_drift",
            phase=EvolutionPhase.EVOLVING,
            trigger=EvolutionTrigger.WORKLOAD_DRIFT,
            capsule=evidence.champion,
            slo_violated=True,
        ),
        _controller_event(
            seed=seed,
            sequence=1,
            logical_time_tick=2,
            action="inject_physical_degradation",
            phase=EvolutionPhase.EVOLVING,
            trigger=EvolutionTrigger.FABRIC_DEGRADATION,
            capsule=evidence.champion,
            slo_violated=True,
        ),
        _controller_event(
            seed=seed,
            sequence=2,
            logical_time_tick=3,
            action="register_rejected_challenger",
            phase=EvolutionPhase.EVOLVING,
            trigger=EvolutionTrigger.WORKLOAD_DRIFT,
            capsule=rejected_reference,
            slo_violated=True,
            adaptation_cost_units=1,
        ),
        _controller_event(
            seed=seed,
            sequence=3,
            logical_time_tick=4,
            action="shadow_runtime_replay",
            phase=EvolutionPhase.SHADOW_VALIDATED,
            trigger=EvolutionTrigger.WORKLOAD_DRIFT,
            capsule=evidence.challenger,
            evidence_digest=evidence.shadow_observation.evidence_digest,
            source=EventSource.LOCAL_RUNTIME,
            slo_violated=True,
            adaptation_cost_units=evidence.shadow_observation.sample_count,
        ),
        _controller_event(
            seed=seed,
            sequence=4,
            logical_time_tick=6,
            action="canary_runtime_replay",
            phase=EvolutionPhase.READY_TO_PROMOTE,
            trigger=EvolutionTrigger.WORKLOAD_DRIFT,
            capsule=evidence.challenger,
            evidence_digest=evidence.canary_observation.evidence_digest,
            source=EventSource.LOCAL_RUNTIME,
            slo_violated=True,
            adaptation_cost_units=evidence.canary_observation.sample_count,
        ),
        _controller_event(
            seed=seed,
            sequence=5,
            logical_time_tick=8,
            action="promote",
            phase=EvolutionPhase.PROMOTED,
            trigger=EvolutionTrigger.WORKLOAD_DRIFT,
            capsule=evidence.challenger,
            slo_violated=evidence.rolled_back,
            adaptation_cost_units=2,
        ),
    ]
    if evidence.rolled_back:
        events.append(
            _controller_event(
                seed=seed,
                sequence=6,
                logical_time_tick=10,
                action="rollback",
                phase=EvolutionPhase.ROLLED_BACK,
                trigger=EvolutionTrigger.PERFORMANCE_REGRESSION,
                capsule=evidence.champion,
                slo_violated=False,
                adaptation_cost_units=1,
            )
        )
    events.append(
        _controller_event(
            seed=seed,
            sequence=len(events),
            logical_time_tick=10 if evidence.rolled_back else 8,
            action="active_stream_preserved",
            phase=(EvolutionPhase.ROLLED_BACK if evidence.rolled_back else EvolutionPhase.PROMOTED),
            trigger=(
                EvolutionTrigger.PERFORMANCE_REGRESSION
                if evidence.rolled_back
                else EvolutionTrigger.WORKLOAD_DRIFT
            ),
            capsule=evidence.champion,
            slo_violated=False,
        )
    )
    return tuple(events)


def _genesis_run(
    *,
    output: Path,
    seed: int,
    champion: CapsuleReference,
    challenger: CapsuleReference,
    capsule_validator: CapsuleValidator,
    runtime_evidence_mode: RuntimeEvidenceMode,
    gate_collector: GateCollector,
    gate_validator: GateValidator,
    transition_validator: TransitionValidator,
    rollback: bool,
) -> tuple[tuple[EvolutionEvent, ...], StrategyRunMetrics, GenesisRunEvidence]:
    output.mkdir(parents=True)
    gates = output / "runtime-gates"
    isolation = output / "challenger-isolation"
    isolation.mkdir()
    spec = ChallengerSpec(
        candidate_id=f"h7-challenger-{seed}",
        capsule=challenger,
        transition_category=TransitionCategory.REQUEST_BOUNDARY_SWAP,
        isolation=IsolationContract(
            mode=IsolationMode.SANDBOX,
            writable_artifact_root=str(isolation.resolve()),
        ),
        active_stream_compatible=False,
    )
    rejected_reference = CapsuleReference(
        capsule_id=f"h7-rejected-capsule-{seed}",
        capsule_digest=hashlib.sha256(f"h7-rejected-reference:{seed}".encode()).hexdigest(),
        genome_hash=challenger.genome_hash,
        path=challenger.path,
    )
    rejected_spec = ChallengerSpec(
        candidate_id=f"h7-rejected-challenger-{seed}",
        capsule=rejected_reference,
        transition_category=TransitionCategory.REQUEST_BOUNDARY_SWAP,
        isolation=spec.isolation,
        active_stream_compatible=False,
    )
    controller = EvolutionController.initialize(
        store=EvolutionStore(output / "controller-state.json"),
        capsule_validator=capsule_validator,
        gate_evidence_validator=gate_validator,
        transition_compatibility_validator=transition_validator,
        config=EvolutionConfig(
            execution_target=ExecutionTarget.LOCAL,
            minimum_shadow_samples=2,
            minimum_canary_samples=3,
            maximum_p95_ttft_ratio=10.0,
            maximum_p99_tpot_ratio=10.0,
        ),
        deployment_id=f"genesis-h7-{seed}",
        champion=champion,
        seed=seed,
        now_ms=0,
    )
    controller.open_stream(
        f"h7-active-stream-{seed}",
        event_id=f"h7-{seed}-open-stream",
        observed_at_ms=1,
        externally_visible_output=True,
    )
    controller.observe_trigger(
        TriggerObservation(
            event_id=f"h7-{seed}-workload-drift",
            observed_at_ms=10,
            workload_js_divergence=0.40,
            detail="deterministic H7 bimodal workload drift",
        )
    )
    controller.register_challenger(
        rejected_spec,
        event_id=f"h7-{seed}-register-rejected",
        observed_at_ms=16,
    )
    controller.register_challenger(
        spec,
        event_id=f"h7-{seed}-register",
        observed_at_ms=20,
    )
    controller.observe_trigger(
        TriggerObservation(
            event_id=f"h7-{seed}-fabric-degradation",
            observed_at_ms=22,
            fabric_health_ratio=0.70,
            detail="deterministic H7 physical fabric degradation",
        )
    )
    controller.begin_shadow(event_id=f"h7-{seed}-shadow", observed_at_ms=30)
    shadow = gate_collector(
        champion=champion,
        challenger=challenger,
        candidate_id=spec.candidate_id,
        stage=GateStage.SHADOW,
        sample_count=controller.config.minimum_shadow_samples,
        seed=seed,
        observed_at_ms=40,
        output_directory=gates / "shadow",
    )
    controller.record_gate(shadow)
    controller.begin_canary(event_id=f"h7-{seed}-canary", observed_at_ms=50)
    canary = gate_collector(
        champion=champion,
        challenger=challenger,
        candidate_id=spec.candidate_id,
        stage=GateStage.CANARY,
        sample_count=controller.config.minimum_canary_samples,
        seed=seed,
        observed_at_ms=60,
        output_directory=gates / "canary",
    )
    controller.record_gate(canary)
    promoted = controller.promote(event_id=f"h7-{seed}-promote", observed_at_ms=70)
    final = (
        controller.rollback(
            event_id=f"h7-{seed}-rollback",
            observed_at_ms=80,
            reason="deterministic post-promotion regression triggered proof-gated rollback",
        )
        if rollback
        else promoted
    )
    lease = next(
        item for item in final.active_streams if item.stream_id == f"h7-active-stream-{seed}"
    )
    active_stream_preserved = lease.capsule_id == champion.capsule_id
    snapshot_path = output / "final-controller-snapshot.json"
    snapshot_payload = canonical_json(final) + b"\n"
    _write_once(snapshot_path, snapshot_payload)

    if not active_stream_preserved:
        raise RuntimeError("proof-gated promotion interrupted an active stream lease")
    evidence = GenesisRunEvidence(
        seed=seed,
        champion=champion,
        challenger=challenger,
        candidate_id=spec.candidate_id,
        runtime_evidence_mode=runtime_evidence_mode,
        controller_snapshot_path=str(snapshot_path.resolve()),
        controller_snapshot_sha256=_digest(snapshot_payload),
        gate_evidence_root=str(gates.resolve()),
        shadow_observation=shadow,
        canary_observation=canary,
        rolled_back=rollback,
    )
    event_tuple = _expected_genesis_events(evidence)
    metrics = _metrics_from_events(seed, AdaptationStrategy.GENESIS_EVOLUTION, event_tuple)
    return event_tuple, metrics, evidence


def _aggregate(rows: tuple[StrategyRunMetrics, ...]) -> tuple[StrategyAggregate, ...]:
    aggregates: list[StrategyAggregate] = []
    for strategy in AdaptationStrategy:
        selected = tuple(row for row in rows if row.strategy is strategy)
        restored = tuple(
            row.logical_time_to_slo_restoration_ticks
            for row in selected
            if row.logical_time_to_slo_restoration_ticks is not None
        )
        aggregates.append(
            StrategyAggregate(
                strategy=strategy,
                run_count=len(selected),
                restoration_count=len(restored),
                restoration_rate=len(restored) / len(selected),
                mean_logical_time_to_slo_restoration_ticks=(
                    statistics.fmean(restored) if restored else None
                ),
                total_dropped_requests=sum(row.dropped_requests for row in selected),
                total_interrupted_streams=sum(row.interrupted_streams for row in selected),
                mean_adaptation_cost_units=statistics.fmean(
                    row.adaptation_cost_units for row in selected
                ),
                total_rollbacks=sum(row.rollback_count for row in selected),
                total_invalid_challengers=sum(row.invalid_challengers for row in selected),
                total_local_runtime_request_observations=sum(
                    row.local_runtime_request_observations for row in selected
                ),
            )
        )
    return tuple(aggregates)


def run_evolution_campaign(
    output: Path,
    *,
    champion: CapsuleReference,
    challenger: CapsuleReference,
    capsule_validator: CapsuleValidator,
    seeds: tuple[int, ...] = (73129, 73130, 73131),
    rollback_seeds: tuple[int, ...] = (73131,),
    runtime_evidence_mode: RuntimeEvidenceMode = RuntimeEvidenceMode.SANDBOXED_GENERATED_RUNTIME,
    gate_collector: GateCollector = collect_local_gate_evidence,
    gate_validator_factory: Callable[[Path], GateValidator] = local_gate_evidence_validator,
    transition_validator: TransitionValidator = validate_transition_compatibility,
) -> EvolutionCampaignReport:
    """Run H7 without live traffic or hardware mutation.

    ``runtime_evidence_mode=TEST_FIXTURE`` exists only for focused unit tests.  A
    publishable campaign uses the default sandbox collector and validator.
    """

    if not seeds or len(seeds) > _MAXIMUM_SEEDS or len(seeds) != len(set(seeds)):
        raise ValueError(f"seeds must contain 1..{_MAXIMUM_SEEDS} unique values")
    if any(seed < 0 for seed in seeds) or not set(rollback_seeds).issubset(seeds):
        raise ValueError("seeds must be non-negative and rollback_seeds must be a subset")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite H7 campaign: {output}")
    output.mkdir(parents=True)

    events: list[EvolutionEvent] = []
    metrics: list[StrategyRunMetrics] = []
    evidence: list[GenesisRunEvidence] = []
    for seed in seeds:
        for strategy in AdaptationStrategy:
            if strategy is AdaptationStrategy.GENESIS_EVOLUTION:
                seed_root = output / "genesis-runs" / str(seed)
                genesis_events, genesis_metrics, genesis_evidence = _genesis_run(
                    output=seed_root,
                    seed=seed,
                    champion=champion,
                    challenger=challenger,
                    capsule_validator=capsule_validator,
                    runtime_evidence_mode=runtime_evidence_mode,
                    gate_collector=gate_collector,
                    gate_validator=gate_validator_factory(seed_root / "runtime-gates"),
                    transition_validator=transition_validator,
                    rollback=seed in rollback_seeds,
                )
                events.extend(genesis_events)
                metrics.append(genesis_metrics)
                evidence.append(genesis_evidence)
            else:
                comparator_events = _synthetic_comparator_events(seed=seed, strategy=strategy)
                events.extend(comparator_events)
                metrics.append(_metrics_from_events(seed, strategy, comparator_events))

    events_path = output / "raw-events.jsonl"
    events_payload = b"".join(canonical_json(item) + b"\n" for item in events)
    _write_once(events_path, events_payload)
    report_path = output / "report.json"
    report = EvolutionCampaignReport(
        seeds=seeds,
        raw_events_path=str(events_path.resolve()),
        raw_events_sha256=_digest(events_payload),
        run_metrics=tuple(metrics),
        genesis_evidence=tuple(evidence),
        aggregates=_aggregate(tuple(metrics)),
        limitations=(
            "Comparator performance and restoration times are deterministic synthetic model outputs, not measured service latency.",
            "Generated-runtime gates execute on the local CPU sandbox and carry host timing only; actual GPU, network, and multi-node experiments are zero.",
            "The physical degradation is an injected controller observation; no host link or device is modified.",
            "No live traffic is enabled and external promotion remains disabled.",
        ),
        report_path=str(report_path.resolve()),
    )
    _write_once(report_path, canonical_json(report) + b"\n")
    return report


def _contained_path(path_value: str, root: Path, *, directory: bool = False) -> Path:
    path = Path(path_value)
    if path.is_symlink():
        raise EvolutionCampaignValidationError("campaign evidence must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EvolutionCampaignValidationError(
            "campaign evidence escapes its artifact root"
        ) from exc
    if directory != resolved.is_dir() or (not directory and not resolved.is_file()):
        raise EvolutionCampaignValidationError("campaign evidence has the wrong file type")
    return resolved


def _read_events(
    report: EvolutionCampaignReport, campaign_root: Path
) -> tuple[EvolutionEvent, ...]:
    path = _contained_path(report.raw_events_path, campaign_root)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise EvolutionCampaignValidationError("raw event trace is not a bounded regular file")
    payload = path.read_bytes()
    if _digest(payload) != report.raw_events_sha256:
        raise EvolutionCampaignValidationError("raw event trace digest changed")
    try:
        return tuple(
            EvolutionEvent.model_validate_json(line, strict=True)
            for line in payload.splitlines()
            if line.strip()
        )
    except ValueError as exc:
        raise EvolutionCampaignValidationError("raw event trace is invalid") from exc


def _validate_genesis_evidence(
    evidence: GenesisRunEvidence,
    events: tuple[EvolutionEvent, ...],
    *,
    capsule_validator: CapsuleValidator,
    transition_validator: TransitionValidator,
    campaign_root: Path,
    gate_validator_factory: Callable[[Path], GateValidator] = local_gate_evidence_validator,
    allow_test_fixture: bool = False,
) -> None:
    snapshot_path = _contained_path(evidence.controller_snapshot_path, campaign_root)
    if snapshot_path.stat().st_size > 8 * 1024 * 1024:
        raise EvolutionCampaignValidationError("controller snapshot is not a bounded regular file")
    snapshot_payload = snapshot_path.read_bytes()
    if _digest(snapshot_payload) != evidence.controller_snapshot_sha256:
        raise EvolutionCampaignValidationError("controller snapshot digest changed")
    try:
        snapshot = EvolutionSnapshot.model_validate_json(snapshot_payload, strict=True)
    except ValueError as exc:
        raise EvolutionCampaignValidationError("controller snapshot is invalid") from exc
    expected_phase = EvolutionPhase.ROLLED_BACK if evidence.rolled_back else EvolutionPhase.PROMOTED
    expected_champion = evidence.champion if evidence.rolled_back else evidence.challenger
    if (
        snapshot.seed != evidence.seed
        or snapshot.phase is not expected_phase
        or snapshot.champion != expected_champion
        or snapshot.active_trigger is not EvolutionTrigger.WORKLOAD_DRIFT
        or not any(item.action == "coalesce_trigger" for item in snapshot.audit)
        or not any(item.action == "promote" for item in snapshot.audit)
        or (evidence.rolled_back != any(item.action == "rollback" for item in snapshot.audit))
    ):
        raise EvolutionCampaignValidationError("controller snapshot contradicts the H7 flow")
    leases = tuple(
        item for item in snapshot.active_streams if item.capsule_id == evidence.champion.capsule_id
    )
    if len(leases) != 1 or not leases[0].externally_visible_output:
        raise EvolutionCampaignValidationError(
            "active stream was not preserved on its original capsule"
        )
    record = next(
        (item for item in snapshot.challengers if item.spec.candidate_id == evidence.candidate_id),
        None,
    )
    if (
        record is None
        or record.shadow_observation != evidence.shadow_observation
        or record.canary_observation != evidence.canary_observation
    ):
        raise EvolutionCampaignValidationError("persisted gates differ from the controller record")
    expected_status = (
        ChallengerStatus.ROLLED_BACK if evidence.rolled_back else ChallengerStatus.PROMOTED
    )
    if record.status is not expected_status:
        raise EvolutionCampaignValidationError("challenger terminal status is inconsistent")
    rejected_record = next(
        (
            item
            for item in snapshot.challengers
            if item.spec.candidate_id == f"h7-rejected-challenger-{evidence.seed}"
        ),
        None,
    )
    if (
        rejected_record is None
        or rejected_record.status is not ChallengerStatus.CAPSULE_REJECTED
        or rejected_record.rejection_reason != "independent capsule validation rejected promotion"
    ):
        raise EvolutionCampaignValidationError(
            "the invalid challenger was not independently rejected"
        )
    for reference in (evidence.champion, evidence.challenger):
        report = capsule_validator(Path(reference.path))
        if (
            not report.local_evolution_eligible
            or report.capsule_digest is None
            or report.capsule_digest.value != reference.capsule_digest
            or report.candidate_genome_hash is None
            or report.candidate_genome_hash.value != reference.genome_hash
        ):
            raise EvolutionCampaignValidationError("capsule failed independent revalidation")
    compatibility = transition_validator(
        evidence.champion,
        evidence.challenger,
        TransitionCategory.REQUEST_BOUNDARY_SWAP,
    )
    if not compatibility.compatible:
        raise EvolutionCampaignValidationError("active-stream transition is incompatible")
    spec = record.spec
    gate_root = _contained_path(evidence.gate_evidence_root, campaign_root, directory=True)
    gate_validator = gate_validator_factory(gate_root)
    if (
        evidence.runtime_evidence_mode is RuntimeEvidenceMode.TEST_FIXTURE
        and not allow_test_fixture
    ):
        raise EvolutionCampaignValidationError("test-fixture runtime evidence is not publishable")
    for observation in (evidence.shadow_observation, evidence.canary_observation):
        if not gate_validator(observation, spec, evidence.champion):
            raise EvolutionCampaignValidationError("local runtime evidence failed replay")
    if not any(item.action == "active_stream_preserved" for item in events):
        raise EvolutionCampaignValidationError("Genesis raw events omit stream preservation")


def validate_evolution_campaign(
    report_path: Path,
    *,
    capsule_validator: CapsuleValidator,
    transition_validator: TransitionValidator = validate_transition_compatibility,
    allow_test_fixture: bool = False,
    test_gate_validator_factory: Callable[[Path], GateValidator] | None = None,
) -> EvolutionCampaignReport:
    """Recompute H7 metrics and replay every publishable local runtime gate."""

    if (
        report_path.is_symlink()
        or not report_path.is_file()
        or report_path.stat().st_size > 8 * 1024 * 1024
    ):
        raise EvolutionCampaignValidationError("campaign report is not a bounded regular file")
    try:
        report = EvolutionCampaignReport.model_validate_json(report_path.read_bytes(), strict=True)
    except ValueError as exc:
        raise EvolutionCampaignValidationError("campaign report schema validation failed") from exc
    if Path(report.report_path).resolve() != report_path.resolve():
        raise EvolutionCampaignValidationError("campaign report path is not self-consistent")
    campaign_root = report_path.resolve().parent
    events = _read_events(report, campaign_root)
    evidence_by_seed = {item.seed: item for item in report.genesis_evidence}
    grouped: dict[tuple[int, AdaptationStrategy], tuple[EvolutionEvent, ...]] = {}
    for seed in report.seeds:
        for strategy in AdaptationStrategy:
            selected = tuple(
                item for item in events if item.seed == seed and item.strategy is strategy
            )
            if not selected or tuple(item.sequence for item in selected) != tuple(
                range(len(selected))
            ):
                raise EvolutionCampaignValidationError(
                    "event sequence is incomplete or non-canonical"
                )
            grouped[(seed, strategy)] = selected
            if (
                strategy is not AdaptationStrategy.GENESIS_EVOLUTION
                and selected != _synthetic_comparator_events(seed=seed, strategy=strategy)
            ):
                raise EvolutionCampaignValidationError(
                    "synthetic comparator trace is not reproducible"
                )
            if (
                strategy is AdaptationStrategy.GENESIS_EVOLUTION
                and selected != _expected_genesis_events(evidence_by_seed[seed])
            ):
                raise EvolutionCampaignValidationError(
                    "Genesis event trace does not derive from controller evidence"
                )
    if sum(len(value) for value in grouped.values()) != len(events):
        raise EvolutionCampaignValidationError("raw trace contains an out-of-matrix event")
    recomputed_metrics = tuple(
        _metrics_from_events(seed, strategy, grouped[(seed, strategy)])
        for seed in report.seeds
        for strategy in AdaptationStrategy
    )
    if (
        recomputed_metrics != report.run_metrics
        or _aggregate(recomputed_metrics) != report.aggregates
    ):
        raise EvolutionCampaignValidationError("campaign metrics do not recompute from raw events")
    for evidence in report.genesis_evidence:
        if (
            evidence.runtime_evidence_mode is RuntimeEvidenceMode.TEST_FIXTURE
            and allow_test_fixture
            and test_gate_validator_factory is None
        ):
            raise EvolutionCampaignValidationError(
                "test fixture validation requires an explicit gate validator"
            )
        _validate_genesis_evidence(
            evidence,
            grouped[(evidence.seed, AdaptationStrategy.GENESIS_EVOLUTION)],
            capsule_validator=capsule_validator,
            transition_validator=transition_validator,
            campaign_root=campaign_root,
            gate_validator_factory=(
                test_gate_validator_factory
                if evidence.runtime_evidence_mode is RuntimeEvidenceMode.TEST_FIXTURE
                and allow_test_fixture
                and test_gate_validator_factory is not None
                else local_gate_evidence_validator
            ),
            allow_test_fixture=allow_test_fixture,
        )
    return report


__all__ = [
    "AdaptationStrategy",
    "EvolutionCampaignReport",
    "EvolutionCampaignValidationError",
    "EvolutionEvent",
    "GenesisRunEvidence",
    "RuntimeEvidenceMode",
    "StrategyAggregate",
    "StrategyRunMetrics",
    "run_evolution_campaign",
    "validate_evolution_campaign",
]
