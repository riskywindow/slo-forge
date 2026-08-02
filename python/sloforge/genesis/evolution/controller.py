"""Deterministic proof-gated champion/challenger evolution controller."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import BaseModel

from ..capsule.models import CapsuleValidationReport, VerificationLevel
from ..ir import canonical_json
from .models import (
    CapsuleReference,
    ChallengerRecord,
    ChallengerSpec,
    ChallengerStatus,
    EvolutionAuditRecord,
    EvolutionConfig,
    EvolutionPhase,
    EvolutionSnapshot,
    EvolutionTrigger,
    ExecutionTarget,
    GateObservation,
    GateStage,
    ProcessedEventRecord,
    StreamLease,
    TransitionCategory,
    TriggerObservation,
)
from .store import EvolutionStore
from .transition import TransitionCompatibility, validate_transition_compatibility

CapsuleValidator = Callable[[Path], CapsuleValidationReport]
GateEvidenceValidator = Callable[[GateObservation, ChallengerSpec, CapsuleReference], bool]
TransitionCompatibilityValidator = Callable[
    [CapsuleReference, CapsuleReference, TransitionCategory], TransitionCompatibility
]
_LIVE_PROMOTION_ENV = "SLOFORGE_GENESIS_ALLOW_LIVE_PROMOTION"


class EvolutionError(RuntimeError):
    """Raised when a state transition violates an evolution safety guard."""


def _jsonable_event_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable_event_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_event_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _event_digest(action: str, payload: object) -> str:
    return hashlib.sha256(
        canonical_json({"action": action, "payload": _jsonable_event_value(payload)})
    ).hexdigest()


def _legacy_event_digest(event_id: str) -> str:
    return hashlib.sha256(f"legacy-event\0{event_id}".encode()).hexdigest()


def _serialized(
    method: Callable[..., EvolutionSnapshot],
) -> Callable[..., EvolutionSnapshot]:
    @wraps(method)
    def wrapper(self: EvolutionController, *args: Any, **kwargs: Any) -> EvolutionSnapshot:
        with self._mutation_lock:
            return method(self, *args, **kwargs)

    return wrapper


def classify_trigger(
    observation: TriggerObservation, config: EvolutionConfig
) -> EvolutionTrigger | None:
    """Classify one observation using a fixed, deterministic priority order."""

    if observation.model_updated:
        return EvolutionTrigger.MODEL_UPDATE
    if observation.hardware_changed:
        return EvolutionTrigger.HARDWARE_CHANGE
    if observation.dependency_updated:
        return EvolutionTrigger.DEPENDENCY_UPDATE
    if observation.slo_changed:
        return EvolutionTrigger.SLO_CHANGE
    if observation.cost_changed:
        return EvolutionTrigger.COST_CHANGE
    if observation.autopsy_bottleneck_changed:
        return EvolutionTrigger.AUTOPSY_BOTTLENECK_CHANGE
    if observation.fabric_health_ratio < config.fabric_health_floor:
        return EvolutionTrigger.FABRIC_DEGRADATION
    if observation.performance_regression_ratio >= config.performance_regression_threshold:
        return EvolutionTrigger.PERFORMANCE_REGRESSION
    if observation.workload_js_divergence >= config.workload_drift_threshold:
        return EvolutionTrigger.WORKLOAD_DRIFT
    return None


class EvolutionController:
    """Persist every safe transition before exposing its resulting snapshot."""

    def __init__(
        self,
        *,
        store: EvolutionStore,
        capsule_validator: CapsuleValidator,
        gate_evidence_validator: GateEvidenceValidator | None,
        transition_compatibility_validator: TransitionCompatibilityValidator,
        config: EvolutionConfig,
        snapshot: EvolutionSnapshot,
    ) -> None:
        self.store = store
        self.capsule_validator = capsule_validator
        self.gate_evidence_validator = gate_evidence_validator
        self.transition_compatibility_validator = transition_compatibility_validator
        self.config = config
        self._mutation_lock = RLock()
        if snapshot.processed_event_ids and not snapshot.processed_events:
            values = snapshot.model_dump(mode="python")
            values["processed_events"] = tuple(
                ProcessedEventRecord(
                    event_id=event_id,
                    payload_sha256=_legacy_event_digest(event_id),
                )
                for event_id in snapshot.processed_event_ids
            )
            snapshot = EvolutionSnapshot.model_validate(values)
        self.snapshot = snapshot

    @classmethod
    def initialize(
        cls,
        *,
        store: EvolutionStore,
        capsule_validator: CapsuleValidator,
        gate_evidence_validator: GateEvidenceValidator | None = None,
        transition_compatibility_validator: TransitionCompatibilityValidator = (
            validate_transition_compatibility
        ),
        config: EvolutionConfig,
        deployment_id: str,
        champion: CapsuleReference,
        seed: int,
        now_ms: int,
    ) -> EvolutionController:
        if config.execution_target is ExecutionTarget.EXTERNAL and gate_evidence_validator is None:
            raise EvolutionError("external evolution requires a trusted gate-evidence validator")
        if store.exists:
            raise EvolutionError("refusing to overwrite an existing evolution state")
        if seed < 0 or now_ms < 0:
            raise EvolutionError("seed and initialization time must be non-negative")
        champion_report = capsule_validator(Path(champion.path))
        if not cls._capsule_reference_is_eligible(
            champion,
            champion_report,
            require_level_five=config.execution_target is ExecutionTarget.EXTERNAL,
        ):
            raise EvolutionError("initial champion must be an independently validated capsule")
        audit = EvolutionAuditRecord(
            sequence=0,
            event_id="controller-initialized",
            observed_at_ms=now_ms,
            action="initialize",
            phase_before=EvolutionPhase.IDLE,
            phase_after=EvolutionPhase.IDLE,
            reason="independently validated champion loaded; production mutation remains disabled",
        )
        initialization_digest = _event_digest(
            "initialize",
            {
                "deployment_id": deployment_id,
                "champion": champion,
                "seed": seed,
                "now_ms": now_ms,
            },
        )
        snapshot = EvolutionSnapshot(
            deployment_id=deployment_id,
            seed=seed,
            sequence=0,
            phase=EvolutionPhase.IDLE,
            champion=champion,
            retained_capsules=(champion,),
            processed_event_ids=("controller-initialized",),
            processed_events=(
                ProcessedEventRecord(
                    event_id="controller-initialized",
                    payload_sha256=initialization_digest,
                ),
            ),
            audit=(audit,),
            last_observed_at_ms=now_ms,
        )
        store.save(snapshot, expected_sequence=None)
        return cls(
            store=store,
            capsule_validator=capsule_validator,
            gate_evidence_validator=gate_evidence_validator,
            transition_compatibility_validator=transition_compatibility_validator,
            config=config,
            snapshot=snapshot,
        )

    @classmethod
    def restore(
        cls,
        *,
        store: EvolutionStore,
        capsule_validator: CapsuleValidator,
        gate_evidence_validator: GateEvidenceValidator | None = None,
        transition_compatibility_validator: TransitionCompatibilityValidator = (
            validate_transition_compatibility
        ),
        config: EvolutionConfig,
    ) -> EvolutionController:
        if config.execution_target is ExecutionTarget.EXTERNAL and gate_evidence_validator is None:
            raise EvolutionError("external evolution requires a trusted gate-evidence validator")
        snapshot = store.load()
        if len(snapshot.challengers) > config.maximum_challengers:
            raise EvolutionError("persisted challenger population exceeds the configured bound")
        if len(snapshot.active_streams) > config.maximum_active_streams:
            raise EvolutionError("persisted stream registry exceeds the configured bound")
        champion_report = capsule_validator(Path(snapshot.champion.path))
        if not cls._capsule_reference_is_eligible(
            snapshot.champion,
            champion_report,
            require_level_five=config.execution_target is ExecutionTarget.EXTERNAL,
        ):
            raise EvolutionError("restored champion failed independent capsule revalidation")
        if snapshot.selected_candidate_id is not None and snapshot.phase in {
            EvolutionPhase.SHADOWING,
            EvolutionPhase.SHADOW_VALIDATED,
            EvolutionPhase.CANARYING,
            EvolutionPhase.READY_TO_PROMOTE,
        }:
            selected = next(
                (
                    record
                    for record in snapshot.challengers
                    if record.spec.candidate_id == snapshot.selected_candidate_id
                ),
                None,
            )
            if selected is None:
                raise EvolutionError("restored controller selected challenger is missing")
            challenger_report = capsule_validator(Path(selected.spec.capsule.path))
            if not cls._capsule_reference_is_eligible(
                selected.spec.capsule,
                challenger_report,
                require_level_five=config.execution_target is ExecutionTarget.EXTERNAL,
            ):
                raise EvolutionError("restored challenger failed independent capsule revalidation")
        references = cls._snapshot_capsule_references(snapshot)
        for lease in snapshot.active_streams:
            reference = references.get(lease.capsule_id)
            if reference is None:
                raise EvolutionError("restored active stream has no complete capsule reference")
            lease_report = capsule_validator(Path(reference.path))
            if not cls._capsule_reference_is_eligible(
                reference,
                lease_report,
                require_level_five=config.execution_target is ExecutionTarget.EXTERNAL,
            ):
                raise EvolutionError("restored active-stream capsule failed revalidation")
        return cls(
            store=store,
            capsule_validator=capsule_validator,
            gate_evidence_validator=gate_evidence_validator,
            transition_compatibility_validator=transition_compatibility_validator,
            config=config,
            snapshot=snapshot,
        )

    @staticmethod
    def _snapshot_capsule_references(
        snapshot: EvolutionSnapshot,
    ) -> dict[str, CapsuleReference]:
        references = (
            snapshot.champion,
            *((snapshot.previous_champion,) if snapshot.previous_champion is not None else ()),
            *(item.spec.capsule for item in snapshot.challengers),
            *snapshot.retained_capsules,
        )
        by_id: dict[str, CapsuleReference] = {}
        for reference in references:
            existing = by_id.get(reference.capsule_id)
            if existing is not None and existing != reference:
                raise EvolutionError("capsule identifier resolves to conflicting references")
            by_id[reference.capsule_id] = reference
        return by_id

    def _already_processed(self, event_id: str, payload_sha256: str) -> bool:
        for record in self.snapshot.processed_events:
            if record.event_id != event_id:
                continue
            if record.payload_sha256 != payload_sha256:
                raise EvolutionError("idempotency key was reused with a different event payload")
            return True
        if event_id in self.snapshot.processed_event_ids:
            raise EvolutionError(
                "legacy idempotency key cannot be safely replayed without a payload digest"
            )
        return False

    def _commit(
        self,
        *,
        event_id: str,
        observed_at_ms: int,
        action: str,
        payload_sha256: str,
        reason: str,
        phase: EvolutionPhase | None = None,
        candidate_id: str | None = None,
        updates: dict[str, Any] | None = None,
    ) -> EvolutionSnapshot:
        if observed_at_ms < self.snapshot.last_observed_at_ms:
            raise EvolutionError("evolution events must have monotonic timestamps")
        next_phase = phase or self.snapshot.phase
        if len(self.snapshot.processed_events) >= self.config.maximum_processed_events:
            raise EvolutionError(
                "bounded idempotency registry is full; an audited checkpoint is required"
            )
        sequence = self.snapshot.sequence + 1
        audit = EvolutionAuditRecord(
            sequence=sequence,
            event_id=event_id,
            observed_at_ms=observed_at_ms,
            action=action,
            phase_before=self.snapshot.phase,
            phase_after=next_phase,
            reason=reason,
            candidate_id=candidate_id,
        )
        values: dict[str, Any] = {
            **self.snapshot.model_dump(mode="python"),
            "sequence": sequence,
            "phase": next_phase,
            "last_observed_at_ms": observed_at_ms,
            "processed_event_ids": (
                *self.snapshot.processed_event_ids,
                event_id,
            ),
            "processed_events": (
                *self.snapshot.processed_events,
                ProcessedEventRecord(
                    event_id=event_id,
                    payload_sha256=payload_sha256,
                ),
            ),
            "audit": (*self.snapshot.audit, audit)[-self.config.maximum_audit_records :],
        }
        if updates:
            values.update(updates)
        next_snapshot = EvolutionSnapshot.model_validate(values)
        if len(next_snapshot.challengers) > self.config.maximum_challengers:
            raise EvolutionError("challenger population exceeds its configured bound")
        if len(next_snapshot.active_streams) > self.config.maximum_active_streams:
            raise EvolutionError("active stream registry exceeds its configured bound")
        if len(next_snapshot.retained_capsules) > self.config.maximum_challengers + 2:
            raise EvolutionError("retained capsule registry exceeds its configured bound")
        self.store.save(next_snapshot, expected_sequence=self.snapshot.sequence)
        self.snapshot = next_snapshot
        return next_snapshot

    @_serialized
    def observe_trigger(self, observation: TriggerObservation) -> EvolutionSnapshot:
        payload_sha256 = _event_digest("observe_trigger", observation)
        if self._already_processed(observation.event_id, payload_sha256):
            return self.snapshot
        trigger = classify_trigger(observation, self.config)
        if trigger is None:
            return self._commit(
                event_id=observation.event_id,
                observed_at_ms=observation.observed_at_ms,
                action="observe_trigger",
                payload_sha256=payload_sha256,
                reason="observation did not cross an evolution threshold",
            )
        evolution_in_progress = self.snapshot.phase in {
            EvolutionPhase.CHALLENGER_READY,
            EvolutionPhase.SHADOWING,
            EvolutionPhase.SHADOW_VALIDATED,
            EvolutionPhase.CANARYING,
            EvolutionPhase.READY_TO_PROMOTE,
        }
        if evolution_in_progress:
            return self._commit(
                event_id=observation.event_id,
                observed_at_ms=observation.observed_at_ms,
                action="coalesce_trigger",
                payload_sha256=payload_sha256,
                reason=(
                    f"{trigger.value} observed while an existing challenger remains proof-gated"
                ),
            )
        return self._commit(
            event_id=observation.event_id,
            observed_at_ms=observation.observed_at_ms,
            action="begin_evolution",
            payload_sha256=payload_sha256,
            reason=f"{trigger.value}: {observation.detail}",
            phase=EvolutionPhase.EVOLVING,
            updates={"active_trigger": trigger, "selected_candidate_id": None},
        )

    @staticmethod
    def _capsule_reference_is_eligible(
        capsule: CapsuleReference,
        report: CapsuleValidationReport,
        *,
        require_level_five: bool = False,
    ) -> bool:
        eligible_for_target = (
            report.external_production_eligible
            if require_level_five
            else report.local_evolution_eligible
        )
        return (
            eligible_for_target
            and report.capsule_digest is not None
            and report.capsule_digest.value == capsule.capsule_digest
            and report.candidate_genome_hash is not None
            and report.candidate_genome_hash.value == capsule.genome_hash
        )

    def _capsule_is_eligible(self, spec: ChallengerSpec, report: CapsuleValidationReport) -> bool:
        return self._capsule_reference_is_eligible(
            spec.capsule,
            report,
            require_level_five=self.config.execution_target is ExecutionTarget.EXTERNAL,
        )

    def _gate_evidence_is_valid(
        self, observation: GateObservation, record: ChallengerRecord
    ) -> None:
        if observation.candidate_id != record.spec.candidate_id:
            raise EvolutionError("gate observation belongs to another candidate")
        if observation.capsule_digest != record.spec.capsule.capsule_digest:
            raise EvolutionError("gate observation belongs to another capsule")
        if observation.deterministic_seed != self.snapshot.seed:
            raise EvolutionError("gate observation seed does not match the controller seed")
        consumed = {
            item.evidence_digest
            for challenger in self.snapshot.challengers
            for item in (challenger.shadow_observation, challenger.canary_observation)
            if item is not None
        }
        if observation.evidence_digest in consumed:
            raise EvolutionError("gate evidence digest has already been consumed")
        if self.config.execution_target is ExecutionTarget.SIMULATED:
            return
        if (
            self.config.execution_target is ExecutionTarget.EXTERNAL
            and observation.verification_level is not VerificationLevel.HARDWARE_OPERATIONAL
        ):
            raise EvolutionError("external gate observations require Level-5 evidence")
        if self.gate_evidence_validator is None or not self.gate_evidence_validator(
            observation, record.spec, self.snapshot.champion
        ):
            raise EvolutionError(
                f"{self.config.execution_target.value} gate evidence failed trusted validation"
            )

    def _revalidate_recorded_gates(self, record: ChallengerRecord) -> None:
        if self.config.execution_target is ExecutionTarget.SIMULATED:
            return
        if self.gate_evidence_validator is None:
            raise EvolutionError("promotion requires a trusted gate-evidence validator")
        observations = (record.shadow_observation, record.canary_observation)
        if any(item is None for item in observations):
            raise EvolutionError("promotion requires persisted shadow and canary observations")
        for observation in observations:
            assert observation is not None
            if not self.gate_evidence_validator(observation, record.spec, self.snapshot.champion):
                raise EvolutionError(
                    f"persisted {observation.stage.value} evidence failed trusted revalidation"
                )

    @_serialized
    def register_challenger(
        self, spec: ChallengerSpec, *, event_id: str, observed_at_ms: int
    ) -> EvolutionSnapshot:
        payload_sha256 = _event_digest(
            "register_challenger",
            {"spec": spec, "event_id": event_id, "observed_at_ms": observed_at_ms},
        )
        if self._already_processed(event_id, payload_sha256):
            return self.snapshot
        if self.snapshot.phase not in {EvolutionPhase.EVOLVING, EvolutionPhase.ROLLED_BACK}:
            raise EvolutionError("challengers may be registered only for an active evolution")
        if any(item.spec.candidate_id == spec.candidate_id for item in self.snapshot.challengers):
            raise EvolutionError("challenger identifier already exists")
        report = self.capsule_validator(Path(spec.capsule.path))
        eligible = self._capsule_is_eligible(spec, report)
        record = ChallengerRecord(
            spec=spec,
            status=(
                ChallengerStatus.CAPSULE_VALIDATED
                if eligible
                else ChallengerStatus.CAPSULE_REJECTED
            ),
            capsule_validation=report,
            rejection_reason=None
            if eligible
            else "independent capsule validation rejected promotion",
        )
        return self._commit(
            event_id=event_id,
            observed_at_ms=observed_at_ms,
            action="register_challenger",
            payload_sha256=payload_sha256,
            reason=(
                "challenger remains isolated and passed capsule validation"
                if eligible
                else "challenger remains isolated and was rejected by capsule validation"
            ),
            phase=EvolutionPhase.CHALLENGER_READY if eligible else EvolutionPhase.EVOLVING,
            candidate_id=spec.candidate_id,
            updates={
                "challengers": (*self.snapshot.challengers, record),
                "selected_candidate_id": spec.candidate_id if eligible else None,
            },
        )

    def _selected(self) -> tuple[int, ChallengerRecord]:
        selected = self.snapshot.selected_candidate_id
        for index, record in enumerate(self.snapshot.challengers):
            if record.spec.candidate_id == selected:
                return index, record
        raise EvolutionError("no selected challenger exists")

    def _replace_challenger(
        self, index: int, replacement: ChallengerRecord
    ) -> tuple[ChallengerRecord, ...]:
        records = list(self.snapshot.challengers)
        records[index] = replacement
        return tuple(records)

    @_serialized
    def begin_shadow(self, *, event_id: str, observed_at_ms: int) -> EvolutionSnapshot:
        payload_sha256 = _event_digest(
            "begin_shadow", {"event_id": event_id, "observed_at_ms": observed_at_ms}
        )
        if self._already_processed(event_id, payload_sha256):
            return self.snapshot
        if self.snapshot.phase is not EvolutionPhase.CHALLENGER_READY:
            raise EvolutionError("shadowing requires a capsule-validated challenger")
        index, record = self._selected()
        report = self.capsule_validator(Path(record.spec.capsule.path))
        if not self._capsule_is_eligible(record.spec, report):
            raise EvolutionError("challenger failed capsule revalidation before shadowing")
        replacement = ChallengerRecord.model_validate(
            {
                **record.model_dump(mode="python"),
                "status": ChallengerStatus.SHADOWING,
                "capsule_validation": report,
            }
        )
        return self._commit(
            event_id=event_id,
            observed_at_ms=observed_at_ms,
            action="begin_shadow",
            payload_sha256=payload_sha256,
            reason="deterministic replay is routed to the isolated challenger",
            phase=EvolutionPhase.SHADOWING,
            candidate_id=record.spec.candidate_id,
            updates={"challengers": self._replace_challenger(index, replacement)},
        )

    def _gate_passes(self, observation: GateObservation) -> bool:
        minimum = (
            self.config.minimum_shadow_samples
            if observation.stage is GateStage.SHADOW
            else self.config.minimum_canary_samples
        )
        return (
            observation.sample_count >= minimum
            and observation.error_rate <= self.config.maximum_error_rate
            and observation.p95_ttft_ratio <= self.config.maximum_p95_ttft_ratio
            and observation.p99_tpot_ratio <= self.config.maximum_p99_tpot_ratio
            and observation.quality_regression <= self.config.maximum_quality_regression
            and observation.interrupted_streams == 0
        )

    @_serialized
    def record_gate(self, observation: GateObservation) -> EvolutionSnapshot:
        payload_sha256 = _event_digest("record_gate", observation)
        if self._already_processed(observation.event_id, payload_sha256):
            return self.snapshot
        expected_phase = {
            GateStage.SHADOW: EvolutionPhase.SHADOWING,
            GateStage.CANARY: EvolutionPhase.CANARYING,
        }.get(observation.stage)
        if expected_phase is None or self.snapshot.phase is not expected_phase:
            raise EvolutionError("gate observation does not match the controller phase")
        index, record = self._selected()
        self._gate_evidence_is_valid(observation, record)
        passed = self._gate_passes(observation)
        if observation.stage is GateStage.SHADOW:
            status = (
                ChallengerStatus.SHADOW_VALIDATED if passed else ChallengerStatus.SHADOW_REJECTED
            )
            phase = EvolutionPhase.SHADOW_VALIDATED if passed else EvolutionPhase.EVOLVING
            fields = {"shadow_observation": observation}
        else:
            status = (
                ChallengerStatus.CANARY_VALIDATED if passed else ChallengerStatus.CANARY_REJECTED
            )
            phase = EvolutionPhase.READY_TO_PROMOTE if passed else EvolutionPhase.EVOLVING
            fields = {"canary_observation": observation}
        replacement = ChallengerRecord.model_validate(
            {
                **record.model_dump(mode="python"),
                **fields,
                "status": status,
                "rejection_reason": None if passed else f"{observation.stage.value} gate failed",
            }
        )
        retained = self.snapshot.retained_capsule_ids
        if not passed and any(
            stream.capsule_id == record.spec.capsule.capsule_id
            for stream in self.snapshot.active_streams
        ):
            retained = tuple(dict.fromkeys((*retained, record.spec.capsule.capsule_id)))
        return self._commit(
            event_id=observation.event_id,
            observed_at_ms=observation.observed_at_ms,
            action=f"record_{observation.stage.value}_gate",
            payload_sha256=payload_sha256,
            reason=(
                f"{observation.stage.value} evidence satisfied all promotion criteria"
                if passed
                else f"{observation.stage.value} evidence failed a promotion criterion"
            ),
            phase=phase,
            candidate_id=record.spec.candidate_id,
            updates={
                "challengers": self._replace_challenger(index, replacement),
                "selected_candidate_id": record.spec.candidate_id if passed else None,
                "retained_capsule_ids": retained,
            },
        )

    def _require_live_opt_in(self) -> None:
        if not self.config.live_traffic:
            return
        enabled = os.environ.get(_LIVE_PROMOTION_ENV, "").lower() in {"1", "true", "yes"}
        if not self.config.live_promotion_authorized or not enabled:
            raise EvolutionError(
                f"live traffic requires config authorization and {_LIVE_PROMOTION_ENV}=1"
            )

    @_serialized
    def begin_canary(self, *, event_id: str, observed_at_ms: int) -> EvolutionSnapshot:
        payload_sha256 = _event_digest(
            "begin_canary", {"event_id": event_id, "observed_at_ms": observed_at_ms}
        )
        if self._already_processed(event_id, payload_sha256):
            return self.snapshot
        if self.snapshot.phase is not EvolutionPhase.SHADOW_VALIDATED:
            raise EvolutionError("canarying requires accepted shadow evidence")
        self._require_live_opt_in()
        index, record = self._selected()
        report = self.capsule_validator(Path(record.spec.capsule.path))
        if not self._capsule_is_eligible(record.spec, report):
            raise EvolutionError("challenger failed capsule revalidation before canarying")
        replacement = ChallengerRecord.model_validate(
            {
                **record.model_dump(mode="python"),
                "status": ChallengerStatus.CANARYING,
                "capsule_validation": report,
            }
        )
        return self._commit(
            event_id=event_id,
            observed_at_ms=observed_at_ms,
            action="begin_canary",
            payload_sha256=payload_sha256,
            reason="bounded canary routing enabled after shadow acceptance",
            phase=EvolutionPhase.CANARYING,
            candidate_id=record.spec.candidate_id,
            updates={"challengers": self._replace_challenger(index, replacement)},
        )

    def _route_request_locked(self, request_id: str) -> CapsuleReference:
        existing = next(
            (item for item in self.snapshot.active_streams if item.stream_id == request_id), None
        )
        if existing is not None:
            reference = self._snapshot_capsule_references(self.snapshot).get(existing.capsule_id)
            if reference is None:
                raise EvolutionError("active stream has no complete capsule reference")
            return reference
        if self.snapshot.phase is not EvolutionPhase.CANARYING:
            return self.snapshot.champion
        _, challenger = self._selected()
        digest = hashlib.sha256(f"{self.snapshot.seed}:{request_id}".encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / float(2**64)
        return (
            challenger.spec.capsule
            if fraction < self.config.canary_fraction
            else self.snapshot.champion
        )

    def route_request(self, request_id: str) -> CapsuleReference:
        """Route unary work or recover the immutable assignment of an open stream.

        Streaming callers must use :meth:`open_stream_and_route` before starting
        execution; separate route-then-open calls cannot establish a lease.
        """

        with self._mutation_lock:
            return self._route_request_locked(request_id)

    def _open_stream_locked(
        self,
        stream_id: str,
        *,
        event_id: str,
        observed_at_ms: int,
        externally_visible_output: bool = False,
    ) -> tuple[EvolutionSnapshot, CapsuleReference]:
        payload_sha256 = _event_digest(
            "open_stream",
            {
                "stream_id": stream_id,
                "event_id": event_id,
                "observed_at_ms": observed_at_ms,
                "externally_visible_output": externally_visible_output,
            },
        )
        if self._already_processed(event_id, payload_sha256):
            existing = next(
                (item for item in self.snapshot.active_streams if item.stream_id == stream_id), None
            )
            if existing is None:
                raise EvolutionError("closed stream assignment cannot be replayed as active")
            reference = self._snapshot_capsule_references(self.snapshot).get(existing.capsule_id)
            if reference is None:
                raise EvolutionError("replayed stream has no complete capsule reference")
            return self.snapshot, reference
        if any(stream.stream_id == stream_id for stream in self.snapshot.active_streams):
            raise EvolutionError("stream identifier is already active")
        capsule = self._route_request_locked(stream_id)
        lease = StreamLease(
            stream_id=stream_id,
            capsule_id=capsule.capsule_id,
            opened_sequence=self.snapshot.sequence + 1,
            externally_visible_output=externally_visible_output,
        )
        snapshot = self._commit(
            event_id=event_id,
            observed_at_ms=observed_at_ms,
            action="open_stream",
            payload_sha256=payload_sha256,
            reason=f"stream is pinned to capsule {capsule.capsule_id} until completion",
            updates={"active_streams": (*self.snapshot.active_streams, lease)},
        )
        return snapshot, capsule

    def open_stream_and_route(
        self,
        stream_id: str,
        *,
        event_id: str,
        observed_at_ms: int,
        externally_visible_output: bool = False,
    ) -> tuple[EvolutionSnapshot, CapsuleReference]:
        """Atomically persist a stream lease and return its assigned capsule."""

        with self._mutation_lock:
            return self._open_stream_locked(
                stream_id,
                event_id=event_id,
                observed_at_ms=observed_at_ms,
                externally_visible_output=externally_visible_output,
            )

    def open_stream(
        self,
        stream_id: str,
        *,
        event_id: str,
        observed_at_ms: int,
        externally_visible_output: bool = False,
    ) -> EvolutionSnapshot:
        """Compatibility wrapper; new streaming integrations use the atomic API."""

        snapshot, _capsule = self.open_stream_and_route(
            stream_id,
            event_id=event_id,
            observed_at_ms=observed_at_ms,
            externally_visible_output=externally_visible_output,
        )
        return snapshot

    @_serialized
    def close_stream(
        self, stream_id: str, *, event_id: str, observed_at_ms: int
    ) -> EvolutionSnapshot:
        payload_sha256 = _event_digest(
            "close_stream",
            {"stream_id": stream_id, "event_id": event_id, "observed_at_ms": observed_at_ms},
        )
        if self._already_processed(event_id, payload_sha256):
            return self.snapshot
        if not any(stream.stream_id == stream_id for stream in self.snapshot.active_streams):
            raise EvolutionError("cannot close an unknown stream")
        active = tuple(
            stream for stream in self.snapshot.active_streams if stream.stream_id != stream_id
        )
        leased = {stream.capsule_id for stream in active}
        rollback_capsule = (
            self.snapshot.previous_champion.capsule_id
            if self.snapshot.previous_champion is not None
            else None
        )
        retained = tuple(
            capsule_id
            for capsule_id in self.snapshot.retained_capsule_ids
            if capsule_id in leased or capsule_id == rollback_capsule
        )
        retained_references = tuple(
            reference
            for reference in self.snapshot.retained_capsules
            if reference.capsule_id in retained
            or reference.capsule_id == self.snapshot.champion.capsule_id
            or reference.capsule_id == rollback_capsule
        )
        return self._commit(
            event_id=event_id,
            observed_at_ms=observed_at_ms,
            action="close_stream",
            payload_sha256=payload_sha256,
            reason="stream lease released without interrupting visible output",
            updates={
                "active_streams": active,
                "retained_capsule_ids": retained,
                "retained_capsules": retained_references,
            },
        )

    @_serialized
    def promote(self, *, event_id: str, observed_at_ms: int) -> EvolutionSnapshot:
        payload_sha256 = _event_digest(
            "promote", {"event_id": event_id, "observed_at_ms": observed_at_ms}
        )
        if self._already_processed(event_id, payload_sha256):
            return self.snapshot
        if self.snapshot.phase is not EvolutionPhase.READY_TO_PROMOTE:
            raise EvolutionError("promotion requires accepted shadow and canary gates")
        self._require_live_opt_in()
        index, record = self._selected()
        if record.spec.transition_category is TransitionCategory.OPERATOR_REQUIRED:
            raise EvolutionError("operator-required transitions cannot be autonomously promoted")
        report = self.capsule_validator(Path(record.spec.capsule.path))
        if not self._capsule_is_eligible(record.spec, report):
            replacement = ChallengerRecord.model_validate(
                {
                    **record.model_dump(mode="python"),
                    "status": ChallengerStatus.CAPSULE_REJECTED,
                    "capsule_validation": report,
                    "rejection_reason": "capsule revalidation failed immediately before promotion",
                }
            )
            return self._commit(
                event_id=event_id,
                observed_at_ms=observed_at_ms,
                action="reject_promotion",
                payload_sha256=payload_sha256,
                reason="capsule revalidation failed immediately before promotion",
                phase=EvolutionPhase.EVOLVING,
                candidate_id=record.spec.candidate_id,
                updates={
                    "challengers": self._replace_challenger(index, replacement),
                    "selected_candidate_id": None,
                },
            )
        self._revalidate_recorded_gates(record)
        champion_report = self.capsule_validator(Path(self.snapshot.champion.path))
        if not self._capsule_reference_is_eligible(
            self.snapshot.champion,
            champion_report,
            require_level_five=self.config.execution_target is ExecutionTarget.EXTERNAL,
        ):
            raise EvolutionError("champion failed independent revalidation before transition")
        active = self.snapshot.active_streams
        if active:
            try:
                compatibility = self.transition_compatibility_validator(
                    self.snapshot.champion,
                    record.spec.capsule,
                    record.spec.transition_category,
                )
            except (OSError, ValueError) as error:
                raise EvolutionError(
                    f"trusted transition compatibility validation failed: {error}"
                ) from error
            if not compatibility.compatible:
                raise EvolutionError(
                    f"transition cannot preserve active streams; drain is required: "
                    f"{compatibility.reason}"
                )
        replacement = ChallengerRecord.model_validate(
            {
                **record.model_dump(mode="python"),
                "status": ChallengerStatus.PROMOTED,
                "capsule_validation": report,
            }
        )
        retained = tuple(
            dict.fromkeys((*self.snapshot.retained_capsule_ids, self.snapshot.champion.capsule_id))
        )
        retained_references = tuple(
            {
                item.capsule_id: item
                for item in (
                    *self.snapshot.retained_capsules,
                    self.snapshot.champion,
                    record.spec.capsule,
                )
            }.values()
        )
        return self._commit(
            event_id=event_id,
            observed_at_ms=observed_at_ms,
            action="promote",
            payload_sha256=payload_sha256,
            reason=(
                "capsule revalidated; new requests use the challenger while existing stream leases "
                "remain pinned"
            ),
            phase=EvolutionPhase.PROMOTED,
            candidate_id=record.spec.candidate_id,
            updates={
                "champion": record.spec.capsule,
                "previous_champion": self.snapshot.champion,
                "challengers": self._replace_challenger(index, replacement),
                "retained_capsule_ids": retained,
                "retained_capsules": retained_references,
            },
        )

    @_serialized
    def rollback(self, *, event_id: str, observed_at_ms: int, reason: str) -> EvolutionSnapshot:
        payload_sha256 = _event_digest(
            "rollback",
            {"event_id": event_id, "observed_at_ms": observed_at_ms, "reason": reason},
        )
        if self._already_processed(event_id, payload_sha256):
            return self.snapshot
        if self.snapshot.phase is not EvolutionPhase.PROMOTED:
            raise EvolutionError("rollback is available only after promotion")
        previous = self.snapshot.previous_champion
        if previous is None:
            raise EvolutionError("rollback champion is unavailable")
        previous_report = self.capsule_validator(Path(previous.path))
        if not self._capsule_reference_is_eligible(
            previous,
            previous_report,
            require_level_five=self.config.execution_target is ExecutionTarget.EXTERNAL,
        ):
            raise EvolutionError("rollback champion failed independent capsule revalidation")
        current = self.snapshot.champion
        records = []
        for record in self.snapshot.challengers:
            if record.spec.capsule.capsule_id == current.capsule_id:
                record = ChallengerRecord.model_validate(
                    {**record.model_dump(mode="python"), "status": ChallengerStatus.ROLLED_BACK}
                )
            records.append(record)
        leased = {stream.capsule_id for stream in self.snapshot.active_streams}
        retained = tuple(
            dict.fromkeys(
                (
                    *(
                        item
                        for item in self.snapshot.retained_capsule_ids
                        if item != previous.capsule_id
                    ),
                    *((current.capsule_id,) if current.capsule_id in leased else ()),
                )
            )
        )
        retained_references = tuple(
            {
                item.capsule_id: item
                for item in (*self.snapshot.retained_capsules, previous, current)
                if item.capsule_id in {*retained, previous.capsule_id, current.capsule_id}
            }.values()
        )
        return self._commit(
            event_id=event_id,
            observed_at_ms=observed_at_ms,
            action="rollback",
            payload_sha256=payload_sha256,
            reason=reason,
            phase=EvolutionPhase.ROLLED_BACK,
            updates={
                "champion": previous,
                "previous_champion": current,
                "challengers": tuple(records),
                "retained_capsule_ids": retained,
                "retained_capsules": retained_references,
                "selected_candidate_id": None,
            },
        )
