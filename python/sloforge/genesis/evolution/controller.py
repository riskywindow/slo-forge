"""Deterministic proof-gated champion/challenger evolution controller."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..capsule.models import CapsuleValidationReport
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
    GateObservation,
    GateStage,
    StreamLease,
    TransitionCategory,
    TriggerObservation,
)
from .store import EvolutionStore

CapsuleValidator = Callable[[Path], CapsuleValidationReport]
_LIVE_PROMOTION_ENV = "SLOFORGE_GENESIS_ALLOW_LIVE_PROMOTION"


class EvolutionError(RuntimeError):
    """Raised when a state transition violates an evolution safety guard."""


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
        config: EvolutionConfig,
        snapshot: EvolutionSnapshot,
    ) -> None:
        self.store = store
        self.capsule_validator = capsule_validator
        self.config = config
        self.snapshot = snapshot

    @classmethod
    def initialize(
        cls,
        *,
        store: EvolutionStore,
        capsule_validator: CapsuleValidator,
        config: EvolutionConfig,
        deployment_id: str,
        champion: CapsuleReference,
        seed: int,
        now_ms: int,
    ) -> EvolutionController:
        if store.exists:
            raise EvolutionError("refusing to overwrite an existing evolution state")
        if seed < 0 or now_ms < 0:
            raise EvolutionError("seed and initialization time must be non-negative")
        audit = EvolutionAuditRecord(
            sequence=0,
            event_id="controller-initialized",
            observed_at_ms=now_ms,
            action="initialize",
            phase_before=EvolutionPhase.IDLE,
            phase_after=EvolutionPhase.IDLE,
            reason="trusted champion loaded; production mutation remains disabled",
        )
        snapshot = EvolutionSnapshot(
            deployment_id=deployment_id,
            seed=seed,
            sequence=0,
            phase=EvolutionPhase.IDLE,
            champion=champion,
            processed_event_ids=("controller-initialized",),
            audit=(audit,),
            last_observed_at_ms=now_ms,
        )
        store.save(snapshot)
        return cls(
            store=store,
            capsule_validator=capsule_validator,
            config=config,
            snapshot=snapshot,
        )

    @classmethod
    def restore(
        cls,
        *,
        store: EvolutionStore,
        capsule_validator: CapsuleValidator,
        config: EvolutionConfig,
    ) -> EvolutionController:
        snapshot = store.load()
        if len(snapshot.challengers) > config.maximum_challengers:
            raise EvolutionError("persisted challenger population exceeds the configured bound")
        if len(snapshot.active_streams) > config.maximum_active_streams:
            raise EvolutionError("persisted stream registry exceeds the configured bound")
        return cls(
            store=store,
            capsule_validator=capsule_validator,
            config=config,
            snapshot=snapshot,
        )

    def _already_processed(self, event_id: str) -> bool:
        return event_id in self.snapshot.processed_event_ids

    def _commit(
        self,
        *,
        event_id: str,
        observed_at_ms: int,
        action: str,
        reason: str,
        phase: EvolutionPhase | None = None,
        candidate_id: str | None = None,
        updates: dict[str, Any] | None = None,
    ) -> EvolutionSnapshot:
        if observed_at_ms < self.snapshot.last_observed_at_ms:
            raise EvolutionError("evolution events must have monotonic timestamps")
        next_phase = phase or self.snapshot.phase
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
            )[-self.config.maximum_processed_events :],
            "audit": (*self.snapshot.audit, audit)[-self.config.maximum_audit_records :],
        }
        if updates:
            values.update(updates)
        next_snapshot = EvolutionSnapshot.model_validate(values)
        if len(next_snapshot.challengers) > self.config.maximum_challengers:
            raise EvolutionError("challenger population exceeds its configured bound")
        if len(next_snapshot.active_streams) > self.config.maximum_active_streams:
            raise EvolutionError("active stream registry exceeds its configured bound")
        self.store.save(next_snapshot)
        self.snapshot = next_snapshot
        return next_snapshot

    def observe_trigger(self, observation: TriggerObservation) -> EvolutionSnapshot:
        if self._already_processed(observation.event_id):
            return self.snapshot
        trigger = classify_trigger(observation, self.config)
        if trigger is None:
            return self._commit(
                event_id=observation.event_id,
                observed_at_ms=observation.observed_at_ms,
                action="observe_trigger",
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
                reason=(
                    f"{trigger.value} observed while an existing challenger remains proof-gated"
                ),
            )
        return self._commit(
            event_id=observation.event_id,
            observed_at_ms=observation.observed_at_ms,
            action="begin_evolution",
            reason=f"{trigger.value}: {observation.detail}",
            phase=EvolutionPhase.EVOLVING,
            updates={"active_trigger": trigger, "selected_candidate_id": None},
        )

    @staticmethod
    def _capsule_is_eligible(spec: ChallengerSpec, report: CapsuleValidationReport) -> bool:
        return (
            report.promotion_eligible
            and report.capsule_digest is not None
            and report.capsule_digest.value == spec.capsule.capsule_digest
        )

    def register_challenger(
        self, spec: ChallengerSpec, *, event_id: str, observed_at_ms: int
    ) -> EvolutionSnapshot:
        if self._already_processed(event_id):
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

    def begin_shadow(self, *, event_id: str, observed_at_ms: int) -> EvolutionSnapshot:
        if self._already_processed(event_id):
            return self.snapshot
        if self.snapshot.phase is not EvolutionPhase.CHALLENGER_READY:
            raise EvolutionError("shadowing requires a capsule-validated challenger")
        index, record = self._selected()
        replacement = ChallengerRecord.model_validate(
            {**record.model_dump(mode="python"), "status": ChallengerStatus.SHADOWING}
        )
        return self._commit(
            event_id=event_id,
            observed_at_ms=observed_at_ms,
            action="begin_shadow",
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

    def record_gate(self, observation: GateObservation) -> EvolutionSnapshot:
        if self._already_processed(observation.event_id):
            return self.snapshot
        expected_phase = {
            GateStage.SHADOW: EvolutionPhase.SHADOWING,
            GateStage.CANARY: EvolutionPhase.CANARYING,
        }.get(observation.stage)
        if expected_phase is None or self.snapshot.phase is not expected_phase:
            raise EvolutionError("gate observation does not match the controller phase")
        index, record = self._selected()
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

    def begin_canary(self, *, event_id: str, observed_at_ms: int) -> EvolutionSnapshot:
        if self._already_processed(event_id):
            return self.snapshot
        if self.snapshot.phase is not EvolutionPhase.SHADOW_VALIDATED:
            raise EvolutionError("canarying requires accepted shadow evidence")
        self._require_live_opt_in()
        index, record = self._selected()
        replacement = ChallengerRecord.model_validate(
            {**record.model_dump(mode="python"), "status": ChallengerStatus.CANARYING}
        )
        return self._commit(
            event_id=event_id,
            observed_at_ms=observed_at_ms,
            action="begin_canary",
            reason="bounded canary routing enabled after shadow acceptance",
            phase=EvolutionPhase.CANARYING,
            candidate_id=record.spec.candidate_id,
            updates={"challengers": self._replace_challenger(index, replacement)},
        )

    def route_request(self, request_id: str) -> CapsuleReference:
        """Choose a runtime deterministically without mutating controller state."""

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

    def open_stream(
        self,
        stream_id: str,
        *,
        event_id: str,
        observed_at_ms: int,
        externally_visible_output: bool = False,
    ) -> EvolutionSnapshot:
        if self._already_processed(event_id):
            return self.snapshot
        if any(stream.stream_id == stream_id for stream in self.snapshot.active_streams):
            raise EvolutionError("stream identifier is already active")
        capsule = self.route_request(stream_id)
        lease = StreamLease(
            stream_id=stream_id,
            capsule_id=capsule.capsule_id,
            opened_sequence=self.snapshot.sequence + 1,
            externally_visible_output=externally_visible_output,
        )
        return self._commit(
            event_id=event_id,
            observed_at_ms=observed_at_ms,
            action="open_stream",
            reason=f"stream is pinned to capsule {capsule.capsule_id} until completion",
            updates={"active_streams": (*self.snapshot.active_streams, lease)},
        )

    def close_stream(
        self, stream_id: str, *, event_id: str, observed_at_ms: int
    ) -> EvolutionSnapshot:
        if self._already_processed(event_id):
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
        return self._commit(
            event_id=event_id,
            observed_at_ms=observed_at_ms,
            action="close_stream",
            reason="stream lease released without interrupting visible output",
            updates={"active_streams": active, "retained_capsule_ids": retained},
        )

    def promote(self, *, event_id: str, observed_at_ms: int) -> EvolutionSnapshot:
        if self._already_processed(event_id):
            return self.snapshot
        if self.snapshot.phase is not EvolutionPhase.READY_TO_PROMOTE:
            raise EvolutionError("promotion requires accepted shadow and canary gates")
        self._require_live_opt_in()
        index, record = self._selected()
        if record.spec.transition_category is TransitionCategory.OPERATOR_REQUIRED:
            raise EvolutionError("operator-required transitions cannot be autonomously promoted")
        active = self.snapshot.active_streams
        boundary_safe = record.spec.transition_category in {
            TransitionCategory.POLICY_ONLY_HOT_SWAP,
            TransitionCategory.REQUEST_BOUNDARY_SWAP,
        }
        if active and not (boundary_safe or record.spec.active_stream_compatible):
            raise EvolutionError("transition cannot preserve active streams; drain is required")
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
                reason="capsule revalidation failed immediately before promotion",
                phase=EvolutionPhase.EVOLVING,
                candidate_id=record.spec.candidate_id,
                updates={
                    "challengers": self._replace_challenger(index, replacement),
                    "selected_candidate_id": None,
                },
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
        return self._commit(
            event_id=event_id,
            observed_at_ms=observed_at_ms,
            action="promote",
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
            },
        )

    def rollback(self, *, event_id: str, observed_at_ms: int, reason: str) -> EvolutionSnapshot:
        if self._already_processed(event_id):
            return self.snapshot
        if self.snapshot.phase is not EvolutionPhase.PROMOTED:
            raise EvolutionError("rollback is available only after promotion")
        previous = self.snapshot.previous_champion
        if previous is None:
            raise EvolutionError("rollback champion is unavailable")
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
        return self._commit(
            event_id=event_id,
            observed_at_ms=observed_at_ms,
            action="rollback",
            reason=reason,
            phase=EvolutionPhase.ROLLED_BACK,
            updates={
                "champion": previous,
                "previous_champion": current,
                "challengers": tuple(records),
                "retained_capsule_ids": retained,
                "selected_candidate_id": None,
            },
        )
